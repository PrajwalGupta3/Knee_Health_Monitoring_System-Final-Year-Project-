"""
app.py — dotenv + Groq + per-user metadata + BMI + download final.csv

Notes:
 - Requires python-dotenv (pip install python-dotenv)
 - Requires requests (pip install requests)
 - Make sure your ML scripts and model files exist as before
"""

import os
import time
import json
import math
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import safe_join 
import re
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from joblib import load

# ---------- load .env ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)
# --------------------------------

# ========== CONFIG ============
PROJECT_ROOT = Path(__file__).resolve().parent

ML_DIR        = PROJECT_ROOT / "ML"
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
DATA_DIR      = PROJECT_ROOT / "data"
USER_DATA_DIR = DATA_DIR / "users"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Firebase vars from .env (app expects Firebase to be correctly configured elsewhere)
FIREBASE_CRED = Path(os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json"))
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL", "")

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_HOST = os.getenv("GROQ_API_HOST", "").rstrip("/")  # e.g. https://api.groq.com
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Polling config
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "3"))
POLL_TIMEOUT  = float(os.getenv("POLL_TIMEOUT", "300"))

# ML artifacts
MODEL_FILE         = ML_DIR / "logistic_model_impaired.joblib"
FEATURE_ORDER_FILE = ML_DIR / "feature_order_impaired.json"
THRESH_FILE        = ML_DIR / "thresholds_impaired.json"
# --- ADD THIS ---
PROFILES_FILE      = ML_DIR / "gait_model_profiles.json"

# load model + artifacts
clf = load(MODEL_FILE)
feature_order = json.load(open(FEATURE_ORDER_FILE))
thresholds = json.load(open(THRESH_FILE))
# --- ADD THIS ---
model_profiles = json.load(open(PROFILES_FILE))

# flask
app = Flask(__name__, static_folder=str(PROJECT_ROOT.parent / "web" / "static"))

# ---------- Helper: user storage ----------
def user_json_path(user_id):
    p = USER_DATA_DIR / f"{user_id}.json"
    return p

def load_user_record(user_id):
    p = user_json_path(user_id)
    if not p.exists():
        default = {"user_id": user_id, "profile": {}, "history": []}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(default, indent=2))
        return default
    return json.load(open(p))

def save_user_record(user_id, data):
    p = user_json_path(user_id)
    json.dump(data, open(p, "w"), indent=2)

# ---------- Utility: BMI + risk ----------
def compute_bmi_and_risk(profile):
    """
    profile: dict may contain height_cm and weight_kg
    returns: {bmi, bmi_category}
    """
    height_cm = profile.get("height_cm")
    weight_kg = profile.get("weight_kg")
    if height_cm is None or weight_kg is None:
        return {"bmi": None, "bmi_category": None}
    try:
        h_m = float(height_cm) / 100.0
        w = float(weight_kg)
        bmi = w / (h_m * h_m) if h_m > 0 else None
        if bmi is None:
            return {"bmi": None, "bmi_category": None}
        bmi = round(float(bmi), 2)
        # simple categories (WHO-ish)
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return {"bmi": bmi, "bmi_category": cat}
    except Exception:
        return {"bmi": None, "bmi_category": None}

# ---------- File serving helper (secure) ----------
def serve_user_file(user_id, relpath):
    """
    Serve a file from USER_DATA_DIR / user_id / relpath.
    """
    # 1. Construct the full absolute path
    user_base_dir = (USER_DATA_DIR / user_id).resolve()
    target_file = (user_base_dir / relpath).resolve()

    # 2. Security check: Ensure target is inside user_base_dir
    if not str(target_file).startswith(str(user_base_dir)):
        abort(403) # Forbidden

    # 3. Check existence
    if not target_file.exists():
        abort(404) # Not Found

    # 4. Send file
    # send_from_directory takes (directory, filename)
    return send_from_directory(str(target_file.parent), target_file.name)

# ---------- Routes ----------

@app.route("/login", methods=["POST"])
def login():
    """
    Accepts:
      { "user_id": "...", "profile": { optional profile fields } }
    If profile provided, save it.
    """
    data = request.json or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "missing user_id"}), 400

    record = load_user_record(user_id)
    profile = data.get("profile")
    if profile:
        # merge profile (only allowed keys)
        allowed = ["age", "height_cm", "weight_kg", "gender", "knee_side"]
        for k in allowed:
            if k in profile:
                record["profile"][k] = profile[k]
        save_user_record(user_id, record)

    return jsonify({"ok": True, "user_id": user_id, "profile": record.get("profile", {})})

@app.route("/update_profile", methods=["POST"])
def update_profile():
    """
    Body: { user_id:..., profile: {height_cm, weight_kg, age, gender, knee_side} }
    """
    data = request.json or {}
    user_id = data.get("user_id")
    profile = data.get("profile", {})
    if not user_id:
        return jsonify({"ok": False, "error": "missing user_id"}), 400
    record = load_user_record(user_id)
    record["profile"].update(profile)
    save_user_record(user_id, record)
    return jsonify({"ok": True, "profile": record["profile"]})

@app.route("/wait_for_new_reading", methods=["POST"])
def wait_for_new_reading():
    """
    Polls firebase for a new session timestamp (same behavior as before).
    Expects { last_seen_timestamp: <str|null> }
    """
    # keep same Firebase polling code as before - uses admin SDK; assume initialized elsewhere
    from firebase_admin import db
    last_ts = (request.json or {}).get("last_seen_timestamp")
    ref = db.reference("/sessions")
    start = time.time()
    while True:
        sessions = ref.get() or {}
        keys = sorted(sessions.keys())
        if last_ts is None:
            if keys:
                return jsonify({"ok": True, "new_timestamp": keys[-1]})
        else:
            newer = [k for k in keys if k > last_ts]
            if newer:
                return jsonify({"ok": True, "new_timestamp": newer[-1]})
        if time.time() - start > POLL_TIMEOUT:
            return jsonify({"ok": False, "error": "timeout"}), 408
        time.sleep(POLL_INTERVAL)

@app.route("/get_session_json", methods=["POST"])
def get_session_json():
    from firebase_admin import db
    timestamp = (request.json or {}).get("timestamp")
    if not timestamp:
        return jsonify({"ok": False, "error": "missing timestamp"}), 400
    ref = db.reference(f"/sessions/{timestamp}")
    sess = ref.get()
    if not sess:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "data": sess})

# Internal pipeline runner — same as before
def run_processing(raw_json, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw.json"
    json.dump({"sessions": {"single": raw_json}}, open(raw_path, "w"))
    # call json_to_csv.py
    rc1 = os.system(f"python {SCRIPTS_DIR/'json_to_csv.py'} --input {raw_path} --outdir {outdir}")
    if rc1 != 0:
        raise RuntimeError("json_to_csv.py failed")
    # call qc_and_plot.py
    rc2 = os.system(f"python {SCRIPTS_DIR/'qc_and_plot.py'} --input-dir {outdir}")
    if rc2 != 0:
        raise RuntimeError("qc_and_plot.py failed")
    final = outdir / "final.csv"
    if not final.exists():
        raise RuntimeError("final.csv not produced")
    return final

def classify_final_csv(final_csv, user_profile):
    df = pd.read_csv(final_csv)
    if df.empty:
        return {"ok": False, "error": "empty final.csv"}
    
    row = df.iloc[0]
    
    # 1. Build Model Input (Safely handle NaNs for the model)
    # We replace NaN/Inf with 0.0 so the classifier doesn't crash
    x_list = []
    for f in feature_order:
        val = row.get(f, 0.0)
        if pd.isna(val) or np.isinf(val):
            val = 0.0
        x_list.append(float(val))
        
    x = np.array(x_list).reshape(1, -1)
    pred = clf.predict(x)[0]
    probs = clf.predict_proba(x)[0]
    prob_map = {str(clf.classes_[i]): float(probs[i]) for i in range(len(probs))}
    
    # 2. Threshold Flags (Safely handle NaNs)
    flags = {}
    for f, th in thresholds.items():
        val = row.get(f, 0.0)
        val_safe = 0.0 if (pd.isna(val) or np.isinf(val)) else float(val)
        flags[f] = "HIGH" if val_safe > float(th) else "OK"
        
    # 3. BMI
    bmi_info = compute_bmi_and_risk(user_profile or {})
    
    # 4. CLEAN UP RAW FEATURES (The Fix)
    # Convert Pandas/Numpy NaNs to Python None (which becomes JSON null)
    clean_features = {}
    for k, v in row.to_dict().items():
        # Check if value is scalar NaN (using pandas isna handles None/NaN/NaT)
        if pd.isna(v): 
            clean_features[k] = None
        # Check specifically for float('nan') or infinite
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean_features[k] = None
        else:
            # If it's a numpy type, convert to native python type
            if isinstance(v, (np.integer, np.int64)):
                clean_features[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean_features[k] = float(v)
            else:
                clean_features[k] = v

    # Consolidated result
    res = {
        "ok": True,
        "prediction": str(pred),
        "probabilities": prob_map,
        "raw_features": clean_features, # <--- Now safe for JSON
        "threshold_flags": flags,
        "bmi": bmi_info
    }
    return res

@app.route("/process_session", methods=["POST"])
def process_session():
    data = request.json or {}
    user_id = data.get("user_id")
    session_json = data.get("session_data")
    
    if not user_id or not session_json:
        return jsonify({"ok": False, "error": "missing params"}), 400
        
    # 1. Setup Directories
    user_dir = USER_DATA_DIR / user_id / "sessions"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    sess_name = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = user_dir / sess_name
    
    try:
        # 2. Run Processing
        final_csv = run_processing(session_json, sess_dir)
        
        # 3. Load Profile & Classify
        record = load_user_record(user_id)
        profile = record.get("profile", {})
        result = classify_final_csv(final_csv, profile)
        
        # 4. FIND THE PLOT IMAGE
        # We look inside the 'plots' folder created by qc_and_plot.py
        plot_url = None
        plots_dir = sess_dir / "plots"
        if plots_dir.exists():
            # Find any .png file in that folder
            found_plots = list(plots_dir.glob("*.png"))
            if found_plots:
                # Get relative path for URL (e.g., sessions/2025.../plots/image.png)
                # IMPORTANT: Replace backslashes with forward slashes for URLs on Windows
                rel_plot_path = found_plots[0].relative_to(USER_DATA_DIR / user_id)
                plot_url = f"/user_file/{user_id}/{str(rel_plot_path).replace('\\', '/')}"

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to terminal for debugging
        return jsonify({"ok": False, "error": str(e)}), 500

    # 5. Save History
    record = load_user_record(user_id)
    
    # Store CSV path safely
    rel_csv_path = final_csv.relative_to(USER_DATA_DIR / user_id)
    
    hist_entry = {
        "session_id": sess_name, 
        "result": result, 
        "final_csv": str(rel_csv_path).replace('\\', '/'),
        "plot_url": plot_url  # <--- Saving the plot URL here
    }
    
    record.setdefault("history", []).append(hist_entry)
    record["history"] = record["history"][-5:] # Keep last 5
    save_user_record(user_id, record)
    
    # 6. Return Response
    download_url = f"/user_file/{user_id}/{str(rel_csv_path).replace('\\', '/')}"
    
    return jsonify({
        "ok": True, 
        "session_id": sess_name, 
        "result": result, 
        "download_csv": download_url,
        "plot_url": plot_url # <--- Sending it to frontend
    })

@app.route("/user_file/<user_id>/<path:relpath>")
def user_file(user_id, relpath):
    # serve file from USER_DATA_DIR/<user_id> safely
    return serve_user_file(user_id, relpath)

@app.route("/get_history", methods=["POST"])
def get_history():
    user_id = (request.json or {}).get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "missing user_id"}), 400
    record = load_user_record(user_id)
    return jsonify({"ok": True, "history": record.get("history", []), "profile": record.get("profile", {})})

# ---------- Chatbot — Groq integration (UPDATED) ----------
def call_groq(system_prompt, user_prompt, max_tokens=1200, temperature=0.2):
    """
    Updated Groq call using the OpenAI-compatible endpoint.
    Compatible with models like llama-3.3-70b-versatile.
    """
    if not GROQ_API_KEY:
        return {"ok": False, "error": "Groq API Key missing"}

    # 1. Use the standard OpenAI-compatible chat completions endpoint
    # (Even if GROQ_API_HOST is set in .env, we force the correct path here for safety)
    base_url = GROQ_API_HOST if GROQ_API_HOST else "https://api.groq.com/openai/v1"
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 2. Use the standard 'messages' format, not 'input'
    payload = {
        "model": GROQ_MODEL,  # e.g., 'llama-3.3-70b-versatile'
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        
        # 3. Parse the OpenAI-style response structure
        text = j["choices"][0]["message"]["content"]
        return {"ok": True, "text": text}
        
    except Exception as e:
        print(f"Groq Error: {e}")
        # Return the error message so the UI can display it (or the fallback)
        return {"ok": False, "error": str(e)}

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json or {}
    user_id = data.get("user_id")
    question = data.get("question", "").strip()
    
    if not user_id or not question:
        return jsonify({"ok": False, "error": "missing user_id or question"}), 400

    # 1. Security Sanitization
    question = question.replace("<", "&lt;").replace(">", "&gt;")
    if len(question) > 500: question = question[:500]
        
    record = load_user_record(user_id)
    last = record.get("history", [])[-1] if record.get("history") else None
    profile = record.get("profile", {})
    
    context_str = json.dumps(last) if last else "No recent session data available."
    thresholds_str = json.dumps(thresholds)

    # --- 2. NEW: Extract ALL Profile Centroids ---
    # We want Healthy, Recovery, Fatigued, Unstable averages.
    # We skip the heavy 'covariance' matrices to save token space.
    ref_profiles = {}
    raw_profiles = model_profiles.get("profiles", {})
    
    for p_name, p_data in raw_profiles.items():
        # Get the average values (centroid) for this profile type
        ref_profiles[p_name] = p_data.get("centroid", {})
        
    ref_profiles_str = json.dumps(ref_profiles)
    # ---------------------------------------------
    
    system_lines = [
        "### SYSTEM ROLE",
        "You are the 'Knee Monitor Assistant', a specialized biomechanics utility.",
        "Your goal is to interpret sensor data by comparing it against known gait profiles (Healthy, Recovery, Fatigued, Unstable) and offer non-medical management strategies.",
        "",
        "### INPUT DATA",
        "Analyze the following data blocks carefully:",
        f"<user_profile>{json.dumps(profile)}</user_profile>",
        f"<safety_thresholds>{thresholds_str}</safety_thresholds>",
        # --- NEW: PASSING ALL PROFILES ---
        f"<reference_profiles>{ref_profiles_str}</reference_profiles>", 
        # ---------------------------------
        f"<current_session>{context_str}</current_session>",
        "",
        "### CONTEXT NOTE",
        "The 'Impaired' classification in the session data usually corresponds to the 'Fatigued' or 'Unstable' profiles provided in <reference_profiles>.",
        "",
        "### GUARDRAILS & SAFETY (HIGHEST PRIORITY)",
        "1. SCOPE: Reject ANY questions not related to knee health, gait analysis, or the provided data.",
        "2. DANGER: If the user mentions severe pain, swelling, or self-harm, stop analysis and direct them to a doctor immediately.",
        "3. NO DIAGNOSIS: Never use terms like 'fracture', 'tear', 'cure', or 'diagnosis'. Use 'pattern', 'metric', or 'indication'.",
        "",
        "### RESPONSE GUIDELINES",
        "1. COMPARATIVE ANALYSIS (CRITICAL):",
        "   - Compare <current_session> metrics against the <reference_profiles>.",
        "   - DO NOT use general world knowledge for 'fast/slow'. Use the Reference Profiles.",
        "   - Example: 'Your pace (48) is higher than the typical Recovery pace (37) and even the Healthy pace (40).'",
        "   - CONTEXTUALIZE: Don't just list numbers. Use the format: '[Metric] is [Value], which is [High/Normal] (Limit: [Threshold]).'",
        "   - EXPLAIN: Briefly state what that implies (e.g., 'High acceleration suggests you are landing heavily').",
        "",
        "2. ADVICE LOGIC:",
        "   - IF explaining numbers, explain which Profile they resemble most.",
        "   - IF the user specifically asks for a 'cure' or 'treatment': State 'I cannot offer a medical cure,' then pivot to general wellness tips (rest, ice, gait adjustments).",
        "   - IF the user just asks for an explanation: Do NOT offer the 'cure' disclaimer. Just explain the numbers clearly.",
        "3. TONE: Weave disclaimers naturally into the flow. Be concise.",
        "",
        "### CHAIN OF THOUGHT INSTRUCTIONS",
        "Before answering, you MUST perform a step-by-step analysis inside <analysis> tags.",
        "Your analysis must follow this sequence:",
        "1. CHECK SCOPE: Is the query relevant?",
        "2. CHECK DANGER: Are there keywords of pain/swelling?",
        "3. CHECK DATA: Compare 'current_session' values vs 'safety_thresholds'. List exactly which metrics are out of range.",
        "4. FORMULATE STRATEGY: Select the correct advice based on the data.",
        "",
        "### FINAL INSTRUCTION",
        "After your analysis, provide the final response for the user. Start your final response with <answer> and end it with </answer>."
    ]
    
    sys_content_str = "\n".join(system_lines)

    # 2. DEFINE THE USER MESSAGE (The Trigger)
    # This is ONLY what the user actually typed, wrapped in tags for clarity.
    user_content_str = f"The user asks: <query>{question}</query>"

    groq_resp = call_groq(system_prompt=sys_content_str, user_prompt=user_content_str, max_tokens=1200, temperature=0.2)
    
    final_response_text = ""

    if groq_resp.get("ok"):
        raw_text = groq_resp["text"]
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)
        if answer_match:
            final_response_text = answer_match.group(1).strip()
        else:
            parts = raw_text.split("</analysis>")
            final_response_text = parts[-1].strip() if len(parts) > 1 else raw_text
            
        # Debug print
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', raw_text, re.DOTALL)
        if analysis_match:
            print(f"\n[BOT THOUGHT PROCESS]:\n{analysis_match.group(1).strip()}\n")
    else:
        final_response_text = "Assistant (offline): " + str(groq_resp.get("error"))

    return jsonify({"ok": True, "response": final_response_text})

# ---------- Serve static UI files via Flask (optional) ----------
WEB_DIR = PROJECT_ROOT.parent / "web"
@app.route("/", defaults={"path":"index.html"})
@app.route("/<path:path>")
def web_static(path):
    p = WEB_DIR / path
    if p.exists():
        return send_from_directory(str(WEB_DIR), path)
    else:
        # index fallback
        return send_from_directory(str(WEB_DIR), "index.html")

# ---------- Main ----------
if __name__ == "__main__":
    # Initialize Firebase admin after loading .env (if not initialized elsewhere)
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(FIREBASE_CRED))
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
            print("[INFO] Firebase initialized")
    except Exception as e:
        print("[WARN] Firebase init failed:", e)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
