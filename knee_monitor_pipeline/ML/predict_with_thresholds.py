
"""
Loads the saved logistic_model_impaired.joblib and thresholds_impaired.json, to
predict the labels for the input CSV. Uses gating:
  - require prob >= PROB_THRESH
  - require pooled Mahalanobis distance <= class threshold (from thresholds_impaired.json)
If either fails, final label = "UNCERTAIN".

Usage:
    python knee_monitor_pipeline/ML/predict_with_thresholds.py \
        --input knee_monitor_pipeline/ML/healthy_baseline_qc.csv \
        --output knee_monitor_pipeline/ML/predictions_with_gating.csv

    (the healthy_baseline_qc.csv here is used for testing purposes)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

#CONFIG - update if your paths differ
MODEL_FILE = Path("knee_monitor_pipeline/ML/logistic_model_impaired.joblib")
FEATURE_FILE = Path("knee_monitor_pipeline/ML/feature_order_impaired.json")
THRESHOLDS_FILE = Path("knee_monitor_pipeline/ML/thresholds_impaired.json")
GAIT_MODEL_FILE = Path("knee_monitor_pipeline/ML/gait_model_profiles.json")  #optional pooled cov
PROB_THRESH = 0.7
IMPAIRED_STRICT_FACTOR = 0.8  #multiply threshold for Impaired (for stricter classification)
#This strict classification factor is to reduce false negatives for Impaired, as we lack sufficient data to set proper thresholds yet

#Optional fallback training CSVs(only to be used for computing pooled cov if the gait_model_profiles.json lacks it)
TRAIN_FILES = [
    Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
]

EPS = 1e-6

def load_features_order(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_thresholds(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_pooled_cov_inv_from_gait_model(path, feature_order):
    if not path.exists():
        return None
    with open(path, 'r') as f:
        gm = json.load(f)
    pooled = gm.get('pooled', {})
    if not pooled:
        return None
    if 'cov_inv' in pooled:
        return np.array(pooled['cov_inv'], dtype=float)
    if 'cov' in pooled:
        cov = np.array(pooled['cov'], dtype=float)
        if cov.ndim == 0:
            cov = cov.reshape((1,1))
        cov = cov + np.eye(cov.shape[0]) * 1e-8
        return np.linalg.inv(cov)
    return None

def compute_pooled_cov_inv_from_training(files, feature_order):
    rows = []
    for p in files:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'qc_pass' in df.columns:
            df = df[df['qc_pass'] == True]
        if df.empty:
            continue
        #engineering same features
        if 'dom_freq_power' in df.columns:
            df['dom_freq_power_log'] = np.log1p(df['dom_freq_power'].astype(float))
        else:
            df['dom_freq_power_log'] = 0.0
        if 'gyro_mag_mean' in df.columns and 'acc_mag_rms' in df.columns:
            df['gyro_acc_ratio'] = df['gyro_mag_mean'].astype(float) / (df['acc_mag_rms'].astype(float) + EPS)
        else:
            df['gyro_acc_ratio'] = 0.0
        for _,row in df.iterrows():
            try:
                feat = [float(row[f]) for f in feature_order]
            except Exception:
                raise KeyError(f"Missing features in training file {p}. Required: {feature_order}")
            rows.append(feat)
    if len(rows) == 0:
        return None
    X = np.array(rows, dtype=float)
    cov = np.cov(X, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    return np.linalg.inv(cov)

def mahalanobis_distance(x, mu, cov_inv):
    d = x - mu
    return float(np.sqrt(float(d.T.dot(cov_inv).dot(d))))

def predict_file(input_csv, output_csv, prob_thresh=PROB_THRESH):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Run training first.")
    clf = joblib.load(MODEL_FILE)
    feature_order = load_features_order(FEATURE_FILE)
    thresholds = load_thresholds(THRESHOLDS_FILE)
    cov_inv = get_pooled_cov_inv_from_gait_model(GAIT_MODEL_FILE, feature_order)
    if cov_inv is None:
        cov_inv = compute_pooled_cov_inv_from_training(TRAIN_FILES, feature_order)
        if cov_inv is None:
            print("Could not compute pooled covariance inverse. Mahalanobis gating will be skipped.")

    #load the input
    df = pd.read_csv(input_csv)
    #engineer features(only if necessary)
    if 'dom_freq_power' in df.columns:
        df['dom_freq_power_log'] = np.log1p(df['dom_freq_power'].astype(float))
    else:
        df['dom_freq_power_log'] = 0.0
    if 'gyro_mag_mean' in df.columns and 'acc_mag_rms' in df.columns:
        df['gyro_acc_ratio'] = df['gyro_mag_mean'].astype(float) / (df['acc_mag_rms'].astype(float) + EPS)
    else:
        df['gyro_acc_ratio'] = 0.0

    results = []
    for _, row in df.iterrows():
        try:
            x = np.array([float(row[f]) for f in feature_order], dtype=float)
        except Exception as e:
            raise KeyError(f"Missing input feature for prediction: {e}")
        probs = clf.predict_proba(x.reshape(1,-1))[0]
        classes = clf.classes_
        idx = int(np.argmax(probs))
        pred = classes[idx]
        max_prob = float(probs[idx])
        #compute Mahalanobis to predicted centroid(if possible)
        mah = None
        if cov_inv is not None:
            #centroid: take class centroid from training files? try gait_model_profiles first
            centroid = None
            if GAIT_MODEL_FILE.exists():
                with open(GAIT_MODEL_FILE,'r') as f:
                    gm = json.load(f)
                profiles = gm.get('profiles', {})
                #merged model may not be in gait_model_profiles.json; attempt to retrieve centroid keys
                if pred in profiles and 'centroid' in profiles[pred]:
                    centroid = np.array([profiles[pred]['centroid'][f] for f in feature_order], dtype=float)
                else:
                    #if centroid keys are raw numbers (legacy), try mapping
                    try:
                        centroid = np.array([profiles[pred][f] for f in feature_order], dtype=float)
                    except Exception:
                        centroid = None
            #fallback: can't retrieve centroid from gait_model_profiles.json; compute rough centroid from training CSVs
            if centroid is None:
                #compute centroid from TRAIN_FILES
                rows = []
                for p in TRAIN_FILES:
                    if not p.exists(): continue
                    dft = pd.read_csv(p)
                    if 'qc_pass' in dft.columns:
                        dft = dft[dft['qc_pass'] == True]
                    if dft.empty:
                        continue
                    if 'dom_freq_power' in dft.columns:
                        dft['dom_freq_power_log'] = np.log1p(dft['dom_freq_power'].astype(float))
                    else:
                        dft['dom_freq_power_log'] = 0.0
                    if 'gyro_mag_mean' in dft.columns and 'acc_mag_rms' in dft.columns:
                        dft['gyro_acc_ratio'] = dft['gyro_mag_mean'].astype(float) / (dft['acc_mag_rms'].astype(float) + EPS)
                    else:
                        dft['gyro_acc_ratio'] = 0.0
                    #label mapping(Impaired)
                    for _, r2 in dft.iterrows():
                        #Determine label for this row by looking up which source file produced it
                        #If the file is Fatigued or Unstable, that row contributes to Impaired
                        pass
                #If we cannot compute centroid robustly here, skip mah gating
                centroid = None

            if centroid is not None:
                mah = mahalanobis_distance(x, centroid, cov_inv)

        #gating decision
        class_th = thresholds.get(pred, None)
        if pred == "Impaired" and class_th is not None:
            class_th = class_th * IMPAIRED_STRICT_FACTOR
        accept = True
        reason = ""
        if max_prob < prob_thresh:
            accept = False
            reason = f"low_prob({max_prob:.2f})"
        elif class_th is not None and (mah is not None) and (mah > class_th):
            accept = False
            reason = f"mah({mah:.2f})>{class_th:.2f}"

        final = pred if accept else "UNCERTAIN"
        results.append({
            "session_id": row.get("session_id", ""),
            "pred": pred,
            "final": final,
            "prob": max_prob,
            "mah": mah,
            "threshold_used": class_th,
            "reason": reason
        })

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved predictions -> {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", help="Input CSV to predict (default: knee_monitor_pipeline/ML/healthy_baseline_qc.csv)", default="knee_monitor_pipeline/ML/healthy_baseline_qc.csv")
    p.add_argument("--output", "-o", help="Output CSV path", default="knee_monitor_pipeline/ML/predictions_with_gating.csv")
    args = p.parse_args()
    predict_file(args.input, args.output)
