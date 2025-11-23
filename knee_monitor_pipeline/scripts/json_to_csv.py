
"""
json_to_csv.py

Reads a JSON file that contains **exactly one** "sessions" object (one timestamp key)
and extracts MPU and BMP readings into CSVs:

  Usage:
    python json_to_csv.py --input <raw_session.json> --outdir <session_outdir>

Writes:
  <outdir>/mpu_readings.csv
  <outdir>/bmp_readings.csv

Notes:
 - The script expects the raw JSON format where content["mpu"] and content["bmp"]
   are strings containing concatenated JSON objects (the extractor handles that).
 - Raises ValueError if the supplied JSON contains more than one session.
"""
import json
import argparse
import re
from pathlib import Path
import pandas as pd

def extract_json_objects(raw_text):
    """Extract valid JSON dicts from broken concatenated text."""
    objects = []
    # coarse pattern: match {...} non-greedy between braces groups (works for your dumps)
    pattern = re.compile(r"\{[^{}]+\}")
    for match in pattern.findall(raw_text):
        frag = match.strip()
        try:
            obj = json.loads(frag)
            objects.append(obj)
        except json.JSONDecodeError:
            # skip invalid fragments
            continue
    return objects

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to raw session JSON (contains exactly one session under 'sessions')")
    p.add_argument("--outdir", "-o", required=True, help="Directory to write CSV outputs (will be created)")
    args = p.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sessions = raw.get("sessions", {})
    if len(sessions) != 1:
        raise ValueError(f"Expected exactly one session in input JSON (found {len(sessions)})")

    (session_id, content), = sessions.items()

    mpu_records = []
    bmp_records = []

    # MPU
    if "mpu" in content and content["mpu"]:
        objs = extract_json_objects(content["mpu"])
        for i, o in enumerate(objs):
            # keep original keys and add reading_index
            o["reading_index"] = i
            mpu_records.append(o)

    # BMP
    if "bmp" in content and content["bmp"]:
        objs = extract_json_objects(content["bmp"])
        for i, o in enumerate(objs):
            o["reading_index"] = i
            bmp_records.append(o)

    # Write CSVs if non-empty
    if mpu_records:
        mpu_df = pd.DataFrame(mpu_records)
        mpu_out = outdir / "mpu_readings.csv"
        mpu_df.to_csv(mpu_out, index=False)
        print(f"[OK] Wrote MPU CSV: {mpu_out} (rows={len(mpu_df)})")
    else:
        print("[WARN] No MPU records extracted.")

    if bmp_records:
        bmp_df = pd.DataFrame(bmp_records)
        bmp_out = outdir / "bmp_readings.csv"
        bmp_df.to_csv(bmp_out, index=False)
        print(f"[OK] Wrote BMP CSV: {bmp_out} (rows={len(bmp_df)})")
    else:
        print("[WARN] No BMP records extracted.")

    print(f"[SESSION] session_id = {session_id}")

if __name__ == "__main__":
    main()
