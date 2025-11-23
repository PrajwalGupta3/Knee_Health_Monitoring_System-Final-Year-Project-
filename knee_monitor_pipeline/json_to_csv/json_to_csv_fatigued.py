import json
import pandas as pd
import re

with open("knee_monitor_pipeline/data/raw/fatigued.json") as f:
    data = json.load(f)

sessions = data.get("sessions", {})
mpu_records, bmp_records = [], []

def extract_json_objects(raw_text):
    """Extract valid JSON dicts from broken concatenated text.
    (As we utilise 'Arduino legacy software',DB stores broken strings.)"""
    objects = []
    pattern = re.compile(r"\{[^{}]+\}")
    for match in pattern.findall(raw_text):
        frag = match.strip()
        try:
            obj = json.loads(frag)
            objects.append(obj)
        except json.JSONDecodeError:
            continue

    return objects

for session_id, content in sessions.items():
    #MPU readings
    if "mpu" in content:
        raw = content["mpu"]
        objs = extract_json_objects(raw)
        for i, o in enumerate(objs):
            o["session"] = session_id
            o["reading_index"] = i
            mpu_records.append(o)

    #BMP readings
    if "bmp" in content:
        raw = content["bmp"]
        objs = extract_json_objects(raw)
        for i, o in enumerate(objs):
            o["session"] = session_id
            o["reading_index"] = i
            bmp_records.append(o)

mpu_df = pd.DataFrame(mpu_records)
bmp_df = pd.DataFrame(bmp_records)
print("MPU shape:", mpu_df.shape)
print("BMP shape:", bmp_df.shape)
if not mpu_df.empty:
    mpu_df.to_csv("knee_monitor_pipeline/data/processed/mpu_readings_fatigued.csv", index=False)
if not bmp_df.empty:
    bmp_df.to_csv("knee_monitor_pipeline/data/processed/bmp_readings_fatigued.csv", index=False)