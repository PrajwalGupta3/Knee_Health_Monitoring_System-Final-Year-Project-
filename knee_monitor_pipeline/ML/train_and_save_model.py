
"""
Trains a supervised LogisticRegression(multiclass) on the CSVs,
maps "Fatigued" and "Unstable" to "Impaired" prior to training, and saves:
 - knee_monitor_pipeline/ML/logistic_model_impaired.joblib
 - knee_monitor_pipeline/ML/feature_order_impaired.json

"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Input CSV files
FILES = {
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
}
#Base and engineered features
FEATURES = ['pace_spm', 'acc_mag_rms', 'gyro_mag_mean', 'dom_freq_power', 'dom_freq_power_log', 'gyro_acc_ratio']

OUT_MODEL = Path("knee_monitor_pipeline/ML/logistic_model_impaired.joblib")
OUT_FEATURE_ORDER = Path("knee_monitor_pipeline/ML/feature_order_impaired.json")

def load_and_engineer(files_map):
    X = []
    y = []
    for label, p in files_map.items():
        if not p.exists():
            print(f"[WARN] missing file: {p} (skipping {label})")
            continue
        df = pd.read_csv(p)
        if 'qc_pass' in df.columns:
            df = df[df['qc_pass'] == True]
        if df.empty:
            print(f"[WARN] empty after QC: {p}")
            continue
        #engineered features
        if 'dom_freq_power' in df.columns:
            df['dom_freq_power_log'] = np.log1p(df['dom_freq_power'].astype(float))
        else:
            df['dom_freq_power_log'] = 0.0
        eps = 1e-6
        if 'gyro_mag_mean' in df.columns and 'acc_mag_rms' in df.columns:
            df['gyro_acc_ratio'] = df['gyro_mag_mean'].astype(float) / (df['acc_mag_rms'].astype(float) + eps)
        else:
            df['gyro_acc_ratio'] = 0.0

        for _, row in df.iterrows():
            try:
                feats = [float(row[f]) for f in FEATURES]
            except Exception as e:
                raise KeyError(f"Missing feature in {p}: {e}")
            X.append(feats)
            y.append(label)
    if len(X) == 0:
        raise RuntimeError("No training data loaded. Check file paths and QC flags.")
    return np.array(X, dtype=float), np.array(y, dtype=object)

def main():
    print("Loading and engineering features...")
    X, y = load_and_engineer(FILES)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features.")

    #Fatigued combined with Unstable to form Impaired
    y_merged = np.array([ "Impaired" if lab in ("Fatigued","Unstable") else lab for lab in y ], dtype=object)

    print("Label counts after merge:")
    unique, counts = np.unique(y_merged, return_counts=True)
    for u,c in zip(unique, counts):
        print(f"  {u}: {c}")

    #Training pipeline
    print("Training LogisticRegression (balanced)...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs', multi_class='multinomial', random_state=42))
    clf.fit(X, y_merged)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_MODEL)
    print(f"Saved model -> {OUT_MODEL}")

    #Saves the feature order
    OUT_FEATURE_ORDER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FEATURE_ORDER, 'w') as f:
        json.dump(FEATURES, f, indent=2)
    print(f"Saved feature order -> {OUT_FEATURE_ORDER}")

if __name__ == "__main__":
    main()
