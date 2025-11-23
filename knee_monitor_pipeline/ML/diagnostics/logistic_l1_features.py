import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

#Loading dataset assembled earlier in evaluate_models.py method
FILES = {
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
}
FEATURES = ['pace_spm','acc_mag_rms','gyro_mag_mean','dom_freq_power']

def load_all(FILES, FEATURES):
    X, y = [], []
    for label, p in FILES.items():
        if not p.exists(): 
            continue
        df = pd.read_csv(p)
        if 'qc_pass' in df.columns:
            df = df[df['qc_pass']==True]
        for _, row in df.iterrows():
            try:
                X.append([float(row[f]) for f in FEATURES])
            except Exception:
                raise KeyError("Missing feature columns in file", p)
            y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=object)

X, y = load_all(FILES, FEATURES)
print("Loaded", X.shape[0], "samples.")

#L1 logistic (multinomial)
clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', class_weight='balanced', max_iter=5000))

#cross-validated score
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("CV acc:", scores.mean(), "±", scores.std())

#fitting on all data to inspect coefficients
clf.fit(X, y)
log = clf.named_steps['logisticregression']
coefs = log.coef_        #shape (n_classes, n_features)
classes = log.classes_
print("\nClasses:", classes)
for i, c in enumerate(classes):
    print(f"\nClass {c} coefficients (feature:coef):")
    pairs = list(zip(FEATURES, coefs[i].tolist()))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    for feat,coef in pairs_sorted:
        print(f"  {feat}: {coef:.4f}")
