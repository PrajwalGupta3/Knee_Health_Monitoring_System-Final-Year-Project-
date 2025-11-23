import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILES = {
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
}
FEATURES = ['pace_spm','gyro_mag_mean','acc_mag_rms','dom_freq_power']

#load the combined df
frames = []
for lab,p in FILES.items():
    if not p.exists(): continue
    df = pd.read_csv(p)
    if 'qc_pass' in df.columns:
        df = df[df['qc_pass']==True]
    df['label'] = lab
    frames.append(df)
if not frames:
    print("No files found.")
    raise SystemExit(1)
df = pd.concat(frames, ignore_index=True)

#pairwise scatter for a few interesting pairs
pairs = [('pace_spm','gyro_mag_mean'), ('pace_spm','acc_mag_rms'), ('gyro_mag_mean','acc_mag_rms')]
for a,b in pairs:
    plt.figure(figsize=(6,5))
    for lab in df['label'].unique():
        sub = df[df['label']==lab]
        plt.scatter(sub[a], sub[b], label=f"{lab} ({len(sub)})", alpha=0.8)
    plt.xlabel(a); plt.ylabel(b)
    plt.title(f"{a} vs {b}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = Path("knee_monitor_pipeline/ML/plots") / f"{a}_vs_{b}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print("Saved", out)

#histograms per feature
for f in FEATURES:
    plt.figure(figsize=(6,4))
    for lab in df['label'].unique():
        sub = df[df['label']==lab]
        plt.hist(sub[f], bins=12, alpha=0.5, label=lab)
    plt.title(f"Histogram: {f}")
    plt.legend()
    out = Path("knee_monitor_pipeline/ML/plots") / f"hist_{f}.png"
    plt.savefig(out)
    print("Saved", out)
