import pandas as pd
from pathlib import Path

LOOCV_CSV = Path("knee_monitor_pipeline/ML/loocv_results.csv")
FEATURES = ['pace_spm','acc_mag_rms','gyro_mag_mean','dom_freq_power']

if not LOOCV_CSV.exists():
    print("File not found:", LOOCV_CSV)
    raise SystemExit(1)

df = pd.read_csv(LOOCV_CSV)
#normalize column names 
print("LOOCV rows:", len(df))

#find Unstable 'true'  but is mispredicted
unstable_mis = df[(df['true_label']=='Unstable') & (df['pred_label']!='Unstable')]
print("\nUnstable misclassified (count):", len(unstable_mis))
print(unstable_mis.head(20).to_string(index=False))

#Show raw feature columns if available in the original test CSV
orig_files = {
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
}
orig_list = []
for lab, p in orig_files.items():
    if p.exists():
        dfp = pd.read_csv(p)
        if 'qc_pass' in dfp.columns:
            dfp = dfp[dfp['qc_pass']==True]
        dfp['__label__'] = lab
        orig_list.append(dfp)
if orig_list:
    full = pd.concat(orig_list, ignore_index=True)
    #merging based on session_id
    merged = unstable_mis.merge(full, left_on='session_id', right_on='session_id', how='left', suffixes=('','_orig'))
    if not merged.empty:
        print("\nMerged rows with original features (first 10):")
        print(merged[['session_id','true_label','pred_label'] + FEATURES].head(10).to_string(index=False))
    else:
        print("\nNo matching session_id found to merge with original CSVs.")
else:
    print("\nOriginal CSV files not found in expected paths; skipping merge.")
