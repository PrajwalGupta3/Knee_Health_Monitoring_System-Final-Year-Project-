import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, welch

SESSION_DURATION_SEC = 60.0 
ROLLING_WINDOW = 5      #At 10Hz, this is a 0.5-second smooth


#We are now looking for the GAIT CYCLE frequency, which is 0.2-1.0 Hz
#(This corresponds to a pace of 24 to 120 steps per minute)
WALK_FREQ_MIN = 0.2  
WALK_FREQ_MAX = 1.0

SENSOR_SATURATION_LIMIT = 32000
data_dir = Path("knee_monitor_pipeline/data/processed")
plots_dir = Path("knee_monitor_pipeline/data/plots/fatigued_qc")
plots_dir.mkdir(parents=True, exist_ok=True)

try:
    #This script reads the NEW CSV created with the 10Hz data
    mpu_df = pd.read_csv(data_dir / "mpu_readings_fatigued.csv")
    bmp_df = pd.read_csv(data_dir / "bmp_readings_fatigued.csv")
except FileNotFoundError as e:
    print(f"ERROR: Could not read CSV file. Did you run your new json_to_csv.py script first? File: {e.filename}")
    exit()

try:
    session_meta_df = pd.read_csv(data_dir / "session_metadata.csv")
    user_meta_df = pd.read_csv(data_dir / "user_metadata.csv")
    METADATA_LOADED = True
    print("[INFO] Successfully loaded session and user metadata.")
except FileNotFoundError:
    METADATA_LOADED = False
    print("Metadata files (session_metadata.csv or user_metadata.csv) not found.")

#Date parsing
date_format = "%Y-%m-%d_%H-%M-%S"
mpu_df["session_time"] = pd.to_datetime(mpu_df["session"], format=date_format, errors='coerce')
bmp_df["session_time"] = pd.to_datetime(bmp_df["session"], format=date_format, errors='coerce')

mpu_df = mpu_df.dropna(subset=["session_time"])
bmp_df = bmp_df.dropna(subset=["session_time"])

paired_sessions = [] 
mpu_session_times = set(mpu_df["session_time"].unique())
bmp_session_times = set(bmp_df["session_time"].unique())
new_unified_sessions = sorted(list(mpu_session_times.intersection(bmp_session_times)))
for session_time in new_unified_sessions:
    paired_sessions.append((session_time, session_time, SESSION_DURATION_SEC))
print(f"Stage 1: Found {len(new_unified_sessions)} new, unified sessions.")

remaining_mpu_times = sorted(list(mpu_session_times - bmp_session_times))
remaining_bmp_times = sorted(list(bmp_session_times - mpu_session_times))
old_paired_count = 0
for mpu_time in remaining_mpu_times:
    next_bmp = [b for b in remaining_bmp_times if 0 < (b - mpu_time).total_seconds() <= 120]
    if next_bmp:
        bmp_time = next_bmp[0]
        time_offset = (bmp_time - mpu_time).total_seconds()
        paired_sessions.append((mpu_time, bmp_time, time_offset))
        old_paired_count += 1
        remaining_bmp_times.remove(bmp_time) 
print(f"Stage 2: Found and paired {old_paired_count} old, separate sessions.")
print(f"--- Total sessions to process: {len(paired_sessions)} ---")

# (QC functions are unchanged)
def compute_magnitude(df, cols):
    return np.sqrt((df[cols] ** 2).sum(axis=1))
def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))
def qc_session(mpu_block, acc_mag, gyro_mag):
    raw_axes = mpu_block[["ax","ay","az","gx","gy","gz"]]
    qc_pass = True
    qc_fail_reason = []
    if acc_mag.isna().any() or gyro_mag.isna().any():
        qc_pass = False; qc_fail_reason.append("nan_values")
    if acc_mag.std() < 50 or gyro_mag.std() < 10:
        qc_pass = False; qc_fail_reason.append("low_variance")
    if (acc_mag > 50000).any() or (gyro_mag > 50000).any():
        qc_pass = False; qc_fail_reason.append("magnitude_spike")
    if (raw_axes.abs() >= SENSOR_SATURATION_LIMIT).any().any():
        qc_pass = False; qc_fail_reason.append("saturation")
    return qc_pass, ", ".join(qc_fail_reason)

baseline_records = []

for i, (mpu_time, bmp_time, time_offset) in enumerate(paired_sessions, start=1):
    
    mpu_sess = mpu_time.strftime(date_format)
    bmp_sess = bmp_time.strftime(date_format)
    
    session_id = f"knee{i:02d}_{mpu_sess}"

    mpu_block = mpu_df[mpu_df["session_time"] == mpu_time].reset_index(drop=True)
    bmp_block = bmp_df[bmp_df["session_time"] == bmp_time].reset_index(drop=True)
    
    if mpu_block.empty or bmp_block.empty:
        print(f"Skipping session {session_id} due to empty data block (MPU: {len(mpu_block)}, BMP: {len(bmp_block)}).")
        continue

    fs_mpu = len(mpu_block) / SESSION_DURATION_SEC
    fs_bmp = len(bmp_block) / SESSION_DURATION_SEC
    
    print(f"[INFO] Processing {session_id}: {len(mpu_block)} readings @ {fs_mpu:.2f} Hz")
    
    mpu_block['time_sec'] = mpu_block['reading_index'] / fs_mpu
    bmp_block['time_sec'] = (bmp_block['reading_index'] / fs_bmp)

    #Compute Magnitudes & Smoothing
    mpu_block["acc_mag"] = compute_magnitude(mpu_block, ["ax","ay","az"])
    mpu_block["gyro_mag"] = compute_magnitude(mpu_block, ["gx","gy","gz"])
    mpu_block["acc_mag_smooth"] = mpu_block["acc_mag"].rolling(
        window=ROLLING_WINDOW, center=True, min_periods=1).mean()
    mpu_block["gyro_mag_smooth"] = mpu_block["gyro_mag"].rolling(
        window=ROLLING_WINDOW, center=True, min_periods=1).mean()

    qc_pass, qc_reason = qc_session(mpu_block, mpu_block["acc_mag"], mpu_block["gyro_mag"])

    #cadence_spm = len(peaks) 
    
    acc_mag_rms = compute_rms(mpu_block["acc_mag"])
    
    f, Pxx = welch(mpu_block["acc_mag"], fs=fs_mpu, nperseg=min(256, len(mpu_block)))
    
    walk_mask = (f >= WALK_FREQ_MIN) & (f <= WALK_FREQ_MAX)
    
    if walk_mask.any() and Pxx[walk_mask].max() > 0:
        dom_freq_idx = np.argmax(Pxx[walk_mask])
        dom_freq_hz = f[walk_mask][dom_freq_idx]
        dom_freq_power = Pxx[walk_mask][dom_freq_idx]
    else:
        #Fallback (e.g., user was standing still)
        if len(Pxx) > 1:
            dom_freq_idx = np.argmax(Pxx[1:]) + 1 
            dom_freq_hz = f[dom_freq_idx]
            dom_freq_power = Pxx[dom_freq_idx]
        else:
            dom_freq_hz = 0
            dom_freq_power = 0
            

    #pace_spm (Steps per Minute) = dom_freq_hz(Cycles/Sec)*2(Steps/Cycle)*60 (Sec/Min)
    pace_spm = dom_freq_hz * 120

    #Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True) 
    axs[0].plot(mpu_block['time_sec'], mpu_block["acc_mag"], label="Acc Mag (Raw)", alpha=0.3, color='lightblue')
    axs[0].plot(mpu_block['time_sec'], mpu_block["acc_mag_smooth"], label="Acc Mag (Smooth)", color='blue')
    axs[0].set_ylabel("Acc Mag"); axs[0].legend(loc="upper right")
    axs[1].plot(mpu_block['time_sec'], mpu_block["gyro_mag"], label="Gyro Mag (Raw)", alpha=0.3, color='moccasin')
    axs[1].plot(mpu_block['time_sec'], mpu_block["gyro_mag_smooth"], label="Gyro Mag (Smooth)", color='orange')
    axs[1].set_ylabel("Gyro Mag"); axs[1].legend(loc="upper right")
    fig.delaxes(axs[2]); axs[2] = fig.add_subplot(4, 1, 3)
    axs[2].plot(bmp_block['time_sec'], bmp_block["T"], label="Temperature", color='green', marker='.', linestyle='-')
    axs[2].set_ylabel("Temp (°C)"); axs[2].set_xlabel("Time (seconds)"); axs[2].legend(loc="upper right")
    fig.delaxes(axs[3]); axs[3] = fig.add_subplot(4, 1, 4)
    axs[3].plot(f, Pxx, label="PSD"); axs[3].axvline(dom_freq_hz, color='red', linestyle='--', label=f'Dom Freq: {dom_freq_hz:.2f} Hz')
    axs[3].set_xlabel("Frequency (Hz)"); axs[3].set_ylabel("Power"); axs[3].set_xlim(0, max(WALK_FREQ_MAX + 0.5, 5)); axs[3].legend(loc="upper right")
    qc_status_str = 'PASS' if qc_pass else f'FAIL ({qc_reason})'
    plt.suptitle(f"{session_id} — QC: {qc_status_str}"); plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = plots_dir / f"{session_id}.png"
    plt.savefig(plot_path); plt.close(fig)
    print(f"[SAVED] {plot_path} — QC: {qc_status_str}")

    #Baseline Summary Record
    record = {
        "session_id": session_id,
        "mpu_session": mpu_sess,
        "bmp_session": bmp_sess,
        "qc_pass": qc_pass,
        "qc_fail_reason": qc_reason if not qc_pass else None,
        
        #Changed "cadence_spm" to "pace_spm"
        "pace_spm": pace_spm,
        "acc_mag_rms": acc_mag_rms,
        "dom_freq_hz": dom_freq_hz,
        "dom_freq_power": dom_freq_power,
        
        "acc_mag_mean": mpu_block["acc_mag"].mean(),
        "gyro_mag_mean": mpu_block["gyro_mag"].mean(),
        "acc_mag_std": mpu_block["acc_mag"].std(),
        "gyro_mag_std": mpu_block["gyro_mag"].std(),
        
        "temp_mean": bmp_block["T"].mean(),
        "temp_std": bmp_block["T"].std(),
        "temp_min": bmp_block["T"].min(),
        "temp_max": bmp_block["T"].max(),
        
        "ax_mean": mpu_block["ax"].mean(), "ay_mean": mpu_block["ay"].mean(), "az_mean": mpu_block["az"].mean(),
        "gx_mean": mpu_block["gx"].mean(), "gy_mean": mpu_block["gy"].mean(), "gz_mean": mpu_block["gz"].mean(),
        "ax_std": mpu_block["ax"].std(), "ay_std": mpu_block["ay"].std(), "az_std": mpu_block["az"].std(),
        "gx_std": mpu_block["gx"].std(), "gy_std": mpu_block["gy"].std(), "gz_std": mpu_block["gz"].std(),
    }
    baseline_records.append(record)

baseline_df = pd.DataFrame(baseline_records)

#Metadata merge
if METADATA_LOADED and not baseline_df.empty:
    try:
        baseline_df = pd.merge(baseline_df, session_meta_df, on="mpu_session", how="left")
        
        if 'subject_id' in baseline_df.columns:
            baseline_df = pd.merge(baseline_df, user_meta_df, on="subject_id", how="left")
        else:
            print("'subject_id' column not found after merging with session_meta_df. Skipping user_meta_df merge.")

        
        meta_cols = ['subject_id', 'knee_side', 'age', 'gender', 'height_cm', 'weight_kg']
        present_meta_cols = [col for col in meta_cols if col in baseline_df.columns]

        all_cols = ['session_id', 'mpu_session', 'bmp_session', 'qc_pass', 'qc_fail_reason'] + \
                   present_meta_cols + \
                   [col for col in baseline_df.columns if col not in ['session_id', 'mpu_session', 'bmp_session', 'qc_pass', 'qc_fail_reason'] + present_meta_cols]
        
        all_cols = [col for col in all_cols if col in baseline_df.columns]
        
        baseline_df = baseline_df[all_cols]
    
    except KeyError as e:
        print(f"ERROR: A key was missing during metadata merge. This can happen if 'session_metadata.csv' is incorrect. Error: {e}")
        print("Saving baseline.csv *without* merged metadata.")
        
elif METADATA_LOADED and baseline_df.empty:
    print("[INFO] No sessions were processed, so metadata merge was skipped.")


baseline_path = data_dir /"final_csv" / "target_fatigued_qc.csv"
baseline_df.to_csv(baseline_path, index=False)

#Zero records check
if baseline_df.empty:
    print(f"\n[INFO] QC baseline saved with 0 records. (No valid sessions were found to process).")
    print(f"-> {baseline_path}")
else:
    print(f"\n[SUCCESS] QC baseline saved with {len(baseline_df)} records.")
    print(f"-> {baseline_path}")
 