
"""
qc_and_plot.py (engineered features added)

Reads:
  <input_dir>/mpu_readings.csv
  <input_dir>/bmp_readings.csv

Writes:
  <input_dir>/final.csv
  <input_dir>/plots/<session_id>.png

Added engineered features:
  - dom_freq_power_log = log1p(dom_freq_power)
  - gyro_acc_ratio = gyro_mag_mean / (acc_mag_rms + eps)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from datetime import datetime
import math

# Tunables
SESSION_DURATION_SEC = 60.0
ROLLING_WINDOW = 5
WALK_FREQ_MIN = 0.2
WALK_FREQ_MAX = 1.0
SENSOR_SATURATION_LIMIT = 32000
EPS = 1e-9

def compute_magnitude(df, cols):
    return np.sqrt((df[cols] ** 2).sum(axis=1))

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def qc_session(mpu_block, acc_mag, gyro_mag):
    qc_pass = True
    reasons = []
    if acc_mag.isna().any() or gyro_mag.isna().any():
        qc_pass = False; reasons.append("nan_values")
    if acc_mag.std() < 50 or gyro_mag.std() < 10:
        qc_pass = False; reasons.append("low_variance")
    if (acc_mag > 50000).any() or (gyro_mag > 50000).any():
        qc_pass = False; reasons.append("magnitude_spike")
    if (mpu_block[["ax","ay","az","gx","gy","gz"]].abs() >= SENSOR_SATURATION_LIMIT).any().any():
        qc_pass = False; reasons.append("saturation")
    return qc_pass, ";".join(reasons)

def process_pair(mpu_df, bmp_df, session_idx, plots_dir):
    date_format = "%Y-%m-%d_%H-%M-%S"
    mpu_sess = mpu_df["session"].iloc[0] if "session" in mpu_df.columns else datetime.now().strftime(date_format)
    session_id = f"session_{session_idx}_{mpu_sess}"

    # compute sampling frequency
    fs_mpu = max(len(mpu_df) / SESSION_DURATION_SEC, 1.0)
    mpu_df['time_sec'] = mpu_df['reading_index'] / fs_mpu
    bmp_df['time_sec'] = bmp_df['reading_index'] / max(len(bmp_df) / SESSION_DURATION_SEC, 1.0)

    # magnitudes & smoothing
    mpu_df["acc_mag"] = compute_magnitude(mpu_df, ["ax","ay","az"])
    mpu_df["gyro_mag"] = compute_magnitude(mpu_df, ["gx","gy","gz"])
    mpu_df["acc_mag_smooth"] = mpu_df["acc_mag"].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
    mpu_df["gyro_mag_smooth"] = mpu_df["gyro_mag"].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

    qc_pass, qc_reason = qc_session(mpu_df, mpu_df["acc_mag"], mpu_df["gyro_mag"])
    acc_mag_rms = compute_rms(mpu_df["acc_mag"])
    gyro_mag_mean = float(mpu_df["gyro_mag"].mean()) if "gyro_mag" in mpu_df else 0.0

    # PSD / dominant frequency in walking band
    try:
        f, Pxx = welch(mpu_df["acc_mag"].fillna(0).to_numpy(), fs=fs_mpu, nperseg=min(256, len(mpu_df)))
    except Exception:
        f = np.array([0.0])
        Pxx = np.array([0.0])

    walk_mask = (f >= WALK_FREQ_MIN) & (f <= WALK_FREQ_MAX)
    if walk_mask.any() and Pxx[walk_mask].max() > 0:
        dom_idx = np.argmax(Pxx[walk_mask])
        dom_freq_hz = float(f[walk_mask][dom_idx])
        dom_freq_power = float(Pxx[walk_mask][dom_idx])
    else:
        if len(Pxx) > 1:
            dom_idx = int(np.argmax(Pxx[1:]) + 1)
            dom_freq_hz = float(f[dom_idx]) if len(f) > dom_idx else 0.0
            dom_freq_power = float(Pxx[dom_idx]) if len(Pxx) > dom_idx else 0.0
        else:
            dom_freq_hz = 0.0
            dom_freq_power = 0.0

    # pace in steps per minute (2 steps per cycle)
    pace_spm = float(dom_freq_hz * 120.0)

    # engineered features
    dom_freq_power_log = math.log1p(dom_freq_power) if dom_freq_power >= 0 else 0.0
    gyro_acc_ratio = gyro_mag_mean / (acc_mag_rms + EPS)

    # plotting (4 rows)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(mpu_df['time_sec'], mpu_df["acc_mag"], alpha=0.3)
    axs[0].plot(mpu_df['time_sec'], mpu_df["acc_mag_smooth"], color='blue')
    axs[0].set_ylabel("Acc Mag")
    axs[1].plot(mpu_df['time_sec'], mpu_df["gyro_mag"], alpha=0.3)
    axs[1].plot(mpu_df['time_sec'], mpu_df["gyro_mag_smooth"], color='orange')
    axs[1].set_ylabel("Gyro Mag")
    # BMP temp plot - guard if T missing
    if "T" in bmp_df.columns:
        axs[2].plot(bmp_df['time_sec'], bmp_df["T"], marker='.', linestyle='-')
    else:
        axs[2].plot(bmp_df['time_sec'], np.zeros(len(bmp_df)), marker='.', linestyle='-')
    axs[2].set_ylabel("Temp")
    axs[3].plot(f, Pxx)
    axs[3].axvline(dom_freq_hz, color='red', linestyle='--')
    axs[3].set_xlabel("Frequency (Hz)")

    qc_status = "PASS" if qc_pass else f"FAIL ({qc_reason})"
    plt.suptitle(f"{session_id} — QC: {qc_status}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{session_id}.png"
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    record = {
        "session_id": session_id,
        "qc_pass": bool(qc_pass),
        "qc_fail_reason": qc_reason if not qc_pass else "",
        "pace_spm": float(pace_spm),
        "acc_mag_rms": float(acc_mag_rms),
        "gyro_mag_mean": float(gyro_mag_mean),
        "dom_freq_hz": float(dom_freq_hz),
        "dom_freq_power": float(dom_freq_power),
        # engineered
        "dom_freq_power_log": float(dom_freq_power_log),
        "gyro_acc_ratio": float(gyro_acc_ratio),
        # additional stats
        "acc_mag_mean": float(mpu_df["acc_mag"].mean()),
        "gyro_mag_std": float(mpu_df["gyro_mag"].std())
    }

    return record, str(plot_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", required=True, help="Session directory produced by json_to_csv.py")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    # check inputs exist
    mpu_path = input_dir / "mpu_readings.csv"
    bmp_path = input_dir / "bmp_readings.csv"
    if not mpu_path.exists() or not bmp_path.exists():
        raise FileNotFoundError(f"Missing input CSVs in {input_dir}. Found: MPU={mpu_path.exists()}, BMP={bmp_path.exists()}")

    # read
    mpu_df = pd.read_csv(mpu_path)
    bmp_df = pd.read_csv(bmp_path)

    # process as single-session block
    records = []
    plots_dir = input_dir / "plots"
    rec, plot = process_pair(mpu_df, bmp_df, 1, plots_dir)
    records.append(rec)

    # final.csv
    final_df = pd.DataFrame(records)
    final_csv = input_dir / "final.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[OK] final.csv written: {final_csv}")
    print(f"[OK] plot saved: {plot}")

if __name__ == "__main__":
    main()
