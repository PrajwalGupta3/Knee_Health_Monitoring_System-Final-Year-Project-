import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
USER_FILE = Path("knee_monitor_pipeline/data/users/prajwal.json")
OUTPUT_FILE = Path("knee_monitor_pipeline/scripts/plot_history/long_trend.png")
HEALTHY_PACE_MIN = 38  # Lower bound of healthy pace (for shading)
HEALTHY_PACE_MAX = 60  # Upper bound of healthy pace

def generate_plot():
    # 1. Load Data
    if not USER_FILE.exists():
        print(f"Error: {USER_FILE} not found.")
        return

    with open(USER_FILE, 'r') as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("No history found.")
        return

    # 2. Extract Metrics
    dates = []
    paces = []
    impacts = []
    
    # Sort history by session_id to ensure chronological order
    history.sort(key=lambda x: x["session_id"])

    for h in history[-5:]: # Analysis of last 5 sessions
        # Parse Timestamp from Session ID (Format: YYYYMMDD_HHMMSS)
        ts_str = h["session_id"].split('_')[1] # e.g. "110158"
        dt_obj = datetime.strptime(ts_str, "%H%M%S")
        dates.append(dt_obj)
        
        # Get features
        feats = h["result"]["raw_features"]
        paces.append(feats.get("pace_spm", 0))
        impacts.append(feats.get("acc_mag_rms", 0))

    # 3. Setup Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- LEFT AXIS: PACE (Blue) ---
    color_pace = '#1E88E5' # Material Blue
    ax1.set_xlabel('Session Time (HH:MM)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pace (Steps/Min)', color=color_pace, fontsize=12, fontweight='bold')
    
    # Plot Line & Markers
    line1 = ax1.plot(dates, paces, color=color_pace, marker='o', markersize=8, 
                     linewidth=2.5, label='Walking Pace')
    ax1.tick_params(axis='y', labelcolor=color_pace)
    
    # Add Healthy Zone Shading
    ax1.axhspan(HEALTHY_PACE_MIN, HEALTHY_PACE_MAX, color='#4CAF50', alpha=0.15, label='Healthy Pace Target')
    ax1.text(dates[0], HEALTHY_PACE_MAX - 2, " Target Zone", color='#2E7D32', fontsize=10, fontweight='bold')

    # --- RIGHT AXIS: IMPACT (Orange) ---
    ax2 = ax1.twinx()  
    color_impact = '#FF9800' # Material Orange
    ax2.set_ylabel('Impact Load (RMS)', color=color_impact, fontsize=12, fontweight='bold')
    
    # Plot Dashed Line for Correlation
    line2 = ax2.plot(dates, impacts, color=color_impact, marker='s', markersize=6, 
                     linestyle='--', linewidth=2, alpha=0.8, label='Impact Load')
    ax2.tick_params(axis='y', labelcolor=color_impact)

    # --- ANNOTATIONS (The "Smart" Part) ---
    # Detect the biggest drop in pace
    deltas = [paces[i] - paces[i-1] for i in range(1, len(paces))]
    if deltas:
        min_delta = min(deltas)
        min_idx = deltas.index(min_delta) + 1 # Index of the 'after' point
        
        if min_delta < -5: # Only flag if drop is > 5 SPM
            # Draw Arrow
            ax1.annotate(f'Alert: Pace Drop ({min_delta:.1f})',
                         xy=(dates[min_idx], paces[min_idx]), 
                         xytext=(dates[min_idx], paces[min_idx] + 15),
                         arrowprops=dict(facecolor='#D32F2F', shrink=0.05),
                         fontsize=10, color='#D32F2F', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", alpha=0.9))

    # --- FORMATTING ---
    # Combine legends from both axes
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    
    # Date Formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("Longitudinal Recovery Analysis: Pace vs. Impact Correlation", fontsize=14, pad=30)
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 4. Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[OK] Advanced graph saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_plot()

