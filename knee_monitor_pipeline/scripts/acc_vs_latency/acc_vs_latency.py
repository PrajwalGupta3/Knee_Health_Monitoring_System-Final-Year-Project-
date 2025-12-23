import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
# We assume these values based on your previous 'merge_and_eval.py' output.
# If you have the CSV, you can load it. For safety/reproducibility here, I'll hardcode the exact values you got.
data = {
    "model": ["LogisticRegression", "GaussianNB", "SVM_linear", "SVM_rbf", "kNN", "RandomForest"],
    "latency_us": [2.43, 2.85, 3.67, 6.18, 33.38, 76.99],
    "mean_acc": [0.806, 0.835, 0.830, 0.828, 0.838, 0.812]
}
df = pd.DataFrame(data)

OUTPUT_FILE = Path("knee_monitor_pipeline/scripts/acc_vs_latency/model_tradeoff_advanced.png")

def generate_plot():
    # Setup stylish plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Plot Points
    # Color mapping: Green for efficient, Red for slow, Blue for middle
    colors = ['#2E7D32', '#2E7D32', '#1565C0', '#1565C0', '#C62828', '#C62828']
    markers = ['o', 'D', 's', '^', 'X', 'P']
    
    for i in range(len(df)):
        ax.scatter(df.latency_us[i], df.mean_acc[i], s=250, c=colors[i], marker=markers[i], 
                   edgecolor='black', linewidth=1.5, zorder=5, label=df.model[i])

    # 2. Add "Real-Time Constraint" Zone
    # Draw a line at 10us (arbitrary "Edge limit")
    ax.axvline(x=10, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(11, 0.806, "Edge Latency Limit (10µs)", rotation=90, verticalalignment='bottom', color='#555', fontsize=10)

    # Shade the "Viable for Edge" area (Green) vs "Too Slow" area (Red)
    ax.axvspan(0, 10, alpha=0.1, color='#4CAF50')  # Green zone
    ax.axvspan(10, 85, alpha=0.05, color='#F44336') # Red zone
    
    ax.text(2, 0.840, "REGION I:\nReal-Time Viable", color='#1B5E20', fontweight='bold', fontsize=11)
    ax.text(38, 0.840, "REGION II:\nServer-Dependent (High Latency)", color='#B71C1C', fontweight='bold', fontsize=11)

    # 3. Add Trade-off Annotation (Arrow)
    # Pointing from kNN (Highest Acc) to LR (Fastest)
    knn_row = df[df.model == "kNN"].iloc[0]
    lr_row = df[df.model == "LogisticRegression"].iloc[0]
    
    ax.annotate(
        f"13x Speedup\n(Selected)",
        xy=(lr_row.latency_us, lr_row.mean_acc), 
        xytext=(20, 0.815),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black', lw=2),
        fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )

    # 4. Labels & Titles
    ax.set_title("Hardware Efficiency Analysis: Accuracy vs. Latency", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Inference Latency ($\mu$s/sample) [Log Scale recommended visually]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Classification Accuracy (Mean CV)", fontsize=12, fontweight='bold')
    
    # Adjust limits to make room for text
    ax.set_xlim(0, 85)
    ax.set_ylim(0.800, 0.845)
    
    # Legend
    ax.legend(title="Model Architecture", loc="upper right", frameon=True, shadow=True)

    # 5. Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[OK] Advanced scatter plot saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_plot()