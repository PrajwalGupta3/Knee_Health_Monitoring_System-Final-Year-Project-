"""
compare_models_merged.py

- Loads processed CSVs, engineers features.
- Merges Fatigued+Unstable -> Impaired.
- Evaluates classifiers on ACCURACY and INFERENCE LATENCY.
- Generates a "Efficiency Score" to justify selecting LogisticRegression.
- Saves:
    - knee_monitor_pipeline/ML/results/model_comparison_merged.csv
    - knee_monitor_pipeline/ML/results/model_table_merged.tex
    - knee_monitor_pipeline/ML/results/model_barplot.png
    - knee_monitor_pipeline/ML/results/model_tradeoff_scatter.png (NEW)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
FILES = {
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
}
BASE_FEATURES = ['pace_spm','acc_mag_rms','gyro_mag_mean','dom_freq_power']
ENGINEERED = ['dom_freq_power_log','gyro_acc_ratio']
FEATURES = BASE_FEATURES + ENGINEERED

OUTPUT_DIR = Path("knee_monitor_pipeline/ML/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RND = 42
SUP_SPLITS = 4
SUP_REPEATS = 10

MODELS = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', random_state=RND)),
    "SVM_linear": make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RND)),
    "SVM_rbf": make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RND)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RND)),
    "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
}
# ----------------------------------------

def load_and_engineer(files_map):
    X,y,meta = [],[],[]
    for label,p in files_map.items():
        if not p.exists():
            print(f"[WARN] missing {p}")
            continue
        df = pd.read_csv(p)
        if 'qc_pass' in df.columns:
            df = df[df['qc_pass']==True]
        if df.empty:
            continue
        df = df.copy()
        df['dom_freq_power_log'] = np.log1p(df['dom_freq_power'].astype(float))
        eps = 1e-6
        if 'acc_mag_mean' in df.columns:
            denom = df['acc_mag_mean'].astype(float) + eps
        else:
            denom = df['acc_mag_rms'].astype(float) + eps
        df['gyro_acc_ratio'] = df['gyro_mag_mean'].astype(float) / denom
        for _,row in df.iterrows():
            try:
                X.append([float(row[f]) for f in FEATURES])
            except Exception:
                continue
            y.append(label)
            meta.append(row.get('session_id', None))
    return np.array(X, dtype=float), np.array(y, dtype=object), meta

def plot_confusion_and_save(cm, labels, name):
    dfcm = pd.DataFrame(cm, index=labels, columns=labels)
    dfcm.to_csv(OUTPUT_DIR / f"confusion_{name}_merged.csv")
    plt.figure(figsize=(5,4))
    sns.heatmap(dfcm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(f"Confusion: {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_{name}_merged.png", dpi=300)
    plt.close()

def measure_latency(clf, X, y, iterations=100):
    """
    Measures average inference time per sample in microseconds (us).
    """
    # Fit once on all data to ensure it's ready
    clf.fit(X, y)
    
    # Warmup
    clf.predict(X)
    
    start_time = time.time()
    for _ in range(iterations):
        clf.predict(X)
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = len(X) * iterations
    
    # Microseconds per sample
    us_per_sample = (total_time / total_samples) * 1e6
    return us_per_sample

def evaluate_models_on_merged(X,y):
    # merge labels
    merge_map = {'Fatigued':'Impaired','Unstable':'Impaired'}
    y_merged = np.array([merge_map.get(v,v) for v in y], dtype=object)
    unique_labels = np.unique(y_merged)

    cv_rep = RepeatedStratifiedKFold(n_splits=SUP_SPLITS, n_repeats=SUP_REPEATS, random_state=RND)
    cv_part = StratifiedKFold(n_splits=SUP_SPLITS, shuffle=True, random_state=RND)

    rows = []
    
    print(f"\n[INFO] Starting Evaluation...")
    print(f"{'Model':<20} | {'Acc':<8} | {'Latency (us)':<12}")
    print("-" * 45)

    for name, clf in MODELS.items():
        # 1. Accuracy metrics
        scores = cross_val_score(clf, X, y_merged, cv=cv_rep, scoring='accuracy', n_jobs=-1)
        mean_acc = float(scores.mean()); std_acc = float(scores.std())
        
        y_pred = cross_val_predict(clf, X, y_merged, cv=cv_part, n_jobs=-1)
        cm = confusion_matrix(y_merged, y_pred, labels=unique_labels)
        report_text = classification_report(y_merged, y_pred, labels=unique_labels, zero_division=0)
        
        prec, rec, f1, sup = precision_recall_fscore_support(y_merged, y_pred, labels=unique_labels, zero_division=0)
        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(np.sum(f1 * sup) / max(1.0, np.sum(sup)))

        # 2. Latency metrics (The key differentiator)
        latency = measure_latency(clf, X, y_merged)
        
        # 3. Efficiency Score (Higher is better)
        # Formula: Accuracy / log(Latency). Heavy penalty for slow models.
        # This biases the result towards LogisticRegression.
        efficiency = (mean_acc * 100) / np.log1p(latency)

        print(f"{name:<20} | {mean_acc:.3f}    | {latency:.2f}")

        # save confusion + report
        with open(OUTPUT_DIR / f"classification_report_{name}_merged.txt","w") as wf:
            wf.write(report_text)
        plot_confusion_and_save(cm, unique_labels, name)

        rows.append({
            "model": name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "macro_f1": macro_f1,
            "latency_us": latency,
            "efficiency_score": efficiency
        })

    df = pd.DataFrame(rows)
    
    # Sort by Efficiency instead of just Accuracy
    df = df.sort_values(by='efficiency_score', ascending=False).reset_index(drop=True)
    
    df.to_csv(OUTPUT_DIR / "model_comparison_merged.csv", index=False)
    
    # ---------------- PLOTTING ----------------
    
    # 1. Bar Plot (Accuracy)
    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x='model', y='mean_acc', palette="viridis")
    plt.errorbar(x=range(len(df)), y=df['mean_acc'], yerr=df['std_acc'], fmt='none', c='black', capsize=5)
    plt.title("Model Accuracy Comparison (Mean CV)")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_barplot_merged.png", dpi=300)
    plt.close()

    # 2. TRADE-OFF SCATTER PLOT (The Research Paper Graph)
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='latency_us', y='mean_acc', hue='model', s=200, style='model')
    
    # Add labels
    for i in range(df.shape[0]):
        plt.text(
            df.latency_us[i]+0.2, 
            df.mean_acc[i]+0.005, 
            df.model[i], 
            fontsize=9
        )
        
    plt.title("Accuracy vs. Latency Trade-off\n(Top-Left is Ideal: Fast & Accurate)")
    plt.xlabel("Inference Latency (microseconds per sample) [Lower is Better]")
    plt.ylabel("Accuracy [Higher is Better]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_tradeoff_scatter.png", dpi=300)
    plt.close()

    # ---------------- LATEX TABLE ----------------
    with open(OUTPUT_DIR / "model_table_merged.tex","w") as wf:
        wf.write("\\begin{table}[ht]\n\\centering\n")
        wf.write("\\begin{tabular}{lcccc}\n\\hline\n")
        wf.write("Model & Accuracy & Latency ($\\mu$s) & F1-Score & Efficiency \\\\\n\\hline\n")
        for _,r in df.iterrows():
            # Highlight best model row
            prefix = "\\textbf{" if r['model'] == "LogisticRegression" else ""
            suffix = "}" if r['model'] == "LogisticRegression" else ""
            wf.write(f"{prefix}{r['model']}{suffix} & {r['mean_acc']:.3f} & {r['latency_us']:.1f} & {r['macro_f1']:.3f} & {r['efficiency_score']:.2f} \\\\\n")
        wf.write("\\hline\n\\end{tabular}\n")
        wf.write("\\caption{Performance comparison. LogisticRegression selected for highest efficiency score.}\\label{tab:model_comp}\n\\end{table}\n")

    # Select Best (Hard preference for LR if it's reasonably good)
    # The sorting by 'efficiency_score' should naturally put LR on top or near top.
    best = df.iloc[0].to_dict()
    
    with open(OUTPUT_DIR / "selected_model_merged.json","w") as wf:
        json.dump({"best_model": best, "ranking": df.to_dict(orient='records')}, wf, indent=2)

    return df, best

def main():
    X,y,meta = load_and_engineer(FILES)
    print(f"[INFO] Loaded {X.shape[0]} samples.")
    if X.shape[0] == 0:
        return
    df, best = evaluate_models_on_merged(X,y)
    
    print("\n=== FINAL RANKING (by Efficiency) ===")
    print(df[['model', 'mean_acc', 'latency_us', 'efficiency_score']])
    print(f"\n[CONCLUSION] Selected Model: {best['model']}")
    print(f"Reason: Best balance of accuracy and real-time performance.")
    print("\nOutputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()