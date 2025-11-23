
"""
Merged Fatigued and Unstable into a single label "Impaired"),
then evaluated supervised logistic and centroid LOOCV, printing confusion matrices and reports.
This is to decide which ML model to be utilised.

"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings("ignore")

FILES = {
    'Healthy': Path("knee_monitor_pipeline/data/processed/final_csv/healthy_baseline_qc.csv"),
    'Recovery': Path("knee_monitor_pipeline/data/processed/final_csv/target_recovery_qc.csv"),
    'Fatigued': Path("knee_monitor_pipeline/data/processed/final_csv/target_fatigued_qc.csv"),
    'Unstable': Path("knee_monitor_pipeline/data/processed/final_csv/target_unstable_qc.csv"),
}
BASE_FEATURES = ['pace_spm','acc_mag_rms','gyro_mag_mean','dom_freq_power']

ENGINEERED = ['dom_freq_power_log','gyro_acc_ratio']
FEATURES = BASE_FEATURES + ENGINEERED

SUP_SPLITS = 4
SUP_REPEATS = 10
RND = 42

def load_and_engineer(files_map):
    X = []
    y = []
    meta = []
    for label,p in files_map.items():
        if not p.exists(): 
            print("missing:", p)
            continue
        df = pd.read_csv(p)
        if 'qc_pass' in df.columns:
            df = df[df['qc_pass']==True]
        if df.empty:
            continue
        df['dom_freq_power_log'] = np.log1p(df['dom_freq_power'].astype(float))
        eps = 1e-6
        df['gyro_acc_ratio'] = df['gyro_mag_mean'].astype(float) / (df['acc_mag_rms'].astype(float) + eps)
        for _,row in df.iterrows():
            X.append([float(row[f]) for f in FEATURES])
            y.append(label)
            meta.append(row.get('session_id', None))
    return np.array(X, dtype=float), np.array(y, dtype=object), meta

def merge_labels(y, merge_map):
    y_new = np.array([merge_map.get(v, v) for v in y], dtype=object)
    return y_new

def run_supervised_cv(X,y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RND))
    cv_rep = RepeatedStratifiedKFold(n_splits=SUP_SPLITS, n_repeats=SUP_REPEATS, random_state=RND)
    scores = cross_val_score(clf, X, y, cv=cv_rep, scoring='accuracy', n_jobs=-1)
    mean_acc, std_acc = float(scores.mean()), float(scores.std())
    cv_part = StratifiedKFold(n_splits=SUP_SPLITS, shuffle=True, random_state=RND)
    y_pred = cross_val_predict(clf, X, y, cv=cv_part, n_jobs=-1)
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    report = classification_report(y, y_pred, zero_division=0)
    return mean_acc, std_acc, cm, report

def pooled_cov_inv(X):
    if X.shape[0] < 2:
        return np.eye(X.shape[1]) * 1e-6, np.linalg.inv(np.eye(X.shape[1]) * 1e-6)
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    cov = lw.covariance_
    cov += np.eye(cov.shape[0]) * 1e-8
    cov_inv = np.linalg.inv(cov)
    return cov, cov_inv

def run_loocv_centroid(X,y):
    n = X.shape[0]
    labels_unique = np.unique(y)
    true = []
    pred = []
    dist_true_list = {lab:[] for lab in labels_unique}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        xt, yt = X[i], y[i]

        #centroids
        centroids = {}
        for lab in labels_unique:
            m = Xtr[ytr==lab]
            if len(m)>0:
                centroids[lab] = m.mean(axis=0)

        #pooled covariance inverse
        _, cov_inv = pooled_cov_inv(Xtr)

        #the distances
        dists = {lab: float(np.sqrt(((xt - c).T.dot(cov_inv).dot(xt - c)))) for lab,c in centroids.items()}
        if len(dists)==0:
            p=None
        else:
            p = min(dists, key=dists.get)
        pred.append(p)
        true.append(yt)
        if yt in dists:
            dist_true_list[yt].append(dists[yt])
    cm = confusion_matrix(true, pred, labels=labels_unique)
    acc = accuracy_score(true, pred)
    report = classification_report(true, pred, zero_division=0)
    thresholds = {lab: (None if len(vals)==0 else float(np.quantile(vals,0.95))) for lab,vals in dist_true_list.items()}
    return acc, cm, report, thresholds

def pretty_print_cm(cm, labels):
    df = pd.DataFrame(cm, index=labels, columns=labels)
    print(df)

def main():
    X,y,meta = load_and_engineer(FILES)
    print("Loaded:", X.shape[0], "samples")

    #baseline: the original labels
    print("\n=== Original labels evaluation ===")
    mean_acc, std_acc, cm_sup, rep_sup = run_supervised_cv(X,y)
    print(f"Supervised CV acc: {mean_acc:.3f} +/- {std_acc:.3f}")
    print("Supervised CM:")
    pretty_print_cm(cm_sup, np.unique(y))
    print("Supervised report:\n", rep_sup)
    acc_loocv, cm_loocv, rep_loocv, thr = run_loocv_centroid(X,y)
    print(f"\nLOOCV centroid acc: {acc_loocv:.3f}")
    print("LOOCV CM:")
    pretty_print_cm(cm_loocv, np.unique(y))
    print("LOOCV report:\n", rep_loocv)
    print("LOOCV 95% thresholds:", thr)

    #merged labels
    merge_map = {'Fatigued':'Impaired','Unstable':'Impaired'}
    y_merged = merge_labels(y, merge_map)
    print("\n=== After merge: Fatigued+Unstable => Impaired ===")
    unique, counts = np.unique(y_merged, return_counts=True)
    for u,c in zip(unique,counts):
        print(f"  {u}: {c}")

    mean_acc_m, std_acc_m, cm_sup_m, rep_sup_m = run_supervised_cv(X,y_merged)
    print(f"\nSupervised CV acc (merged): {mean_acc_m:.3f} +/- {std_acc_m:.3f}")
    print("Supervised CM (merged):")
    pretty_print_cm(cm_sup_m, np.unique(y_merged))
    print("Supervised report (merged):\n", rep_sup_m)

    acc_loocv_m, cm_loocv_m, rep_loocv_m, thr_m = run_loocv_centroid(X,y_merged)
    print(f"\nLOOCV centroid acc (merged): {acc_loocv_m:.3f}")
    print("LOOCV CM (merged):")
    pretty_print_cm(cm_loocv_m, np.unique(y_merged))
    print("LOOCV report (merged):\n", rep_loocv_m)
    print("LOOCV 95% thresholds (merged):", thr_m)

if __name__=="__main__":
    main()
