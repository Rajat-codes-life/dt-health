"""Train the heart-disease likelihood model on REAL clinical data (UCI Heart Disease, ml/data/heart.csv)
and an anomaly detector for live vitals.

Data notes (important for the report):
 * The Kaggle heart.csv has 1025 rows but only 302 unique patients -> we DROP DUPLICATES,
   otherwise identical rows land in both train and test and inflate the scores.
 * In this file target=1 means NO disease (older patients, low max-HR, high ST-depression, more blocked
   vessels all sit in target=0). We therefore define  risk = 1 - target.
 * Anomaly detector: there is no public real-vitals file here, so it is trained on simulated normal vitals.
"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models"); os.makedirs(OUT, exist_ok=True)
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]

def load():
    raw = pd.read_csv(os.path.join(HERE, "data", "heart.csv"))
    df = raw.drop_duplicates().reset_index(drop=True)
    df["risk"] = 1 - df["target"]          # target=1 means healthy in this file
    print(f"Rows: {len(raw)} raw -> {len(df)} unique patients | disease rate {df['risk'].mean():.1%}")
    return raw, df

def main():
    raw, df = load()
    X, y = df[FEATURES], df["risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cands = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = {}
    for n, m in cands.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        cv_auc[n] = (round(s.mean(), 3), round(s.std(), 3))
        print(f"  5-fold CV ROC-AUC  {n:20s} {s.mean():.3f} +/- {s.std():.3f}")
    best = max(cv_auc, key=lambda k: cv_auc[k][0])
    model = cands[best].fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    m = {"rows_raw": len(raw), "rows_unique": len(df), "best_model": best,
         "cv_roc_auc": {k: {"mean": v[0], "std": v[1]} for k, v in cv_auc.items()},
         "test": {"n": len(yte), "accuracy": round(accuracy_score(yte, pred), 3),
                  "precision": round(precision_score(yte, pred), 3), "recall": round(recall_score(yte, pred), 3),
                  "f1": round(f1_score(yte, pred), 3), "roc_auc": round(roc_auc_score(yte, proba), 3),
                  "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}}
    pi = permutation_importance(model, Xte, yte, n_repeats=20, random_state=42, scoring="roc_auc")
    m["feature_importance"] = dict(sorted(zip(FEATURES, pi.importances_mean.round(4).tolist()), key=lambda t: -t[1]))
    joblib.dump(model, os.path.join(OUT, "heart_model.joblib"))
    # baseline = median patient of the TRAINING data (reference for Shapley explanations)
    json.dump({k: float(v) for k, v in Xtr.median().items()}, open(os.path.join(OUT, "baseline.json"), "w"), indent=2)
    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"\nBEST: {best}\nHeld-out test ({len(yte)} patients): {m['test']}")
    print("Top features:", list(m["feature_importance"].items())[:5])

    # Anomaly detector on simulated NORMAL vitals
    r = np.random.default_rng(1); n = 5000
    V = pd.DataFrame({"hr": r.normal(76, 10, n), "spo2": np.clip(r.normal(97, 1.2, n), 90, 100),
                      "sys_bp": r.normal(122, 12, n), "dia_bp": r.normal(80, 8, n),
                      "temp": r.normal(36.8, 0.3, n), "glucose": r.normal(105, 18, n)})
    joblib.dump(IsolationForest(n_estimators=150, contamination=0.03, random_state=1).fit(V),
                os.path.join(OUT, "anomaly_iso.joblib"))

    try:   # report figures (optional)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        cm = confusion_matrix(yte, pred); ax[0].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax[0].text(j, i, v, ha="center", va="center", fontsize=14)
        ax[0].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1],
                  xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
        fpr, tpr, _ = roc_curve(yte, proba); ax[1].plot(fpr, tpr, lw=2); ax[1].plot([0, 1], [0, 1], "--", c="grey")
        ax[1].set(title=f"ROC curve (AUC {m['test']['roc_auc']})", xlabel="False positive rate", ylabel="True positive rate")
        fi = list(m["feature_importance"].items())[:8][::-1]
        ax[2].barh([k for k, _ in fi], [v for _, v in fi], color="#2f5d8a"); ax[2].set(title="Feature importance (permutation)")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "evaluation.png"), dpi=150)
        print("Saved ml/models/evaluation.png")
    except Exception as e:
        print("(matplotlib not installed, skipping figure)", e)

if __name__ == "__main__":
    main()
