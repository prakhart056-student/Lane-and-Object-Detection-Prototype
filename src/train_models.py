"""Train and evaluate risk-anticipation models.

Protocol
--------
* US-101 is split by *vehicle* (GroupShuffleSplit) into train / validation /
  test, so no driver appears in more than one split.
* The decision threshold of each model is tuned on validation (max F1).
* I-80 (different freeway, different day, different congestion) is held out
  completely as an out-of-distribution test of generalisation.
* Physics baselines (TTC, MTTC, CRI on the current state) are evaluated with
  the same protocol.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import CONTEXT, ONBOARD, build_dataset
from .preprocess import load


def get_models():
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5, n_jobs=-1,
            random_state=rs),
        "MLP Neural Network": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                            early_stopping=True, max_iter=300,
                                            random_state=rs)),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=rs),
    }


BASELINES = {"TTC rule": "inv_ttc", "MTTC rule": "inv_mttc", "CRI rule": "cri"}


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def evaluate(y, s, thr):
    yhat = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "f1": f1_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": thr,
    }


def split_us101(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=C.RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df["vehicle_id"]))
    train_full, test = df.iloc[tr], df.iloc[te]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.RANDOM_STATE)
    tr2, va = next(gss2.split(train_full, groups=train_full["vehicle_id"]))
    return train_full.iloc[tr2], train_full.iloc[va], test


def main():
    t0 = time.time()
    data = {s: build_dataset(load(f"pairs_{s}")) for s in C.SITES}
    for s, d in data.items():
        d.to_pickle(C.DATA_PROCESSED / f"dataset_{s}.pkl")
        print(f"{s}: {len(d):,} samples, positive rate {d.label.mean():.3f}")

    train, val, test = split_us101(data["us-101"])
    ood = data["i-80"]
    sets = {"US-101 test": test, "I-80 (OOD)": ood}

    results, curves, scores_store = {}, {}, {}

    # ---------- physics baselines ----------
    for name, col in BASELINES.items():
        thr = best_threshold(val["label"], val[col])
        results[name] = {k: evaluate(d["label"], d[col].to_numpy(), thr) for k, d in sets.items()}
        scores_store[name] = {k: d[col].to_numpy() for k, d in sets.items()}

    # ---------- ML models ----------
    fitted = {}
    for fs_name, feats in [("onboard", ONBOARD), ("context", CONTEXT)]:
        for name, model in get_models().items():
            key = f"{name} [{fs_name}]"
            tic = time.time()
            model.fit(train[feats], train["label"])
            thr = best_threshold(val["label"], model.predict_proba(val[feats])[:, 1])
            res = {}
            scores_store[key] = {}
            for sname, d in sets.items():
                s = model.predict_proba(d[feats])[:, 1]
                res[sname] = evaluate(d["label"], s, thr)
                res[sname]["brier"] = brier_score_loss(d["label"], s)
                scores_store[key][sname] = s
            res["train_seconds"] = time.time() - tic
            results[key] = res
            fitted[key] = (model, feats, thr)
            print(f"{key:40s} AUC={res['US-101 test']['roc_auc']:.3f} "
                  f"PR-AUC={res['US-101 test']['pr_auc']:.3f} | I-80 AUC="
                  f"{res['I-80 (OOD)']['roc_auc']:.3f} ({res['train_seconds']:.1f}s)")

    # ---------- select deployable onboard model by validation PR-AUC ----------
    val_pr = {k: average_precision_score(val["label"], m.predict_proba(val[f])[:, 1])
              for k, (m, f, _) in fitted.items() if "[onboard]" in k}
    best_key = max(val_pr, key=val_pr.get)
    model, feats, thr = fitted[best_key]
    joblib.dump({"model": model, "features": feats, "threshold": thr, "name": best_key},
                C.MODELS / "risk_model.joblib")
    print(f"Selected onboard model: {best_key} (val PR-AUC {val_pr[best_key]:.3f})")

    # ---------- permutation importance ----------
    sub = test.sample(min(15000, len(test)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(model, sub[feats], sub["label"], n_repeats=5,
                                scoring="average_precision", random_state=C.RANDOM_STATE,
                                n_jobs=-1)
    importance = dict(sorted(zip(feats, pi.importances_mean.tolist()),
                             key=lambda kv: -kv[1]))

    # ---------- curves ----------
    for key in list(BASELINES) + [best_key] + [k for k in fitted if "[onboard]" in k]:
        curves[key] = {}
        for sname, d in sets.items():
            s = scores_store[key][sname]
            fpr, tpr, _ = roc_curve(d["label"], s)
            pr, rc, _ = precision_recall_curve(d["label"], s)
            i1 = np.unique(np.r_[np.linspace(0, len(fpr) - 1, 400).astype(int), len(fpr) - 1])
            i2 = np.unique(np.r_[np.linspace(0, len(pr) - 1, 400).astype(int), len(pr) - 1])
            curves[key][sname] = {"fpr": fpr[i1].tolist(), "tpr": tpr[i1].tolist(),
                                  "precision": pr[i2].tolist(), "recall": rc[i2].tolist()}

    # ---------- anticipation: recall by time-to-event at matched false-alarm ----------
    lead = {}
    for sname, d in sets.items():
        neg = d["label"] == 0
        s_model = scores_store[best_key][sname]
        far_model = results[best_key][sname]["false_alarm_rate"]
        s_ttc = d["inv_ttc"].to_numpy()
        thr_ttc = np.quantile(s_ttc[neg], 1 - far_model)      # same false-alarm rate
        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.01]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d["label"] == 1) & (d["time_to_event"] > lo) & (d["time_to_event"] <= hi)
            m = m.to_numpy()
            rows.append({"bin": f"{lo:.1f}-{min(hi,3.0):.1f}", "n": int(m.sum()),
                         "model_recall": float((s_model[m] >= thr).mean()) if m.any() else np.nan,
                         "ttc_recall": float((s_ttc[m] >= thr_ttc).mean()) if m.any() else np.nan})
        lead[sname] = {"false_alarm_rate": far_model, "bins": rows}

    out = {
        "dataset": {s: {"samples": int(len(d)), "vehicles": int(d.vehicle_id.nunique()),
                        "positive_rate": float(d.label.mean())} for s, d in data.items()},
        "splits": {"train": len(train), "val": len(val), "test": len(test), "ood": len(ood)},
        "results": results, "best_model": best_key, "importance": importance,
        "anticipation": lead, "runtime_s": time.time() - t0,
    }
    with open(C.REPORTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(C.REPORTS / "curves.json", "w") as f:
        json.dump(curves, f)

    # reliability (calibration) data for best model
    calib = {}
    for sname, d in sets.items():
        s = scores_store[best_key][sname]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        calib[sname] = [{"pred": float(s[idx == b].mean()), "obs": float(d["label"].to_numpy()[idx == b].mean()),
                         "n": int((idx == b).sum())} for b in range(10) if (idx == b).sum() > 30]
    with open(C.REPORTS / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
