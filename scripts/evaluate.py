#!/usr/bin/env python3
"""Score both saved models and rebuild every figure referenced by the README.

Produces
--------
reports/metrics.json        every number in the README's results tables
reports/metrics_chart.png   ROC, precision-recall, calibration, threshold sweep
reports/shap_summary.png    SHAP beeswarm for the tuned model
reports/eda_overview.png    failure modes, load regime, operating envelope

Kept separate from training on purpose: scoring reads the pickles, so the
figures can be regenerated without refitting a single tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_predict import (FEATURES, MODE_COLUMNS, TARGET, default_data_path,  # noqa: E402
                           load_dataset)

RED, INK, GREY = "#E53E3E", "#111111", "#9CA3AF"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
})

MODELS = {"baseline": ROOT / "src" / "model.pkl", "tuned": ROOT / "src" / "model_tuned.pkl"}


def operating_point(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average=None, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"precision": round(float(p[1]), 3), "recall": round(float(r[1]), 3),
            "f1": round(float(f1[1]), 3), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


def best_f1_threshold(y_true, prob):
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def main() -> None:
    df = load_dataset(default_data_path())
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    any_mode = (df[MODE_COLUMNS].sum(axis=1) > 0).astype(int)
    out = {
        "dataset": {
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "failures": int(y.sum()), "failure_rate_pct": round(100 * y.mean(), 2),
            "missing_values": int(df.isna().sum().sum()),
            "type_counts": {k: int(v) for k, v in df["Type"].value_counts().items()},
            "mode_counts": {m: int(df[m].sum()) for m in MODE_COLUMNS},
            "rows_with_multiple_modes": int((df[MODE_COLUMNS].sum(axis=1) > 1).sum()),
            "label_inconsistencies": {
                "flag_1_no_mode": int(((y == 1) & (any_mode == 0)).sum()),
                "mode_set_flag_0": int(((y == 0) & (any_mode == 1)).sum()),
            },
            "feature_range": {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                                  "mean": round(float(df[f].mean()), 2)} for f in FEATURES},
            "correlations": {"torque_vs_rpm": round(float(df[FEATURES[2]].corr(df[FEATURES[3]])), 3),
                             "air_vs_process_temp": round(float(df[FEATURES[0]].corr(df[FEATURES[1]])), 3)},
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "test_positives": int(y_test.sum()),
        }
    }

    results, probs = {}, {}
    for name, path in MODELS.items():
        clf = joblib.load(path)
        prob = clf.predict_proba(X_test)[:, 1]
        probs[name] = prob
        thr = best_f1_threshold(y_test, prob)
        per_mode = {}
        for m in MODE_COLUMNS:
            mask = (df.loc[y_test.index, m] == 1).to_numpy()
            per_mode[m] = {
                "n": int(mask.sum()),
                "flagged_at_0.5": round(float((prob[mask] >= 0.5).mean()), 3) if mask.any() else None,
                "mean_prob": round(float(prob[mask].mean()), 3) if mask.any() else None,
            }
        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, prob)), 4),
            "brier": round(float(brier_score_loss(y_test, prob)), 4),
            "at_0.5": operating_point(y_test, prob, 0.5),
            "best_f1_threshold": round(thr, 3),
            "at_best_f1": operating_point(y_test, prob, thr),
            "per_mode": per_mode,
            "params": {k: v for k, v in clf.get_params().items()
                       if k in ("n_estimators", "max_depth", "min_samples_leaf",
                                "max_features", "class_weight")},
        }
    out["metrics"] = results

    tuned = joblib.load(MODELS["tuned"])
    cv = cross_val_score(tuned, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc", n_jobs=-1)
    out["cv_on_train"] = {"folds": [round(float(s), 4) for s in cv],
                          "mean": round(float(cv.mean()), 4), "std": round(float(cv.std()), 4)}

    pi = permutation_importance(tuned, X_test, y_test, n_repeats=10, random_state=42,
                                scoring="roc_auc", n_jobs=-1)
    out["permutation_importance"] = {f: round(float(v), 4) for f, v in
                                     sorted(zip(FEATURES, pi.importances_mean), key=lambda t: -t[1])}

    # Ablation: does the categorical product-quality class add anything?
    Xtr2 = pd.concat([X_train, pd.get_dummies(df.loc[X_train.index, "Type"], prefix="type", dtype=int)], axis=1)
    Xte2 = pd.concat([X_test, pd.get_dummies(df.loc[X_test.index, "Type"], prefix="type", dtype=int)], axis=1)
    Xte2 = Xte2.reindex(columns=Xtr2.columns, fill_value=0)
    from sklearn.ensemble import RandomForestClassifier
    clf2 = RandomForestClassifier(**tuned.get_params()).fit(Xtr2, y_train)
    auc2 = float(roc_auc_score(y_test, clf2.predict_proba(Xte2)[:, 1]))
    out["ablation_type_feature"] = {"roc_auc_with_type": round(auc2, 4),
                                    "delta_vs_five_sensors": round(auc2 - results["tuned"]["roc_auc"], 4)}

    sv = shap.TreeExplainer(tuned).shap_values(X_test)
    sv = sv[..., 1] if getattr(sv, "ndim", 2) == 3 else (sv[1] if isinstance(sv, list) else sv)
    sv = np.asarray(sv)
    out["shap"] = dict(sorted({f: {"mean_abs": round(float(np.abs(sv[:, i]).mean()), 4),
                                   "direction_corr": round(float(np.corrcoef(X_test[f].to_numpy(), sv[:, i])[0, 1]), 3)}
                               for i, f in enumerate(FEATURES)}.items(), key=lambda t: -t[1]["mean_abs"]))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))

    for name, color in (("baseline", GREY), ("tuned", RED)):
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        ax[0, 0].plot(fpr, tpr, color=color, lw=1.7,
                      label=f"{name} · AUC {results[name]['roc_auc']:.3f}")
    ax[0, 0].plot([0, 1], [0, 1], ls="--", color="#D1D5DB", lw=1)
    ax[0, 0].set(xlabel="False positive rate", ylabel="True positive rate",
                 title="ROC — held-out test set (2000 rows, 68 failures)")
    ax[0, 0].legend(loc="lower right", frameon=False, fontsize=8)

    for name, color in (("baseline", GREY), ("tuned", RED)):
        prec, rec, _ = precision_recall_curve(y_test, probs[name])
        ax[0, 1].plot(rec, prec, color=color, lw=1.7,
                      label=f"{name} · AP {results[name]['pr_auc']:.3f}")
    ax[0, 1].axhline(y_test.mean(), ls="--", color="#D1D5DB", lw=1)
    ax[0, 1].text(0.02, y_test.mean() + 0.02, "prevalence 3.4%", fontsize=7, color=GREY)
    ax[0, 1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (AP = area under this curve)")
    ax[0, 1].legend(loc="upper right", frameon=False, fontsize=8)

    for name, color in (("baseline", GREY), ("tuned", RED)):
        frac_pos, mean_pred = calibration_curve(y_test, probs[name], n_bins=8, strategy="quantile")
        ax[1, 0].plot(mean_pred, frac_pos, marker="o", ms=3, color=color, lw=1.4,
                      label=f"{name} · Brier {results[name]['brier']:.3f}")
    ax[1, 0].plot([0, 1], [0, 1], ls="--", color="#D1D5DB", lw=1)
    ax[1, 0].set(xlabel="Mean predicted probability", ylabel="Observed failure rate",
                 title="Calibration — does 0.5 mean 50%?", xlim=(0, 1), ylim=(0, 1))
    ax[1, 0].legend(loc="upper left", frameon=False, fontsize=8)

    grid = np.linspace(0.01, 0.99, 99)
    for name, color in (("baseline", GREY), ("tuned", RED)):
        p = [operating_point(y_test, probs[name], t)["precision"] for t in grid]
        r = [operating_point(y_test, probs[name], t)["recall"] for t in grid]
        ax[1, 1].plot(grid, p, color=color, lw=1.5, label=f"{name} precision")
        ax[1, 1].plot(grid, r, color=color, lw=1.5, ls=":", label=f"{name} recall")
        ax[1, 1].axvline(results[name]["best_f1_threshold"], color=color, lw=0.8, alpha=0.35)
    ax[1, 1].set(xlabel="Decision threshold", ylabel="Score", title="Operating point — precision vs recall",
                 ylim=(0, 1.02))
    ax[1, 1].legend(frameon=False, fontsize=7, ncol=2, loc="lower center")

    fig.suptitle("AI4I 2020 failure classifier — verified metrics", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "reports" / "metrics_chart.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))

    modes = [m for m in MODE_COLUMNS]
    counts = [out["dataset"]["mode_counts"][m] for m in modes]
    bars = ax[0, 0].bar(modes, counts, color=RED, width=0.55)
    ax[0, 0].bar_label(bars, fontsize=8)
    ax[0, 0].set(title="Failure modes are rare and uneven", ylabel="rows flagged",
                 xticks=range(len(modes)),
                 xticklabels=["TWF\ntool wear", "HDF\nheat diss.", "PWF\npower",
                              "OSF\noverstrain", "RNF\nrandom"])
    ax[0, 0].tick_params(axis="x", labelsize=7)

    by_type = df.groupby("Type")[TARGET].agg(["mean", "size"]).reindex(["L", "M", "H"])
    bars = ax[0, 1].bar(by_type.index, 100 * by_type["mean"], color=RED, width=0.5)
    ax[0, 1].bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax[0, 1].set(title="Failure rate falls as the quality variant improves", ylabel="% failing",
                 xlabel="product quality variant (L = cheapest)")
    for i, (t, row) in enumerate(by_type.iterrows()):
        ax[0, 1].text(i, 0.15, f"n={int(row['size'])}", ha="center", fontsize=7, color="white")

    ok, bad = df[df[TARGET] == 0], df[df[TARGET] == 1]
    ax[1, 0].scatter(ok["Rotational speed [rpm]"], ok["Torque [Nm]"], s=3, c="#D1D5DB",
                     label="no failure", rasterized=True)
    for m, c in zip(("PWF", "OSF", "HDF"), ("#111111", RED, "#B45309")):
        sub = df[df[m] == 1]
        ax[1, 0].scatter(sub["Rotational speed [rpm]"], sub["Torque [Nm]"], s=16, c=c, label=m)
    ax[1, 0].set(xlabel="Rotational speed [rpm]", ylabel="Torque [Nm]",
                 title="Operating envelope: power failures at low speed, overstrain at high")
    ax[1, 0].legend(frameon=False, fontsize=7, markerscale=1.2)

    groups = [ok["Tool wear [min]"], bad["Tool wear [min]"]]
    bp = ax[1, 1].boxplot(groups, tick_labels=["no failure", "failure"], patch_artist=True, widths=0.45,
                          medianprops={"color": INK}, flierprops={"markersize": 2,
                                                                  "markerfacecolor": GREY,
                                                                  "markeredgecolor": GREY})
    for patch, color in zip(bp["boxes"], ("#E5E7EB", RED)):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax[1, 1].set(ylabel="Tool wear [min]", title="Failed rows sit further into tool life")

    fig.suptitle("AI4I 2020 — what the data looks like before modelling", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "reports" / "eda_overview.png", dpi=170)
    plt.close(fig)

    plt.figure(figsize=(7.4, 4.2))
    shap.summary_plot(sv, X_test, show=False, plot_size=None)
    plt.title("SHAP — per-row impact on P(failure), tuned model", fontsize=10)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "shap_summary.png", dpi=170)
    plt.close()

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("metrics", "cv_on_train", "ablation_type_feature",
                                         "permutation_importance", "shap")}, indent=2))


if __name__ == "__main__":
    main()
