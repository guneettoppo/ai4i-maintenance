#!/usr/bin/env python3
"""Baseline model: five sensor channels -> P(machine failure).

Trains one RandomForestClassifier with fixed, hand-picked settings, then writes
the model and its test-set predictions. This is the reference point the tuned
model in ``scripts/train_with_optuna.py`` is measured against.

Artifacts
---------
src/model.pkl            fitted estimator (joblib)
outputs/predictions.csv  actual / predicted / prob_failure on the test split
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

# The five physical channels a machine controller already publishes.
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
# One-hot flag per failure mode; kept out of the feature matrix (they are labels).
MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def default_data_path() -> Path:
    """First existing copy of the dataset: data/ inside the repo, then its parent."""
    for candidate in (ROOT / "data" / "ai4i2020.csv", ROOT.parent / "ai4i2020.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "ai4i2020.csv not found. Put it in data/ or pass --data /path/to/ai4i2020.csv"
    )


def load_dataset(path: Path) -> pd.DataFrame:
    # utf-8-sig: the distributed CSV carries a BOM on the first header cell.
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="path to ai4i2020.csv")
    parser.add_argument("--model-out", type=Path, default=ROOT / "src" / "model.pkl")
    parser.add_argument("--pred-out", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = parser.parse_args()

    df = load_dataset(args.data or default_data_path())
    print(f"rows={len(df)} columns={df.shape[1]} failures={int(df[TARGET].sum())}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",  # 3.4% positives; unweighted trees ignore them
        n_jobs=-1,
    ).fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred, digits=3))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    pd.DataFrame(
        {"actual": y_test.to_numpy(), "predicted": pred, "prob_failure": prob.round(6)}
    ).to_csv(args.pred_out, index=False)
    print(f"saved {args.model_out} and {args.pred_out}")


if __name__ == "__main__":
    main()
