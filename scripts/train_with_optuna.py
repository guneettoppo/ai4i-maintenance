#!/usr/bin/env python3
"""Tune the failure classifier with Optuna, selecting on cross-validated AUC.

Why CV and not the test split: choosing hyperparameters by their test AUC leaks
the test set into model selection, and the reported number stops being an
estimate of generalisation. The objective here is 5-fold stratified CV AUC on
the *training* split; the test split is scored once, after the fit.

Artifacts
---------
src/model_tuned.pkl          fitted estimator (joblib)
reports/optuna_trials.csv    every trial, its parameters and CV score
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_predict import FEATURES, TARGET, default_data_path, load_dataset  # noqa: E402

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-out", type=Path, default=ROOT / "src" / "model_tuned.pkl")
    args = parser.parse_args()

    df = load_dataset(args.data or default_data_path())
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=10),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.6, 1.0]),
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=args.trials)

    print(f"best CV AUC: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    clf = RandomForestClassifier(
        **study.best_params, class_weight="balanced", random_state=SEED, n_jobs=-1
    ).fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"held-out test AUC: {test_auc:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)

    trials_path = ROOT / "reports" / "optuna_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"saved {args.model_out} and {trials_path}")


if __name__ == "__main__":
    main()
