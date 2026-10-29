"""Smoke tests: the saved pickles load, match the training schema, and score sanely."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_predict import FEATURES, MODE_COLUMNS, TARGET, load_dataset  # noqa: E402

MODELS = [pytest.param(ROOT / "src" / "model.pkl", id="baseline"),
          pytest.param(ROOT / "src" / "model_tuned.pkl", id="tuned")]


@pytest.fixture(scope="module")
def frame():
    df = load_dataset(ROOT / "data" / "ai4i2020.csv")
    assert len(df) == 10000
    assert int(df[TARGET].sum()) == 339
    return df


@pytest.mark.parametrize("path", MODELS)
def test_model_expects_the_five_sensor_channels(path):
    clf = joblib.load(path)
    assert list(clf.feature_names_in_) == FEATURES
    assert hasattr(clf, "predict_proba")


@pytest.mark.parametrize("path", MODELS)
def test_probabilities_are_in_range_and_ordered(path, frame):
    prob = joblib.load(path).predict_proba(frame[FEATURES])[:, 1]
    assert prob.shape == (len(frame),)
    assert np.all((prob >= 0) & (prob <= 1))
    failing = prob[frame[TARGET] == 1].mean()
    healthy = prob[frame[TARGET] == 0].mean()
    assert failing > healthy, "model should score failing rows higher on average"


def test_mode_columns_are_excluded_from_features():
    # The per-mode flags are labels, not inputs: leaking them would fake the score.
    assert not set(MODE_COLUMNS) & set(FEATURES)


def test_predictions_file_matches_dataset(frame):
    preds = pd.read_csv(ROOT / "outputs" / "predictions.csv")
    assert list(preds.columns) == ["actual", "predicted", "prob_failure"]
    assert len(preds) == 2000
    assert set(preds["predicted"].unique()) <= {0, 1}
    assert preds["actual"].sum() == 68  # 3.4% of the 2000-row stratified test split
