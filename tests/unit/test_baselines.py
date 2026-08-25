"""Tests for Phase 6 point-prediction baselines and endpoint metrics."""

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.evaluation.metrics import (
    evaluate_endpoint_predictions,
    nasa_asymmetric_score,
    rul_metrics,
)
from calibrated_reliability.models.baselines import fit_baseline_models, predict_baselines


def features() -> pd.DataFrame:
    """Create a small model-ready feature matrix without metadata."""
    return pd.DataFrame({"cycle_index": [1.0, 2.0, 3.0, 4.0], "sensor_1": [4.0, 3.0, 2.0, 1.0]})


def test_baselines_fit_and_predict_deterministically() -> None:
    """All declared baselines fit on train data and return finite predictions."""
    X = features()
    y = np.array([3.0, 2.0, 1.0, 0.0])
    first = predict_baselines(fit_baseline_models(X, y, random_state=13), X)
    second = predict_baselines(fit_baseline_models(X, y, random_state=13), X)
    assert set(first) == {"mean", "ridge", "hist_gradient_boosting"}
    for name in first:
        np.testing.assert_allclose(first[name], second[name])


@pytest.mark.parametrize(
    "column",
    [
        "engine_id",
        "rul",
        "rul_raw",
        "rul_capped",
        "target",
        "target_rul",
        "remaining_useful_life",
        "y",
    ],
)
def test_baselines_reject_metadata_and_targets(column: str) -> None:
    """Identifiers and labels cannot enter the model feature matrix."""
    with pytest.raises(ValueError):
        fit_baseline_models(features().assign(**{column: 1.0}), [3, 2, 1, 0])


def test_baselines_reject_bad_shapes_and_nonfinite_values() -> None:
    """Feature-target shape and finite-value contracts fail closed."""
    with pytest.raises(ValueError, match="row counts"):
        fit_baseline_models(features(), [1, 0])
    with pytest.raises(ValueError, match="finite"):
        fit_baseline_models(features().assign(sensor_1=np.nan), [3, 2, 1, 0])


def test_rul_metrics_and_asymmetric_score() -> None:
    """Metrics use signed error and the declared asymmetric penalty directions."""
    truth = [10.0, 10.0]
    prediction = [8.0, 12.0]
    expected = (np.exp(2.0 / 13.0) - 1.0) + (np.exp(2.0 / 10.0) - 1.0)
    assert nasa_asymmetric_score(truth, prediction) == pytest.approx(expected)
    result = rul_metrics(truth, prediction)
    assert result["rmse"] == pytest.approx(2.0)
    assert result["mae"] == pytest.approx(2.0)
    assert result["signed_error"] == pytest.approx(0.0)
    assert result["nasa_score"] == pytest.approx(expected)


def test_endpoint_evaluation_requires_one_row_per_engine() -> None:
    """Endpoint metrics reject duplicated engine rows."""
    endpoints = pd.DataFrame({"engine_id": [1, 2], "y_true": [3.0, 1.0], "y_pred": [2.0, 2.0]})
    result = evaluate_endpoint_predictions(endpoints)
    assert result["mae"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one row per engine"):
        evaluate_endpoint_predictions(pd.concat([endpoints, endpoints.iloc[[0]]]))
