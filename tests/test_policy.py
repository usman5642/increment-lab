import numpy as np
import pandas as pd
import pytest
from uplift_lab.metrics import *
from uplift_lab.pipeline import heuristic
from uplift_lab.data import FEATURES


def test_half_rounding_and_endpoints():
    assert top_policy([1, 2, 3, 4, 5], 0.5).sum() == 3
    assert top_policy([1, 1], 0).sum() == 0
    assert top_policy([1, 1], 1).sum() == 2
    assert np.array_equal(top_policy(np.ones(20), 0.4), top_policy(np.ones(20), 0.4))


def test_known_ipw_and_dr():
    y = np.array([1, 0, 0, 0])
    t = np.array([1, 1, 0, 0])
    assert transformed_outcome(y, t).mean() == 0.5
    assert np.allclose(aipw(y, t, np.zeros(4), np.zeros(4)), transformed_outcome(y, t))


def test_invalid_assignment_not_truncated():
    with pytest.raises(ValueError):
        transformed_outcome([1, 0], [1, 0.4])
    with pytest.raises(ValueError):
        top_policy([np.nan], 0.4)


def test_no_outcomes_in_features_or_rule():
    assert not set(FEATURES) & {"visit", "conversion", "spend", "treatment", "segment"}
    d = pd.DataFrame({"mens": [0, 1, 1], "recency": [1, 4, 1], "history": [100, 1, 2]})
    assert list(np.argsort(-heuristic(d))) == [2, 1, 0]


def test_bootstrap_paired_identical_models():
    names, point, draws = bootstrap_policies(
        {"a": np.arange(20), "b": np.arange(20)},
        np.tile([0, 1], 10),
        np.arange(20),
        draws=30,
    )
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert np.all(draws[:, -1] == 0)


def test_curve_exact_step_area():
    q, g, area = curve([2, 1], [1, 0], [1, 0])
    assert np.allclose(g, [0, 1, 1])
    assert area == 0
