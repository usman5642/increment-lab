"""Small pre-specified model comparison; preprocessing learns from train only."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from .data import FEATURES, NUMERIC, CATEGORICAL


def fit_candidates(train, seed):
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    x = pre.fit_transform(train[FEATURES])
    t, y = train.treatment.to_numpy(), train.visit.to_numpy()

    def learner(kind):
        if kind == "logistic":
            return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        return HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=7,
            learning_rate=0.05,
            min_samples_leaf=150,
            l2_regularization=5,
            early_stopping=False,
            random_state=seed,
        )

    bundle = {"preprocessor": pre, "models": {}, "features": FEATURES}
    for name, kind in [("T-logistic", "logistic"), ("T-gradient", "gradient")]:
        pair = [learner(kind).fit(x[t == arm], y[t == arm]) for arm in [0, 1]]
        bundle["models"][name] = pair
    bundle["models"]["S-gradient"] = learner("gradient").fit(np.c_[x, t], y)
    return bundle


def score(bundle, frame):
    x = bundle["preprocessor"].transform(frame[FEATURES])
    result = {}
    for name in ["T-logistic", "S-gradient", "T-gradient"]:
        model = bundle["models"][name]
        if name.startswith("T-"):
            p0, p1 = [m.predict_proba(x)[:, 1] for m in model]
        else:
            p0 = model.predict_proba(np.c_[x, np.zeros(len(x))])[:, 1]
            p1 = model.predict_proba(np.c_[x, np.ones(len(x))])[:, 1]
        result[name] = p1 - p0
        if name == "T-logistic":
            result["Response ranking"] = p1
            result["mu0"] = p0
            result["mu1"] = p1
    return result
