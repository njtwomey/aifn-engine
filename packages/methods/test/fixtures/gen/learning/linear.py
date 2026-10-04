"""Golden values for aifn-methods learning/linear and learning/generalised/glm (logistic regression).

From scikit-learn's LinearRegression, Ridge and LogisticRegression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    n, d = 40, 3
    x = rng.normal(size=(n, d))
    y = x @ np.array([1.5, -2.0, 0.5]) + 0.7 + rng.normal(scale=0.3, size=n)

    ols = LinearRegression().fit(x, y)
    residuals = y - ols.predict(x)
    ridge = Ridge(alpha=2.5).fit(x, y)
    x_test = rng.normal(size=(5, d))

    # Rank-deficient design: the third column is the sum of the first two.
    x_deficient = np.column_stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 1]])
    deficient = LinearRegression().fit(x_deficient, y)

    logits = x @ np.array([1.0, -1.5, 0.5]) - 0.3
    y_binary = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    binary = LogisticRegression(C=1 / 0.5, tol=1e-12, max_iter=100000).fit(x, y_binary)

    scores = x @ np.array([[1.0, -1.0, 0.0], [0.5, 0.5, -1.0], [-1.0, 0.0, 1.0]])
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    y_multi = np.array([rng.choice(3, p=p) for p in probs])
    multi = LogisticRegression(C=1 / 0.8, tol=1e-12, max_iter=100000).fit(x, y_multi)

    return {
        "x": x,
        "y": y,
        "x_test": x_test,
        "ols": {
            "coef": ols.coef_,
            "intercept": ols.intercept_,
            "predict": ols.predict(x_test),
            "noise_sd": np.sqrt((residuals**2).sum() / (n - d - 1)),
        },
        "ridge": {"alpha": 2.5, "coef": ridge.coef_, "intercept": ridge.intercept_, "predict": ridge.predict(x_test)},
        "deficient": {"x": x_deficient, "coef": deficient.coef_, "intercept": deficient.intercept_},
        "binary": {
            "l2": 0.5,
            "y": y_binary,
            "coef": binary.coef_[0],
            "intercept": binary.intercept_[0],
            "proba": binary.predict_proba(x_test)[:, 1],
            "predict": binary.predict(x_test),
        },
        "multinomial": {
            "l2": 0.8,
            "y": y_multi,
            "coef": multi.coef_.T,
            "intercept": multi.intercept_,
            "proba": multi.predict_proba(x_test),
            "predict": multi.predict(x_test),
        },
    }
