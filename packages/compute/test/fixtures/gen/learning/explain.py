"""Reference values for aifn-compute/learning/explain: the `shap` package's TreeExplainer (path-dependent) on a
scikit-learn
regression tree and a small random forest, exported node by node; shap's KernelExplainer with every coalition
enumerated (exact) on a fixed function; scikit-learn's `permutation_importance` (with the row shuffles it draws,
recomputed from its seed) and `partial_dependence` (brute force, custom grid, average and individual curves). Under
`attribution` and `probe`, the cases of gen/learning/_explain_attribution.py (shap interaction values; Captum's
DeepLift, DeepLiftShap, integrated gradients and Occlusion) and _explain_probe.py (scikit-learn's L2 logistic
regression)."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import shap  # pyright: ignore[reportMissingTypeStubs]
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).parent))
from _explain_attribution import attribution_cases
from _explain_probe import probe_cases


def fixed(x: np.ndarray) -> np.ndarray:
    """The function explained by KernelSHAP, permutation importance and partial dependence."""
    return x[:, 0] * x[:, 1] + np.sin(x[:, 2]) + 0.5 * x[:, 3] ** 2


class Fixed(RegressorMixin, BaseEstimator):
    """A fitted-looking estimator whose predictions are `fixed`."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Fixed":
        self.fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return fixed(np.asarray(x))


def export(tree: Any) -> dict[str, Any]:
    t = tree.tree_
    return {
        "left": t.children_left,
        "right": t.children_right,
        "feature": t.feature,
        "threshold": t.threshold,
        "weight": t.weighted_n_node_samples,
        "value": t.value[:, 0, 0],
    }


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(120, 5))
    y = 2 * x[:, 0] - x[:, 1] * (x[:, 2] > 0) + 0.3 * x[:, 3] ** 2 + 0.1 * rng.normal(size=120)
    queries = x[:6]

    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, y)
    te = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent")
    forest = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0).fit(x, y)
    fe = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent")

    background = rng.normal(size=(8, 4))
    point = np.array([0.7, -1.2, 0.4, 1.5])
    ke = shap.KernelExplainer(fixed, background)
    kernel = np.asarray(ke.shap_values(point, nsamples=2**4, silent=True))  # pyright: ignore[reportUnknownMemberType]

    xp = rng.normal(size=(40, 4))
    yp = fixed(xp) + 0.2 * rng.normal(size=40)
    est = Fixed().fit(xp, yp)
    pi = permutation_importance(est, xp, yp, scoring="r2", n_repeats=3, random_state=0)
    seed = np.random.RandomState(0).randint(np.iinfo(np.int32).max + 1)
    permutations: list[list[list[int]]] = []
    for _ in range(4):
        rs = np.random.RandomState(seed)
        idx = np.arange(40)
        shuffles: list[list[int]] = []
        for _ in range(3):
            rs.shuffle(idx)
            shuffles.append(idx.tolist())
        permutations.append(shuffles)

    grid = np.linspace(-2, 2, 9)
    pd = partial_dependence(est, xp, [2], kind="both", custom_values={2: grid}, method="brute")

    return {
        "x": x,
        "queries": queries,
        "tree": {
            **export(tree),
            "shap": te.shap_values(queries),  # pyright: ignore[reportUnknownMemberType]
            "expected": float(np.ravel(te.expected_value)[0]),  # pyright: ignore[reportArgumentType]
            "predicted": tree.predict(queries),
        },
        "forest": {
            "trees": [export(t) for t in forest.estimators_],
            "shap": fe.shap_values(queries),  # pyright: ignore[reportUnknownMemberType]
            "expected": float(np.ravel(fe.expected_value)[0]),  # pyright: ignore[reportArgumentType]
        },
        "kernel": {
            "background": background,
            "point": point,
            "shap": kernel,
            "expected": float(np.ravel(ke.expected_value)[0]),  # pyright: ignore[reportArgumentType]
        },
        "permutation": {
            "x": xp,
            "y": yp,
            "permutations": permutations,
            "importances": pi["importances"],
            "mean": pi["importances_mean"],
            "std": pi["importances_std"],
        },
        "partialDependence": {"grid": grid, "average": pd["average"][0], "individual": pd["individual"][0]},
        "attribution": attribution_cases(),
        "probe": probe_cases(),
    }
