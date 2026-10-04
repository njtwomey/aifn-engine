"""Reference values for the attributions added to aifn-compute/learning/explain: the `shap` package's exact tree SHAP
interaction values (TreeExplainer, path-dependent) on a scikit-learn regression tree and a small forest, and Captum's
DeepLift (rescale rule), DeepLiftShap, Occlusion (2-D windows with strides that crop at the edge) and integrated
gradients (Riemann midpoint) on small torch networks whose weights are exported."""

from typing import Any

import numpy as np
import shap  # pyright: ignore[reportMissingTypeStubs]
import torch
from captum.attr import (  # pyright: ignore[reportMissingTypeStubs]
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Occlusion,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def export_tree(tree: Any) -> dict[str, Any]:
    t = tree.tree_
    return {
        "left": t.children_left,
        "right": t.children_right,
        "feature": t.feature,
        "threshold": t.threshold,
        "weight": t.weighted_n_node_samples,
        "value": t.value[:, 0, 0],
    }


def network(sizes: list[int], activation: str, seed: int) -> torch.nn.Sequential:
    """An MLP with `activation` between layers and a linear output."""
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    for k in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[k], sizes[k + 1]))
        if k + 2 < len(sizes):
            layers.append(torch.nn.Tanh() if activation == "tanh" else torch.nn.ReLU())
    return torch.nn.Sequential(*layers).double()


def export_net(net: torch.nn.Sequential, activation: str) -> dict[str, Any]:
    linear = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {
        "weights": [m.weight.detach().numpy().T for m in linear],
        "biases": [m.bias.detach().numpy() for m in linear],
        "activation": activation,
    }


def trees() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(150, 5))
    y = x[:, 0] * x[:, 1] + np.sin(2 * x[:, 2]) + 0.5 * x[:, 3] * (x[:, 0] > 0) + 0.05 * rng.normal(size=150)
    queries = x[:4]
    tree = DecisionTreeRegressor(max_depth=5, random_state=0).fit(x, y)
    forest = RandomForestRegressor(n_estimators=3, max_depth=4, random_state=1).fit(x, y)
    ti = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent").shap_interaction_values(queries)  # pyright: ignore[reportUnknownMemberType]
    fi = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent").shap_interaction_values(queries)  # pyright: ignore[reportUnknownMemberType]
    return {
        "queries": queries,
        "tree": {**export_tree(tree), "interactions": np.asarray(ti)},
        "forest": {
            "trees": [export_tree(e) for e in forest.estimators_],  # pyright: ignore[reportUnknownMemberType]
            "interactions": np.asarray(fi),
        },
    }


def deep() -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(9)
    x = torch.tensor(rng.normal(size=(3, 4)))
    baseline = torch.tensor(rng.normal(size=(1, 4)) * 0.5)
    background = torch.tensor(rng.normal(size=(5, 4)))
    for activation in ("tanh", "relu"):
        net = network([4, 6, 5, 2], activation, seed=3 if activation == "tanh" else 4)
        dl = DeepLift(net).attribute(x, baselines=baseline, target=1)  # pyright: ignore[reportUnknownMemberType]
        ds = DeepLiftShap(net).attribute(x, baselines=background, target=1)  # pyright: ignore[reportUnknownMemberType]
        ig = IntegratedGradients(net).attribute(  # pyright: ignore[reportUnknownMemberType]
            x, baselines=baseline, target=1, n_steps=20, method="riemann_middle"
        )
        out[activation] = {
            "net": export_net(net, activation),
            "deepLift": dl.detach().numpy(),  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            "deepShap": ds.detach().numpy(),  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            "integratedGradients": ig.detach().numpy(),  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        }
    return {"x": x.numpy(), "baseline": baseline.numpy()[0], "background": background.numpy(), **out}


def occlusion() -> dict[str, Any]:
    rng = np.random.default_rng(13)
    net = network([30, 8, 1], "tanh", seed=7)

    def image_model(images: torch.Tensor) -> torch.Tensor:
        return net(images.reshape(images.shape[0], 30))[:, 0]

    image = torch.tensor(rng.normal(size=(1, 5, 6)))
    cases: list[dict[str, Any]] = []
    for window, strides, baseline in (((2, 2), (1, 1), 0.0), ((2, 3), (2, 2), 0.5), ((3, 2), (2, 1), -0.25)):
        attr = Occlusion(image_model).attribute(  # pyright: ignore[reportUnknownMemberType]
            image, sliding_window_shapes=window, strides=strides, baselines=baseline
        )
        cases.append(
            {
                "window": list(window),
                "strides": list(strides),
                "baseline": baseline,
                "values": attr.detach().numpy().reshape(-1),  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            }
        )
    return {"net": export_net(net, "tanh"), "image": image.numpy().reshape(-1), "cases": cases}


def attribution_cases() -> dict[str, Any]:
    return {"trees": trees(), "deep": deep(), "occlusion": occlusion()}
