"""Reference values for aifn-methods/optimisation/portfolio: greedy and optimal portfolios on a loss table.

The loss of a portfolio is the mean over tasks of its best member's loss, capped at `worst`. The greedy portfolio is
built by a plain loop over the candidates (ties to the lower index), the optimum by itertools.combinations over every
subset, the leave-one-task-out losses by rebuilding the greedy portfolio without each task, and the average ranks by
scipy.stats.rankdata (ties averaged).
"""

from itertools import combinations

import numpy as np
from scipy.stats import rankdata


def loss(table: np.ndarray, members, worst: float) -> float:
    if len(members) == 0:
        return worst
    return float(np.minimum(table[list(members)].min(axis=0), worst).mean())


def greedy(table: np.ndarray, k: int, worst: float):
    order: list[int] = []
    losses: list[float] = []
    for _ in range(k):
        scores = [loss(table, [*order, c], worst) if c not in order else np.inf for c in range(len(table))]
        pick = int(np.argmin(scores))  # argmin returns the first of equal values
        order.append(pick)
        losses.append(scores[pick])
    gains = [worst - losses[0]] + [losses[i - 1] - losses[i] for i in range(1, k)]
    return order, losses, gains


def optimal(table: np.ndarray, k: int, worst: float):
    best = min(combinations(range(len(table)), k), key=lambda s: loss(table, s, worst))
    return list(best), loss(table, best, worst)


def case(table: np.ndarray, k: int, worst: float) -> dict:
    order, losses, gains = greedy(table, k, worst)
    best, best_loss = optimal(table, k, worst)
    T = table.shape[1]
    loto = np.zeros((T, k))
    orders = []
    for held in range(T):
        rest = np.delete(table, held, axis=1)
        o, _, _ = greedy(rest, k, worst)
        orders.append(o)
        for s in range(k):
            loto[held, s] = min(worst, table[o[: s + 1], held].min())
    per_task = np.array([np.minimum(table[order[: s + 1]].min(axis=0), worst) for s in range(k)])
    return {
        "table": table,
        "k": k,
        "worst": worst,
        "order": order,
        "loss": losses,
        "gain": gains,
        "perTask": per_task,
        "optimal": best,
        "optimalLoss": best_loss,
        "lotoOrder": orders,
        "lotoLoss": loto,
        "lotoMean": loto.mean(axis=0),
        "rank": rankdata(table, axis=0).mean(axis=1),
    }


def cases() -> dict:
    rng = np.random.default_rng(20150125)
    # Continuous losses: no ties.
    table = rng.uniform(0, 1, size=(12, 6))
    # Integer losses: ties in the ranks and between candidates (exact in floating point, so the tie-breaking to the
    # lower index is the same in both languages), and a cap below the largest loss.
    coarse = np.round(rng.uniform(0, 1, size=(9, 5)) * 10)
    return {"random": case(table, 4, float(table.max())), "coarse": case(coarse, 3, 8.0)}
