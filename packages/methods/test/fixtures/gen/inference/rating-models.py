"""Golden values for aifn-methods/inference/rating-models: Bradley–Terry and Plackett–Luce maximum-likelihood and MAP
strengths by `choix`'s MM algorithm (Hunter, 2004) on seeded comparison data; online TrueSkill over a game sequence by
the `trueskill` package (Lee's reference implementation); and TrueSkill Through Time by `trueskillthroughtime`
(Landfried and Mocskos's implementation of Dangauthier et al., 2007)."""

import math

import choix
import numpy as np
import trueskill  # pyright: ignore[reportMissingTypeStubs]
import trueskillthroughtime as ttt  # pyright: ignore[reportMissingTypeStubs]

# (a, b, score of a) per game; the online sequence is between two players, one game per round.
ONLINE = [(0, 1, 1.0), (0, 1, 1.0), (1, 0, 0.5), (1, 0, 1.0), (0, 1, 0.0), (0, 1, 0.5), (1, 0, 0.0), (0, 1, 1.0)]
# (round, a, b, score of a); every player plays in round 0, as the package starts a player's prior where it first plays.
SMOOTH = [
    (0, 0, 1, 1.0),
    (0, 1, 2, 1.0),
    (1, 0, 2, 0.5),
    (3, 1, 0, 1.0),
    (4, 2, 1, 1.0),
    (4, 0, 1, 1.0),
    (7, 0, 2, 0.0),
    (7, 2, 1, 0.5),
]
MU, SIGMA, BETA, TAU, P_DRAW = 25.0, 25 / 3, 25 / 6, 25 / 300, 0.1


def online() -> dict[str, object]:
    env = trueskill.TrueSkill(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU, draw_probability=P_DRAW)
    rate = trueskill.rate_1vs1  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    r = [env.create_rating(), env.create_rating()]  # pyright: ignore[reportUnknownMemberType]
    after: list[list[float]] = []
    for a, b, s in ONLINE:
        if s == 0.5:
            r[a], r[b] = rate(r[a], r[b], drawn=True, env=env)
        elif s == 1.0:
            r[a], r[b] = rate(r[a], r[b], env=env)
        else:
            r[b], r[a] = rate(r[b], r[a], env=env)
        after.append([r[0].mu, r[0].sigma, r[1].mu, r[1].sigma])  # pyright: ignore[reportUnknownMemberType]
    return {"games": [list(g) for g in ONLINE], "after": after}


def smoothed() -> dict[str, object]:
    composition = [[[str(a)], [str(b)]] for _, a, b, _ in SMOOTH]
    results = [[0.0, 0.0] if s == 0.5 else ([1.0, 0.0] if s == 1.0 else [0.0, 1.0]) for _, _, _, s in SMOOTH]
    times = [t for t, _, _, _ in SMOOTH]
    h = ttt.History(composition, results=results, times=times, mu=MU, sigma=SIGMA, beta=BETA, gamma=TAU, p_draw=P_DRAW)
    h.convergence(epsilon=1e-10, iterations=500, verbose=False)  # pyright: ignore[reportUnknownMemberType]
    curves = h.learning_curves()  # pyright: ignore[reportUnknownMemberType]
    out: list[list[float]] = []
    for p in ("0", "1", "2"):
        for t, g in curves[p]:  # pyright: ignore[reportUnknownVariableType]
            out.append([int(p), t, g.mu, g.sigma])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return {"games": [list(g) for g in SMOOTH], "posteriors": out}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    players = 6
    skills = rng.normal(0, 1, size=players)
    pairs: list[tuple[int, int]] = []
    for _ in range(300):
        a, b = rng.choice(players, size=2, replace=False)
        p = 1 / (1 + np.exp(skills[b] - skills[a]))
        pairs.append((int(a), int(b)) if rng.random() < p else (int(b), int(a)))
    items = 7
    strengths = rng.normal(0, 1, size=items)
    rankings: list[tuple[int, ...]] = []
    for _ in range(150):
        left = [int(i) for i in rng.choice(items, size=4, replace=False)]
        ranking: list[int] = []
        while left:
            w = np.exp(strengths[left])
            k = int(rng.choice(len(left), p=w / w.sum()))
            ranking.append(left.pop(k))
        rankings.append(tuple(ranking))
    return {
        "pairwise": {
            "players": players,
            "pairs": pairs,
            "ml": choix.mm_pairwise(players, pairs, tol=1e-12, max_iter=100000),
            "map": choix.mm_pairwise(players, pairs, alpha=0.5, tol=1e-12, max_iter=100000),
        },
        "rankings": {
            "items": items,
            "rankings": rankings,
            "ml": choix.mm_rankings(items, rankings, tol=1e-12, max_iter=100000),
        },
        "dynamics": {
            "constants": {"mu": MU, "sigma": SIGMA, "beta": BETA, "tau": TAU, "drawProbability": P_DRAW},
            "sigmaAtStart": math.sqrt(SIGMA**2 - TAU**2),
            "online": online(),
            "smoothed": smoothed(),
        },
    }
