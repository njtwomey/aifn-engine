/**
 * `aifn-methods/optimisation`: hyperparameter optimisation and the search for an optimum of an expensive black box,
 * each method a step-through algorithm whose states hold what a figure draws.
 *
 * - `aifn-methods/optimisation/bayesian`: Bayesian optimisation with a Gaussian-process surrogate, one evaluation per
 *   step, and its acquisition functions (expected improvement, probability of improvement, the upper confidence
 *   bound, Thompson sampling and max-value entropy search).
 * - `aifn-methods/optimisation/tree-parzen`: the tree-structured Parzen estimator, which models the good and bad
 *   configurations by densities $\ell(\lambda)$ and $g(\lambda)$ and proposes where $\ell / g$ is largest.
 * - `aifn-methods/optimisation/multi-fidelity`: successive halving, Hyperband and asynchronous successive halving over
 *   given learning curves, spending little budget on poor configurations.
 * - `aifn-methods/optimisation/population`: population-based training, which exploits and explores hyperparameters
 *   while the population trains.
 * - `aifn-methods/optimisation/importance`: hyperparameter importance by the functional ANOVA of a tabulated surface.
 * - `aifn-methods/optimisation/multi-objective`: Pareto fronts, the exact hypervolume, expected hypervolume
 *   improvement and ParEGO's Chebyshev scalarisation.
 * - `aifn-methods/optimisation/portfolio`: greedy portfolios of configurations over earlier tasks, for warm-starting.
 *
 * Objectives to search come from `aifn-methods/data`: the test surfaces of `aifn-methods/data/objectives` (Forrester,
 * Branin and a tabulated SVM tuning surface among them) and the learning curves of `aifn-methods/data/synthetic`. The
 * general optimisers (gradient methods, Nelder–Mead, CMA-ES, linear and quadratic programming) are in
 * `aifn-compute/optim`, and grid and random search with cross-validation in `aifn-compute/learning/validate`. The area
 * has no shared layer: import each method from its module.
 */

export {}
