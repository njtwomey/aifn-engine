/**
 * `aifn-methods/optimisation/multi-objective`: Pareto fronts, the hypervolume indicator and the acquisitions of
 * multi-objective Bayesian optimisation.
 *
 * - Dominance: `paretoFront` returns the points no other point dominates, ordered along the front.
 * - Hypervolume, exact: `hypervolume` of a set against a reference point (a sweep for two objectives, slicing along
 *   the last objective for more), and `hypervolumeImprovement`, the growth from adding each candidate.
 * - Acquisition: `expectedHypervolumeImprovement` under independent Gaussian posteriors per objective, exact for one
 *   and two objectives by a strip decomposition, and a Monte Carlo average with common random numbers for more.
 * - Scalarisation, as ParEGO: `chebyshevScalarisation`, the augmented Chebyshev function
 *   $\max_j \lambda_j y_j + \rho\sum_j \lambda_j y_j$, and `simplexLattice`, the weight vectors it draws from.
 *
 * Every function takes a `minimise` option, a flag for all objectives or one per objective (default: minimise all).
 * Points are $n \times M$ matrices, one row per point and one column per objective; results are numbers or vectors.
 * To run a scalarised search, apply the single-objective acquisitions of `aifn-methods/optimisation/bayesian` to
 * `chebyshevScalarisation` of the observations. `multiObjectiveFunctions` registers the functions.
 */

export { paretoFront, type Directions, type ParetoOptions } from './pareto'
export {
  expectedHypervolumeImprovement,
  hypervolume,
  hypervolumeImprovement,
  type ExpectedHypervolumeOptions,
  type HypervolumeOptions,
} from './hypervolume'
export { chebyshevScalarisation, simplexLattice, type ChebyshevOptions } from './scalarisation'
export { multiObjectiveFunctions } from './registry'
