/**
 * `aifn-compute/transport`: optimal transport between weighted point sets, as POT (Python Optimal Transport).
 *
 * - Building a problem: `costMatrix` ($C_{ij} = \norm{\xvec_i - \yvec_j}^p$, squared Euclidean by default) and
 *   `uniformWeights`.
 * - Exact transport: `exactTransport` returns an optimal plan $\Pmat$ minimising $\inner{\Cmat}{\Pmat}$, by the
 *   Hungarian algorithm for equal numbers of equally weighted points and the simplex method of
 *   `aifn-compute/optim/programming` otherwise (with dual potentials). Dense, so for small problems.
 * - Entropic transport: `sinkhorn` runs to convergence and `sinkhornSteps` is the same log-domain iteration as a
 *   traceable algorithm; the regularisation $\varepsilon$ trades exactness for speed and smoothness.
 * - On the line, where the optimal plan matches quantiles: `wasserstein1d` (exact $W_p$ between weighted samples),
 *   `monotonePlan` (the plan between histograms), `barycenter1d` (the $W_2$ barycentre by quantile averaging), and
 *   `slicedWasserstein`, which averages 1-D distances over random directions to compare clouds in $d$ dimensions.
 * - Matching spaces without a common cost: `gromovWasserstein` and `gromovWassersteinSteps` (entropic
 *   Gromov–Wasserstein with the square loss, from intra-space distance matrices).
 * - Registries: `transportAlgorithms` and `transportFunctions`, for addressing them by key.
 *
 * Weights are vectors with equal totals (each sample's weights are normalised on the line); matrices are row-major
 * tensors or arrays of rows. The iterative solvers report convergence and divergence in their state's `converged` and
 * `diverged` flags rather than throwing; bad shapes and inputs throw.
 */

export {
  costMatrix,
  exactTransport,
  sinkhorn,
  sinkhornSteps,
  uniformWeights,
  type CostInput,
  type PointsInput,
  type SinkhornOptions,
  type SinkhornStart,
  type SinkhornState,
  type TransportPlan,
  type WeightsInput,
} from './discrete'
export {
  barycenter1d,
  monotonePlan,
  slicedWasserstein,
  wasserstein1d,
  type Barycenter1d,
  type MonotonePlan,
  type SlicedWasserstein,
} from './oneD'
export {
  gromovWasserstein,
  gromovWassersteinSteps,
  type GromovOptions,
  type GromovProblem,
  type GromovState,
} from './gromov'
export { transportAlgorithms, transportFunctions } from './registry'
