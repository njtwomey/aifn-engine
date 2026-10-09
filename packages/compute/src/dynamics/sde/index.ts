/**
 * `aifn-compute/dynamics/sde`: stochastic differential equations $dX = a(t, X) \, dt + b(t, X) \, dW$ with diagonal
 * noise, integrated for a whole cloud of paths at once.
 *
 * - Schemes: `eulerMaruyama` (strong order $\tfrac{1}{2}$, weak order 1; differentiable through `unrolled`),
 *   `milstein` (strong order 1; the derivative $\partial b / \partial x$ by autodiff unless given),
 *   `stochasticRungeKutta` (Platen's derivative-free strong order 1 scheme).
 * - Exact solutions: `ornsteinUhlenbeck` and `geometricBrownianMotion` (moments, transition laws and exact samplers
 *   on the same streams), and `brownianMotion`.
 * - Noise and output: `increments` draws the Brownian increments the schemes use, and `paths` extracts a path matrix
 *   from a trace. The density of a scalar SDE by its Fokker–Planck equation (`densityEvolution`) is in
 *   `aifn-methods/dynamics`, beside the named PDEs.
 *
 * Every scheme is a traceable `Algorithm`; the Brownian increments of step $k$ come from the runner's step stream
 * `child(root, 'step', k)` in row-major order, so path $i$'s increments depend only on the root key, the step and
 * $i$: raising the number of paths adds paths without changing the existing ones, and two schemes (or an exact
 * solution) run from the same root stream see the same Brownian path. States count steps in `t` and hold the time in
 * `time`; a path that stops being finite is counted in `nonFinite`, and the run is `diverged` only when all have.
 * `sdeAlgorithms` and `sdeFunctions` register the schemes and functions.
 *
 * References: Kloeden & Platen (1992), "Numerical Solution of Stochastic Differential Equations", §9–11; Higham
 * (2001), "An algorithmic introduction to numerical simulation of stochastic differential equations", SIAM Review 43.
 */

export {
  eulerMaruyama,
  increments,
  milstein,
  paths,
  stochasticRungeKutta,
  type Sde,
  type SdeInitial,
  type SdeOptions,
  type SdeState,
} from './integrators'
export { brownianMotion, geometricBrownianMotion, ornsteinUhlenbeck, type ExactSde } from './processes'
export { sdeAlgorithms, sdeFunctions } from './registry'
