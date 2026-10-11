/**
 * `aifn-methods/neural/pinn`: physics-informed neural networks (Raissi, Perdikaris & Karniadakis, 2019), a network
 * $u_{\thetavec}(x, t)$ trained to satisfy a partial differential equation with its initial and boundary conditions
 * rather than to fit data.
 *
 * - Problems: `burgers` (viscous Burgers' equation, which steepens into a near-shock) and `heat`, each a
 *   `PinnProblem` (a residual in $u$, $u_t$, $u_x$, $u_{xx}$, an initial profile and Dirichlet ends) with its exact
 *   solution; `burgersSolution` gives Burgers' by the Cole–Hopf transform.
 * - Training: `pinnRun` streams Adam then L-BFGS on the residual at a fixed Sobol set of collocation points plus the
 *   initial and boundary misfits, with the network, its residual and its error against the exact solution on a grid.
 *
 * The derivatives inside the residual are taken by `partials` of `aifn-compute/dynamics/fields` (nested forward mode
 * over the batch), and the loss is differentiated in the weights by reverse mode, so no finite differences appear
 * anywhere. Runs are deterministic in their seed and report an error in their last snapshot rather than throwing.
 */

export { burgers, burgersSolution, heat, type PinnDerivatives, type PinnProblem } from './problems'
export { pinnRun, type PinnRun, type PinnRunOptions } from './run'
export { pinnFunctions } from './registry'
