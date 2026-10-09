/**
 * `aifn-compute/dynamics`: dynamical systems: ordinary and stochastic differential equations, vector fields and their
 * flows, and feedback control.
 *
 * - `aifn-compute/dynamics/ode`: initial-value problems $\xvec' = \fvec(t, \xvec)$: explicit, implicit, adaptive and
 *   symplectic solvers as step algorithms, `solveIvp` in one call, events, stability regions, and gradients through
 *   the solution (the adjoint method, neural ODEs), as scipy.integrate's `solve_ivp`.
 * - `aifn-compute/dynamics/sde`: stochastic differential equations $dX = a(t, X) \, dt + b(t, X) \, dW$ for a cloud of
 *   paths at once (Euler–Maruyama, Milstein, stochastic Runge–Kutta), and processes with exact solutions.
 * - `aifn-compute/dynamics/fields`: vector and scalar fields on $\reals^n$: Jacobian, divergence and curl by
 *   autodiff, flow maps and streamlines, density transport, and fixed points with their classification and invariant
 *   manifolds.
 * - `aifn-compute/dynamics/control`: feedback for linear plants $\xvec' = \Amat\xvec + \Bmat\uvec$: LQR, pole
 *   placement, model predictive control and LQG, as python-control's `lqr`, `dlqr` and `acker`.
 *
 * Step methods (the ODE and SDE solvers, receding-horizon control) are `Algorithm`s, run with `run` or `trace` and
 * seeded by the runner's root stream. This index re-exports the most used functions of its children: `rungeKutta`,
 * `dormandPrince`, `eulerMaruyama`, `trajectory`, `fixedPoints` and `lqr`.
 */

export { rungeKutta, dormandPrince } from './ode'
export { eulerMaruyama } from './sde'
export { trajectory, fixedPoints } from './fields'
export { lqr } from './control'
