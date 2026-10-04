/**
 * `aifn-compute/dynamics`: dynamical systems: ordinary and stochastic differential equations, vector fields and flows, and state feedback.
 * Children: ode, sde, fields, control.
 */

export { rungeKutta, dormandPrince } from './ode'
export { eulerMaruyama } from './sde'
export { trajectory, fixedPoints } from './fields'
export { lqr } from './control'
