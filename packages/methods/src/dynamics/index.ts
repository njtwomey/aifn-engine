/**
 * `aifn-methods/dynamics`: named dynamical systems, in discrete and continuous time, and their analysis.
 *
 * - `maps`: iterated maps $x_{n+1} = f(x_n)$ (logistic, tent, sine, Hénon, standard), orbits, cobwebs, bifurcation
 *   diagrams and Lyapunov exponents; `logisticMap` is re-exported here.
 * - `nonlinear`: flows $\dot{\xvec} = \fvec(\xvec)$: Poincaré sections, limit cycles with their Floquet multipliers,
 *   and Lyapunov-function checks on a grid; `limitCycle` is re-exported here.
 * - `pde`: the heat, transport, wave and Fokker–Planck equations by the method of lines, with stability numbers.
 * - `control`: a PID loop around a linear plant, with actuator limits and anti-windup.
 *
 * The generic machinery (ODE and SDE integrators, vector fields, linear systems) is in `aifn-compute/dynamics` and
 * `aifn-compute/systems`; the modules here build the named systems and methods on it.
 */

export { logisticMap } from './maps'
export { limitCycle } from './nonlinear'
