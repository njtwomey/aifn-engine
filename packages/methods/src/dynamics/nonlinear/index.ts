/**
 * `aifn-methods/dynamics/nonlinear`: limit cycles, Poincaré sections and Lyapunov-function checks for autonomous flows
 * $\dot{\xvec} = \fvec(\xvec)$.
 *
 * - Periodic orbits: `poincareSection` gives the successive crossings of a trajectory with a `Section` (the orbit of
 *   the first-return map, by event detection on an adaptive Dormand–Prince integration); `limitCycle` iterates that
 *   map to a fixed point and reports the cycle's period, orbit and Floquet multiplier as a `LimitCycle`. Both take
 *   `SectionOptions`, whose `reverse` integrates backwards in time to find repelling cycles.
 * - Stability: `lyapunovCheck` tests a candidate Lyapunov function $V$ about an equilibrium of a planar flow on a grid
 *   (positive definite, $\dot{V} \le 0$, $\dot{V} < 0$) and returns a `LyapunovCheck`.
 * - `nonlinearFunctions`: the registry entries of the three.
 *
 * Vector fields are those of `aifn-compute/dynamics/fields`. A grid check is evidence on the sampled region, not a
 * proof.
 */
export { limitCycle, type LimitCycle, poincareSection, type Section, type SectionOptions } from './flow'
export { lyapunovCheck, type LyapunovCheck } from './lyapunov'
export { nonlinearFunctions } from './registry'
