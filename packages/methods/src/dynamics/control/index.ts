/**
 * `aifn-methods/dynamics/control`: PID control of linear systems, simulated sample by sample.
 *
 * - `pidLoop` closes a discrete PID loop around a SISO `LtiSystem` (continuous plants are discretised by zero-order
 *   hold) as a traceable `Algorithm`: gains `PidGains` ($k_p$, $k_i$, $k_d$ and a derivative filter), options
 *   `PidOptions` (set point, load disturbance, actuator limits, input delay, derivative on the error or on the
 *   measurement) and the anti-windup scheme `AntiWindup`. Each `PidState` reports the output, the error, the P, I and
 *   D contributions, the applied input and whether it saturated.
 * - `pidAlgorithms`: its registry entry.
 *
 * The plant must be strictly proper; invalid settings throw `DomainError`.
 */

export { pidLoop, type AntiWindup, type PidGains, type PidOptions, type PidState } from './pid'
export { pidAlgorithms } from './registry'
