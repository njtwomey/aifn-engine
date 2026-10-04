/**
 * Reparameterising a log-density through a change of variables, part of `aifn-compute/probability/bijectors`. If θ = T(u)
 * for a bijection T, the density of u is p_u(u) = p_θ(T(u)) |det ∂T/∂u|, so
 *
 *   log p_u(u) = log p_θ(T(u)) + log |det J_T(u)|.
 *
 * The new log-density is written with primitives (T, the target and the log-Jacobian all are), so samplers and
 * variational methods differentiate it as usual. Two kinds of map are accepted: a scalar `Bijector` applied to every
 * coordinate (its log |f′| summed: the constrained-to-unconstrained transforms of Stan and PyMC), or a vector map
 * `Reparameterisation` such as the non-centred parameterisation of a hierarchical model, (v, z) ↦ (v, z e^{v/2})
 * for Neal's funnel (Papaspiliopoulos, Roberts & Sköld, 2007, Statistical Science 22(1)). A vector map without a
 * log-Jacobian gets one from the Jacobian matrix (`jacobian`, then `logDet`), O(d³) per evaluation.
 */

import { jacobian } from 'aifn-compute/foundation/autodiff'
import type { Bijector, LogDensity } from 'aifn-compute/foundation/contracts'
import { add, reshape, shapeOfValue, sum, type Value } from 'aifn-compute/foundation/tensor'
import { logDet } from 'aifn-compute/numerics/linalg'

/** A bijection θ = T(u) between vectors of length d. */
export type Reparameterisation = {
  readonly name?: string
  /** θ = T(u). */
  forward(u: Value): Value
  /** u = T⁻¹(θ); used to map starting points into u (`fromOriginal`). */
  inverse?(theta: Value): Value
  /** log |det ∂T/∂u| at u; when absent it is computed from the Jacobian matrix by autodiff. */
  logAbsDetJacobian?(u: Value): Value
}

/** A log-density over u = T⁻¹(θ), with the maps between the two parameterisations. */
export type TransformedLogDensity = LogDensity & {
  /** θ = T(u): a draw in the new parameterisation read in the original one. */
  toOriginal(u: Value): Value
  /** u = T⁻¹(θ), when the map has an inverse (always for a `Bijector`). */
  fromOriginal?(theta: Value): Value
}

const isBijector = (t: Bijector | Reparameterisation): t is Bijector =>
  'domain' in t && 'codomain' in t && 'increasing' in t

/** log |det ∂T/∂u| of a vector map from its d × d Jacobian. */
function jacobianLogDet(forward: (u: Value) => Value, u: Value): Value {
  const d = shapeOfValue(u)[0] ?? 1
  const J = jacobian(forward)(u) as Value
  return logDet(reshape(J, [d, d]))
}

/**
 * The log-density of u where θ = T(u) has log-density `target`: log p(T(u)) + log |det J_T(u)|. `map` is a scalar
 * `Bijector` (applied to every coordinate; u ranges over its domain) or a vector `Reparameterisation`. The result keeps
 * the target's dimension and normalisation (a change of variables preserves the integral), drops its closed-form
 * gradient and truth (they are in θ), and carries `toOriginal` (and `fromOriginal` when T can be inverted) to move
 * points between the parameterisations.
 *
 * @example
 * // Neal's funnel, non-centred: θ = (v, z e^{v/2}) has log |det J| = (d − 1) v/2.
 * const nc = transformLogDensity(funnel, {
 *   forward: (u) => concat([slice(u, [0, 1]), mul(slice(u, [1, null]), exp(mul(0.5, get(u, 0))))]),
 *   logAbsDetJacobian: (u) => mul((d - 1) / 2, get(u, 0)),
 * })
 */
export function transformLogDensity(target: LogDensity, map: Bijector | Reparameterisation): TransformedLogDensity {
  const forward = (u: Value) => map.forward(u)
  const logJac: (u: Value) => Value = isBijector(map)
    ? (u) => sum(map.logAbsDetJacobian(u))
    : map.logAbsDetJacobian
      ? (u) => map.logAbsDetJacobian!(u)
      : (u) => jacobianLogDet(forward, u)
  const inverse = map.inverse ? (theta: Value) => map.inverse!(theta) : undefined
  const name = `${target.name ?? 'log-density'} through ${map.name ?? 'a reparameterisation'}`
  return {
    kind: 'log-density',
    name,
    dim: target.dim,
    normalised: target.normalised,
    ...(isBijector(map) ? { support: { type: 'interval' as const, ...map.domain } } : {}),
    logDensity: (u: Value) => add(target.logDensity(forward(u)), logJac(u)),
    toOriginal: forward,
    ...(inverse ? { fromOriginal: inverse } : {}),
  }
}
