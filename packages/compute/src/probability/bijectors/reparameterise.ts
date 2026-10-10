/**
 * Reparameterising a log-density through a change of variables. If $\thetavec = T(\uvec)$ for a bijection $T$, the
 * density of $\uvec$ is $p_u(\uvec) = p_\theta(T(\uvec)) \lvert \det \Jmat_T(\uvec) \rvert$, so
 * $\log p_u(\uvec) = \log p_\theta(T(\uvec)) + \log \lvert \det \Jmat_T(\uvec) \rvert$.
 *
 * The new log-density is written with primitives ($T$, the target and the log-Jacobian all are), so samplers and
 * variational methods differentiate it as usual. Two kinds of map are accepted: a scalar `Bijector` applied to every
 * coordinate (its $\log \lvert f'(u_i) \rvert$ summed over the last axis: the constrained-to-unconstrained transforms
 * of Stan and PyMC),
 * or a vector map `Reparameterisation` such as the non-centred parameterisation of a hierarchical model,
 * $(v, \zvec) \mapsto (v, \zvec e^{v/2})$ for Neal's funnel (Papaspiliopoulos, Roberts and Sköld, 2007, Statistical
 * Science 22(1)). A vector map without a log-Jacobian gets one from the Jacobian matrix (`jacobian`, then `logDet`), at
 * $O(d^3)$ per evaluation.
 */

import { jacobian } from 'aifn-compute/foundation/autodiff'
import type { Bijector, LogDensity } from 'aifn-compute/foundation/contracts'
import { add, reshape, shapeOfValue, sum, type Value } from 'aifn-compute/foundation/tensor'
import { logDet } from 'aifn-compute/numerics/linalg'

/** A bijection $\thetavec = T(\uvec)$ between vectors of length $d$. */
export type Reparameterisation = {
  /** A readable name, used in the transformed density's name. */
  readonly name?: string
  /** $\thetavec = T(\uvec)$. */
  forward(u: Value): Value
  /** $\uvec = T^{-1}(\thetavec)$; used to map starting points into $\uvec$ (`fromOriginal`). */
  inverse?(theta: Value): Value
  /**
   * $\log \lvert \det \Jmat_T(\uvec) \rvert$ at $\uvec$; when absent it is computed from the Jacobian matrix by
   * autodiff.
   */
  logAbsDetJacobian?(u: Value): Value
}

/** A log-density over $\uvec = T^{-1}(\thetavec)$, with the maps between the two parameterisations. */
export type TransformedLogDensity = LogDensity & {
  /** $\thetavec = T(\uvec)$: a draw in the new parameterisation read in the original one. */
  toOriginal(u: Value): Value
  /** $\uvec = T^{-1}(\thetavec)$, when the map has an inverse (always for a `Bijector`). */
  fromOriginal?(theta: Value): Value
}

/**
 * Whether a map is a scalar `Bijector` (it has a domain, a codomain and a direction) rather than a vector
 * `Reparameterisation`.
 *
 * @param t The map.
 * @returns True for a `Bijector`.
 */
const isBijector = (t: Bijector | Reparameterisation): t is Bijector =>
  'domain' in t && 'codomain' in t && 'increasing' in t

/**
 * $\log \lvert \det \Jmat_T(\uvec) \rvert$ of a vector map from its $d \times d$ Jacobian, by autodiff and `logDet`
 * ($O(d^3)$).
 *
 * @param forward The map $T$, from a vector of length $d$ to one of length $d$.
 * @param u The point $\uvec$, a vector of length $d$ (a scalar counts as $d = 1$).
 * @returns The log absolute determinant of the Jacobian at $\uvec$ (differentiable).
 */
function jacobianLogDet(forward: (u: Value) => Value, u: Value): Value {
  const d = shapeOfValue(u)[0] ?? 1
  const J = jacobian(forward)(u) as Value
  return logDet(reshape(J, [d, d]))
}

/**
 * The log-density of $\uvec$ where $\thetavec = T(\uvec)$ has log-density `target`:
 * $\log p(T(\uvec)) + \log \lvert \det \Jmat_T(\uvec) \rvert$. The result keeps the target's dimension and
 * normalisation (a change of variables preserves the integral), drops its closed-form gradient and truth (they are in
 * $\thetavec$), and carries `toOriginal` (and `fromOriginal` when $T$ can be inverted) to move points between the
 * parameterisations. With a `Bijector` its support is the bijector's domain.
 *
 * @param target The log-density of $\thetavec$.
 * @param map The map $T$: a scalar `Bijector`, applied to every coordinate with its log-Jacobians summed over the last
 *   axis (so a batch of points, one per row, works when the target takes one), or a vector `Reparameterisation`,
 *   whose log-Jacobian is computed by autodiff when it does not give one.
 * @returns The log-density of $\uvec$, named `<target> through <map>`.
 *
 * @example An exponential density on the log scale
 * const target = { kind: 'log-density', dim: 1, normalised: true, logDensity: (t) => neg(sum(t)) }
 * const onLogScale = transformLogDensity(target, expBijector)
 * print('log p(u = 0) (-1 + 0):', onLogScale.logDensity(tensor([0])))
 * print('theta at u = 1:', onLogScale.toOriginal(tensor([1])))
 * print('u at theta = 1:', onLogScale.fromOriginal(tensor([1])))
 *
 * @example A vector map whose log-Jacobian comes from autodiff
 * const target = { kind: 'log-density', dim: 2, normalised: false, logDensity: (t) => mul(-0.5, sum(mul(t, t))) }
 * const stretched = transformLogDensity(target, { name: 'stretch', forward: (u) => mul(tensor([1, 2]), u) })
 * print(stretched.name)
 * print('log p(1, 1) (-2.5 + log 2):', stretched.logDensity(tensor([1, 1])))
 */
export function transformLogDensity(target: LogDensity, map: Bijector | Reparameterisation): TransformedLogDensity {
  const forward = (u: Value) => map.forward(u)
  const logJac: (u: Value) => Value = isBijector(map)
    ? (u) => (shapeOfValue(u).length > 1 ? sum(map.logAbsDetJacobian(u), -1) : sum(map.logAbsDetJacobian(u)))
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
