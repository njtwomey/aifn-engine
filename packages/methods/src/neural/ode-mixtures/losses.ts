/**
 * The losses of stochastic vector field mixtures (Twomey, Kozłowski & Santos-Rodríguez, 2020, §2.3), on the moments
 * `propagate` carries over the grid t₀ … t_T:
 *
 * - **TLoss** (eq. 8), the transportation loss 1/T Σᵢ ‖h(tᵢ) − h(tᵢ₋₁)‖², least when every step points the same way;
 * - **VLoss** (eq. 9), the variance loss 1/T Σᵢ ‖∇h(tᵢ) − E[∇h(t)]‖² with E[∇h(t)] = 1/T Σᵢ ∇h(tᵢ), least when the VF is
 *   constant along the path; TLoss + VLoss is TVLoss. Both are computed per component and weighted by π;
 * - **MDLoss** (eq. 10), −log Σₖ p(h(t) | μ⁽ᵏ⁾(t), τ⁽ᵏ⁾(t)) πₖ(t): the mixture density of a target under the model's
 *   output (compute `mixtureDensityNll`); for class labels, the mixture of the components' class likelihoods;
 * - **FLoss** (eq. 11), the forecasting loss 1/T Σᵢ ℓ(h(tᵢ), interp(X, t_X, tᵢ)) against a path interpolated at the grid
 *   times (cubic spline), with ℓ the squared error for a single VF and MDLoss otherwise.
 *
 * FLoss penalises any deviation from the path, so it is incompatible with TLoss and VLoss (§4.1.3); `svfmObjective`
 * refuses the combination. λ weighs the path regularisers against the predictive loss.
 */

import { cubicSpline, evaluatePiecewise } from 'aifn-compute/numerics/interpolate'
import {
  add,
  concat,
  div,
  exp,
  fromData,
  logsumexp,
  mean,
  mul,
  neg,
  permute,
  reshape,
  shapeOfValue,
  slice,
  square,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { mixtureDensityNll } from 'aifn-compute/learning/losses'
import type { Propagation, Svfm, SvfmParams } from './model'
import { DomainError } from 'aifn-compute/foundation/errors'

/** π(tᵢ) as [K, B] weights (from log π [B, K]). */
const weightsAt = (p: Propagation, i: number): Value => permute(exp(p.logWeights[i]), [1, 0])

/**
 * TLoss (eq. 8): 1/T Σᵢ Σₖ πₖ(tᵢ₋₁) ‖m_k(tᵢ) − m_k(tᵢ₋₁)‖², the distance each component's state travels per interval
 * (from where interval i starts to where it arrives), averaged over the batch.
 */
export function transportLoss(p: Propagation): Value {
  const T = p.times.length - 1
  let total: Value = 0
  for (let i = 1; i <= T; i++) {
    const step = sum(square(sub(p.arrived[i], p.states[i - 1])), -1) // [K, B]
    total = add(total, mean(sum(mul(weightsAt(p, i - 1), step), 0)))
  }
  return div(total, T)
}

/**
 * VLoss (eq. 9): 1/T Σᵢ Σₖ πₖ(tᵢ) ‖∇h⁽ᵏ⁾(tᵢ) − E[∇h⁽ᵏ⁾]‖² over i = 1 … T, with E[∇h⁽ᵏ⁾] = 1/T Σᵢ ∇h⁽ᵏ⁾(tᵢ) the expected VF
 * along component k's path; averaged over the batch.
 */
export function varianceLoss(p: Propagation): Value {
  const T = p.times.length - 1
  let avg: Value = 0
  for (let i = 1; i <= T; i++) avg = add(avg, div(p.fields[i], T))
  let total: Value = 0
  for (let i = 1; i <= T; i++) {
    const dev = sum(square(sub(p.fields[i], avg)), -1) // [K, B]
    total = add(total, mean(sum(mul(weightsAt(p, i), dev), 0)))
  }
  return div(total, T)
}

/**
 * MDLoss (eq. 10) of targets y [B, D] under the mixture output at grid time i: −log Σₖ πₖ N(y; μₖ, σₖ² I), averaged
 * over the batch (compute `mixtureDensityNll`, its head packed as [log π | μ | log σ]).
 */
export function mixtureDensityLoss(model: Svfm, params: SvfmParams, p: Propagation, i: number, y: Value): Value {
  const { logWeights, means, logScales } = model.output(params, p, i)
  const [B, K, D] = shapeOfValue(means)
  const head = concat([logWeights, reshape(means, [B, K * D]), reshape(logScales, [B, K * D])], 1)
  return mixtureDensityNll(head, y as Tensor, { components: K, dims: D, scale: 'exp', floor: 0 })
}

/**
 * MDLoss for class labels y [B] (int): −log Σₖ πₖ(t_T) p(y | component k), averaged over the batch. With one
 * deterministic component it is the cross-entropy of the readout.
 */
export function classMixtureLoss(model: Svfm, params: SvfmParams, p: Propagation, y: Tensor, classes: number): Value {
  const logLik = model.classLogLikelihoods(params, p) // [B, K, C]
  const B = y.shape[0]
  const labels = toFlat(y)
  const onehot = new Float64Array(B * classes)
  for (let b = 0; b < B; b++) onehot[b * classes + labels[b]] = 1
  const picked = sum(mul(logLik, reshape(fromData(onehot, [B, classes]), [B, 1, classes])), -1) // [B, K]
  return neg(mean(logsumexp(add(p.logWeights[p.logWeights.length - 1], picked), -1)))
}

/** The squared error ‖m(tᵢ) − y‖² / D of a single component's state (the baselines' ℓ), averaged over the batch. */
export function squaredErrorAt(model: Svfm, p: Propagation, i: number, y: Value): Value {
  const D = model.options.dim
  const m = reshape(slice(p.states[i], [0, 1], null, [0, D]), shapeOfValue(y))
  return mean(div(sum(square(sub(m, y)), -1), D))
}

/**
 * Path targets on the grid: interp(X, t_X, tᵢ) for each path (eq. 11), by a not-a-knot cubic spline through its
 * samples. `paths` is [n, M, D] at `times` [M] on [0, 1]; returns [n, T + 1, D].
 */
export function interpolatePaths(
  paths: Float64Array,
  times: ArrayLike<number>,
  D: number,
  grid: readonly number[],
): Tensor {
  const M = times.length
  const n = paths.length / (M * D)
  const out = new Float64Array(n * grid.length * D)
  const tt = fromData(Float64Array.from(times))
  for (let i = 0; i < n; i++)
    for (let d = 0; d < D; d++) {
      const ys = Float64Array.from({ length: M }, (_, j) => paths[(i * M + j) * D + d])
      const spline = cubicSpline(tt, fromData(ys))
      const at = toFlat(evaluatePiecewise(spline, fromData(Float64Array.from(grid))) as Tensor)
      grid.forEach((_, g) => (out[(i * grid.length + g) * D + d] = at[g]))
    }
  return fromData(out, [n, grid.length, D])
}

/** Which losses an objective adds up (§2.3); the predictive one follows the task. */
export type SvfmLossSettings = {
  /** The predictive ℓ: `'auto'` is the squared error for one deterministic VF and MDLoss otherwise. Default auto. */
  predictive?: 'auto' | 'squared-error' | 'mixture-density'
  /** TLoss (eq. 8) and VLoss (eq. 9); both together are TVLoss. */
  transport?: boolean
  variance?: boolean
  /** FLoss (eq. 11) over the whole grid rather than the end point only (forecasting tasks). */
  forecast?: boolean
  /** λ, the weight of TLoss and VLoss. Default 0.1. */
  lambda?: number
  /** Separate weights of TLoss and VLoss (default λ each). */
  transportWeight?: number
  varianceWeight?: number
}

/** A training batch: starts x [B, D], optional context c [B, C], and labels, end targets or grid targets. */
export type SvfmBatch = {
  x: Tensor
  c?: Tensor
  /** Class labels [B] (int32). */
  labels?: Tensor
  /** End targets [B, D] (endpoint regression). */
  targets?: Tensor
  /** Path targets on the grid [B, T + 1, D] (forecasting). */
  path?: Tensor
}

/** The parts of an objective's value. */
export type SvfmLossParts = { total: Value; predictive: Value; transport: Value; variance: Value }

/** Refuse loss settings the paper rules out: FLoss with TLoss or VLoss (§4.1.3). */
export function checkLossSettings(losses: SvfmLossSettings): void {
  if (losses.forecast && (losses.transport || losses.variance))
    throw new DomainError(
      'svfm',
      'svfm: FLoss penalises deviation from the path, so it cannot be combined with TLoss or VLoss',
    )
  if (losses.lambda !== undefined && !(losses.lambda >= 0)) throw new DomainError('svfm', 'svfm: λ must be ≥ 0')
}

/**
 * The objective of a batch: the predictive loss (class mixture likelihood; ℓ at t_T; or FLoss over the grid) plus
 * λ_T·TLoss + λ_V·VLoss as selected (both λ by default). Returns the parts, for curves.
 */
export function svfmObjective(model: Svfm, losses: SvfmLossSettings = {}) {
  checkLossSettings(losses)
  const { transport = false, variance = false, forecast = false, lambda = 0.1 } = losses
  const wT = losses.transportWeight ?? lambda
  const wV = losses.varianceWeight ?? lambda
  const single = model.options.components === 1 && !model.options.stochastic
  const predictive = losses.predictive ?? 'auto'
  const squared = predictive === 'squared-error' || (predictive === 'auto' && single)
  if (squared && model.options.components > 1)
    throw new DomainError('svfm', 'svfm: the squared error needs a single component (K = 1)')
  const ell = (params: SvfmParams, p: Propagation, i: number, y: Value) =>
    squared ? squaredErrorAt(model, p, i, y) : mixtureDensityLoss(model, params, p, i, y)
  return (params: SvfmParams, batch: SvfmBatch, onSolve?: Parameters<Svfm['propagate']>[3]): SvfmLossParts => {
    const p = model.propagate(params, batch.x, batch.c ?? null, onSolve)
    const T = p.times.length - 1
    let pred: Value
    if (batch.labels) pred = classMixtureLoss(model, params, p, batch.labels, model.options.classes)
    else if (forecast) {
      if (!batch.path) throw new DomainError('svfm', 'svfm: FLoss needs path targets')
      pred = 0
      for (let i = 1; i <= T; i++) {
        const yi = reshape(slice(batch.path, null, [i, i + 1], null), [batch.path.shape[0], model.options.dim])
        pred = add(pred, div(ell(params, p, i, yi), T))
      }
    } else {
      const y =
        batch.targets ??
        (batch.path
          ? reshape(slice(batch.path, null, [T, T + 1], null), [batch.path.shape[0], model.options.dim])
          : null)
      if (!y) throw new DomainError('svfm', 'svfm: no targets')
      pred = ell(params, p, T, y)
    }
    const tl = transport ? transportLoss(p) : 0
    const vl = variance ? varianceLoss(p) : 0
    const total = add(pred, add(mul(wT, tl), mul(wV, vl)))
    return { total, predictive: pred, transport: tl, variance: vl }
  }
}
