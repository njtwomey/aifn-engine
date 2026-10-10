/**
 * The losses of stochastic vector field mixtures (Twomey, Kozłowski & Santos-Rodríguez, 2020, §2.3), on the moments
 * `propagate` carries over the grid $t_0, \dots, t_T$:
 *
 * - **TLoss** (eq. 8), the transportation loss $\frac{1}{T} \sum_i \norm{\hvec(t_i) - \hvec(t_{i-1})}^2$, least when
 *   every step points the same way;
 * - **VLoss** (eq. 9), the variance loss $\frac{1}{T} \sum_i \norm{\nabla \hvec(t_i) - \expect[\nabla \hvec(t)]}^2$
 *   with $\expect[\nabla \hvec(t)] = \frac{1}{T} \sum_i \nabla \hvec(t_i)$, least when the VF is constant along the
 *   path; TLoss + VLoss is TVLoss. Both are computed per component and weighted by $\pivec$;
 * - **MDLoss** (eq. 10), $-\log \sum_k p(\hvec(t) \mid \muvec^{(k)}(t), \tau^{(k)}(t)) \, \pi_k(t)$: the mixture
 *   density of a target under the model's output (compute `mixtureDensityNll`); for class labels, the mixture of the
 *   components' class likelihoods;
 * - **FLoss** (eq. 11), the forecasting loss
 *   $\frac{1}{T} \sum_i \ell(\hvec(t_i), \operatorname{interp}(\Xmat, t_X, t_i))$ against a path interpolated at the
 *   grid times (cubic spline), with $\ell$ the squared error for a single VF and MDLoss otherwise.
 *
 * FLoss penalises any deviation from the path, so it is incompatible with TLoss and VLoss (§4.1.3); `svfmObjective`
 * refuses the combination. $\lambda$ weighs the path regularisers against the predictive loss. Every loss is
 * differentiable in the parameters and averaged over the batch.
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

/**
 * $\pivec(t_i)$ as weights $[K, B]$.
 *
 * @param p The propagation, whose `logWeights` hold $\log \pivec$ $[B, K]$ per grid time.
 * @param i The grid index.
 * @returns The weights, components first.
 */
const weightsAt = (p: Propagation, i: number): Value => permute(exp(p.logWeights[i]), [1, 0])

/**
 * TLoss (eq. 8): $\frac{1}{T} \sum_{i=1}^T \sum_k \pi_k(t_{i-1}) \norm{\mvec_k(t_i) - \mvec_k(t_{i-1})}^2$, the squared
 * distance each component's state travels per interval (from where interval $i$ starts to where it arrives), averaged
 * over the batch.
 *
 * @param p The propagation of a batch, from `propagate`.
 * @returns The loss, a scalar.
 *
 * @example A constant field $f = v$ travels $v / T$ per interval, and has no VLoss; the field $f = t$ has some
 * // One VF with no hidden layer: weight 0 on (h, t) and bias 0.5 is the constant field 0.5
 * const model = svfm({ dim: 1, layers: 0, grid: 4 })
 * const constant = { ...model.init(stream(0)), fields: [{ weight: zeros([1, 2, 1]), bias: tensor([[[0.5]]]) }] }
 * const p = model.propagate(constant, tensor([[0], [1]]), null)
 * print('constant: TLoss', transportLoss(p), ' (v / T)^2', (0.5 / 4) ** 2, ' VLoss', varianceLoss(p))
 * const timed = { ...constant, fields: [{ weight: tensor([[[0], [1]]]), bias: tensor([[[0]]]) }] }
 * const q = model.propagate(timed, tensor([[0], [1]]), null)
 * print('f = t: TLoss', transportLoss(q), ' VLoss', varianceLoss(q))
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
 * VLoss (eq. 9):
 * $\frac{1}{T} \sum_{i=1}^T \sum_k \pi_k(t_i) \norm{\nabla \hvec^{(k)}(t_i) - \expect[\nabla \hvec^{(k)}]}^2$, with
 * $\expect[\nabla \hvec^{(k)}] = \frac{1}{T} \sum_i \nabla \hvec^{(k)}(t_i)$ the expected VF along component $k$'s
 * path (the mean VF at the arrived states); averaged over the batch.
 *
 * @param p The propagation of a batch, from `propagate`.
 * @returns The loss, a scalar.
 *
 * @example The field $f = t$ at $t_i = i/4$ deviates from its mean $0.625$ by $\pm 0.125$ and $\pm 0.375$
 * const model = svfm({ dim: 1, layers: 0, grid: 4 })
 * const timed = { ...model.init(stream(0)), fields: [{ weight: tensor([[[0], [1]]]), bias: tensor([[[0]]]) }] }
 * const p = model.propagate(timed, tensor([[0]]), null)
 * print('VLoss:', varianceLoss(p), ' by hand:', (2 * 0.125 ** 2 + 2 * 0.375 ** 2) / 4)
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
 * MDLoss (eq. 10) of targets under the mixture output at grid time $t_i$:
 * $-\log \sum_k \pi_k \Gauss(\yvec; \muvec_k, \sigma_k^2 \Imat)$ with $\sigma_k^2 = r_k^2 + \sigma_0^2$, averaged over
 * the batch (compute `mixtureDensityNll`, its head packed as $[\log \pivec \mid \muvec \mid \log \sigmavec]$).
 *
 * @param model The SVFM.
 * @param params Its parameters (for the noise floor $\sigma_0$).
 * @param p The propagation of the batch.
 * @param i The grid index of the output, 0 to $T$.
 * @param y The targets, $[B, D]$.
 * @returns The loss, a scalar.
 *
 * @example One deterministic VF: MDLoss is the Gaussian negative log-likelihood with the noise floor $\sigma_0 = 0.1$
 * const model = svfm({ dim: 1, layers: 0, grid: 4 })
 * const params = { ...model.init(stream(0)), fields: [{ weight: zeros([1, 2, 1]), bias: tensor([[[0.5]]]) }] }
 * const p = model.propagate(params, tensor([[0], [1]]), null)
 * // h(1) is 0.5 and 1.5: the targets miss by 0 and 0.5
 * const y = tensor([[0.5], [2]])
 * print('MDLoss at t_T:', mixtureDensityLoss(model, params, p, 4, y))
 * const logN = (r) => -0.5 * (r / 0.1) ** 2 - Math.log(0.1) - 0.5 * Math.log(2 * Math.PI)
 * print('by hand:', -(logN(0) + logN(0.5)) / 2)
 */
export function mixtureDensityLoss(model: Svfm, params: SvfmParams, p: Propagation, i: number, y: Value): Value {
  const { logWeights, means, logScales } = model.output(params, p, i)
  const [B, K, D] = shapeOfValue(means)
  const head = concat([logWeights, reshape(means, [B, K * D]), reshape(logScales, [B, K * D])], 1)
  return mixtureDensityNll(head, y as Tensor, { components: K, dims: D, scale: 'exp', floor: 0 })
}

/**
 * MDLoss for class labels: $-\log \sum_k \pi_k(t_T) \, p(y \mid k)$, averaged over the batch, with $p(y \mid k)$ the
 * readout's softmax at component $k$'s state (probit-scaled for SVF units). With one deterministic component it is the
 * cross-entropy of the readout.
 *
 * @param model The SVFM, with a classifier readout.
 * @param params Its parameters.
 * @param p The propagation of the batch.
 * @param y The labels, $[B]$ (int32), each in $[0, \text{classes})$.
 * @param classes The number of classes.
 * @returns The loss, a scalar.
 *
 * @example With one deterministic VF it is the readout's cross-entropy
 * const model = svfm({ dim: 1, classes: 2, layers: 0, grid: 2 })
 * const params = model.init(stream(0))
 * const p = model.propagate(params, tensor([[0.5], [-1]]), null)
 * print('class mixture loss:', classMixtureLoss(model, params, p, fromData(Int32Array.of(0, 1)), 2))
 * const z = toArray(add(matmul(p.states.at(-1), params.readout.weight), params.readout.bias))[0]
 * const ce = [0, 1].map((i) => Math.log(Math.exp(z[i][0]) + Math.exp(z[i][1])) - z[i][i])
 * print('cross-entropy of the readout:', (ce[0] + ce[1]) / 2)
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

/**
 * The squared error $\norm{\mvec(t_i) - \yvec}^2 / D$ of a single component's state (the baselines' $\ell$), on its
 * first $D$ coordinates, averaged over the batch.
 *
 * @param model The SVFM (the first component is read).
 * @param p The propagation of the batch.
 * @param i The grid index, 0 to $T$.
 * @param y The targets, $[B, D]$.
 * @returns The loss, a scalar.
 *
 * @example The constant field 0.5 carries 0 and 1 to 0.5 and 1.5
 * const model = svfm({ dim: 1, layers: 0, grid: 4 })
 * const params = { ...model.init(stream(0)), fields: [{ weight: zeros([1, 2, 1]), bias: tensor([[[0.5]]]) }] }
 * const p = model.propagate(params, tensor([[0], [1]]), null)
 * print('squared error:', squaredErrorAt(model, p, 4, tensor([[0.5], [2]])), ' by hand:', (0 ** 2 + 0.5 ** 2) / 2)
 */
export function squaredErrorAt(model: Svfm, p: Propagation, i: number, y: Value): Value {
  const D = model.options.dim
  const m = reshape(slice(p.states[i], [0, 1], null, [0, D]), shapeOfValue(y))
  return mean(div(sum(square(sub(m, y)), -1), D))
}

/**
 * Path targets on the grid: $\operatorname{interp}(\Xmat, t_X, t_i)$ for each path (eq. 11), by a not-a-knot cubic
 * spline through its samples, coordinate by coordinate.
 *
 * @param paths The sampled paths, $[n, M, D]$ row-major.
 * @param times The $M$ sample times $t_X$, increasing, on $[0, 1]$.
 * @param D The dimension of a path's state.
 * @param grid The times to interpolate at (the grid $t_0, \dots, t_T$).
 * @returns The targets, $[n, T + 1, D]$.
 *
 * @example Three samples of $t^2$ pin the spline to it
 * const at = interpolatePaths(Float64Array.of(0, 0.25, 1), [0, 0.5, 1], 1, [0, 0.25, 0.5, 0.75, 1])
 * print('t^2 at the grid times:', at)
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
  /** Add TLoss (eq. 8). Default false. */
  transport?: boolean
  /** Add VLoss (eq. 9); with TLoss, TVLoss. Default false. */
  variance?: boolean
  /** FLoss (eq. 11) over the whole grid rather than the end point only (forecasting tasks). Default false. */
  forecast?: boolean
  /** $\lambda$, the weight of TLoss and VLoss. Default 0.1. */
  lambda?: number
  /** A separate weight of TLoss (default $\lambda$). */
  transportWeight?: number
  /** A separate weight of VLoss (default $\lambda$). */
  varianceWeight?: number
}

/** A training batch: starts, an optional context, and labels, end targets or grid targets. */
export type SvfmBatch = {
  /** Starts $\xvec$, $[B, D]$. */
  x: Tensor
  /** Context $\cvec$, $[B, C]$. */
  c?: Tensor
  /** Class labels $[B]$ (int32). */
  labels?: Tensor
  /** End targets $[B, D]$ (endpoint regression). */
  targets?: Tensor
  /** Path targets on the grid $[B, T + 1, D]$ (forecasting). */
  path?: Tensor
}

/**
 * The parts of an objective's value: the `total`, the `predictive` loss, and the unweighted `transport` (TLoss) and
 * `variance` (VLoss), each 0 when not selected.
 */
export type SvfmLossParts = { total: Value; predictive: Value; transport: Value; variance: Value }

/**
 * Refuse loss settings the paper rules out, FLoss with TLoss or VLoss (§4.1.3), and a $\lambda$ that is negative or
 * NaN: each throws `DomainError`.
 *
 * @param losses The settings to check.
 *
 * @example FLoss with TLoss is refused; TVLoss is fine
 * try {
 *   checkLossSettings({ forecast: true, transport: true })
 * } catch (e) {
 *   print(e.message)
 * }
 * checkLossSettings({ transport: true, variance: true, lambda: 0.5 })
 * print('TVLoss with lambda 0.5: accepted')
 */
export function checkLossSettings(losses: SvfmLossSettings): void {
  if (losses.forecast && (losses.transport || losses.variance))
    throw new DomainError(
      'svfm',
      'svfm: FLoss penalises deviation from the path, so it cannot be combined with TLoss or VLoss',
    )
  if (losses.lambda !== undefined && !(losses.lambda >= 0)) throw new DomainError('svfm', 'svfm: λ must be ≥ 0')
}

/**
 * The objective of a batch: the predictive loss (the class mixture likelihood; $\ell$ at $t_T$; or FLoss over the
 * grid) plus $\lambda_T \cdot \text{TLoss} + \lambda_V \cdot \text{VLoss}$ as selected (both $\lambda$ by default).
 * $\ell$ is the squared error for one deterministic VF and MDLoss otherwise, unless `predictive` says. The settings are
 * checked first (`checkLossSettings`), and asking for the squared error with $K > 1$ throws `DomainError`.
 *
 * @param model The SVFM.
 * @param losses Which losses to add, and their weights; see `SvfmLossSettings`.
 * @returns A function of the parameters, a batch and an optional `onSolve` callback giving the parts of the
 *   objective, for curves; it is differentiable in the parameters, and throws `DomainError` when the batch lacks the
 *   targets the losses need.
 *
 * @example The parts of the objective with TVLoss
 * const model = svfm({ dim: 1, grid: 4, hidden: 8 })
 * const objective = svfmObjective(model, { transport: true, variance: true, lambda: 0.1 })
 * const parts = objective(model.init(stream(0)), { x: tensor([[-1], [1]]), targets: tensor([[-2], [2]]) })
 * print('predictive', parts.predictive, ' TLoss', parts.transport, ' VLoss', parts.variance, ' total', parts.total)
 *
 * @example Gradient steps on the objective fit a scaling
 * const model = svfm({ dim: 1, grid: 4, layers: 0 })
 * const objective = svfmObjective(model)
 * const batch = { x: tensor([[-1], [1]]), targets: tensor([[-2], [2]]) }
 * let params = model.init(stream(0))
 * const lossOf = (fields) => objective({ ...params, fields }, batch).total
 * const step = (l, g) => ({ weight: sub(l.weight, mul(0.1, g.weight)), bias: sub(l.bias, mul(0.1, g.bias)) })
 * for (let k = 0; k <= 30; k++) {
 *   const { value, grad: g } = valueAndGrad(lossOf)(params.fields)
 *   if (k % 10 === 0) print(`step ${k}: squared error`, value)
 *   params = { ...params, fields: params.fields.map((l, j) => step(l, g[j])) }
 * }
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
