/**
 * Gradient attributions of a differentiable scalar function $f(\xvec)$, $\xvec \in \reals^d$, by
 * `aifn-compute/foundation/autodiff`:
 *
 * - `inputGradient`: the saliency $\nabla f(\xvec)$ (Simonyan, Vedaldi and Zisserman, 2014), and gradient times input.
 * - `integratedGradients` (Sundararajan, Taly and Yan, 2017):
 *   $(\xvec - \xvec') \odot \int_0^1 \nabla f(\xvec' + \alpha(\xvec - \xvec'))\, d\alpha$ along the straight path
 *   from a baseline $\xvec'$, by the trapezoid rule on `steps` intervals (or a left, right or midpoint Riemann sum).
 *   Its attributions sum to $f(\xvec) - f(\xvec')$ as the steps grow (completeness); `delta` reports the gap.
 * - `smoothGrad` (Smilkov et al., 2017): the mean gradient over $\xvec + \epsilonvec$,
 *   $\epsilonvec \sim \Gauss(\zeros, \sigma^2\Imat)$, which averages out the gradient's local noise.
 *
 * The gradients along a path or over noisy copies are taken in one batch with `vmap(grad(f))`.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A differentiable scalar function of one input vector ($d$ values), returning a number or a one-element tensor. */
export type Differentiable = (x: Tensor) => Value

/**
 * A scalar function's value as a number.
 *
 * @param v A number, or a tensor whose first element is read.
 * @returns The number.
 */
const scalar = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

/**
 * Gradients of $f$ at each of $m$ points, in one `vmap(grad(f))` call.
 *
 * @param f The function differentiated.
 * @param rows The points, row-major ($m \times d$ values); not modified.
 * @param m The number of points.
 * @param d The number of features.
 * @returns The gradients, row-major ($m \times d$ values): row $k$ is $\nabla f$ at point $k$.
 */
function batchGradients(f: Differentiable, rows: Float64Array, m: Size, d: Size): Float64Array {
  const g = vmap(grad(f))(fromData(rows, [m, d])) as Tensor
  return Float64Array.from(toFlat(g))
}

/**
 * The saliency $\nabla f(\xvec)$ and gradient times input $\xvec \odot \nabla f(\xvec)$.
 *
 * @param f The function explained.
 * @param x The instance $\xvec$ ($d$ values).
 * @returns `gradient`, $\nabla f(\xvec)$, and `timesInput`, $\xvec \odot \nabla f(\xvec)$ ($d$ values each).
 *
 * @example A linear function's gradient is its weights
 * const f = (x) => sum(mul(tensor([2, -1, 0.5]), x))
 * print(inputGradient(f, [1, 2, 3]))
 */
export function inputGradient(f: Differentiable, x: VectorLike): { gradient: Float64Array; timesInput: Float64Array } {
  const xv = dense.toF64(x, 'inputGradient')
  const gradient = batchGradients(f, Float64Array.from(xv), 1, xv.length)
  return { gradient, timesInput: Float64Array.from(gradient, (g, i) => g * xv[i]) }
}

/**
 * The quadrature rule of `integratedGradients`: the trapezoid rule on the `steps + 1` points $\alpha = k/\text{steps}$,
 * or a Riemann sum at the left ends, right ends or midpoints of the `steps` intervals.
 */
export type PathRule = 'trapezoid' | 'left' | 'right' | 'midpoint'

/**
 * Integrated gradients of $f$ at $\xvec$ from a baseline $\xvec'$ (see the file comment), with the path's gradients
 * taken in one batch. Throws `DomainError` when `steps` is not a positive integer.
 *
 * @param f The function explained.
 * @param x The instance $\xvec$ ($d$ values).
 * @param options The baseline and the quadrature.
 * @param options.baseline The baseline $\xvec'$ ($d$ values; default all zeros).
 * @param options.steps The number of intervals the path $[0, 1]$ is cut into (default 50).
 * @param options.rule The quadrature rule (default `'trapezoid'`).
 * @returns `values`, the attributions ($d$); `delta`, $\sum_i \text{values}_i - (f(\xvec) - f(\xvec'))$, the
 *   completeness gap from the quadrature; `output`, $f(\xvec)$; and `baselineOutput`, $f(\xvec')$.
 *
 * @example A linear model: weight times (x minus the baseline)
 * const f = (x) => sum(mul(tensor([2, -1, 0.5]), x))
 * const r = integratedGradients(f, [1, 2, 3], { baseline: [1, 0, 1] })
 * print('values =', r.values)
 * print('w (x - baseline) =', [2 * 0, -1 * 2, 0.5 * 2])
 *
 * @example A product shares its output, and completeness holds
 * const f = (x) => mul(get(x, 0), get(x, 1))
 * const r = integratedGradients(f, [1, 2], { steps: 10 })
 * print('values =', r.values, ' output =', r.output, ' delta =', r.delta)
 */
export function integratedGradients(
  f: Differentiable,
  x: VectorLike,
  options: { baseline?: VectorLike; steps?: Size; rule?: PathRule } = {},
): { values: Float64Array; delta: number; output: number; baselineOutput: number } {
  const xv = dense.toF64(x, 'integratedGradients')
  const d = xv.length
  const base = options.baseline ? dense.toF64(options.baseline, 'integratedGradients') : new Float64Array(d)
  const { steps = 50, rule = 'trapezoid' } = options
  if (!(Number.isInteger(steps) && steps >= 1))
    throw new DomainError('integratedGradients', 'integratedGradients: steps must be ≥ 1')
  const alphas: number[] = []
  const w: number[] = []
  if (rule === 'trapezoid')
    for (let k = 0; k <= steps; k++) {
      alphas.push(k / steps)
      w.push((k === 0 || k === steps ? 0.5 : 1) / steps)
    }
  else
    for (let k = 0; k < steps; k++) {
      alphas.push((k + (rule === 'left' ? 0 : rule === 'right' ? 1 : 0.5)) / steps)
      w.push(1 / steps)
    }
  const m = alphas.length
  const rows = new Float64Array(m * d)
  alphas.forEach((a, k) => {
    for (let i = 0; i < d; i++) rows[k * d + i] = base[i] + a * (xv[i] - base[i])
  })
  const G = batchGradients(f, rows, m, d)
  const values = new Float64Array(d)
  for (let k = 0; k < m; k++) for (let i = 0; i < d; i++) values[i] += w[k] * G[k * d + i]
  for (let i = 0; i < d; i++) values[i] *= xv[i] - base[i]
  const output = scalar(f(fromData(Float64Array.from(xv), [d])))
  const baselineOutput = scalar(f(fromData(Float64Array.from(base), [d])))
  const sum = values.reduce((a, b) => a + b, 0)
  return { values, delta: sum - (output - baselineOutput), output, baselineOutput }
}

/**
 * SmoothGrad: the mean of $\nabla f(\xvec + \epsilonvec)$ over draws of
 * $\epsilonvec \sim \Gauss(\zeros, \sigma^2\Imat)$, with $\sigma$ in the units of $\xvec$ (not relative to the
 * input's range, as in the paper).
 *
 * @param f The function explained.
 * @param x The instance $\xvec$ ($d$ values).
 * @param stream The random stream the noise is drawn from.
 * @param options The number of draws and their spread.
 * @param options.samples The number of noisy copies averaged (default 50).
 * @param options.noise The noise's standard deviation $\sigma$ (default 0.1).
 * @returns `values`, the mean gradient ($d$), and `samples`, the noisy copies ($\text{samples} \times d$).
 *
 * @example Noise smooths the gradient of a ReLU at its kink
 * const f = (x) => maximum(get(x, 0), 0)
 * print('gradient at 0:', inputGradient(f, [0]).gradient)
 * print('smoothed:', smoothGrad(f, [0], stream(0), { samples: 200, noise: 0.5 }).values)
 */
export function smoothGrad(
  f: Differentiable,
  x: VectorLike,
  stream: Stream,
  options: { samples?: Size; noise?: number } = {},
): { values: Float64Array; samples: Tensor } {
  const xv = dense.toF64(x, 'smoothGrad')
  const d = xv.length
  const { samples: m = 50, noise = 0.1 } = options
  const eps = toFlat(normal(stream, 0, noise, { shape: [m * d] }))
  const rows = Float64Array.from({ length: m * d }, (_, t) => xv[t % d] + eps[t])
  const G = batchGradients(f, rows, m, d)
  const values = new Float64Array(d)
  for (let k = 0; k < m; k++) for (let i = 0; i < d; i++) values[i] += G[k * d + i] / m
  return { values, samples: fromData(rows, [m, d]) }
}
