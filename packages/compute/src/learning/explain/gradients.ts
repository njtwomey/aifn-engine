/**
 * Gradient attributions of a differentiable scalar function f(x), x ∈ ℝ^d, by `aifn-compute/foundation/autodiff`:
 *
 * - `inputGradient`: the saliency ∇f(x) (Simonyan, Vedaldi and Zisserman, 2014), and gradient × input.
 * - `integratedGradients` (Sundararajan, Taly and Yan, 2017): (x − x′) ⊙ ∫₀¹ ∇f(x′ + α(x − x′)) dα along the straight
 *   path from a baseline x′, by the trapezoid rule on `steps` intervals (or a left/right/midpoint Riemann sum). Its
 *   attributions sum to f(x) − f(x′) as the steps grow (completeness); `delta` reports the gap.
 * - `smoothGrad` (Smilkov et al., 2017): the mean gradient over x + ε, ε ~ N(0, σ²I), which averages out the gradient's
 *   local noise.
 *
 * The gradients along a path or over noisy copies are taken in one batch with `vmap(grad(f))`.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A differentiable scalar function of one input vector [d]. */
export type Differentiable = (x: Tensor) => Value

const scalar = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

/** Gradients of f at each row of X [m, d], as rows [m, d]. */
function batchGradients(f: Differentiable, rows: Float64Array, m: Size, d: Size): Float64Array {
  const g = vmap(grad(f))(fromData(rows, [m, d])) as Tensor
  return Float64Array.from(toFlat(g))
}

/** ∇f(x) and gradient × input x ⊙ ∇f(x). */
export function inputGradient(f: Differentiable, x: VectorLike): { gradient: Float64Array; timesInput: Float64Array } {
  const xv = dense.toF64(x, 'inputGradient')
  const gradient = batchGradients(f, Float64Array.from(xv), 1, xv.length)
  return { gradient, timesInput: Float64Array.from(gradient, (g, i) => g * xv[i]) }
}

/** The quadrature rule of `integratedGradients`. */
export type PathRule = 'trapezoid' | 'left' | 'right' | 'midpoint'

/** Integrated gradients of f at x [d] from `baseline` (default 0) with `steps` (default 50) path intervals. */
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

/** SmoothGrad: the mean of ∇f(x + ε) over `samples` (default 50) draws of ε ~ N(0, σ²I) from `stream`, σ = `noise`. */
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
