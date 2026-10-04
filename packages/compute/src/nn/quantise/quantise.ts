/**
 * Quantisation of tensors to b-bit integers, after torch.ao.quantization (Jacob et al., 2018, "Quantization and
 * training of neural networks for efficient integer-arithmetic-only inference", CVPR; Nagel et al., 2021, "A white
 * paper on neural network quantization", arXiv:2106.08295).
 *
 * - A quantiser maps x to q = clamp(round(x/s) + z, q_min, q_max) and back to x̂ = s(q − z): the scale s and zero point
 *   z per tensor, or per channel along an axis (`quantisationParams`; affine uses the range [min, max] ∪ {0},
 *   symmetric s = max|x| / ((q_max − q_min)/2) with z at the middle of the range).
 * - Rounding to nearest (halves to even, as torch) or stochastic rounding ⌊x + u⌋, u ~ U[0, 1), which is unbiased.
 * - `fakeQuantise`: quantise and dequantise in the forward pass, and pass the gradient straight through where x lies
 *   inside the representable range (zero where it was clipped): the straight-through estimator (Bengio, Léonard &
 *   Courville, 2013), as `torch.fake_quantize_per_tensor_affine`, built with `customVjp`.
 * - `quantisedMatmul`: an int8 matmul simulated exactly: integer products accumulated in int32 range, then rescaled.
 * - `gptqQuantise`: GPTQ's column-by-column rounding with second-order error feedback from the layer's inputs (Frantar
 *   et al., 2023, "GPTQ: accurate post-training quantization for generative pre-trained transformers", ICLR).
 * - `awqQuantise`: activation-aware scaling of the input channels before rounding, the scale's exponent searched on a
 *   grid (Lin et al., 2024, "AWQ: activation-aware weight quantization for LLM compression and acceleration", MLSys).
 */

import { customVjp } from 'aifn-compute/foundation/autodiff'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  clip,
  dense,
  div,
  fromData,
  greaterEqual,
  lessEqual,
  map,
  mul,
  sub,
  toFlat,
  type Matrix,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { cholesky, inverse } from 'aifn-compute/numerics/linalg'

/** Round half to even (banker's rounding), as `torch.round` and C's `nearbyint`. */
export function roundHalfEven(v: Scalar): Scalar {
  const r = Math.round(v)
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/** The integer range of a b-bit quantiser: signed [−2^{b−1}, 2^{b−1} − 1] or unsigned [0, 2^b − 1]. */
export function integerRange(bits: Size, signed: boolean): { qmin: number; qmax: number } {
  if (!(Number.isInteger(bits) && bits >= 1 && bits <= 16))
    throw new DomainError('integerRange', 'integerRange: bits must be an integer in 1 … 16')
  return signed ? { qmin: -(2 ** (bits - 1)), qmax: 2 ** (bits - 1) - 1 } : { qmin: 0, qmax: 2 ** bits - 1 }
}

/** The parameters of a quantiser: scale and zero point per tensor (length 1) or per channel along `axis`. */
export interface QuantisationParams {
  scale: Float64Array
  zeroPoint: Float64Array
  qmin: number
  qmax: number
  bits: Size
  /** The channel axis, or null for per-tensor. */
  axis: number | null
  scheme: 'affine' | 'symmetric'
}

/** Options for `quantisationParams`. */
export interface QuantisationOptions {
  /** Bits per value. Default 8. */
  bits?: Size
  /** `affine` (asymmetric, unsigned range by default) or `symmetric` (signed range by default). Default `affine`. */
  scheme?: 'affine' | 'symmetric'
  /** Use a signed integer range. Default: true for symmetric, false for affine. */
  signed?: boolean
  /** Per-channel along this axis (e.g. 0 for a weight matrix's output rows); omitted: per tensor. */
  axis?: number
}

/** The channel index of each flat element of a tensor of `shape` along `axis`. */
function channelOf(shape: readonly number[], axis: number): (i: number) => number {
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const n = shape[axis]
  return (i) => Math.floor(i / inner) % n
}

/**
 * The scale and zero point of a quantiser fitted to the range of x (min–max calibration, as torch's MinMaxObserver).
 * Affine: the range is widened to include 0, s = (max − min)/(q_max − q_min) and z = q_min − round(min/s), clamped to
 * the integer range, so 0 is exactly representable. Symmetric: s = max(|min|, |max|)/((q_max − q_min)/2) and z the
 * middle of the range (0 for a signed range, 2^{b−1} for an unsigned one). A zero range gives s = 1.
 */
export function quantisationParams(x: Value, options: QuantisationOptions = {}): QuantisationParams {
  const t = x as Tensor
  const v = toFlat(t)
  const shape = t.shape ?? [v.length]
  const bits = options.bits ?? 8
  const scheme = options.scheme ?? 'affine'
  const signed = options.signed ?? scheme === 'symmetric'
  const { qmin, qmax } = integerRange(bits, signed)
  const axis = options.axis ?? null
  if (axis !== null && (axis < 0 || axis >= shape.length))
    throw new ShapeError('quantisationParams', `quantisationParams: axis ${axis} out of range`)
  const channels = axis === null ? 1 : shape[axis]
  const of = axis === null ? () => 0 : channelOf(shape, axis)
  const lo = new Float64Array(channels).fill(Infinity)
  const hi = new Float64Array(channels).fill(-Infinity)
  v.forEach((x, i) => {
    const c = of(i)
    if (x < lo[c]) lo[c] = x
    if (x > hi[c]) hi[c] = x
  })
  const scale = new Float64Array(channels)
  const zeroPoint = new Float64Array(channels)
  for (let c = 0; c < channels; c++) {
    const mn = Math.min(lo[c], 0)
    const mx = Math.max(hi[c], 0)
    if (scheme === 'symmetric') {
      const m = Math.max(-mn, mx)
      scale[c] = m > 0 ? m / ((qmax - qmin) / 2) : 1
      zeroPoint[c] = signed ? 0 : Math.ceil((qmin + qmax) / 2)
    } else {
      scale[c] = mx > mn ? (mx - mn) / (qmax - qmin) : 1
      zeroPoint[c] = Math.min(qmax, Math.max(qmin, qmin - roundHalfEven(mn / scale[c])))
    }
  }
  return { scale, zeroPoint, qmin, qmax, bits, axis, scheme }
}

/** Per-element scale and zero point of params broadcast over a tensor of `shape`. */
function expand(params: QuantisationParams, shape: readonly number[], n: number): { s: Float64Array; z: Float64Array } {
  if (params.axis === null)
    return { s: new Float64Array(n).fill(params.scale[0]), z: new Float64Array(n).fill(params.zeroPoint[0]) }
  const of = channelOf(shape, params.axis)
  return {
    s: Float64Array.from({ length: n }, (_, i) => params.scale[of(i)]),
    z: Float64Array.from({ length: n }, (_, i) => params.zeroPoint[of(i)]),
  }
}

/** Options for `quantise`. */
export interface RoundingOptions {
  /** `nearest` (halves to even) or `stochastic` (⌊x + u⌋, u ~ U[0, 1), needing `stream`). Default `nearest`. */
  rounding?: 'nearest' | 'stochastic'
  stream?: Stream
}

/** The integers q = clamp(round(x/s) + z, q_min, q_max) (stored as float64 values). */
export function quantise(x: Tensor, params: QuantisationParams, options: RoundingOptions = {}): Tensor {
  const v = toFlat(x)
  const { s, z } = expand(params, x.shape, v.length)
  let u: number[] | null = null
  if (options.rounding === 'stochastic') {
    if (!options.stream) throw new DomainError('quantise', 'quantise: stochastic rounding needs a stream')
    u = toFlat(uniform(child(options.stream, 'round'), 0, 1, { shape: [v.length] }) as Tensor)
  }
  const out = Float64Array.from(v, (val, i) => {
    const r = val / s[i]
    const q = u ? Math.floor(r + u[i]) : roundHalfEven(r)
    return Math.min(params.qmax, Math.max(params.qmin, q + z[i]))
  })
  return fromData(out, x.shape)
}

/** The real values x̂ = s(q − z) of integers q. */
export function dequantise(q: Tensor, params: QuantisationParams): Tensor {
  const v = toFlat(q)
  const { s, z } = expand(params, q.shape, v.length)
  return fromData(
    Float64Array.from(v, (val, i) => s[i] * (val - z[i])),
    q.shape,
  )
}

/**
 * Fake quantisation x ↦ s(clamp(round(x/s) + z, q_min, q_max) − z) with the straight-through estimator (Bengio,
 * Léonard & Courville, 2013; Jacob et al., 2018): the backward pass multiplies the cotangent by 1 where
 * q_min ≤ round(x/s) + z ≤ q_max and by 0 where the value was clipped, as `torch.fake_quantize_per_tensor_affine`.
 * Written with `customVjp`, so it can sit inside a model trained by `grad`; quantisation-aware training uses it on the
 * weights and activations.
 */
export function fakeQuantise(x: Value, params: QuantisationParams): Value {
  if (params.axis !== null && params.scale.length > 1)
    throw new DomainError('fakeQuantise', 'fakeQuantise: per-channel parameters need a concrete tensor; use quantise')
  const s = params.scale[0]
  const z = params.zeroPoint[0]
  const lo = params.qmin
  const hi = params.qmax
  const integer = (v: Value): Value => add(map(div(v, s), roundHalfEven), z)
  const forward = (v: Value): Value => mul(sub(clip(integer(v), lo, hi), z), s)
  const ste = customVjp(
    forward,
    (v: Value) => {
      const q = integer(v)
      return { out: mul(sub(clip(q, lo, hi), z), s), residuals: mul(greaterEqual(q, lo), lessEqual(q, hi)) }
    },
    (inside: Value, g: Value) => [mul(g, inside)],
  )
  return ste(x)
}

/** The error of a quantiser on x: mean squared error, signal-to-quantisation-noise ratio (dB), the clipped share. */
export function quantisationError(
  x: Tensor,
  params: QuantisationParams,
  options: RoundingOptions = {},
): { mse: Scalar; sqnr: Scalar; clipped: Scalar; maxError: Scalar } {
  const v = toFlat(x)
  const back = toFlat(dequantise(quantise(x, params, options), params))
  const { s, z } = expand(params, x.shape, v.length)
  let se = 0
  let ss = 0
  let clipped = 0
  let maxError = 0
  v.forEach((val, i) => {
    const e = val - back[i]
    se += e * e
    ss += val * val
    maxError = Math.max(maxError, Math.abs(e))
    const q = roundHalfEven(val / s[i]) + z[i]
    if (q < params.qmin || q > params.qmax) clipped++
  })
  return { mse: se / v.length, sqnr: 10 * Math.log10(ss / Math.max(se, 1e-300)), clipped: clipped / v.length, maxError }
}

/** The result of `quantisedMatmul`. */
export interface QuantisedMatmul {
  /** The rescaled output Y ≈ AB (m × n). */
  y: Matrix
  /** The integer operands and the int32 accumulator Σ_k (q_A − z_A)(q_B − z_B). */
  qa: Matrix
  qb: Matrix
  accumulator: Matrix
  /** The quantisers used for A (per tensor, affine) and B (per output column, symmetric). */
  paramsA: QuantisationParams
  paramsB: QuantisationParams
  /** The largest |accumulator|: it must stay below 2³¹ for int32 accumulation. */
  maxAccumulator: Scalar
}

/**
 * An integer matmul simulated exactly (Jacob et al., 2018, §2.2): A (m × k activations) is quantised per tensor with
 * an affine b-bit quantiser and B (k × n weights) per output column with a symmetric one; the products of the
 * zero-point-shifted integers are summed in integer arithmetic (exact in float64 at these sizes), and the output is
 * rescaled, Y[i, j] = s_A s_B[j] Σ_k (q_A[i, k] − z_A) q_B[k, j].
 */
export function quantisedMatmul(A: MatrixLike, B: MatrixLike, { bits = 8 }: { bits?: Size } = {}): QuantisedMatmul {
  const a = dense.toMatrixF64(A, 'quantisedMatmul A')
  const b = dense.toMatrixF64(B, 'quantisedMatmul B', a.n)
  const ta = fromData(a.data, [a.m, a.n])
  const tb = fromData(b.data, [b.m, b.n])
  const paramsA = quantisationParams(ta, { bits, scheme: 'affine' })
  const paramsB = quantisationParams(tb, { bits, scheme: 'symmetric', axis: 1 })
  const qa = quantise(ta, paramsA)
  const qb = quantise(tb, paramsB)
  const qad = dense.data(qa).map((v) => v - paramsA.zeroPoint[0])
  const qbd = dense.data(qb)
  const acc = dense.matMul(qad, qbd, a.m, a.n, b.n)
  const y = Float64Array.from(acc, (v, i) => v * paramsA.scale[0] * paramsB.scale[i % b.n])
  return {
    y: fromData(y, [a.m, b.n]),
    qa,
    qb,
    accumulator: fromData(acc, [a.m, b.n]),
    paramsA,
    paramsB,
    maxAccumulator: Math.max(...Array.from(acc, Math.abs)),
  }
}

/** The result of `gptqQuantise`. */
export interface GptqResult {
  /** The quantised-and-dequantised weights (out × in). */
  weights: Matrix
  /** Per-output-row quantisers (symmetric). */
  params: QuantisationParams
  /** ‖X Wᵀ − X Ŵᵀ‖²_F / n for GPTQ and for round-to-nearest with the same quantisers. */
  outputError: Scalar
  roundToNearestError: Scalar
}

/**
 * GPTQ-lite (Frantar et al., 2023): quantise the weights W (out × in) of a linear layer y = Wx column by column; after
 * rounding column j, its error, divided by [H⁻¹]_jj, is spread over the not-yet-quantised columns through row j of
 * H⁻¹, where H = 2XᵀX/n + λI is the Hessian of the layer's squared output error over calibration inputs X (n × in).
 * Uses the Cholesky form of H⁻¹ (upper factor) as the paper does, and per-row symmetric quantisers fitted to W.
 * Smaller layers only (the dense H⁻¹ is in × in).
 */
export function gptqQuantise(
  W: MatrixLike,
  X: MatrixLike,
  { bits = 4, damping = 0.01 }: { bits?: Size; damping?: Scalar } = {},
): GptqResult {
  const w = dense.toMatrixF64(W, 'gptqQuantise W')
  const x = dense.toMatrixF64(X, 'gptqQuantise X', undefined, w.n)
  const d = w.n
  const H = dense.scale(2 / x.m, dense.gram(x.data, x.m, d))
  const meanDiag = Array.from({ length: d }, (_, i) => H[i * d + i]).reduce((a, b) => a + b, 0) / d
  for (let i = 0; i < d; i++) H[i * d + i] += damping * meanDiag + 1e-12
  const Hinv = dense.data(inverse(fromData(H, [d, d])))
  // Upper Cholesky factor U of H⁻¹ (H⁻¹ = UᵀU): U = Lᵀ with L the lower factor.
  const L = dense.data(cholesky(fromData(dense.symmetrise(Hinv, d), [d, d])).L)
  const U = dense.transpose(L, d, d)
  const params = quantisationParams(fromData(w.data, [w.m, d]), { bits, scheme: 'symmetric', axis: 0 })
  const q1 = (v: number, row: number) => roundTrip(v, params, row)
  const Wq = Float64Array.from(w.data)
  const out = new Float64Array(w.m * d)
  for (let j = 0; j < d; j++) {
    const ujj = U[j * d + j]
    for (let r = 0; r < w.m; r++) {
      const val = Wq[r * d + j]
      const q = q1(val, r)
      out[r * d + j] = q
      const err = (val - q) / ujj
      for (let k = j + 1; k < d; k++) Wq[r * d + k] -= err * U[j * d + k]
    }
  }
  const rtn = Float64Array.from(w.data, (v, i) => q1(v, Math.floor(i / d)))
  return {
    weights: fromData(out, [w.m, d]),
    params,
    outputError: layerOutputError(w, x, out),
    roundToNearestError: layerOutputError(w, x, rtn),
  }
}

type Dense = { data: Float64Array; m: number; n: number }

/** One value through channel `c` of a quantiser and back: s(clamp(round(v/s) + z) − z). */
function roundTrip(v: number, params: QuantisationParams, c: number): number {
  const s = params.scale[c]
  const z = params.zeroPoint[c]
  return s * (Math.min(params.qmax, Math.max(params.qmin, roundHalfEven(v / s) + z)) - z)
}

/** ‖X Wᵀ − X Ŵᵀ‖²_F / n: the squared output error of a linear layer with weights Ŵ in place of W. */
function layerOutputError(w: Dense, x: Dense, Wq: Float64Array): number {
  const diff = Float64Array.from(w.data, (v, i) => v - Wq[i])
  const Y = dense.matMul(x.data, dense.transpose(diff, w.m, w.n), x.m, w.n, w.m)
  return Y.reduce((a, v) => a + v * v, 0) / x.m
}

/** Per-output-row symmetric round-to-nearest of W (out × in) and back, with the quantisers fitted to W. */
function roundToNearestRows(data: Float64Array, rows: number, cols: number, bits: Size) {
  const params = quantisationParams(fromData(data, [rows, cols]), { bits, scheme: 'symmetric', axis: 0 })
  return { params, weights: Float64Array.from(data, (v, i) => roundTrip(v, params, Math.floor(i / cols))) }
}

/** The result of `awqQuantise`. */
export interface AwqResult {
  /** The quantised-and-dequantised weights Q(W diag(s)) diag(s)⁻¹ (out × in). */
  weights: Matrix
  /** The per-input-channel scales s (length in), normalised so that max(s)·min(s) = 1. */
  scales: Tensor
  /** The chosen exponent α of s_j ∝ (mean_i |X_ij|)^α. */
  alpha: Scalar
  /** The searched grid of α and the output error at each. */
  grid: Tensor
  errors: Tensor
  /** ‖X Wᵀ − X Ŵᵀ‖²_F / n for AWQ, and for round-to-nearest (α = 0). */
  outputError: Scalar
  roundToNearestError: Scalar
  /** Per-output-row quantisers of the scaled weights. */
  params: QuantisationParams
}

/**
 * AWQ-lite (Lin et al., 2024): weights that meet large activations matter most, so scale input channel j of W
 * (out × in) up by s_j before rounding and divide the dequantised column by s_j after, Ŵ = Q(W diag(s)) diag(s)⁻¹.
 * The relative rounding error of a scaled-up column shrinks while the per-row step size barely moves. The scale is
 * s_j = max((mean_i |X_ij|)^α, 10⁻⁴) over calibration inputs X (n × in), divided by √(max s · min s) as in the
 * reference code, and α is chosen on the grid k/`gridSize`, k = 0 … `gridSize` (the reference stops before 1), to
 * minimise the layer's output error ‖X Wᵀ − X Ŵᵀ‖²_F / n; α = 0 is round-to-nearest. Per-output-row symmetric
 * quantisers (the reference uses groups of 128 input channels).
 */
export function awqQuantise(
  W: MatrixLike,
  X: MatrixLike,
  { bits = 4, gridSize = 20 }: { bits?: Size; gridSize?: Size } = {},
): AwqResult {
  const w = dense.toMatrixF64(W, 'awqQuantise W')
  const x = dense.toMatrixF64(X, 'awqQuantise X', undefined, w.n)
  if (!(Number.isInteger(gridSize) && gridSize >= 1))
    throw new DomainError('awqQuantise', 'awqQuantise: gridSize must be a positive integer')
  const d = w.n
  const mag = new Float64Array(d)
  for (let i = 0; i < x.m; i++) for (let j = 0; j < d; j++) mag[j] += Math.abs(x.data[i * d + j]) / x.m
  const grid = Float64Array.from({ length: gridSize + 1 }, (_, k) => k / gridSize)
  const errors = new Float64Array(grid.length)
  let best: {
    alpha: number
    s: Float64Array
    weights: Float64Array
    params: QuantisationParams
    error: number
  } | null = null
  for (let k = 0; k < grid.length; k++) {
    const alpha = grid[k]
    const raw = Float64Array.from(mag, (m) => Math.max(m ** alpha, 1e-4))
    const norm = Math.sqrt(Math.max(...raw) * Math.min(...raw))
    const sc = raw.map((v) => v / norm)
    const scaled = Float64Array.from(w.data, (v, i) => v * sc[i % d])
    const q = roundToNearestRows(scaled, w.m, d, bits)
    const back = q.weights.map((v, i) => v / sc[i % d])
    errors[k] = layerOutputError(w, x, back)
    if (!best || errors[k] < best.error) best = { alpha, s: sc, weights: back, params: q.params, error: errors[k] }
  }
  const b = best!
  return {
    weights: fromData(b.weights, [w.m, d]),
    scales: fromData(b.s, [d]),
    alpha: b.alpha,
    grid: fromData(grid, [grid.length]),
    errors: fromData(errors, [grid.length]),
    outputError: b.error,
    roundToNearestError: errors[0],
    params: b.params,
  }
}
