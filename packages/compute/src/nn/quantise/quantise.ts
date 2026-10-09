/**
 * Quantisation of tensors to $b$-bit integers, after torch.ao.quantization (Jacob et al., 2018, "Quantization and
 * training of neural networks for efficient integer-arithmetic-only inference", CVPR; Nagel et al., 2021, "A white
 * paper on neural network quantization", arXiv:2106.08295).
 *
 * - A quantiser maps $x$ to $q = \mathrm{clamp}(\mathrm{round}(x/s) + z, q_{\min}, q_{\max})$ and back to
 *   $\hat{x} = s(q - z)$: the scale $s$ and zero point $z$ per tensor, or per channel along an axis
 *   (`quantisationParams`; affine uses the range $[\min x, \max x] \cup \{0\}$, symmetric
 *   $s = \max\lvert x \rvert / ((q_{\max} - q_{\min})/2)$ with $z$ at the middle of the range).
 * - Rounding to nearest (halves to even, as torch) or stochastic rounding $\lfloor x + u \rfloor$,
 *   $u \sim \mathcal{U}[0, 1)$, which is unbiased.
 * - `fakeQuantise`: quantise and dequantise in the forward pass, and pass the gradient straight through where $x$ lies
 *   inside the representable range (zero where it was clipped): the straight-through estimator (Bengio, Léonard &
 *   Courville, 2013), as `torch.fake_quantize_per_tensor_affine`, built with `customVjp`.
 * - `quantisedMatmul`: an int8 matmul simulated exactly: integer products accumulated in int32 range, then rescaled.
 * - `gptqQuantise`: GPTQ's column-by-column rounding with second-order error feedback from the layer's inputs (Frantar
 *   et al., 2023, "GPTQ: accurate post-training quantization for generative pre-trained transformers", ICLR).
 * - `awqQuantise`: activation-aware scaling of the input channels before rounding, the scale's exponent searched on a
 *   grid (Lin et al., 2024, "AWQ: activation-aware weight quantization for LLM compression and acceleration", MLSys).
 *
 * Integers are stored as float64 values. Only `fakeQuantise` is differentiable; the others work on concrete tensors.
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

/**
 * Round half to even (banker's rounding), as `torch.round` and C's `nearbyint`: a value halfway between two integers
 * goes to the even one, so halves are not biased upward.
 *
 * @param v The number to round.
 * @returns The nearest integer, the even one on a tie.
 *
 * @example Ties go to the even neighbour
 * print([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4].map(roundHalfEven))
 */
export function roundHalfEven(v: Scalar): Scalar {
  const r = Math.round(v)
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/**
 * The integer range of a $b$-bit quantiser: signed $[-2^{b-1}, 2^{b-1} - 1]$ or unsigned $[0, 2^b - 1]$. Throws
 * `DomainError` unless $b$ is an integer from 1 to 16.
 *
 * @param bits The number of bits $b$.
 * @param signed Whether the integers are signed (two's complement) or unsigned.
 * @returns The smallest and largest integers, `qmin` and `qmax`.
 *
 * @example Signed and unsigned 8 bits, and signed 4 bits
 * print('int8:', integerRange(8, true))
 * print('uint8:', integerRange(8, false))
 * print('int4:', integerRange(4, true))
 */
export function integerRange(bits: Size, signed: boolean): { qmin: number; qmax: number } {
  if (!(Number.isInteger(bits) && bits >= 1 && bits <= 16))
    throw new DomainError('integerRange', 'integerRange: bits must be an integer in 1 … 16')
  return signed ? { qmin: -(2 ** (bits - 1)), qmax: 2 ** (bits - 1) - 1 } : { qmin: 0, qmax: 2 ** bits - 1 }
}

/** The parameters of a quantiser: scale and zero point per tensor (length 1) or per channel along `axis`. */
export interface QuantisationParams {
  /** The scale $s$, the real value of one integer step: one per channel (one value per tensor). */
  scale: Float64Array
  /** The zero point $z$, the integer that represents 0: one per channel (one value per tensor). */
  zeroPoint: Float64Array
  /** The smallest integer $q_{\min}$. */
  qmin: number
  /** The largest integer $q_{\max}$. */
  qmax: number
  /** The number of bits $b$. */
  bits: Size
  /** The channel axis, or null for per-tensor. */
  axis: number | null
  /** How the range was fitted: `affine` (min–max) or `symmetric` (about 0). */
  scheme: 'affine' | 'symmetric'
}

/** Options for `quantisationParams`. */
export interface QuantisationOptions {
  /** Bits per value $b$, 1 to 16. Default 8. */
  bits?: Size
  /** `affine` (asymmetric, unsigned range by default) or `symmetric` (signed range by default). Default `affine`. */
  scheme?: 'affine' | 'symmetric'
  /** Use a signed integer range. Default: true for symmetric, false for affine. */
  signed?: boolean
  /** Per-channel along this axis (e.g. 0 for a weight matrix's output rows); omitted: per tensor. */
  axis?: number
}

/**
 * The channel index of each flat element of a tensor of `shape` along `axis`.
 *
 * @param shape The tensor's shape (row-major).
 * @param axis The channel axis.
 * @returns A function from the flat index of an element to its index along `axis`.
 */
function channelOf(shape: readonly number[], axis: number): (i: number) => number {
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const n = shape[axis]
  return (i) => Math.floor(i / inner) % n
}

/**
 * The scale and zero point of a quantiser fitted to the range of $x$ (min–max calibration, as torch's
 * MinMaxObserver). Affine: the range is widened to include 0, $s = (\max - \min)/(q_{\max} - q_{\min})$ and
 * $z = q_{\min} - \mathrm{round}(\min/s)$, clamped to the integer range, so 0 is exactly representable. Symmetric:
 * $s = \max(\lvert \min \rvert, \lvert \max \rvert)/((q_{\max} - q_{\min})/2)$ and $z$ the middle of the range
 * (0 for a signed range, $2^{b-1}$ for an unsigned one). A zero range gives $s = 1$. As in torch, the range's end maps
 * to $q_{\max} + 1/2$, so the largest value may be clipped by half a step. Throws `ShapeError` for an axis out of
 * range and `DomainError` for an invalid bit width.
 *
 * @param x The concrete tensor to calibrate on (not traced).
 * @param options The bit width, scheme, signedness and channel axis.
 * @returns The quantiser's parameters.
 *
 * @example A symmetric int8 quantiser of a short vector, and its round trip
 * const x = tensor([-1, -0.5, 0, 0.3, 1])
 * const p = quantisationParams(x, { scheme: 'symmetric' })
 * print('scale =', p.scale[0], 'zero point =', p.zeroPoint[0])
 * const q = quantise(x, p)
 * print('q =', q)
 * print('round trip error =', sub(dequantise(q, p), x))
 *
 * @example Per channel keeps a row of small weights that per tensor rounds to zero
 * const W = tensor([[1, -2], [0.01, 0.02]])
 * const perTensor = quantisationParams(W, { scheme: 'symmetric', bits: 4 })
 * const perRow = quantisationParams(W, { scheme: 'symmetric', bits: 4, axis: 0 })
 * print('per tensor:', dequantise(quantise(W, perTensor), perTensor))
 * print('per row:', dequantise(quantise(W, perRow), perRow))
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

/**
 * Per-element scale and zero point of `params` broadcast over a tensor of `shape`.
 *
 * @param params The quantiser, per tensor or per channel.
 * @param shape The tensor's shape, which gives each element's channel.
 * @param n The number of elements of the tensor.
 * @returns The scale `s` and zero point `z` of each of the $n$ elements, in row-major order.
 */
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
  /**
   * `nearest` (halves to even) or `stochastic` ($\lfloor x/s + u \rfloor$, $u \sim \mathcal{U}[0, 1)$, needing
   * `stream`). Default `nearest`.
   */
  rounding?: 'nearest' | 'stochastic'
  /** The random stream for stochastic rounding (its child `'round'` is drawn from); unused by `nearest`. */
  stream?: Stream
}

/**
 * The integers $q = \mathrm{clamp}(\mathrm{round}(x/s) + z, q_{\min}, q_{\max})$, stored as float64 values. Throws
 * `DomainError` for stochastic rounding without a stream.
 *
 * @param x The concrete tensor to quantise.
 * @param params The quantiser, from `quantisationParams`; per-channel parameters follow the tensor's axis.
 * @param options Nearest or stochastic rounding.
 * @returns The integers $q$, with the shape of `x`.
 *
 * @example Stochastic rounding is unbiased on average
 * // A 2-bit quantiser of [0, 1] has step 1/3, so 0.4 lies between the levels 1/3 and 2/3.
 * const p = quantisationParams(tensor([0, 1]), { bits: 2 })
 * const x = tensor(Array(1000).fill(0.4))
 * print('nearest:', mean(dequantise(quantise(x, p), p)))
 * print('stochastic:', mean(dequantise(quantise(x, p, { rounding: 'stochastic', stream: stream(0) }), p)))
 */
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

/**
 * The real values $\hat{x} = s(q - z)$ of integers $q$.
 *
 * @param q The integers, as `quantise` returns them.
 * @param params The quantiser that produced them.
 * @returns The dequantised values $\hat{x}$, with the shape of `q`.
 *
 * @example An affine uint8 round trip
 * const x = tensor([-1, -0.5, 0, 0.3, 1])
 * const p = quantisationParams(x)
 * const q = quantise(x, p)
 * print('q =', q)
 * print('x hat =', dequantise(q, p))
 */
export function dequantise(q: Tensor, params: QuantisationParams): Tensor {
  const v = toFlat(q)
  const { s, z } = expand(params, q.shape, v.length)
  return fromData(
    Float64Array.from(v, (val, i) => s[i] * (val - z[i])),
    q.shape,
  )
}

/**
 * Fake quantisation $x \mapsto s(\mathrm{clamp}(\mathrm{round}(x/s) + z, q_{\min}, q_{\max}) - z)$ with the
 * straight-through estimator (Bengio, Léonard & Courville, 2013; Jacob et al., 2018): the backward pass multiplies the
 * cotangent by 1 where $q_{\min} \le \mathrm{round}(x/s) + z \le q_{\max}$ and by 0 where the value was clipped, as
 * `torch.fake_quantize_per_tensor_affine`. Written with `customVjp`, so it can sit inside a model trained by `grad`;
 * quantisation-aware training uses it on the weights and activations. Per tensor only: per-channel parameters with
 * more than one channel throw `DomainError`.
 *
 * @param x The values to quantise, traced or not, any shape.
 * @param params A per-tensor quantiser.
 * @returns The dequantised values, with the shape of `x`.
 *
 * @example The gradient passes through inside the range and stops where clipped
 * const p = quantisationParams(tensor([-1, 1]), { scheme: 'symmetric', bits: 4 })
 * const x = tensor([0.1, 0.5, 2])
 * print('fake quantised:', fakeQuantise(x, p))
 * print('gradient:', grad((v) => sum(fakeQuantise(v, p)))(x))
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

/**
 * The error of a quantiser on $x$: the mean squared error of the round trip, the signal-to-quantisation-noise ratio
 * $10 \log_{10}(\sum x^2 / \sum (x - \hat{x})^2)$ in dB, the share of values clipped, and the largest absolute error.
 *
 * @param x The concrete tensor to quantise.
 * @param params The quantiser.
 * @param options The rounding of the round trip; the clipped share is always counted with rounding to nearest.
 * @returns `mse`, `sqnr` (dB), `clipped` (a fraction from 0 to 1) and `maxError`.
 *
 * @example Int8 against int4 on the same vector
 * const x = tensor([-1, -0.5, 0, 0.3, 1])
 * print('int8:', quantisationError(x, quantisationParams(x, { scheme: 'symmetric' })))
 * print('int4:', quantisationError(x, quantisationParams(x, { scheme: 'symmetric', bits: 4 })))
 */
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
  /** The rescaled output $\Ymat \approx \Amat\Bmat$ ($m \times n$). */
  y: Matrix
  /** The integers $q_A$ of $\Amat$ ($m \times k$). */
  qa: Matrix
  /** The integers $q_B$ of $\Bmat$ ($k \times n$). */
  qb: Matrix
  /** The integer accumulator $\sum_k (q_A - z_A)(q_B - z_B)$ ($m \times n$; $z_B = 0$). */
  accumulator: Matrix
  /** The quantiser of $\Amat$ (per tensor, affine). */
  paramsA: QuantisationParams
  /** The quantiser of $\Bmat$ (per output column, symmetric). */
  paramsB: QuantisationParams
  /** The largest $\lvert \text{accumulator} \rvert$: it must stay below $2^{31}$ for int32 accumulation. */
  maxAccumulator: Scalar
}

/**
 * An integer matmul simulated exactly (Jacob et al., 2018, §2.2): $\Amat$ ($m \times k$ activations) is quantised per
 * tensor with an affine $b$-bit quantiser and $\Bmat$ ($k \times n$ weights) per output column with a symmetric one;
 * the products of the zero-point-shifted integers are summed in integer arithmetic (exact in float64 at these sizes),
 * and the output is rescaled, $Y_{ij} = s_A s_{B,j} \sum_k (q_{A,ik} - z_A) q_{B,kj}$.
 *
 * @param A The activations $\Amat$, $m \times k$.
 * @param B The weights $\Bmat$, $k \times n$.
 * @param options The bit width.
 * @param options.bits The bit width $b$ of both quantisers.
 * @returns The output with the integer operands, the accumulator and the quantisers.
 *
 * @example An int8 matmul of two 2 by 2 matrices against the exact product
 * const A = tensor([[0.5, -1], [2, 0.25]])
 * const B = tensor([[1, 0.5], [-0.5, 2]])
 * const r = quantisedMatmul(A, B)
 * print('int8:', r.y)
 * print('exact:', matmul(A, B))
 * print('accumulator:', r.accumulator)
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
  /** The quantised-and-dequantised weights $\hat{\Wmat}$ (out $\times$ in). */
  weights: Matrix
  /** Per-output-row quantisers (symmetric), fitted to the original $\Wmat$. */
  params: QuantisationParams
  /** $\lVert \Xmat\Wmat^\top - \Xmat\hat{\Wmat}^\top \rVert_F^2 / n$ for GPTQ. */
  outputError: Scalar
  /** The same error for round-to-nearest with the same quantisers. */
  roundToNearestError: Scalar
}

/**
 * GPTQ-lite (Frantar et al., 2023): quantise the weights $\Wmat$ (out $\times$ in) of a linear layer
 * $\yvec = \Wmat\xvec$ column by column, feeding each column's rounding error forward into the columns not yet
 * quantised. $\Hmat = 2\Xmat^\top\Xmat/n + \lambda\Imat$ is the Hessian of the layer's squared output error over
 * calibration inputs $\Xmat$ ($n \times$ in), and $\Umat$ the upper Cholesky factor of $\Hmat^{-1}$
 * ($\Hmat^{-1} = \Umat^\top\Umat$), as the paper uses: after rounding column $j$, its error divided by $U_{jj}$ is
 * subtracted from the later columns in proportion to row $j$ of $\Umat$. Per-row symmetric quantisers fitted to
 * $\Wmat$. Smaller layers only (the dense $\Hmat^{-1}$ is in $\times$ in).
 *
 * @param W The weights $\Wmat$, out $\times$ in (one row per output).
 * @param X The calibration inputs $\Xmat$, $n \times$ in (one row per sample).
 * @param options The bit width and the dampening.
 * @param options.bits The bit width $b$ of the per-row quantisers.
 * @param options.damping The dampening $\lambda$ as a fraction of the mean diagonal of $2\Xmat^\top\Xmat/n$ (the
 *   reference code's `percdamp`); it keeps $\Hmat$ invertible.
 * @returns The quantised weights, the quantisers, and the output error of GPTQ beside that of round-to-nearest.
 *
 * @example 2-bit GPTQ beats round-to-nearest on the layer's output
 * const W = normals(stream(0), [4, 6])
 * const X = normals(stream(1), [16, 6])
 * const r = gptqQuantise(W, X, { bits: 2 })
 * print('GPTQ error:', r.outputError)
 * print('round-to-nearest error:', r.roundToNearestError)
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

/**
 * One value through channel `c` of a quantiser and back:
 * $s(\mathrm{clamp}(\mathrm{round}(v/s) + z, q_{\min}, q_{\max}) - z)$.
 *
 * @param v The value.
 * @param params The quantiser.
 * @param c The channel whose scale and zero point are used (0 for a per-tensor quantiser).
 * @returns The dequantised value.
 */
function roundTrip(v: number, params: QuantisationParams, c: number): number {
  const s = params.scale[c]
  const z = params.zeroPoint[c]
  return s * (Math.min(params.qmax, Math.max(params.qmin, roundHalfEven(v / s) + z)) - z)
}

/**
 * $\lVert \Xmat\Wmat^\top - \Xmat\hat{\Wmat}^\top \rVert_F^2 / n$: the squared output error of a linear layer
 * with weights $\hat{\Wmat}$ in place of $\Wmat$.
 *
 * @param w The original weights $\Wmat$, row-major out $\times$ in.
 * @param x The calibration inputs $\Xmat$, row-major $n \times$ in.
 * @param Wq The replacement weights $\hat{\Wmat}$, row-major, the shape of `w`.
 * @returns The mean over the $n$ inputs of the squared output error.
 */
function layerOutputError(w: Dense, x: Dense, Wq: Float64Array): number {
  const diff = Float64Array.from(w.data, (v, i) => v - Wq[i])
  const Y = dense.matMul(x.data, dense.transpose(diff, w.m, w.n), x.m, w.n, w.m)
  return Y.reduce((a, v) => a + v * v, 0) / x.m
}

/**
 * Per-output-row symmetric round-to-nearest of $\Wmat$ (out $\times$ in) and back, with the quantisers fitted to
 * $\Wmat$.
 *
 * @param data The weights, row-major.
 * @param rows The number of rows (outputs).
 * @param cols The number of columns (inputs).
 * @param bits The bit width.
 * @returns The per-row quantisers and the dequantised weights, row-major.
 */
function roundToNearestRows(data: Float64Array, rows: number, cols: number, bits: Size) {
  const params = quantisationParams(fromData(data, [rows, cols]), { bits, scheme: 'symmetric', axis: 0 })
  return { params, weights: Float64Array.from(data, (v, i) => roundTrip(v, params, Math.floor(i / cols))) }
}

/** The result of `awqQuantise`. */
export interface AwqResult {
  /**
   * The quantised-and-dequantised weights $Q(\Wmat \diag(\svec)) \diag(\svec)^{-1}$ (out $\times$ in).
   */
  weights: Matrix
  /** The per-input-channel scales $\svec$ (length in), normalised so that $\max(\svec)\min(\svec) = 1$. */
  scales: Tensor
  /** The chosen exponent $\alpha$ of $s_j \propto (\mathrm{mean}_i \lvert X_{ij} \rvert)^\alpha$. */
  alpha: Scalar
  /** The searched grid of $\alpha$. */
  grid: Tensor
  /** The output error at each $\alpha$ of the grid. */
  errors: Tensor
  /** $\lVert \Xmat\Wmat^\top - \Xmat\hat{\Wmat}^\top \rVert_F^2 / n$ for AWQ. */
  outputError: Scalar
  /** The same error for round-to-nearest ($\alpha = 0$). */
  roundToNearestError: Scalar
  /** Per-output-row quantisers of the scaled weights. */
  params: QuantisationParams
}

/**
 * AWQ-lite (Lin et al., 2024): weights that meet large activations matter most, so scale input channel $j$ of
 * $\Wmat$ (out $\times$ in) up by $s_j$ before rounding and divide the dequantised column by $s_j$ after,
 * $\hat{\Wmat} = Q(\Wmat \diag(\svec)) \diag(\svec)^{-1}$. The relative rounding error of a scaled-up column shrinks
 * while the per-row step size barely moves. The scale is
 * $s_j = \max((\mathrm{mean}_i \lvert X_{ij} \rvert)^\alpha, 10^{-4})$ over calibration inputs $\Xmat$
 * ($n \times$ in), divided by $\sqrt{\max \svec \cdot \min \svec}$ as in the reference code, and $\alpha$ is chosen
 * on the grid $k/\text{gridSize}$, $k = 0, \dots, \text{gridSize}$ (the reference stops before 1), to minimise the
 * layer's output error $\lVert \Xmat\Wmat^\top - \Xmat\hat{\Wmat}^\top \rVert_F^2 / n$; $\alpha = 0$ is
 * round-to-nearest.
 * Per-output-row symmetric quantisers (the reference uses groups of 128 input channels). Throws `DomainError` unless
 * `gridSize` is a positive integer.
 *
 * @param W The weights $\Wmat$, out $\times$ in (one row per output).
 * @param X The calibration inputs $\Xmat$, $n \times$ in (one row per sample).
 * @param options The bit width and the grid.
 * @param options.bits The bit width $b$ of the per-row quantisers.
 * @param options.gridSize The number of steps of the grid of $\alpha$ on $[0, 1]$, which has $\text{gridSize} + 1$
 *   points.
 * @returns The quantised weights, the scales and exponent chosen, the error over the grid, and the output error beside
 *   that of round-to-nearest.
 *
 * @example A salient input channel with small weights
 * // Channel 0 carries activations twenty times larger than the others, through weights near zero.
 * const W = tensor([[0.05, 1, -0.8], [-0.04, 0.6, 1]])
 * const X = mul(normals(stream(1), [16, 3]), tensor([20, 1, 1]))
 * const r = awqQuantise(W, X, { bits: 3, gridSize: 4 })
 * print('errors over alpha = 0, 0.25, ..., 1:', r.errors)
 * print('alpha =', r.alpha, 'scales =', r.scales)
 * print('AWQ error:', r.outputError, 'round-to-nearest error:', r.roundToNearestError)
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
