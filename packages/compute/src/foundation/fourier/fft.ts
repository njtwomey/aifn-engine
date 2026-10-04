/**
 * The discrete Fourier transform X[k] = Σₙ x[n] e^{−2πi kn/N} as primitives on complex128 tensors, with numpy.fft's
 * conventions: `fft`, `ifft`, `rfft`, `irfft` along any axis, with `{ axis = −1, n, norm }`. Every line along the axis
 * is transformed independently, so leading and trailing axes are batches. Power-of-two lengths use the iterative
 * radix-2 Cooley–Tukey FFT (Cooley and Tukey, 1965, Math. Comp. 19); other lengths use Bluestein's chirp-z algorithm
 * (Bluestein, 1970, IEEE Trans. Audio Electroacoust. 18(4)), which rewrites a length-N DFT as a circular convolution of
 * power-of-two length ≥ 2N − 1. Both are O(N log N).
 *
 * **Derivatives (design K §8.2).** The four maps are ℝ-linear, so each is a linear primitive whose jvp is itself and
 * whose vjp is its declared `transpose`, the ℝ² adjoint (the conjugate transpose). With F the unnormalised DFT matrix
 * (Fⱼₖ = e^{−2πijk/n}, symmetric) and `norm` scaling the forward map by 1, 1/√n or 1/n ('backward', 'ortho',
 * 'forward') and the inverse by 1/n, 1/√n or 1:
 *
 * - fft(norm)ᴴ = s·Fᴴ = ifft(dual(norm)), where dual swaps 'backward' and 'forward' and keeps 'ortho';
 * - ifft(norm)ᴴ = fft(dual(norm));
 * - rfft (real n → complex ⌊n/2⌋ + 1): the adjoint zero-pads the cotangent to n, applies ifft(dual) and keeps the real
 *   part (the input is real);
 * - irfft (⌊n/2⌋ + 1 bins → real n): the adjoint is c ⊙ rfft(dual)(ȳ) with cₖ = 1 at k = 0 and at k = n/2 (n even), else
 *   2, because each interior bin appears twice in the Hermitian extension. Using irfft's own inverse as the adjoint
 *   double-counts the interior bins.
 */

import { DTypeError, DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  avalOf,
  batchToFront,
  concat,
  copy,
  definePrimitive,
  fromData,
  isTensor,
  isTraced,
  matmul,
  mul,
  permute,
  realPart,
  shapeOfValue,
  slice,
  zeros,
  type Aval,
  type Op,
  type SliceSpec,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'

const TAU = 2 * Math.PI

/** True when n is a positive power of two. */
export const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0

/** The smallest power of two ≥ n (1 for n ≤ 1). */
export const nextPowerOfTwo = (n: number): number => 2 ** Math.ceil(Math.log2(Math.max(1, n)))

// ── Kernels on raw arrays ────────────────────────────────────────────────────────────────────────────────────────────

/** In-place radix-2 FFT; `inverse` flips the sign of the exponent (no 1/N). Length must be a power of two. */
function radix2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]
      re[i] = re[j]
      re[j] = t
      t = im[i]
      im[i] = im[j]
      im[j] = t
    }
  }
  const sign = inverse ? 1 : -1
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1
    // Twiddles computed directly per k (not by repeated multiplication) to keep rounding error O(log n).
    const step = (sign * TAU) / len
    for (let k = 0; k < half; k++) {
      const wr = Math.cos(step * k)
      const wi = Math.sin(step * k)
      for (let start = 0; start < n; start += len) {
        const a = start + k
        const b = a + half
        const tr = re[b] * wr - im[b] * wi
        const ti = re[b] * wi + im[b] * wr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
      }
    }
  }
}

/** In-place Bluestein transform of any length (no 1/N on the inverse). */
function bluestein(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  const m = nextPowerOfTwo(2 * n - 1)
  const sign = inverse ? 1 : -1
  // Chirp w[k] = exp(sign · iπ k²/n); k² is reduced mod 2n so the angle stays small and accurate for large k.
  const cr = new Float64Array(n)
  const ci = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const angle = (sign * Math.PI * ((k * k) % (2 * n))) / n
    cr[k] = Math.cos(angle)
    ci[k] = Math.sin(angle)
  }
  const ar = new Float64Array(m)
  const ai = new Float64Array(m)
  for (let k = 0; k < n; k++) {
    ar[k] = re[k] * cr[k] - im[k] * ci[k]
    ai[k] = re[k] * ci[k] + im[k] * cr[k]
  }
  // b[k] = conj(w[k]) for |k| < n, wrapped circularly.
  const br = new Float64Array(m)
  const bi = new Float64Array(m)
  br[0] = cr[0]
  bi[0] = -ci[0]
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = cr[k]
    bi[k] = bi[m - k] = -ci[k]
  }
  radix2(ar, ai, false)
  radix2(br, bi, false)
  for (let k = 0; k < m; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k]
    const i = ar[k] * bi[k] + ai[k] * br[k]
    ar[k] = r
    ai[k] = i
  }
  radix2(ar, ai, true)
  for (let k = 0; k < n; k++) {
    const r = ar[k] / m
    const i = ai[k] / m
    re[k] = r * cr[k] - i * ci[k]
    im[k] = r * ci[k] + i * cr[k]
  }
}

/**
 * The DFT of any length, in place on separate real and imaginary arrays: radix-2 Cooley–Tukey for powers of two,
 * Bluestein's chirp z-transform otherwise. `inverse` gives the unscaled inverse (divide by n). The raw-array kernel
 * under the primitives; spectral estimators with their own buffers call it directly.
 */
function transformInPlace(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (n <= 1) return
  if (isPowerOfTwo(n)) radix2(re, im, inverse)
  else bluestein(re, im, inverse)
}

// ── Normalisation ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Which direction carries the 1/n: numpy's `norm`. 'ortho' splits it as 1/√n on both. */
export type FftNorm = 'backward' | 'ortho' | 'forward'

/** Options of the one-axis transforms. */
export type FftOptions = {
  /** The axis transformed (default −1, the last). */
  axis?: number
  /** Length of the transform: the input is zero-padded or truncated along the axis first (irfft: output length). */
  n?: number
  /** Normalisation (default 'backward': the forward transform unscaled, the inverse divided by n). */
  norm?: FftNorm
}

/** The scale of the forward transform (`inverse` false) or of the inverse one under `norm`, for length n. */
function scaleOf(norm: FftNorm, n: number, inverse: boolean): number {
  if (norm === 'ortho') return 1 / Math.sqrt(n)
  return (norm === 'backward') === inverse ? 1 / n : 1
}

/** The normalisation whose forward scale is the inverse scale of `norm` (and vice versa): the adjoint's. */
const dual = (norm: FftNorm): FftNorm => (norm === 'backward' ? 'forward' : norm === 'forward' ? 'backward' : 'ortho')

// ── Lines along an axis ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Apply `line` to every line of `x` along axis `axis` (negative): each line is read as complex (real inputs promoted)
 * into `re`/`im` of length `nIn`, and `line` leaves the output line in `outRe`/`outIm` of length `nOut`. The result is
 * complex128, or float64 (real parts) when `real`.
 */
function mapLines(
  x: Tensor,
  axis: number,
  nOut: number,
  real: boolean,
  line: (re: Float64Array, im: Float64Array, outRe: Float64Array, outIm: Float64Array) => void,
): Tensor {
  const shape = x.shape
  const a = shape.length + axis
  const nIn = shape[a]
  const outer = shape.slice(0, a).reduce((p, d) => p * d, 1)
  const inner = shape.slice(a + 1).reduce((p, d) => p * d, 1)
  const src = copy(x, 'complex128').data as Float64Array
  const outShape = shape.map((d, k) => (k === a ? nOut : d))
  const width = real ? 1 : 2
  const out = new Float64Array(outer * nOut * inner * width)
  const re = new Float64Array(nIn)
  const im = new Float64Array(nIn)
  const outRe = new Float64Array(nOut)
  const outIm = new Float64Array(nOut)
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      for (let j = 0; j < nIn; j++) {
        const p = 2 * ((o * nIn + j) * inner + i)
        re[j] = src[p]
        im[j] = src[p + 1]
      }
      line(re, im, outRe, outIm)
      for (let j = 0; j < nOut; j++) {
        const q = (o * nOut + j) * inner + i
        if (real) out[q] = outRe[j]
        else {
          out[2 * q] = outRe[j]
          out[2 * q + 1] = outIm[j]
        }
      }
    }
  return fromData(out, outShape, real ? 'float64' : 'complex128')
}

/** The tensor input of a transform, checked: rank ≥ 1 and `axis` (negative) in range. */
function lineInput(x: number | Tensor, axis: number, what: string): Tensor {
  if (typeof x === 'number' || x.shape.length === 0)
    throw new ShapeError(
      what,
      `${what}: needs a tensor of rank ≥ 1, got a ${typeof x === 'number' ? 'number' : 'scalar'}`,
    )
  if (-axis > x.shape.length)
    throw new ShapeError(what, `${what}: axis ${axis} is out of range for rank ${x.shape.length}`)
  return x
}

/** The abstract value of a transform's result: `aval` with the axis resized to `n`. */
function resized(aval: Aval, axis: number, n: number, real: boolean): Aval {
  const shape = [...aval.shape]
  shape[shape.length + axis] = n
  return { shape, dtype: real ? 'float64' : 'complex128', number: false }
}

// ── The primitives ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Parameters of every transform primitive: the axis (negative, so a batch axis in front leaves it unchanged). */
type LineParams = { axis: number; norm: FftNorm }
/** rfft and irfft also carry the real length n. */
type RealParams = LineParams & { n: number }

/** Registry test cases of fft and ifft: three norms, a trailing and a leading axis, complex inputs too. */
const complexCases = {
  complex: true,
  secondOrder: true,
  cases: (draw: (shape: readonly number[]) => Tensor) => [
    { inputs: [draw([8])], params: { axis: -1, norm: 'backward' } },
    { inputs: [draw([2, 5])], params: { axis: -1, norm: 'ortho' } },
    { inputs: [draw([3, 2])], params: { axis: -2, norm: 'forward' } },
  ],
}

function fullTransform(inverse: boolean): (x: Tensor, p: LineParams) => Tensor {
  return (x, { axis, norm }) => {
    const n = x.shape[x.shape.length + axis]
    const s = scaleOf(norm, n, inverse)
    return mapLines(x, axis, n, false, (re, im, outRe, outIm) => {
      transformInPlace(re, im, inverse)
      for (let k = 0; k < n; k++) {
        outRe[k] = re[k] * s
        outIm[k] = im[k] * s
      }
    })
  }
}

const fftKernel = fullTransform(false)
const ifftKernel = fullTransform(true)

const fftOp: Op<LineParams> = definePrimitive<LineParams>({
  id: 'foundation/fourier/fft',
  arity: 1,
  dtype: 'complex',
  impl: ([x], p) => fftKernel(lineInput(x, p.axis, 'fft'), p),
  linear: 'linear',
  transpose: (ct, _xs, _which, p) => ifftOp([ct], { ...p, norm: dual(p.norm) }),
  shape: ([x], { axis }) => resized(x, axis, x.shape[x.shape.length + axis], false),
  batch: ([x], [b], p) => [fftOp([batchToFront(x, b ?? 0)], p), 0],
  doc: {
    note: 'fast-fourier-transform',
    summary: 'The discrete Fourier transform along an axis (radix-2 or Bluestein).',
    formula: 'X_k = s \\sum_{t=0}^{n-1} x_t e^{-2\\pi i kt/n}',
  },
  test: complexCases,
})

const ifftOp: Op<LineParams> = definePrimitive<LineParams>({
  id: 'foundation/fourier/ifft',
  arity: 1,
  dtype: 'complex',
  impl: ([x], p) => ifftKernel(lineInput(x, p.axis, 'ifft'), p),
  linear: 'linear',
  transpose: (ct, _xs, _which, p) => fftOp([ct], { ...p, norm: dual(p.norm) }),
  shape: ([x], { axis }) => resized(x, axis, x.shape[x.shape.length + axis], false),
  batch: ([x], [b], p) => [ifftOp([batchToFront(x, b ?? 0)], p), 0],
  doc: {
    note: 'discrete-fourier-transform',
    summary: 'The inverse discrete Fourier transform along an axis.',
    formula: 'x_t = s \\sum_{k=0}^{n-1} X_k e^{2\\pi i kt/n}',
  },
  test: complexCases,
})

/** The weights cₖ of irfft's adjoint on m = ⌊n/2⌋ + 1 bins, shaped to broadcast along `axis`. */
function hermitianWeights(n: number, axis: number): Tensor {
  const m = Math.floor(n / 2) + 1
  const c = Float64Array.from({ length: m }, (_, k) => (k === 0 || (n % 2 === 0 && k === n / 2) ? 1 : 2))
  return fromData(c, [m, ...new Array<number>(-axis - 1).fill(1)])
}

const rfftOp: Op<RealParams> = definePrimitive<RealParams>({
  id: 'foundation/fourier/rfft',
  arity: 1,
  dtype: 'complex',
  impl: ([x], { axis, norm, n }) => {
    const t = lineInput(x, axis, 'rfft')
    if (t.dtype === 'complex128') throw new DTypeError('rfft', 'rfft: needs a real input (use fft for complex)')
    const m = Math.floor(n / 2) + 1
    const s = scaleOf(norm, n, false)
    return mapLines(t, axis, m, false, (re, im, outRe, outIm) => {
      transformInPlace(re, im, false)
      for (let k = 0; k < m; k++) {
        outRe[k] = re[k] * s
        outIm[k] = im[k] * s
      }
    })
  },
  linear: 'linear',
  // x̄ = Re(s Fᴴ pad(ȳ)): zero-pad the m bins to n, invert with the dual scale, keep the real part.
  transpose: (ct, _xs, _which, { axis, norm, n }) => {
    const m = Math.floor(n / 2) + 1
    return realPart(ifftOp([padAxis(ct, axis, n - m)], { axis, norm: dual(norm) }))
  },
  shape: ([x], { axis, n }) => {
    if (x.dtype === 'complex128') throw new DTypeError('rfft', 'rfft: needs a real input (use fft for complex)')
    return resized(x, axis, Math.floor(n / 2) + 1, false)
  },
  batch: ([x], [b], p) => [rfftOp([batchToFront(x, b ?? 0)], p), 0],
  doc: {
    note: 'fast-fourier-transform',
    summary: 'The DFT of a real signal at its non-negative frequencies k = 0, …, ⌊n/2⌋.',
    formula: 'X_k = s \\sum_{t=0}^{n-1} x_t e^{-2\\pi i kt/n}, \\; 0 \\le k \\le \\lfloor n/2 \\rfloor',
  },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([8])], params: { axis: -1, norm: 'backward', n: 8 } },
      { inputs: [draw([2, 5])], params: { axis: -1, norm: 'ortho', n: 5 } },
      { inputs: [draw([6, 2])], params: { axis: -2, norm: 'forward', n: 6 } },
    ],
  },
})

const irfftOp: Op<RealParams> = definePrimitive<RealParams>({
  id: 'foundation/fourier/irfft',
  arity: 1,
  dtype: 'real',
  impl: ([x], { axis, norm, n }) => {
    const t = lineInput(x, axis, 'irfft')
    const m = Math.floor(n / 2) + 1
    if (t.shape[t.shape.length + axis] !== m)
      throw new ShapeError('irfft', `irfft: expected ${m} bins along the axis for n = ${n}`)
    const s = scaleOf(norm, n, true)
    const full = new Float64Array(n)
    const fullIm = new Float64Array(n)
    return mapLines(t, axis, n, true, (re, im, outRe) => {
      // Hermitian extension: the imaginary parts of DC and (n even) Nyquist are dropped; bin n − k = conj(bin k).
      for (let k = 0; k < n; k++) {
        const mirrored = k >= m
        const j = mirrored ? n - k : k
        full[k] = re[j]
        fullIm[k] = j === 0 || (n % 2 === 0 && j === n / 2) ? 0 : mirrored ? -im[j] : im[j]
      }
      transformInPlace(full, fullIm, true)
      for (let k = 0; k < n; k++) outRe[k] = full[k] * s
    })
  },
  linear: 'linear',
  // X̄ = c ⊙ rfft(dual)(ȳ): the interior bins appear twice in the extension (design K §8.2).
  transpose: (ct, _xs, _which, { axis, norm, n }) =>
    mul(hermitianWeights(n, axis), rfftOp([ct], { axis, norm: dual(norm), n })),
  shape: ([x], { axis, n }) => resized(x, axis, n, true),
  batch: ([x], [b], p) => [irfftOp([batchToFront(x, b ?? 0)], p), 0],
  doc: {
    note: 'discrete-fourier-transform',
    summary: 'The real signal of length n whose non-negative half-spectrum is the input (Hermitian extension).',
    formula: 'x_t = s \\sum_{k=0}^{n-1} X_k e^{2\\pi i kt/n}, \\; X_{n-k} = \\overline{X_k}',
  },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([5])], params: { axis: -1, norm: 'backward', n: 8 } },
      { inputs: [draw([2, 3])], params: { axis: -1, norm: 'ortho', n: 5 } },
      { inputs: [draw([4, 2])], params: { axis: -2, norm: 'forward', n: 6 } },
    ],
  },
})

// ── Public transforms ────────────────────────────────────────────────────────────────────────────────────────────────

/** A tensor or traced value from a transform's input (arrays become float64 vectors). */
function asValue(x: Value | ArrayLike<number>): Value {
  if (typeof x === 'number' || isTensor(x) || isTraced(x)) return x as Value
  return fromData(Float64Array.from(x as ArrayLike<number>))
}

/** The axis as a negative index into a value of rank `rank`. */
function negativeAxis(axis: number, rank: number, what: string): number {
  const a = axis < 0 ? axis : axis - rank
  if (!Number.isInteger(a) || a < -rank || a >= 0 || rank === 0)
    throw new ShapeError(what, `${what}: axis ${axis} is out of range for rank ${rank}`)
  return a
}

/** Slice specs selecting [0, m) along negative axis `axis` of a rank-`rank` value. */
function prefix(rank: number, axis: number, m: number): SliceSpec[] {
  const specs: SliceSpec[] = new Array<SliceSpec>(rank + axis).fill(null)
  specs.push([0, m])
  return specs
}

/** `v` zero-padded by `extra` entries at the end of negative axis `axis`. */
function padAxis(v: Value, axis: number, extra: number): Value {
  if (extra === 0) return v
  const aval = avalOf(v)
  const shape = [...aval.shape]
  shape[shape.length + axis] = extra
  return concat([v, zeros(shape, aval.dtype === 'complex128' ? 'complex128' : 'float64')], shape.length + axis)
}

/** `v` zero-padded or truncated to length n along negative axis `axis`. */
function fitAxis(v: Value, axis: number, n: number): Value {
  const shape = shapeOfValue(v)
  const len = shape[shape.length + axis]
  if (n === len) return v
  return n < len ? slice(v, ...prefix(shape.length, axis, n)) : padAxis(v, axis, n - len)
}

/** Resolve the options of a transform of `x`: the negative axis, the length n (default: the axis length), the norm. */
function resolve(x: Value, o: FftOptions, what: string): { axis: number; n: number; norm: FftNorm; len: number } {
  const shape = shapeOfValue(x)
  const axis = negativeAxis(o.axis ?? -1, shape.length, what)
  const len = shape[shape.length + axis]
  const n = o.n ?? len
  if (!Number.isInteger(n) || n < 1) throw new DomainError(what, `${what}: n must be a positive integer, got ${n}`)
  return { axis, n, norm: o.norm ?? 'backward', len }
}

/** The result kind of a transform: traced for traced inputs, else a tensor. */
type Out<X> = X extends Traced ? Traced : Tensor

/**
 * The DFT along an axis, as `numpy.fft.fft`: X[k] = s Σₜ x[t] e^{−2πi kt/n}, complex128 (real inputs are promoted).
 * `n` zero-pads or truncates the input first (any length is fast); `norm` sets s (1, 1/√n or 1/n).
 */
export function fft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'fft')
  return fftOp([fitAxis(v, axis, n)], { axis, norm }) as Out<X>
}

/** The inverse DFT along an axis, as `numpy.fft.ifft`: x[t] = s Σₖ X[k] e^{2πi kt/n}, s = 1/n by default. */
export function ifft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'ifft')
  return ifftOp([fitAxis(v, axis, n)], { axis, norm }) as Out<X>
}

/** The DFT of a real signal at its non-negative frequencies k = 0, …, ⌊n/2⌋, as `numpy.fft.rfft`. */
export function rfft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'rfft')
  return rfftOp([fitAxis(v, axis, n)], { axis, norm, n }) as Out<X>
}

/**
 * The inverse of `rfft`, as `numpy.fft.irfft`: the real signal of length n (default 2(m − 1) for m input bins) whose
 * spectrum has the given non-negative half. The bins are truncated or zero-padded to ⌊n/2⌋ + 1 first; the imaginary
 * parts of the DC and (n even) Nyquist bins are ignored. float64.
 */
export function irfft<X extends Value>(x: X, options: FftOptions = {}): Out<X> {
  const { axis, len } = resolve(x, { ...options, n: undefined }, 'irfft')
  const n = options.n ?? 2 * (len - 1)
  if (!Number.isInteger(n) || n < 1) throw new DomainError('irfft', `irfft: n must be a positive integer, got ${n}`)
  const norm = options.norm ?? 'backward'
  return irfftOp([fitAxis(x, axis, Math.floor(n / 2) + 1)], { axis, norm, n }) as Out<X>
}

/** Options of the n-dimensional transforms. */
export type FftnOptions = {
  /** Axes transformed (default: every axis, or the last `s.length`). */
  axes?: readonly number[]
  /** Lengths per transformed axis (pad or truncate). */
  s?: readonly number[]
  norm?: FftNorm
}

function nd(one: typeof fft, what: string) {
  return <X extends Value>(x: X, { axes, s, norm }: FftnOptions = {}): Out<X> => {
    const rank = shapeOfValue(x).length
    const list = axes ?? Array.from({ length: s?.length ?? rank }, (_, k) => k - (s?.length ?? rank))
    if (s && s.length !== list.length) throw new ShapeError(what, `${what}: s and axes differ in length`)
    let v: Value = x
    list.forEach((axis, k) => {
      v = one(v, { axis, n: s?.[k], norm })
    })
    return v as Out<X>
  }
}

/** The n-dimensional DFT, as `numpy.fft.fftn`: `fft` along each axis in turn (the transforms commute). */
export const fftn = nd(fft, 'fftn')
/** The n-dimensional inverse DFT, as `numpy.fft.ifftn`. */
export const ifftn = nd(ifft, 'ifftn')

/** The 2-D DFT over the last two axes, as `numpy.fft.fft2`. */
export function fft2<X extends Value>(x: X, { axes = [-2, -1], s, norm }: FftnOptions = {}): Out<X> {
  return fftn(x, { axes, s, norm })
}

/** The inverse 2-D DFT over the last two axes, as `numpy.fft.ifft2`. */
export function ifft2<X extends Value>(x: X, { axes = [-2, -1], s, norm }: FftnOptions = {}): Out<X> {
  return ifftn(x, { axes, s, norm })
}

// ── Frequency grids and shifts ───────────────────────────────────────────────────────────────────────────────────────

/** Frequencies of the DFT bins, as `numpy.fft.fftfreq`: k/(n d) for k = 0, …, ⌈n/2⌉ − 1, then the negative ones. */
export function fftfreq(n: number, d = 1): Tensor {
  const out = new Float64Array(n)
  for (let k = 0; k < n; k++) out[k] = (k < Math.ceil(n / 2) ? k : k - n) / (n * d)
  return fromData(out)
}

/** Frequencies of the `rfft` bins, k/(n d) for k = 0, …, ⌊n/2⌋, as `numpy.fft.rfftfreq`. */
export function rfftfreq(n: number, d = 1): Tensor {
  return fromData(Float64Array.from({ length: Math.floor(n / 2) + 1 }, (_, k) => k / (n * d)))
}

/** A circular roll by `shift` along each axis in `axes` (a composition of slices and a concat, so linear). */
function roll(x: Value, axes: readonly number[] | undefined, shift: (n: number) => number): Value {
  const shape = shapeOfValue(x)
  const list = axes ?? shape.map((_, k) => k)
  let v = x
  for (const axis of list) {
    const a = axis < 0 ? axis + shape.length : axis
    const n = shape[a]
    const k = ((shift(n) % n) + n) % n
    if (k === 0) continue
    const lead = new Array<SliceSpec>(a).fill(null)
    v = concat([slice(v, ...lead, [n - k, n]), slice(v, ...lead, [0, n - k])], a)
  }
  return v
}

/** Move the zero-frequency bin to the centre, as `numpy.fft.fftshift` (every axis by default). */
export function fftshift<X extends Value | ArrayLike<number>>(x: X, axes?: readonly number[]): Out<X> {
  return roll(asValue(x), axes, (n) => Math.floor(n / 2)) as Out<X>
}

/** The inverse of `fftshift`, as `numpy.fft.ifftshift`. */
export function ifftshift<X extends Value | ArrayLike<number>>(x: X, axes?: readonly number[]): Out<X> {
  return roll(asValue(x), axes, (n) => -Math.floor(n / 2)) as Out<X>
}

// ── The DFT matrix: the definition ───────────────────────────────────────────────────────────────────────────────────

/**
 * The n × n DFT matrix, complex128: Fⱼₖ = s e^{−2πi jk/n} (with `inverse`, s e^{+2πi jk/n}), s from `norm` as for
 * `fft`/`ifft`. Symmetric; with 'ortho' it is unitary. The O(n²) operator whose fast algorithm is `fft`.
 */
export function dftMatrix(
  n: number,
  { norm = 'backward', inverse = false }: { norm?: FftNorm; inverse?: boolean } = {},
): Tensor {
  if (!Number.isInteger(n) || n < 1)
    throw new DomainError('dftMatrix', `dftMatrix: n must be a positive integer, got ${n}`)
  const s = scaleOf(norm, n, inverse)
  const sign = inverse ? 1 : -1
  const out = new Float64Array(2 * n * n)
  for (let j = 0; j < n; j++)
    for (let k = 0; k < n; k++) {
      // jk is reduced mod n so the angle stays in [0, 2π) and accurate.
      const angle = (sign * TAU * ((j * k) % n)) / n
      out[2 * (j * n + k)] = s * Math.cos(angle)
      out[2 * (j * n + k) + 1] = s * Math.sin(angle)
    }
  return fromData(out, [n, n], 'complex128')
}

/**
 * The DFT by its definition: the product with `dftMatrix(n)` along `axis`, O(n²) per line. The same map as `fft`
 * (which computes it in O(n log n)); differentiable through `matmul`.
 */
export function dft<X extends Value | ArrayLike<number>>(
  x: X,
  { axis = -1, norm = 'backward', inverse = false }: { axis?: number; norm?: FftNorm; inverse?: boolean } = {},
): Out<X> {
  const v = asValue(x)
  const rank = shapeOfValue(v).length
  const a = rank + negativeAxis(axis, rank, 'dft')
  const F = dftMatrix(shapeOfValue(v)[a], { norm, inverse })
  // x[..., n] · F applies F to every line along the last axis (F is symmetric).
  if (a === rank - 1) return matmul(v, F) as Out<X>
  // Move the axis last, multiply, move it back.
  const order = [...Array.from({ length: rank }, (_, k) => k).filter((k) => k !== a), a]
  const back = order.map((_, k) => order.indexOf(k))
  return permute(matmul(permute(v, order), F), back) as Out<X>
}
