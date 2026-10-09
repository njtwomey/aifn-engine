/**
 * The discrete Fourier transform $X_k = s \sum_{t=0}^{n-1} x_t e^{-2\pi i kt/n}$ as primitives on complex128 tensors,
 * with numpy.fft's conventions: `fft`, `ifft`, `rfft`, `irfft` along any axis, with `{ axis = -1, n, norm }`.
 * Every line along the axis is transformed independently, so leading and trailing axes are batches. Power-of-two
 * lengths use the iterative radix-2 Cooley–Tukey FFT (Cooley and Tukey, 1965, Math. Comp. 19); other lengths use
 * Bluestein's chirp-z algorithm (Bluestein, 1970, IEEE Trans. Audio Electroacoust. 18(4)), which rewrites a length-$n$
 * DFT as a circular convolution of power-of-two length $m \ge 2n - 1$. Both are $O(n \log n)$.
 *
 * **Derivatives (design K §8.2).** The four maps are $\reals$-linear, so each is a linear primitive whose jvp is itself
 * and whose vjp is its declared `transpose`, the $\reals^2$ adjoint (the conjugate transpose). Let $\Fmat$ be the
 * unnormalised DFT matrix, $F_{jk} = e^{-2\pi i jk/n}$ (symmetric). `norm` scales the forward map by $s = 1$,
 * $1/\sqrt{n}$ or $1/n$ (`'backward'`, `'ortho'`, `'forward'`) and the inverse by $1/n$, $1/\sqrt{n}$ or $1$. Then:
 *
 * - the adjoint of `fft` is $s\Fmat^{\mathsf{H}}$, which is `ifft` under `dual(norm)`; `dual` swaps `'backward'` and
 *   `'forward'` and keeps `'ortho'`;
 * - the adjoint of `ifft` is `fft` under `dual(norm)`;
 * - `rfft` (real, length $n$, to $\lfloor n/2 \rfloor + 1$ complex bins): the adjoint zero-pads the cotangent to $n$,
 *   applies `ifft` under `dual(norm)` and keeps the real part (the input is real);
 * - `irfft` ($\lfloor n/2 \rfloor + 1$ bins to a real signal of length $n$): the adjoint is
 *   $\cvec \odot \operatorname{rfft}(\bar{\yvec})$ under `dual(norm)`, with $c_k = 1$ at $k = 0$ and (for even $n$)
 *   at $k = n/2$ and $c_k = 2$ otherwise, because each interior bin appears twice in the Hermitian extension. Taking
 *   `rfft` alone as the adjoint would count each interior bin once instead of twice.
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

/**
 * True when $n$ is a positive power of two ($1, 2, 4, \dots$), the lengths `fft` transforms by radix-2 rather than by
 * Bluestein's algorithm. It is a 32-bit bit test, meant for integer lengths below $2^{31}$.
 *
 * @param n The length to test.
 * @returns Whether $n = 2^k$ for some integer $k \ge 0$.
 *
 * @example Which lengths take the radix-2 path
 * print('8:', isPowerOfTwo(8))
 * print('6:', isPowerOfTwo(6))
 * print('1:', isPowerOfTwo(1))
 * print('0:', isPowerOfTwo(0))
 */
export const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0

/**
 * The smallest power of two $2^k \ge n$ ($1$ for $n \le 1$): the length to zero-pad a signal to before an FFT.
 *
 * @param n The length to round up; need not be an integer.
 * @returns The smallest power of two that is at least $n$.
 *
 * @example Round lengths up for zero-padding
 * print('5 ->', nextPowerOfTwo(5))
 * print('8 ->', nextPowerOfTwo(8))
 * print('1000 ->', nextPowerOfTwo(1000))
 * print('0 ->', nextPowerOfTwo(0))
 */
export const nextPowerOfTwo = (n: number): number => 2 ** Math.ceil(Math.log2(Math.max(1, n)))

// ── Kernels on raw arrays ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * In-place iterative radix-2 FFT (Cooley and Tukey, 1965): a bit-reversal permutation, then butterflies of length
 * $2, 4, \dots, n$. Unscaled in both directions. The length is not checked: it must be a power of two.
 *
 * @param re The real parts of the line, length $n$; overwritten with the real parts of its transform.
 * @param im The imaginary parts, length $n$; overwritten likewise.
 * @param inverse False for the forward exponent $e^{-2\pi i kt/n}$, true for $e^{+2\pi i kt/n}$ (with no $1/n$).
 */
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

/**
 * In-place Bluestein transform of any length $n$ (Bluestein, 1970). As $kt = (k^2 + t^2 - (k - t)^2)/2$, the DFT is
 * a product with the chirp $w_k = e^{-\pi i k^2/n}$, a circular convolution with $\bar{w}$ (three radix-2 FFTs of
 * length $m = 2^{\lceil \log_2(2n - 1) \rceil}$), and a product with $w$ again. Unscaled in both directions;
 * allocates its work arrays.
 *
 * @param re The real parts of the line, length $n$; overwritten with the real parts of its transform.
 * @param im The imaginary parts, length $n$; overwritten likewise.
 * @param inverse False for the forward exponent $e^{-2\pi i kt/n}$, true for $e^{+2\pi i kt/n}$ (with no $1/n$).
 */
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
 * Bluestein's chirp z-transform otherwise, and nothing for $n \le 1$. Unscaled: the inverse must still be divided by
 * $n$. The raw-array kernel under the primitives (it is not exported by the module).
 *
 * @param re The real parts of the line, length $n$; overwritten with the real parts of its transform.
 * @param im The imaginary parts, of the same length; overwritten likewise.
 * @param inverse False (default) for the forward transform, true for the inverse one without its $1/n$.
 */
function transformInPlace(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (n <= 1) return
  if (isPowerOfTwo(n)) radix2(re, im, inverse)
  else bluestein(re, im, inverse)
}

// ── Normalisation ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Which direction carries the $1/n$, as numpy's `norm`: `'backward'` puts it on the inverse, `'forward'` on the forward
 * transform, and `'ortho'` splits it as $1/\sqrt{n}$ on both (which makes both unitary).
 */
export type FftNorm = 'backward' | 'ortho' | 'forward'

/** Options of the one-axis transforms. */
export type FftOptions = {
  /** The axis transformed (default `-1`, the last); negative values count from the end. */
  axis?: number
  /** Length of the transform: the input is zero-padded or truncated along the axis first (irfft: output length). */
  n?: number
  /** Normalisation (default 'backward': the forward transform unscaled, the inverse divided by n). */
  norm?: FftNorm
}

/**
 * The scale $s$ that multiplies the unscaled transform: $1$, $1/\sqrt{n}$ or $1/n$.
 *
 * @param norm The normalisation.
 * @param n The transform length.
 * @param inverse False for the scale of the forward transform, true for that of the inverse.
 * @returns $1/\sqrt{n}$ under `'ortho'`; $1/n$ for the inverse under `'backward'` and the forward under `'forward'`;
 *   otherwise $1$.
 */
function scaleOf(norm: FftNorm, n: number, inverse: boolean): number {
  if (norm === 'ortho') return 1 / Math.sqrt(n)
  return (norm === 'backward') === inverse ? 1 / n : 1
}

/**
 * The normalisation whose forward scale is the inverse scale of `norm` (and vice versa): the one the adjoint uses.
 *
 * @param norm The normalisation of the transform being transposed.
 * @returns `'forward'` for `'backward'`, `'backward'` for `'forward'`, and `'ortho'` for `'ortho'`.
 */
const dual = (norm: FftNorm): FftNorm => (norm === 'backward' ? 'forward' : norm === 'forward' ? 'backward' : 'ortho')

// ── Lines along an axis ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Apply `line` to every line of `x` along one axis. Each line is read as complex (real inputs promoted) into `re` and
 * `im` of the axis length, and `line` leaves the output line in `outRe` and `outIm` of length `nOut`. The four buffers
 * are reused from line to line.
 *
 * @param x The input tensor, of rank at least 1; not modified.
 * @param axis The axis whose lines are transformed, as a negative index (`-1` is the last).
 * @param nOut The length of each output line, the size of that axis in the result.
 * @param real Whether to keep only the real parts (`outIm` is then ignored).
 * @param line The transform of one line: it reads `re` and `im` (which it may overwrite) and must fill `outRe` (and
 *   `outIm` unless `real`) in full.
 * @returns A new tensor with the shape of `x` except `nOut` along `axis`: complex128, or float64 when `real`.
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

/**
 * The tensor input of a transform primitive, checked: a `ShapeError` for a number, a scalar tensor, or an axis below
 * $-\text{rank}$.
 *
 * @param x The primitive's input.
 * @param axis The transformed axis, as a negative index.
 * @param what The caller's name for error messages.
 * @returns `x`, now known to be a tensor of rank at least 1.
 */
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

/**
 * The abstract value (shape and dtype) of a transform's result.
 *
 * @param aval The abstract value of the input.
 * @param axis The transformed axis, as a negative index.
 * @param n The length of that axis in the result.
 * @param real Whether the result is real (float64) rather than complex128.
 * @returns `aval`'s shape with the axis resized to $n$, and the result's dtype.
 */
function resized(aval: Aval, axis: number, n: number, real: boolean): Aval {
  const shape = [...aval.shape]
  shape[shape.length + axis] = n
  return { shape, dtype: real ? 'float64' : 'complex128', number: false }
}

// ── The primitives ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Parameters of every transform primitive: the axis (negative, so a batch axis in front leaves it unchanged). */
type LineParams = { axis: number; norm: FftNorm }
/** `rfft` and `irfft` also carry the length $n$ of the real signal. */
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

/**
 * The kernel of the `fft` or `ifft` primitive: every line along the axis transformed and multiplied by the scale of
 * `norm`.
 *
 * @param inverse False for the kernel of `fft`, true for that of `ifft`.
 * @returns A function of a tensor and its `{ axis, norm }` that returns the complex128 transform, of the same shape.
 */
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

/**
 * The weights $c_k$ of the adjoint of `irfft` on its $m = \lfloor n/2 \rfloor + 1$ bins: $1$ at $k = 0$ and (for even
 * $n$) at $k = n/2$, else $2$.
 *
 * @param n The length of the real signal.
 * @param axis The axis of the bins, as a negative index.
 * @returns A float64 tensor of shape $[m, 1, \dots, 1]$, with one trailing 1 per axis after `axis`, so that it
 *   broadcasts along that axis.
 */
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

/**
 * A transform's input as a value: numbers, tensors and traced values pass through, and arrays become float64
 * vectors.
 *
 * @param x The input as the caller gave it.
 * @returns `x` as a number, tensor or traced value.
 */
function asValue(x: Value | ArrayLike<number>): Value {
  if (typeof x === 'number' || isTensor(x) || isTraced(x)) return x as Value
  return fromData(Float64Array.from(x as ArrayLike<number>))
}

/**
 * An axis as a negative index into a value of rank `rank`. Throws `ShapeError` when it is not an integer in
 * $[-\text{rank}, \text{rank})$, or when the rank is 0.
 *
 * @param axis The axis, negative (counted from the end) or non-negative (counted from the start).
 * @param rank The rank of the value.
 * @param what The caller's name for error messages.
 * @returns The same axis counted from the end, in $[-\text{rank}, -1]$.
 */
function negativeAxis(axis: number, rank: number, what: string): number {
  const a = axis < 0 ? axis : axis - rank
  if (!Number.isInteger(a) || a < -rank || a >= 0 || rank === 0)
    throw new ShapeError(what, `${what}: axis ${axis} is out of range for rank ${rank}`)
  return a
}

/**
 * Slice specs selecting the first $m$ entries along one axis and everything along the others.
 *
 * @param rank The rank of the value sliced.
 * @param axis The axis to cut, as a negative index.
 * @param m The number of entries kept, from the start of the axis.
 * @returns One `null` (all) per axis before `axis`, then `[0, m]`; the axes after it are left whole.
 */
function prefix(rank: number, axis: number, m: number): SliceSpec[] {
  const specs: SliceSpec[] = new Array<SliceSpec>(rank + axis).fill(null)
  specs.push([0, m])
  return specs
}

/**
 * `v` zero-padded at the end of one axis, through `concat` (so differentiable). The zeros are complex128 when `v` is,
 * float64 otherwise.
 *
 * @param v The value to pad; not modified.
 * @param axis The axis to pad, as a negative index.
 * @param extra The number of zeros to append along it; 0 returns `v` itself.
 * @returns `v` with `extra` more entries along `axis`.
 */
function padAxis(v: Value, axis: number, extra: number): Value {
  if (extra === 0) return v
  const aval = avalOf(v)
  const shape = [...aval.shape]
  shape[shape.length + axis] = extra
  return concat([v, zeros(shape, aval.dtype === 'complex128' ? 'complex128' : 'float64')], shape.length + axis)
}

/**
 * `v` zero-padded or truncated to length $n$ along one axis (the `n` option of the transforms).
 *
 * @param v The value; not modified.
 * @param axis The axis to fit, as a negative index.
 * @param n The length wanted along it.
 * @returns `v` itself when the axis already has length $n$; else its first $n$ entries, or `v` followed by zeros.
 */
function fitAxis(v: Value, axis: number, n: number): Value {
  const shape = shapeOfValue(v)
  const len = shape[shape.length + axis]
  if (n === len) return v
  return n < len ? slice(v, ...prefix(shape.length, axis, n)) : padAxis(v, axis, n - len)
}

/**
 * Resolve the options of a transform of `x`, with their defaults. Throws `ShapeError` for a bad axis and `DomainError`
 * when `n` is not a positive integer.
 *
 * @param x The input of the transform.
 * @param o The options as the caller gave them.
 * @param what The caller's name for error messages.
 * @returns `axis` as a negative index (default `-1`), `n` (default: the axis length), `norm` (default `'backward'`)
 *   and `len`, the current length of the axis.
 */
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
 * The DFT along an axis, as `numpy.fft.fft`: $X_k = s \sum_{t=0}^{n-1} x_t e^{-2\pi i kt/n}$, complex128 (real inputs
 * are promoted). Any length is fast: radix-2 for powers of two, Bluestein otherwise. Linear and differentiable in both
 * modes, and batched by `vmap`. Throws `ShapeError` for a scalar or an axis out of range.
 *
 * @param x The signal: a real or complex tensor of rank at least 1 (every line along the axis is transformed), or an
 *   array of numbers (a float64 vector). A traced value makes the result differentiable.
 * @param options The axis, the length $n$ (the input is zero-padded or truncated to it first) and `norm`, which sets
 *   $s$ ($1$ by default, or $1/\sqrt{n}$ or $1/n$).
 * @returns The complex128 spectrum, with the shape of `x` except $n$ along the axis.
 *
 * @example An impulse has a flat spectrum, a constant only a zero-frequency bin
 * print('fft(impulse) =', fft([1, 0, 0, 0]))
 * print('fft(constant) =', fft([1, 1, 1, 1]))
 *
 * @example Length 3 (Bluestein) and the unitary scaling
 * print('fft([1, 2, 3]) =', fft([1, 2, 3]))
 * print('ortho =', fft([1, 1, 1, 1], { norm: 'ortho' }))
 *
 * @example Zero-padding, and transforming the columns of a matrix
 * print('padded to 4 =', fft([1, 1], { n: 4 }))
 * print('down the columns =', fft(tensor([[1, 2], [3, 4]]), { axis: 0 }))
 */
export function fft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'fft')
  return fftOp([fitAxis(v, axis, n)], { axis, norm }) as Out<X>
}

/**
 * The inverse DFT along an axis, as `numpy.fft.ifft`: $x_t = s \sum_{k=0}^{n-1} X_k e^{2\pi i kt/n}$, with $s = 1/n$
 * by default. Complex128; linear, differentiable and batched like `fft`.
 *
 * @param x The spectrum: a real or complex tensor of rank at least 1, or an array of numbers (a float64 vector).
 * @param options The axis, the length $n$ (the input is zero-padded or truncated to it first) and `norm`, which sets
 *   $s$ ($1/n$ by default, or $1/\sqrt{n}$ or $1$).
 * @returns The complex128 signal, with the shape of `x` except $n$ along the axis.
 *
 * @example A flat spectrum is an impulse
 * print('ifft([1, 1, 1, 1]) =', ifft([1, 1, 1, 1]))
 *
 * @example ifft undoes fft
 * const x = tensor([1, 2, 3])
 * print('x =', x)
 * print('ifft(fft(x)) =', ifft(fft(x)))
 */
export function ifft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'ifft')
  return ifftOp([fitAxis(v, axis, n)], { axis, norm }) as Out<X>
}

/**
 * The DFT of a real signal at its non-negative frequencies $k = 0, \dots, \lfloor n/2 \rfloor$, as `numpy.fft.rfft`:
 * the rest of the spectrum is their complex conjugate, $X_{n-k} = \bar{X}_k$. Complex128; linear, differentiable and
 * batched like `fft`. Throws `DTypeError` for a complex input.
 *
 * @param x The real signal: a real tensor of rank at least 1, or an array of numbers.
 * @param options The axis, the length $n$ (the input is zero-padded or truncated to it first) and `norm`, as for `fft`.
 * @returns The $\lfloor n/2 \rfloor + 1$ complex128 bins, with the shape of `x` otherwise.
 *
 * @example The non-negative half of the spectrum
 * print('rfft([1, 2, 3, 4]) =', rfft([1, 2, 3, 4]))
 * print('fft([1, 2, 3, 4]) =', fft([1, 2, 3, 4]))
 */
export function rfft<X extends Value | ArrayLike<number>>(x: X, options: FftOptions = {}): Out<X> {
  const v = asValue(x)
  const { axis, n, norm } = resolve(v, options, 'rfft')
  return rfftOp([fitAxis(v, axis, n)], { axis, norm, n }) as Out<X>
}

/**
 * The inverse of `rfft`, as `numpy.fft.irfft`: the real signal of length $n$ whose spectrum has the given non-negative
 * half, extended by $X_{n-k} = \bar{X}_k$. The bins are truncated or zero-padded to $\lfloor n/2 \rfloor + 1$ first;
 * the imaginary parts of the DC and (for even $n$) Nyquist bins are ignored. Linear, differentiable and batched like
 * `fft`.
 *
 * @param x The non-negative half of a spectrum, $m$ bins along the axis: a real or complex tensor of rank at least 1.
 * @param options The axis, the output length $n$ (default $2(m - 1)$, so pass it for an odd-length signal) and
 *   `norm`, as for `ifft`.
 * @returns The float64 signal, with the shape of `x` except $n$ along the axis.
 *
 * @example irfft undoes rfft, for even and odd lengths
 * print('even:', irfft(rfft([1, 2, 3, 4])))
 * print('odd, with n:', irfft(rfft([1, 2, 3]), { n: 3 }))
 * print('odd, without n:', irfft(rfft([1, 2, 3])))
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
  /** Lengths per transformed axis, in the order of `axes`: each axis is zero-padded or truncated to its length. */
  s?: readonly number[]
  /** Normalisation of every one-axis transform (default `'backward'`). */
  norm?: FftNorm
}

/**
 * The n-dimensional version of a one-axis transform: `one` applied along each axis in turn. Throws `ShapeError` when
 * `s` and `axes` are both given with different lengths.
 *
 * @param one The one-axis transform (`fft` or `ifft`).
 * @param what The name of the n-dimensional transform, for error messages.
 * @returns The transform of a value and its `FftnOptions`.
 */
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

/**
 * The n-dimensional DFT, as `numpy.fft.fftn`: `fft` along each axis in turn (the transforms commute). It takes the
 * value and `FftnOptions`: by default every axis is transformed, or the last `s.length` when only `s` is given.
 *
 * @example A 2 × 2 impulse has a flat spectrum
 * print('fftn =', fftn(tensor([[1, 0], [0, 0]])))
 *
 * @example Only the axes asked for
 * print('along axis 1 only =', fftn(tensor([[1, 2], [3, 4]]), { axes: [1] }))
 */
export const fftn = nd(fft, 'fftn')
/**
 * The n-dimensional inverse DFT, as `numpy.fft.ifftn`: `ifft` along each axis in turn, with the options of `fftn`.
 *
 * @example ifftn undoes fftn
 * print('ifftn(fftn(x)) =', ifftn(fftn(tensor([[1, 2], [3, 4]]))))
 */
export const ifftn = nd(ifft, 'ifftn')

/**
 * The 2-D DFT over the last two axes, as `numpy.fft.fft2`: `fftn` with `axes` defaulting to `[-2, -1]`, so any
 * leading axes are a batch of images.
 *
 * @param x The array: a real or complex tensor of rank at least 2 (or of rank at least the length of `axes`).
 * @param options The transform's options.
 * @param options.axes The two axes transformed (default `[-2, -1]`, the last two).
 * @param options.s The lengths to zero-pad or truncate those axes to first, in the order of `axes`.
 * @param options.norm The normalisation of each one-axis transform (default `'backward'`).
 * @returns The complex128 2-D spectrum, with the shape of `x` except the lengths in `s`.
 *
 * @example The spectrum of a 2 × 2 matrix
 * print('fft2 =', fft2(tensor([[1, 2], [3, 4]])))
 */
export function fft2<X extends Value>(x: X, { axes = [-2, -1], s, norm }: FftnOptions = {}): Out<X> {
  return fftn(x, { axes, s, norm })
}

/**
 * The inverse 2-D DFT over the last two axes, as `numpy.fft.ifft2`: `ifftn` with `axes` defaulting to `[-2, -1]`.
 *
 * @param x The 2-D spectrum: a real or complex tensor of rank at least 2.
 * @param options The transform's options.
 * @param options.axes The two axes transformed (default `[-2, -1]`, the last two).
 * @param options.s The lengths to zero-pad or truncate those axes to first, in the order of `axes`.
 * @param options.norm The normalisation of each one-axis transform (default `'backward'`, which divides by both
 *   lengths).
 * @returns The complex128 array, with the shape of `x` except the lengths in `s`.
 *
 * @example A flat 2 × 2 spectrum is an impulse
 * print('ifft2 =', ifft2(tensor([[1, 1], [1, 1]])))
 */
export function ifft2<X extends Value>(x: X, { axes = [-2, -1], s, norm }: FftnOptions = {}): Out<X> {
  return ifftn(x, { axes, s, norm })
}

// ── Frequency grids and shifts ───────────────────────────────────────────────────────────────────────────────────────

/**
 * Frequencies of the DFT bins, as `numpy.fft.fftfreq`: $k/(nd)$ for $k = 0, \dots, \lceil n/2 \rceil - 1$, then the
 * negative ones, $(k - n)/(nd)$ for the rest.
 *
 * @param n The number of bins (the transform length).
 * @param d The sample spacing (default 1), so the frequencies are in cycles per unit of `d`; with `d = 1` they are in
 *   cycles per sample.
 * @returns A float64 vector of the $n$ frequencies, in the order `fft` returns the bins.
 *
 * @example Bins in cycles per sample, and in hertz at 10 Hz sampling
 * print('n = 4:', fftfreq(4))
 * print('n = 5, d = 0.1 s:', fftfreq(5, 0.1))
 */
export function fftfreq(n: number, d = 1): Tensor {
  const out = new Float64Array(n)
  for (let k = 0; k < n; k++) out[k] = (k < Math.ceil(n / 2) ? k : k - n) / (n * d)
  return fromData(out)
}

/**
 * Frequencies of the `rfft` bins, $k/(nd)$ for $k = 0, \dots, \lfloor n/2 \rfloor$, as `numpy.fft.rfftfreq`.
 *
 * @param n The length of the real signal (not the number of bins).
 * @param d The sample spacing (default 1); the frequencies are in cycles per unit of `d`.
 * @returns A float64 vector of the $\lfloor n/2 \rfloor + 1$ non-negative frequencies.
 *
 * @example The frequencies of rfft's bins
 * print('n = 4:', rfftfreq(4))
 * print('n = 5:', rfftfreq(5))
 */
export function rfftfreq(n: number, d = 1): Tensor {
  return fromData(Float64Array.from({ length: Math.floor(n / 2) + 1 }, (_, k) => k / (n * d)))
}

/**
 * A circular roll along each axis in `axes`: entry $i$ moves to $(i + k) \bmod n$. A composition of slices and a
 * concat, so linear and differentiable.
 *
 * @param x The value to roll; not modified.
 * @param axes The axes to roll (negative ones count from the end); every axis when undefined.
 * @param shift The shift $k$ for an axis of length $n$ (any integer: it is reduced modulo $n$).
 * @returns The rolled value, of the same shape.
 */
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

/**
 * Move the zero-frequency bin to the centre, as `numpy.fft.fftshift`: a circular roll by $\lfloor n/2 \rfloor$ along
 * each axis, so the frequencies run from the most negative to the most positive. Linear and differentiable.
 *
 * @param x A spectrum (or `fftfreq`'s grid): a tensor, a traced value or an array of numbers.
 * @param axes The axes to shift (negative ones count from the end); every axis when omitted.
 * @returns The shifted value, of the same shape.
 *
 * @example Frequencies in increasing order
 * print('fftfreq(4) =', fftfreq(4))
 * print('shifted =', fftshift(fftfreq(4)))
 * print('fftshift([0, 1, 2, 3, 4]) =', fftshift([0, 1, 2, 3, 4]))
 */
export function fftshift<X extends Value | ArrayLike<number>>(x: X, axes?: readonly number[]): Out<X> {
  return roll(asValue(x), axes, (n) => Math.floor(n / 2)) as Out<X>
}

/**
 * The inverse of `fftshift`, as `numpy.fft.ifftshift`: a circular roll by $-\lfloor n/2 \rfloor$ along each axis. The
 * two differ only for odd lengths.
 *
 * @param x A centred spectrum: a tensor, a traced value or an array of numbers.
 * @param axes The axes to shift back (negative ones count from the end); every axis when omitted.
 * @returns The value with the zero-frequency bin back at index 0, of the same shape.
 *
 * @example ifftshift undoes fftshift for an odd length
 * const x = [0, 1, 2, 3, 4]
 * print('fftshift =', fftshift(x))
 * print('ifftshift(fftshift) =', ifftshift(fftshift(x)))
 * print('fftshift(fftshift) =', fftshift(fftshift(x)))
 */
export function ifftshift<X extends Value | ArrayLike<number>>(x: X, axes?: readonly number[]): Out<X> {
  return roll(asValue(x), axes, (n) => -Math.floor(n / 2)) as Out<X>
}

// ── The DFT matrix: the definition ───────────────────────────────────────────────────────────────────────────────────

/**
 * The $n \times n$ DFT matrix, complex128: $F_{jk} = s e^{-2\pi i jk/n}$ (with `inverse`, $s e^{+2\pi i jk/n}$), $s$
 * from `norm` as for `fft` and `ifft`. Symmetric; with `'ortho'` it is unitary. The $O(n^2)$ operator whose fast
 * algorithm is `fft`. Throws `DomainError` unless $n$ is a positive integer.
 *
 * @param n The transform length: the number of rows and columns.
 * @param options Which matrix.
 * @param options.norm The normalisation that sets $s$ (default `'backward'`: $s = 1$ forward, $1/n$ inverse).
 * @param options.inverse True for the matrix of `ifft` (exponent $+2\pi i jk/n$) instead of `fft`'s (default false).
 * @returns $\Fmat$ as a complex128 tensor of shape $[n, n]$.
 *
 * @example The 2 × 2 and 4 × 4 DFT matrices
 * print('F2 =', dftMatrix(2))
 * print('F4 =', dftMatrix(4))
 *
 * @example The forward and inverse matrices multiply to the identity
 * print('F4⁻¹ F4 =', matmul(dftMatrix(4, { inverse: true }), dftMatrix(4)))
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
 * The DFT by its definition: the product with `dftMatrix(n)` along an axis, $O(n^2)$ per line. The same map as `fft`
 * (which computes it in $O(n \log n)$), but with no `n` option; differentiable through `matmul`.
 *
 * @param x The signal: a real or complex tensor of rank at least 1, or an array of numbers (a float64 vector).
 * @param options Which transform, along which axis.
 * @param options.axis The axis transformed (default `-1`, the last); negative values count from the end.
 * @param options.norm The normalisation, as for `fft` (default `'backward'`).
 * @param options.inverse True for the inverse DFT (the map of `ifft`) instead of the forward one (default false).
 * @returns The complex128 transform, with the shape of `x`.
 *
 * @example The definition agrees with the FFT
 * print('dft([1, 2, 3]) =', dft([1, 2, 3]))
 * print('fft([1, 2, 3]) =', fft([1, 2, 3]))
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
