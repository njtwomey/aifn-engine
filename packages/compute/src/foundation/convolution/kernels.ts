/**
 * Raw kernels of the convolution family, on contiguous row-major `Float64Array`s: the forward convolution and its two
 * adjoints, each by direct summation or by FFT.
 *
 * The three maps share one trilinear form over the taps of a convolution,
 * $T(x, w, y) = \sum_{n, o, i, c, a} x_{n,\, c,\, is - \mathrm{lo} + ad}\, w_{o,\, c',\, \hat a}\, y_{n,\, o,\, i}$,
 * with $i$ the output position, $a$ the kernel tap, $s$ the stride, $d$ the dilation and $\mathrm{lo}$ the padding
 * before (all per spatial axis), $c$ running over the input channels of the group of $o$ and $c'$ its index within the
 * group, and $\hat a$ the tap $a$ reversed on every spatial axis when `flip` and $a$ itself otherwise. `convForward`
 * contracts $x$ and $w$ (the convolution), `convInputAdjoint` contracts $w$ and $y$ (the transposed convolution) and
 * `convWeightAdjoint` contracts $x$ and $y$ (the kernel gradient). Each has a direct kernel (zero-padded input and
 * precomputed tap offsets, the output as the inner loop) and an FFT kernel (the convolution theorem in $n$ dimensions,
 * channels contracted in the frequency domain); the forward also has overlap-add (blocks along the last axis). The
 * `method` chooses among them and changes the result only by rounding.
 */

import { fftn, ifftn, nextPowerOfTwo } from 'aifn-compute/foundation/fourier'
import { fromData } from 'aifn-compute/foundation/tensor'

/**
 * How the kernel computes a convolution: `direct` sums the taps; `fft` and `overlapAdd` (blocks along the last axis,
 * for long inputs and short kernels) use the convolution theorem; `auto` estimates the cost of direct and FFT and runs
 * the cheaper. The adjoints have no overlap-add kernel and run the FFT for it.
 */
export type ConvMethod = 'direct' | 'fft' | 'overlapAdd' | 'auto'

/**
 * The sizes of one convolution of an input `x` of shape `[N, C, ...S]` with kernels `w` of shape `[O, C/groups, ...K]`
 * into an output `y` of shape `[N, O, ...Y]`, with the stride, dilation and zero padding of each spatial axis.
 */
export type ConvDims = {
  /** The number of images (the batch axis of `x` and `y`). */
  readonly N: number
  /** The number of input channels. */
  readonly C: number
  /** The number of output channels (kernels). */
  readonly O: number
  /** The number of channel groups: input and output channels split into this many independent blocks. */
  readonly groups: number
  /** The spatial shape of the input, before padding. */
  readonly S: readonly number[]
  /** The spatial shape of each kernel, in taps (before dilation). */
  readonly K: readonly number[]
  /** The spatial shape of the output. */
  readonly Y: readonly number[]
  /** The step between output positions, per spatial axis. */
  readonly stride: readonly number[]
  /** The spacing between kernel taps, per spatial axis (1 for adjacent taps). */
  readonly dilation: readonly number[]
  /** The zeros added before the input, per spatial axis. */
  readonly lo: readonly number[]
  /** The zeros added after the input, per spatial axis. */
  readonly hi: readonly number[]
  /** True for convolution (the kernel reversed on every spatial axis), false for cross-correlation. */
  readonly flip: boolean
}

/**
 * The product of a list of numbers.
 *
 * @param a The numbers, usually the axis lengths of a shape.
 * @returns Their product (1 for an empty list).
 */
const prod = (a: readonly number[]): number => a.reduce((p, v) => p * v, 1)

/**
 * The row-major strides of a shape: how far apart, in elements, consecutive indices of each axis are.
 *
 * @param dims The axis lengths.
 * @returns One stride per axis: 1 for the last, and each other the product of the lengths after it.
 */
function rowStrides(dims: readonly number[]): number[] {
  const s = new Array<number>(dims.length)
  for (let d = dims.length - 1, t = 1; d >= 0; d--) {
    s[d] = t
    t *= dims[d]
  }
  return s
}

/**
 * The flat offsets $\sum_k a_k\, \mathrm{step}_k\, \mathrm{strides}_k$ of every multi-index $a$ over `extent`, in
 * row-major order of $a$: the positions of a (possibly spaced-out) block inside a larger row-major array.
 *
 * @param extent The length of each axis of the block of multi-indices.
 * @param step The spacing of consecutive indices on each axis (a stride or a dilation; 1 for a contiguous block).
 * @param strides The row-major strides of the array the offsets index into.
 * @returns One offset per multi-index, $\prod_k \mathrm{extent}_k$ in all.
 */
function offsets(extent: readonly number[], step: readonly number[], strides: readonly number[]): Int32Array {
  const out = new Int32Array(prod(extent))
  const D = extent.length
  if (out.length === 0) return out
  const a = new Array<number>(D).fill(0)
  let off = 0
  for (let k = 0; k < out.length; k++) {
    out[k] = off
    for (let d = D - 1; d >= 0; d--) {
      a[d]++
      off += step[d] * strides[d]
      if (a[d] < extent[d]) break
      off -= a[d] * step[d] * strides[d]
      a[d] = 0
    }
  }
  return out
}

/**
 * Sizes derived from a convolution's dimensions: the padded input $P = S + \mathrm{lo} + \mathrm{hi}$, the effective
 * kernel extent $K_e = d(K - 1) + 1$ (both per spatial axis), the channels per group and the element counts.
 *
 * @param d The dimensions of the convolution.
 * @returns `P` and `Ke` per axis; `Cg` and `Og`, the input and output channels per group; `Kp`, `Yp`, `Sp` and `Pp`,
 *   the number of elements of one kernel, output, input and padded input channel.
 */
function derived(d: ConvDims) {
  const P = d.S.map((s, k) => s + d.lo[k] + d.hi[k])
  const Ke = d.K.map((k, j) => d.dilation[j] * (k - 1) + 1)
  return { P, Ke, Cg: d.C / d.groups, Og: d.O / d.groups, Kp: prod(d.K), Yp: prod(d.Y), Sp: prod(d.S), Pp: prod(P) }
}

/**
 * Copy `count` arrays of extent $S$ into zeroed arrays of extent $P$ at offset $\mathrm{lo}$ (zero padding), or, with
 * `crop`, copy the window of extent $S$ at offset $\mathrm{lo}$ out of each array of extent $P$ (its adjoint).
 *
 * @param src The arrays, consecutive and row-major: `count` of extent `S`, or of extent `P` with `crop`. Not modified.
 * @param count The number of arrays (images times channels).
 * @param S The extent of the unpadded arrays, per axis.
 * @param P The extent of the padded arrays, per axis; at least $S + \mathrm{lo}$ on every axis.
 * @param lo Where the unpadded array starts inside the padded one, per axis.
 * @param crop False to pad (`src` of extent `S`), true to crop (`src` of extent `P`).
 * @returns A new array of `count` arrays of extent `P` (padding) or `S` (cropping).
 */
function padOrCrop(
  src: Float64Array,
  count: number,
  S: readonly number[],
  P: readonly number[],
  lo: readonly number[],
  crop: boolean,
): Float64Array {
  const Sp = prod(S)
  const Pp = prod(P)
  const out = new Float64Array(count * (crop ? Sp : Pp))
  if (Sp === 0) return out
  const D = S.length
  const rows = offsets(S.slice(0, D - 1), new Array<number>(D - 1).fill(1), rowStrides(P).slice(0, D - 1))
  const shift = lo.reduce((acc, l, k) => acc + l * rowStrides(P)[k], 0)
  const last = S[D - 1]
  for (let c = 0; c < count; c++)
    for (let r = 0; r < rows.length; r++) {
      const s = c * Sp + r * last
      const p = c * Pp + shift + rows[r]
      if (crop) out.set(src.subarray(p, p + last), s)
      else out.set(src.subarray(s, s + last), p)
    }
  return out
}

// ── Direct ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The tap tables of the direct kernels: output-position and kernel-tap offsets into one padded channel, so the input
 * read by output $i$ and tap $a$ is at `ob[i] + ko[a]`.
 *
 * @param d The dimensions of the convolution.
 * @param P The extent of a padded input channel.
 * @returns `ob`, the offset of each output position (spaced by the stride), and `ko`, the offset of each kernel tap
 *   (spaced by the dilation), both in row-major order.
 */
function taps(d: ConvDims, P: readonly number[]) {
  const st = rowStrides(P)
  return { ob: offsets(d.Y, d.stride, st), ko: offsets(d.K, d.dilation, st) }
}

/**
 * The convolution by direct summation over the taps, the output as the inner loop; zero taps are skipped.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution.
 * @returns The output, row-major `[N, O, ...Y]`.
 */
function directForward(x: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Cg, Og, Kp, Yp, Pp } = derived(d)
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const { ob, ko } = taps(d, P)
  const y = new Float64Array(d.N * d.O * Yp)
  for (let n = 0; n < d.N; n++)
    for (let o = 0; o < d.O; o++) {
      const g = Math.floor(o / Og)
      const yb = (n * d.O + o) * Yp
      for (let c = 0; c < Cg; c++) {
        const xb = (n * d.C + g * Cg + c) * Pp
        const wb = (o * Cg + c) * Kp
        for (let k = 0; k < Kp; k++) {
          const wv = w[wb + (d.flip ? Kp - 1 - k : k)]
          if (wv === 0) continue
          const off = xb + ko[k]
          for (let i = 0; i < Yp; i++) y[yb + i] += wv * xp[off + ob[i]]
        }
      }
    }
  return y
}

/**
 * The transposed convolution by direct summation: each output gradient scattered back through the taps onto the padded
 * input it was read from, then the padding cropped off.
 *
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution whose input adjoint this is.
 * @returns The input gradient, row-major `[N, C, ...S]`.
 */
function directInput(gy: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Cg, Og, Kp, Yp, Pp } = derived(d)
  const gxp = new Float64Array(d.N * d.C * Pp)
  const { ob, ko } = taps(d, P)
  for (let n = 0; n < d.N; n++)
    for (let o = 0; o < d.O; o++) {
      const g = Math.floor(o / Og)
      const yb = (n * d.O + o) * Yp
      for (let c = 0; c < Cg; c++) {
        const xb = (n * d.C + g * Cg + c) * Pp
        const wb = (o * Cg + c) * Kp
        for (let k = 0; k < Kp; k++) {
          const wv = w[wb + (d.flip ? Kp - 1 - k : k)]
          if (wv === 0) continue
          const off = xb + ko[k]
          for (let i = 0; i < Yp; i++) gxp[off + ob[i]] += wv * gy[yb + i]
        }
      }
    }
  return padOrCrop(gxp, d.N * d.C, d.S, P, d.lo, true)
}

/**
 * The kernel gradient by direct summation: for each tap, the inner product of the output gradient with the input it
 * multiplies, summed over the images.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param d The dimensions of the convolution whose kernel adjoint this is.
 * @returns The kernel gradient, row-major `[O, C/groups, ...K]`.
 */
function directWeight(x: Float64Array, gy: Float64Array, d: ConvDims): Float64Array {
  const { P, Cg, Og, Kp, Yp, Pp } = derived(d)
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const { ob, ko } = taps(d, P)
  const gw = new Float64Array(d.O * Cg * Kp)
  for (let n = 0; n < d.N; n++)
    for (let o = 0; o < d.O; o++) {
      const g = Math.floor(o / Og)
      const yb = (n * d.O + o) * Yp
      for (let c = 0; c < Cg; c++) {
        const xb = (n * d.C + g * Cg + c) * Pp
        const wb = (o * Cg + c) * Kp
        for (let k = 0; k < Kp; k++) {
          const off = xb + ko[k]
          let s = 0
          for (let i = 0; i < Yp; i++) s += gy[yb + i] * xp[off + ob[i]]
          gw[wb + (d.flip ? Kp - 1 - k : k)] += s
        }
      }
    }
  return gw
}

// ── FFT ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The $n$-dimensional DFT in place (the inverse unscaled) through the `fftn`/`ifftn` primitives over every axis of the
 * grid $L$: the separate real and imaginary parts are packed into one complex128 tensor and unpacked again.
 *
 * @param re The real part, row-major over the grid, $\prod_k L_k$ values; overwritten with the transform's.
 * @param im The imaginary part, laid out as `re`; overwritten with the transform's.
 * @param L The grid: the length of each axis transformed.
 * @param inverse False for the forward DFT, true for the inverse without the $1 / \prod_k L_k$ factor (callers
 *   divide by it themselves).
 */
function fftN(re: Float64Array, im: Float64Array, L: readonly number[], inverse: boolean): void {
  const packed = new Float64Array(2 * re.length)
  for (let k = 0; k < re.length; k++) {
    packed[2 * k] = re[k]
    packed[2 * k + 1] = im[k]
  }
  const x = fromData(packed, [...L], 'complex128')
  const y = (inverse ? ifftn(x, { norm: 'forward' }) : fftn(x)).data as Float64Array
  for (let k = 0; k < re.length; k++) {
    re[k] = y[2 * k]
    im[k] = y[2 * k + 1]
  }
}

/** A complex array on an FFT grid: `re` and `im`, its real and imaginary parts, each row-major over the grid. */
type Spectrum = { re: Float64Array; im: Float64Array }

/**
 * The spectra on the grid $L$ of `count` arrays of extent $E$ stored consecutively in `src`, each zero-filled to the
 * grid with its element $a$ placed at $a \cdot \mathrm{step}$, or at $(E - 1 - a) \cdot \mathrm{step}$ with
 * `reverse` (both per axis): a dilated or upsampled, possibly reversed, copy.
 *
 * @param src The arrays, consecutive and row-major, `count` of extent `E`. Not modified.
 * @param count The number of arrays.
 * @param E The extent of each array, per axis.
 * @param L The FFT grid, per axis; large enough to hold $(E - 1) \cdot \mathrm{step} + 1$ on every axis.
 * @param step The spacing on the grid of consecutive elements, per axis (a dilation or a stride).
 * @param reverse Whether each array is placed reversed on every axis.
 * @returns One spectrum per array.
 */
function spectra(
  src: Float64Array,
  count: number,
  E: readonly number[],
  L: readonly number[],
  step: readonly number[],
  reverse: boolean,
): Spectrum[] {
  const Ep = prod(E)
  const Lp = prod(L)
  const at = offsets(E, step, rowStrides(L))
  return Array.from({ length: count }, (_, c) => {
    const re = new Float64Array(Lp)
    const im = new Float64Array(Lp)
    for (let k = 0; k < Ep; k++) re[at[reverse ? Ep - 1 - k : k]] = src[c * Ep + k]
    fftN(re, im, L, false)
    return { re, im }
  })
}

/**
 * Multiply-accumulate of spectra, $\mathrm{acc} \leftarrow \mathrm{acc} + a b$, complex and pointwise.
 *
 * @param acc The accumulator; added to in place.
 * @param a The first factor, on the same grid. Not modified.
 * @param b The second factor, on the same grid. Not modified.
 */
function mac(acc: Spectrum, a: Spectrum, b: Spectrum): void {
  for (let k = 0; k < acc.re.length; k++) {
    acc.re[k] += a.re[k] * b.re[k] - a.im[k] * b.im[k]
    acc.im[k] += a.re[k] * b.im[k] + a.im[k] * b.re[k]
  }
}

/**
 * A spectrum of zeros.
 *
 * @param Lp The number of grid points.
 * @returns New zeroed real and imaginary parts of that length.
 */
const zeroSpectrum = (Lp: number): Spectrum => ({ re: new Float64Array(Lp), im: new Float64Array(Lp) })

/**
 * The FFT grid for a linear (not circular) convolution of extents $A$ and $B$: per axis, the smallest power of two
 * $\ge A + B - 1$.
 *
 * @param A The extent of the first operand, per axis.
 * @param B The extent of the second operand, per axis.
 * @returns The grid, per axis.
 */
const grid = (A: readonly number[], B: readonly number[]) => A.map((a, k) => nextPowerOfTwo(a + B[k] - 1))

/**
 * Add to `full` the full linear convolutions, by FFT, of the padded inputs `xp` with the dilated kernels, summed over
 * each group's channels:
 * $\mathrm{full}_{n,o,j} \mathrel{+}= \sum_{c, a} x_{n,\, c,\, j - (K_e - 1) + ad}\, w_{o,\, c',\, \hat a}$.
 * The kernels are placed on the grid reversed when not `flip` (cross-correlation) and as given when `flip`, so that
 * the strided valid part of `full` is the convolution.
 *
 * @param xp The padded inputs, row-major, $N C$ arrays of extent `Pb`. Not modified.
 * @param Pb The extent of each padded input (a block of it, in overlap-add).
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution.
 * @param full Where the results are added: $N O$ arrays of extent `F`, each result of extent $P_b + K_e - 1$ added at
 *   its start; added to in place.
 * @param F The extent of each array of `full`.
 * @param shift The offset, in elements of the last axis, at which each result is added (the block's start in
 *   overlap-add; 0 otherwise).
 */
function accumulateFull(
  xp: Float64Array,
  Pb: readonly number[],
  w: Float64Array,
  d: ConvDims,
  full: Float64Array,
  F: readonly number[],
  shift: number,
): void {
  const { Ke, Cg, Og } = derived(d)
  const L = grid(Pb, Ke)
  const Lp = prod(L)
  const X = spectra(
    xp,
    d.N * d.C,
    Pb,
    L,
    Pb.map(() => 1),
    false,
  )
  const W = spectra(w, d.O * Cg, d.K, L, d.dilation, !d.flip)
  const E = Pb.map((p, k) => p + Ke[k] - 1)
  const fromGrid = offsets(
    E,
    E.map(() => 1),
    rowStrides(L),
  )
  const intoFull = offsets(
    E,
    E.map(() => 1),
    rowStrides(F),
  )
  const Fp = prod(F)
  for (let n = 0; n < d.N; n++)
    for (let o = 0; o < d.O; o++) {
      const g = Math.floor(o / Og)
      const acc = zeroSpectrum(Lp)
      for (let c = 0; c < Cg; c++) mac(acc, X[n * d.C + g * Cg + c], W[o * Cg + c])
      fftN(acc.re, acc.im, L, true)
      const base = (n * d.O + o) * Fp + shift
      for (let k = 0; k < fromGrid.length; k++) full[base + intoFull[k]] += acc.re[fromGrid[k]] / Lp
    }
}

/**
 * The strided valid part of full convolutions: $y_i = \mathrm{full}_{is + K_e - 1}$ per spatial axis.
 *
 * @param full The full convolutions, row-major, $N O$ arrays of extent `F`, as `accumulateFull` leaves them.
 * @param F The extent of each full convolution, $P + K_e - 1$ per axis.
 * @param d The dimensions of the convolution.
 * @returns The output, row-major `[N, O, ...Y]`.
 */
function extractForward(full: Float64Array, F: readonly number[], d: ConvDims): Float64Array {
  const { Ke, Yp } = derived(d)
  const Fst = rowStrides(F)
  const at = offsets(d.Y, d.stride, Fst)
  const shift = Ke.reduce((acc, k, j) => acc + (k - 1) * Fst[j], 0)
  const Fp = prod(F)
  const y = new Float64Array(d.N * d.O * Yp)
  for (let m = 0; m < d.N * d.O; m++) for (let i = 0; i < Yp; i++) y[m * Yp + i] = full[m * Fp + shift + at[i]]
  return y
}

/**
 * The convolution by FFT: the padded input's full convolution with the kernels on one grid per call, then its strided
 * valid part.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution.
 * @returns The output, row-major `[N, O, ...Y]`.
 */
function fftForward(x: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Ke } = derived(d)
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const F = P.map((p, k) => p + Ke[k] - 1)
  const full = new Float64Array(d.N * d.O * prod(F))
  accumulateFull(xp, P, w, d, full, F, 0)
  return extractForward(full, F, d)
}

/**
 * The convolution by overlap-add: the padded input cut into blocks along its last axis, each block's full convolution
 * by FFT on a grid a few times the kernel's length (at least 64 and 8 times the effective kernel extent, rounded up to
 * a power of two), added into the full result at the block's offset; then its strided valid part.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution.
 * @returns The output, row-major `[N, O, ...Y]`.
 */
function overlapAddForward(x: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Ke } = derived(d)
  const D = P.length
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const F = P.map((p, k) => p + Ke[k] - 1)
  const full = new Float64Array(d.N * d.O * prod(F))
  const kl = Ke[D - 1]
  const block = Math.max(1, nextPowerOfTwo(Math.max(8 * kl, 64)) - kl + 1)
  const rows = (d.N * d.C * prod(P)) / P[D - 1]
  for (let b0 = 0; b0 < P[D - 1]; b0 += block) {
    const len = Math.min(block, P[D - 1] - b0)
    const Pb = [...P.slice(0, D - 1), len]
    const xb = new Float64Array(rows * len)
    for (let r = 0; r < rows; r++) xb.set(xp.subarray(r * P[D - 1] + b0, r * P[D - 1] + b0 + len), r * len)
    accumulateFull(xb, Pb, w, d, full, F, b0)
  }
  return extractForward(full, F, d)
}

/**
 * The transposed convolution by FFT: the stride-upsampled output gradient convolved with the dilated kernels (summed
 * over each group's output channels), written into the padded input and cropped to $S$.
 *
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution whose input adjoint this is.
 * @returns The input gradient, row-major `[N, C, ...S]`.
 */
function fftInput(gy: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Ke, Cg, Og, Pp } = derived(d)
  const U = d.Y.map((y, k) => (y - 1) * d.stride[k] + 1)
  const L = grid(U, Ke)
  const Lp = prod(L)
  const G = spectra(gy, d.N * d.O, d.Y, L, d.stride, false)
  const W = spectra(w, d.O * Cg, d.K, L, d.dilation, d.flip)
  const E = U.map((u, k) => Math.min(u + Ke[k] - 1, P[k]))
  const fromGrid = offsets(
    E,
    E.map(() => 1),
    rowStrides(L),
  )
  const intoP = offsets(
    E,
    E.map(() => 1),
    rowStrides(P),
  )
  const gxp = new Float64Array(d.N * d.C * Pp)
  for (let n = 0; n < d.N; n++)
    for (let c = 0; c < d.C; c++) {
      const g = Math.floor(c / Cg)
      const acc = zeroSpectrum(Lp)
      for (let o = g * Og; o < (g + 1) * Og; o++) mac(acc, G[n * d.O + o], W[o * Cg + (c - g * Cg)])
      fftN(acc.re, acc.im, L, true)
      const base = (n * d.C + c) * Pp
      for (let k = 0; k < fromGrid.length; k++) gxp[base + intoP[k]] = acc.re[fromGrid[k]] / Lp
    }
  return padOrCrop(gxp, d.N * d.C, d.S, P, d.lo, true)
}

/**
 * The kernel gradient by FFT: the correlations of the padded input with the stride-upsampled output gradient, summed
 * over the images, read at the dilated tap positions $ad$.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param d The dimensions of the convolution whose kernel adjoint this is.
 * @returns The kernel gradient, row-major `[O, C/groups, ...K]`.
 */
function fftWeight(x: Float64Array, gy: Float64Array, d: ConvDims): Float64Array {
  const { P, Cg, Og, Kp } = derived(d)
  const U = d.Y.map((y, k) => (y - 1) * d.stride[k] + 1)
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const L = grid(P, U)
  const Lp = prod(L)
  const X = spectra(
    xp,
    d.N * d.C,
    P,
    L,
    P.map(() => 1),
    false,
  )
  const G = spectra(gy, d.N * d.O, d.Y, L, d.stride, true)
  const Lst = rowStrides(L)
  const at = offsets(d.K, d.dilation, Lst)
  const shift = U.reduce((acc, u, k) => acc + (u - 1) * Lst[k], 0)
  const gw = new Float64Array(d.O * Cg * Kp)
  for (let o = 0; o < d.O; o++) {
    const g = Math.floor(o / Og)
    for (let c = 0; c < Cg; c++) {
      const acc = zeroSpectrum(Lp)
      for (let n = 0; n < d.N; n++) mac(acc, X[n * d.C + g * Cg + c], G[n * d.O + o])
      fftN(acc.re, acc.im, L, true)
      const wb = (o * Cg + c) * Kp
      for (let k = 0; k < Kp; k++) gw[wb + (d.flip ? Kp - 1 - k : k)] = acc.re[shift + at[k]] / Lp
    }
  }
  return gw
}

// ── Choice ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The kernel a method asks for. `auto` chooses the FFT when its estimated operation count (transforms plus products)
 * is under half the direct kernel's, and direct otherwise; it never chooses overlap-add.
 *
 * @param method The method asked for.
 * @param d The dimensions of the convolution, which the costs are estimated from.
 * @returns `method` itself unless it is `auto`, then `direct` or `fft`.
 */
function resolve(method: ConvMethod, d: ConvDims): 'direct' | 'fft' | 'overlapAdd' {
  if (method !== 'auto') return method
  const { P, Ke, Cg, Kp, Yp } = derived(d)
  const direct = d.N * d.O * Cg * Yp * Kp
  const Lp = prod(grid(P, Ke))
  const transforms = (d.N * d.C + d.O * Cg + d.N * d.O) * Lp * Math.log2(Math.max(2, Lp)) * 3
  const fft = transforms + 4 * d.N * d.O * Cg * Lp
  return 2 * fft < direct ? 'fft' : 'direct'
}

/**
 * The convolution $y_{n,o} = \sum_c x_{n,c} \star w_{o,c'}$ ($c$ over the group of $o$), by the kernel `method`
 * resolves to.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the convolution.
 * @param method The kernel: `direct`, `fft`, `overlapAdd` or `auto`.
 * @returns The output, row-major `[N, O, ...Y]`.
 */
export function convForward(x: Float64Array, w: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  const m = resolve(method, d)
  return m === 'direct' ? directForward(x, w, d) : m === 'fft' ? fftForward(x, w, d) : overlapAddForward(x, w, d)
}

/**
 * The adjoint of the convolution in its input (the transposed convolution). Any method but `direct` (after `auto` is
 * resolved) runs the FFT kernel, as there is no overlap-add adjoint.
 *
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param w The kernels, row-major `[O, C/groups, ...K]`. Not modified.
 * @param d The dimensions of the forward convolution.
 * @param method The kernel: `direct`, `fft`, `overlapAdd` or `auto`.
 * @returns The input gradient, row-major `[N, C, ...S]`.
 */
export function convInputAdjoint(gy: Float64Array, w: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  return resolve(method, d) === 'direct' ? directInput(gy, w, d) : fftInput(gy, w, d)
}

/**
 * The adjoint of the convolution in its kernel (the weight gradient), summed over the images. Any method but `direct`
 * (after `auto` is resolved) runs the FFT kernel, as there is no overlap-add adjoint.
 *
 * @param x The input, row-major `[N, C, ...S]`. Not modified.
 * @param gy The output gradient, row-major `[N, O, ...Y]`. Not modified.
 * @param d The dimensions of the forward convolution.
 * @param method The kernel: `direct`, `fft`, `overlapAdd` or `auto`.
 * @returns The kernel gradient, row-major `[O, C/groups, ...K]`.
 */
export function convWeightAdjoint(x: Float64Array, gy: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  return resolve(method, d) === 'direct' ? directWeight(x, gy, d) : fftWeight(x, gy, d)
}
