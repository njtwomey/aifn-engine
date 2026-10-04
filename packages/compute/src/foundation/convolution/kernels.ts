/**
 * Raw kernels of the convolution family, on contiguous row-major Float64Arrays. Three maps share one trilinear form
 * T(x, w, y) = Σ x[n, c, i·s − lo + a·d] · w[o, c′, â] · y[n, o, i] over the taps of a convolution (c in the group of
 * o, â = a or its reversal when `flip`): `forward` contracts x and w (the convolution), `inputAdjoint` contracts w and
 * y (the transposed convolution), `weightAdjoint` contracts x and y (the kernel gradient). Each has a direct kernel
 * (zero-padded input and precomputed tap offsets, the output as the inner loop) and an FFT kernel (the convolution
 * theorem in n dimensions, channels contracted in the frequency domain); the forward also has overlap-add (blocks
 * along the last axis). `method` chooses among them and never changes the result beyond rounding.
 */

import { fftn, ifftn, nextPowerOfTwo } from 'aifn-compute/foundation/fourier'
import { fromData } from 'aifn-compute/foundation/tensor'

/** How the kernel computes a convolution: `direct` sums the taps; `fft` and `overlapAdd` use the convolution theorem. */
export type ConvMethod = 'direct' | 'fft' | 'overlapAdd' | 'auto'

/** The sizes of one convolution: x [N, C, ...S], w [O, C/groups, ...K], y [N, O, ...Y]; per spatial axis stride, dilation and zero padding. */
export type ConvDims = {
  readonly N: number
  readonly C: number
  readonly O: number
  readonly groups: number
  readonly S: readonly number[]
  readonly K: readonly number[]
  readonly Y: readonly number[]
  readonly stride: readonly number[]
  readonly dilation: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly flip: boolean
}

const prod = (a: readonly number[]): number => a.reduce((p, v) => p * v, 1)

function rowStrides(dims: readonly number[]): number[] {
  const s = new Array<number>(dims.length)
  for (let d = dims.length - 1, t = 1; d >= 0; d--) {
    s[d] = t
    t *= dims[d]
  }
  return s
}

/** Offsets Σ_d a_d·step_d·strides_d for every multi-index a over `extent`, in row-major order. */
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

/** Derived sizes: padded input P, effective kernel extent Ke = d(K − 1) + 1, channels per group. */
function derived(d: ConvDims) {
  const P = d.S.map((s, k) => s + d.lo[k] + d.hi[k])
  const Ke = d.K.map((k, j) => d.dilation[j] * (k - 1) + 1)
  return { P, Ke, Cg: d.C / d.groups, Og: d.O / d.groups, Kp: prod(d.K), Yp: prod(d.Y), Sp: prod(d.S), Pp: prod(P) }
}

/** Copy `count` arrays of extent S into zeroed arrays of extent P at offset lo (or back, with `crop`). */
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

/** The tap tables of the direct kernels: output-position and kernel-tap offsets into one padded channel. */
function taps(d: ConvDims, P: readonly number[]) {
  const st = rowStrides(P)
  return { ob: offsets(d.Y, d.stride, st), ko: offsets(d.K, d.dilation, st) }
}

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
 * The n-dimensional DFT in place (unscaled inverse) through the `fftn`/`ifftn` primitives over every axis of the grid
 * L: the separate real and imaginary parts are packed into one complex128 tensor and unpacked again.
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

type Spectrum = { re: Float64Array; im: Float64Array }

/**
 * The spectra (grid L) of `count` arrays of extent E stored consecutively in `src`, element a placed at a·step, or at
 * (E − 1 − a)·step with `reverse` (both per axis).
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

/** acc += a·b (complex, pointwise). */
function mac(acc: Spectrum, a: Spectrum, b: Spectrum): void {
  for (let k = 0; k < acc.re.length; k++) {
    acc.re[k] += a.re[k] * b.re[k] - a.im[k] * b.im[k]
    acc.im[k] += a.re[k] * b.im[k] + a.im[k] * b.re[k]
  }
}

const zeroSpectrum = (Lp: number): Spectrum => ({ re: new Float64Array(Lp), im: new Float64Array(Lp) })

/** The FFT grid for a linear (not circular) convolution of extents A and B: a power of two ≥ A + B − 1 per axis. */
const grid = (A: readonly number[], B: readonly number[]) => A.map((a, k) => nextPowerOfTwo(a + B[k] - 1))

/**
 * Add to `full` (count N·O, extent F) the full convolutions of padded inputs `xp` (extent Pb) with the reversed,
 * dilated kernels, summed over each group's channels, shifted by `shift` along the last axis.
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

/** The strided valid part of full convolutions: y[i] = full[i·s + Ke − 1]. */
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

function fftForward(x: Float64Array, w: Float64Array, d: ConvDims): Float64Array {
  const { P, Ke } = derived(d)
  const xp = padOrCrop(x, d.N * d.C, d.S, P, d.lo, false)
  const F = P.map((p, k) => p + Ke[k] - 1)
  const full = new Float64Array(d.N * d.O * prod(F))
  accumulateFull(xp, P, w, d, full, F, 0)
  return extractForward(full, F, d)
}

/**
 * Overlap-add: the padded input cut into blocks along its last axis, each block's full convolution by FFT on a grid
 * a few times the kernel's length, added into the full result at the block's offset.
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

/** The transposed convolution by FFT: the stride-upsampled y convolved with the dilated kernels, cropped to S. */
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

/** The kernel gradient by FFT: correlations of the padded x with the upsampled y, summed over the batch, read at a·d. */
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

/** `auto`: the FFT when its operation count (transforms plus products) is well below the direct kernel's. */
function resolve(method: ConvMethod, d: ConvDims): 'direct' | 'fft' | 'overlapAdd' {
  if (method !== 'auto') return method
  const { P, Ke, Cg, Kp, Yp } = derived(d)
  const direct = d.N * d.O * Cg * Yp * Kp
  const Lp = prod(grid(P, Ke))
  const transforms = (d.N * d.C + d.O * Cg + d.N * d.O) * Lp * Math.log2(Math.max(2, Lp)) * 3
  const fft = transforms + 4 * d.N * d.O * Cg * Lp
  return 2 * fft < direct ? 'fft' : 'direct'
}

/** The convolution y[n, o] = Σ_c x[n, c] ⋆ w[o, c]: shape [N, O, ...Y]. */
export function convForward(x: Float64Array, w: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  const m = resolve(method, d)
  return m === 'direct' ? directForward(x, w, d) : m === 'fft' ? fftForward(x, w, d) : overlapAddForward(x, w, d)
}

/** The adjoint in the input (the transposed convolution): shape [N, C, ...S]. */
export function convInputAdjoint(gy: Float64Array, w: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  return resolve(method, d) === 'direct' ? directInput(gy, w, d) : fftInput(gy, w, d)
}

/** The adjoint in the kernel (the weight gradient): shape [O, C/groups, ...K]. */
export function convWeightAdjoint(x: Float64Array, gy: Float64Array, d: ConvDims, method: ConvMethod): Float64Array {
  return resolve(method, d) === 'direct' ? directWeight(x, gy, d) : fftWeight(x, gy, d)
}
