/**
 * Optimal (minimax) linear-phase FIR design by the Parks–McClellan algorithm (Parks and McClellan, 1972, IEEE Trans.
 * Circuit Theory 19(2)), as `scipy.signal.remez` with `type='bandpass'`: the Remez exchange on a dense frequency grid,
 * with the amplitude interpolated in barycentric form through the current extremal set (McClellan, Parks and Rabiner,
 * 1973). Types I (odd length) and II (even length, zero at Nyquist) are designed.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, NumericalError } from 'aifn-compute/foundation/errors'
import { transferFunction, type LtiOf, type TransferFunctionForm } from 'aifn-compute/systems'

/** Options for `remez`. */
export interface RemezOptions {
  /** One weight per band (default 1 each): the error in band b is weighted by `weight[b]`. */
  weight?: readonly Scalar[]
  /** Sampling frequency (default 1: band edges in cycles per sample, up to 0.5). The system gets dt = 1/fs. */
  fs?: Scalar
  /** Grid points per extremal frequency (default 16, as scipy). */
  gridDensity?: Size
  /** The most exchange iterations (default 40). */
  maxiter?: Size
}

/** The design and how it went: the taps, the minimax error δ, the extremal frequencies and the iteration count. */
export type RemezResult = LtiOf<TransferFunctionForm> & {
  /** The weighted ripple |δ| at convergence. */
  readonly ripple: Scalar
  /** The extremal frequencies of the final alternation, in the units of `fs`. */
  readonly extremals: Tensor
  readonly iterations: Size
}

/**
 * A linear-phase FIR filter of `numtaps` taps whose weighted error from a piecewise-constant desired amplitude is
 * minimax over the bands, as `scipy.signal.remez(numtaps, bands, desired, weight, fs=fs)`. `bands` holds edge pairs
 * [f₀, f₁, f₂, f₃, …] (increasing, within [0, fs/2]) and `desired` one gain per band; the gaps between bands are don't
 * care regions. By the alternation theorem the optimum error equiripples with r + 1 extrema, r the number of cosine
 * terms. An even `numtaps` (type II) has a zero at Nyquist, so it cannot design a high-pass.
 */
export function remez(
  numtaps: Size,
  bands: readonly Scalar[],
  desired: readonly Scalar[],
  options: RemezOptions = {},
): RemezResult {
  const { fs = 1, gridDensity = 16, maxiter = 40 } = options
  const weight = options.weight ?? desired.map(() => 1)
  if (!(Number.isInteger(numtaps) && numtaps >= 3))
    throw new DomainError('remez', 'remez: numtaps must be an integer ≥ 3')
  if (bands.length % 2 || bands.length / 2 !== desired.length || weight.length !== desired.length)
    throw new DomainError('remez', 'remez: bands must be edge pairs, with one desired gain and one weight per band')
  const edges = bands.map((f) => f / fs)
  for (let i = 1; i < edges.length; i++)
    if (!(edges[i] >= edges[i - 1])) throw new DomainError('remez', 'remez: band edges must increase')
  if (edges[0] < 0 || edges[edges.length - 1] > 0.5)
    throw new DomainError('remez', 'remez: band edges must lie in [0, fs/2]')
  const odd = numtaps % 2 === 1
  const r = odd ? (numtaps + 1) / 2 : numtaps / 2
  // Type II: A(ω) = cos(ω/2) P(ω), so P approximates D / cos(ω/2) with weight W cos(ω/2); ω = π is left out.
  const step = 0.5 / (gridDensity * r)
  const grid: number[] = []
  const D: number[] = []
  const W: number[] = []
  // The grid of scipy's C code (Janovetz): each band from its lower edge in steps of `step`, its last point moved onto
  // the upper edge. Type II keeps ω = π out (A(π) = 0) unless the desired gain there is zero.
  for (let b = 0; b < desired.length; b++) {
    let f = edges[2 * b]
    let hi = edges[2 * b + 1]
    if (!odd && desired[b] !== 0 && hi > 0.5 - step) hi = 0.5 - step
    const start = grid.length
    while (f <= hi) {
      grid.push(f)
      f += step
    }
    if (grid.length === start) grid.push(hi)
    else grid[grid.length - 1] = hi
    for (let i = start; i < grid.length; i++) {
      const c = odd ? 1 : Math.cos(Math.PI * grid[i])
      D.push(desired[b] / c)
      W.push(weight[b] * c)
    }
  }
  const G = grid.length
  if (G < r + 1) throw new DomainError('remez', 'remez: the bands are too narrow for this many taps')
  const x = grid.map((f) => Math.cos(2 * Math.PI * f))
  // Initial extremal set: r + 1 grid points spread evenly.
  let ext = Array.from({ length: r + 1 }, (_, i) => Math.floor((i * (G - 1)) / r))
  let delta = 0
  let iterations = 0
  const E = new Float64Array(G)
  let interp = { xs: [] as number[], ys: [] as number[], ws: [] as number[] }
  for (; iterations < maxiter; iterations++) {
    // Barycentric weights over the r + 1 extremals give δ; the first r define the interpolant.
    const xe = ext.map((i) => x[i])
    const bw = baryWeights(xe)
    let num = 0
    let den = 0
    ext.forEach((i, k) => {
      num += bw[k] * D[i]
      den += (bw[k] * (k % 2 ? -1 : 1)) / W[i]
    })
    delta = num / den
    const xs = xe.slice(0, r)
    const ws = baryWeights(xs)
    const ys = ext.slice(0, r).map((i, k) => D[i] - ((k % 2 ? -1 : 1) * delta) / W[i])
    interp = { xs, ys, ws }
    for (let g = 0; g < G; g++) E[g] = W[g] * (D[g] - barycentric(xs, ys, ws, x[g]))
    const next = extremals(E, Math.abs(delta), r + 1)
    if (next === null) break
    const same = next.length === ext.length && next.every((v, k) => v === ext[k])
    ext = next
    if (same) break
    let maxE = 0
    for (const i of ext) maxE = Math.max(maxE, Math.abs(E[i]))
    if (maxE - Math.abs(delta) <= 1e-12 * Math.max(1, maxE)) break
  }
  if (!Number.isFinite(delta)) throw new NumericalError('remez', 'remez: the exchange broke down', 'not-converged')
  // The impulse response from the amplitude at numtaps equally spaced frequencies: h[n] = (1/N) Σₘ A(ωₘ) cos(ωₘ(n − c)).
  const N = numtaps
  const A = new Float64Array(N)
  for (let m = 0; m < N; m++) {
    const w = (2 * Math.PI * m) / N
    const p = barycentric(interp.xs, interp.ys, interp.ws, Math.cos(w))
    A[m] = odd ? p : Math.cos(w / 2) * p
  }
  const c = (N - 1) / 2
  const h = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    let s = 0
    for (let m = 0; m < N; m++) s += A[m] * Math.cos(((2 * Math.PI * m) / N) * (n - c))
    h[n] = s / N
  }
  return Object.assign(transferFunction(h, [1], { dt: 1 / fs }), {
    ripple: Math.abs(delta),
    extremals: fromData(
      Float64Array.from(ext, (i) => grid[i] * fs),
      [ext.length],
    ),
    iterations,
  })
}

/** Barycentric weights 1/Πⱼ≠ₖ (xₖ − xⱼ), rescaled to avoid overflow (only ratios matter). */
function baryWeights(xs: readonly number[]): number[] {
  const n = xs.length
  return xs.map((xk, k) => {
    let p = 1
    // Scaling each factor by 2 keeps the product in range for n in the hundreds (the nodes lie in [−1, 1]).
    for (let j = 0; j < n; j++) if (j !== k) p *= 2 * (xk - xs[j])
    return 1 / p
  })
}

/** The barycentric (second form) interpolant through (xs, ys) at x. */
function barycentric(xs: readonly number[], ys: readonly number[], ws: readonly number[], x: number): number {
  let num = 0
  let den = 0
  for (let k = 0; k < xs.length; k++) {
    const d = x - xs[k]
    if (d === 0) return ys[k]
    const t = ws[k] / d
    num += t * ys[k]
    den += t
  }
  return num / den
}

/**
 * The next extremal set: local extrema of the error E with |E| ≥ |δ|, one per run of equal sign (the largest), then
 * trimmed from whichever end is smaller until `count` remain. Null when too few alternations are left.
 */
function extremals(E: Float64Array, delta: number, count: Size): number[] | null {
  const G = E.length
  const candidates: number[] = []
  for (let i = 0; i < G; i++) {
    const v = Math.abs(E[i])
    if (v < delta * (1 - 1e-6)) continue
    const left = i === 0 || Math.abs(E[i - 1]) <= v || Math.sign(E[i - 1]) !== Math.sign(E[i])
    const right = i === G - 1 || Math.abs(E[i + 1]) < v || Math.sign(E[i + 1]) !== Math.sign(E[i])
    if (left && right) candidates.push(i)
  }
  const alternating: number[] = []
  for (const i of candidates) {
    const last = alternating[alternating.length - 1]
    if (last !== undefined && Math.sign(E[last]) === Math.sign(E[i])) {
      if (Math.abs(E[i]) > Math.abs(E[last])) alternating[alternating.length - 1] = i
    } else alternating.push(i)
  }
  if (alternating.length < count) return null
  while (alternating.length > count) {
    if (Math.abs(E[alternating[0]]) < Math.abs(E[alternating[alternating.length - 1]])) alternating.shift()
    else alternating.pop()
  }
  return alternating
}

/** Options of `equiripple`. */
export interface EquirippleOptions {
  btype?: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** Sampling frequency (default 2: edges as fractions of Nyquist, as `firwin`). */
  fs?: Scalar
  /** Stopband weight relative to the passband (default 1). */
  stopWeight?: Scalar
}

/**
 * An equiripple FIR filter specified like `firwin`: band type, cutoff edge(s) and a transition width, the bands built
 * for `remez` with the passband ending (or starting) at each cutoff and the stopband a transition width beyond it. A
 * high-pass or band-stop needs an odd `numtaps`.
 */
export function equiripple(
  numtaps: Size,
  cutoff: Scalar | readonly [Scalar, Scalar],
  transition: Scalar,
  options: EquirippleOptions = {},
): RemezResult {
  const fs = options.fs ?? 2
  const btype = options.btype ?? (typeof cutoff === 'number' ? 'lowpass' : 'bandpass')
  const ny = fs / 2
  const c = (v: number) => Math.min(ny, Math.max(0, v))
  const [f1, f2] = typeof cutoff === 'number' ? [cutoff, cutoff] : cutoff
  const tw = transition
  const spec: Record<NonNullable<EquirippleOptions['btype']>, [number[], number[]]> = {
    lowpass: [
      [0, c(f1), c(f1 + tw), ny],
      [1, 0],
    ],
    highpass: [
      [0, c(f1 - tw), c(f1), ny],
      [0, 1],
    ],
    bandpass: [
      [0, c(f1 - tw), c(f1), c(f2), c(f2 + tw), ny],
      [0, 1, 0],
    ],
    bandstop: [
      [0, c(f1), c(f1 + tw), c(f2 - tw), c(f2), ny],
      [1, 0, 1],
    ],
  }
  const [bands, desired] = spec[btype]
  const w = options.stopWeight ?? 1
  return remez(numtaps, bands, desired, { fs, weight: desired.map((d) => (d === 0 ? w : 1)) })
}
