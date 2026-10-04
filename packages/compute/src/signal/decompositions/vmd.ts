/**
 * Variational mode decomposition (Dragomiretskiy & Zosso, 2014, "Variational mode decomposition", IEEE Trans. Signal
 * Process. 62(3)): K band-limited modes, each compact around a centre frequency, found together by ADMM in the
 * Fourier domain. `vmdSteps` is one ADMM sweep per step (traceable); `vmd` runs it and returns a `Decomposition`.
 */

import { fft, fftshift, ifft, ifftshift } from 'aifn-compute/foundation/fourier'
import { fromData, readonlyData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Decomposition, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { readSamples, type SignalInput } from '../signal'

/** Options for `vmd` and `vmdSteps`. */
export type VmdOptions = {
  /** The number of modes K. */
  modes: Size
  /** The bandwidth penalty α (larger: narrower modes). Default 2000. */
  alpha?: Scalar
  /** The dual ascent step τ (0 lets the modes absorb noise: no exact reconstruction is enforced). Default 0. */
  tau?: Scalar
  /** Hold the first mode at zero frequency (a trend). Default false. */
  dc?: boolean
  /**
   * The initial centre frequencies (cycles per sample, in [0, ½]): `'zero'`, `'uniform'` (ωₖ = k/(2K), the
   * default), or a `Stream` to draw them log-uniformly, as the reference code's `init = 2`.
   */
  init?: 'zero' | 'uniform' | Stream
  /** Stop when Σₖ ‖ûₖⁿ⁺¹ − ûₖⁿ‖²/T falls below this. Default 1e-7. */
  tolerance?: Scalar
}

/** A state of `vmdSteps`. */
export interface VmdState extends Status {
  /** ADMM sweeps made. */
  t: Size
  /** The modes' one-sided spectra on the mirrored signal: complex128 [K, T], fftshifted (index T/2 is 0). */
  spectra: Tensor
  /** The centre frequencies ωₖ in cycles per sample (length K). */
  omega: Vector
  /** The Lagrange multiplier's spectrum λ̂ (complex128 [T]). */
  multiplier: Tensor
  /** The update size Σₖ ‖ûₖⁿ⁺¹ − ûₖⁿ‖²/T of the last sweep (∞ at the start). */
  change: Scalar
  /** The update size fell below the tolerance. */
  converged: boolean
}

/** The mirrored signal (half of each end reflected, as the reference code) and its fftshifted spectrum, zero for f < 0. */
function prepare(x: Float64Array) {
  const v = x.length % 2 ? x.subarray(0, x.length - 1) : x
  const n = v.length
  const half = n >> 1
  const T = 2 * n
  const m = new Float64Array(T)
  for (let i = 0; i < half; i++) m[i] = v[half - 1 - i]
  m.set(v, half)
  for (let i = 0; i < half; i++) m[half + n + i] = v[n - 1 - i]
  const spec = readonlyData(fftshift(fft(m)))!
  const fPlus = new Float64Array(2 * T)
  for (let i = T >> 1; i < T; i++) {
    fPlus[2 * i] = spec[2 * i]
    fPlus[2 * i + 1] = spec[2 * i + 1]
  }
  // freqs = (1..T)/T − ½ − 1/T: the fftshifted grid, 0 at index T/2.
  const freqs = Float64Array.from({ length: T }, (_, i) => (i + 1) / T - 0.5 - 1 / T)
  return { n, T, fPlus, freqs }
}

/**
 * VMD as a traceable algorithm. The signal (trimmed to even length n and mirrored to T = 2n) is split into K modes
 * minimising Σₖ ‖∂ₜ[(δ + j/πt) ∗ uₖ] e^{−jωₖt}‖² (each mode's analytic signal, shifted to baseband, has little
 * bandwidth) subject to Σₖ uₖ = f. Each step is one ADMM sweep over the modes (Gauss–Seidel), in the Fourier domain:
 *
 *   ûₖ ← (f̂ − Σ_{i≠k} ûᵢ − λ̂/2) / (1 + 2α(ω − ωₖ)²)   (a Wiener filter centred on ωₖ; the reference code's α absorbs the 2)
 *   ωₖ ← ∫₀^∞ ω|ûₖ(ω)|² dω / ∫₀^∞ |ûₖ(ω)|² dω          (the centre of gravity of the mode's power)
 *   λ̂ ← λ̂ + τ(Σₖ ûₖ − f̂)
 *
 * as the authors' reference code (ported by vmdpy), whose denominator is 1 + α(ω − ωₖ)². `init` takes no start.
 */
export function vmdSteps(x: SignalInput, options: VmdOptions): Algorithm<undefined, VmdState> {
  const { modes: K, alpha = 2000, tau = 0, dc = false, init = 'uniform', tolerance = 1e-7 } = options
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError('vmd', 'vmd: modes must be a positive integer')
  if (!(alpha > 0)) throw new DomainError('vmd', 'vmd: alpha must be positive')
  const values = readSamples(x, 'vmd').values
  if (values.length < 4) throw new DomainError('vmd', 'vmd: need at least 4 samples')
  const { T, fPlus, freqs } = prepare(values)
  const omega0 = new Float64Array(K)
  if (init === 'uniform') for (let k = 0; k < K; k++) omega0[k] = (0.5 / K) * k
  else if (typeof init === 'object') {
    // Log-uniform on [1/fs, ½] with fs = 1/n samples, sorted (the reference code's init = 2).
    const u = units(child(init, 'omega'), K)
    const fs = 1 / values.length
    for (let k = 0; k < K; k++) omega0[k] = Math.exp(Math.log(fs) + (Math.log(0.5) - Math.log(fs)) * u[k])
    omega0.sort()
  }
  if (dc) omega0[0] = 0
  const lo = T >> 1
  return {
    name: 'vmd',
    init: () => ({
      t: 0,
      spectra: fromData(new Float64Array(2 * K * T), [K, T], 'complex128'),
      omega: fromData(omega0, [K]),
      multiplier: fromData(new Float64Array(2 * T), [T], 'complex128'),
      change: Infinity,
      converged: false,
    }),
    step: (s) => {
      const u = readonlyData(s.spectra)!
      const om = readonlyData(s.omega)!
      const lam = readonlyData(s.multiplier)!
      const next = new Float64Array(2 * K * T)
      const omega = Float64Array.from(om)
      // Σ_{i≠k} ûᵢ with the newest values (Gauss–Seidel): start from Σ_{i≥1} ûᵢⁿ.
      const others = new Float64Array(2 * T)
      for (let k = 1; k < K; k++) for (let i = 0; i < 2 * T; i++) others[i] += u[2 * k * T + i]
      for (let k = 0; k < K; k++) {
        if (k > 0) for (let i = 0; i < 2 * T; i++) others[i] += next[2 * (k - 1) * T + i] - u[2 * k * T + i]
        const wk = omega[k]
        let num = 0
        let den = 0
        for (let i = 0; i < T; i++) {
          const g = 1 / (1 + alpha * (freqs[i] - wk) ** 2)
          const re = (fPlus[2 * i] - others[2 * i] - lam[2 * i] / 2) * g
          const im = (fPlus[2 * i + 1] - others[2 * i + 1] - lam[2 * i + 1] / 2) * g
          next[2 * (k * T + i)] = re
          next[2 * (k * T + i) + 1] = im
          if (i >= lo) {
            const p = re * re + im * im
            num += freqs[i] * p
            den += p
          }
        }
        if (!(dc && k === 0)) omega[k] = den > 0 ? num / den : omega[k]
      }
      const multiplier = Float64Array.from(lam)
      if (tau !== 0)
        for (let i = 0; i < T; i++) {
          let re = -fPlus[2 * i]
          let im = -fPlus[2 * i + 1]
          for (let k = 0; k < K; k++) {
            re += next[2 * (k * T + i)]
            im += next[2 * (k * T + i) + 1]
          }
          multiplier[2 * i] += tau * re
          multiplier[2 * i + 1] += tau * im
        }
      let change = Number.EPSILON
      for (let i = 0; i < 2 * K * T; i++) change += (next[i] - u[i]) ** 2 / T
      return {
        t: s.t + 1,
        spectra: fromData(next, [K, T], 'complex128'),
        omega: fromData(omega, [K]),
        multiplier: fromData(multiplier, [T], 'complex128'),
        change,
        converged: change <= tolerance,
      }
    },
    done: (s) => s.converged,
  }
}

/**
 * The time-domain modes of a VMD state: each mode's one-sided spectrum is completed by Hermitian symmetry, inverted,
 * and the middle n samples (the unmirrored signal) kept. Returns K arrays of length n.
 */
export function vmdModes(s: VmdState): Float64Array[] {
  const [K, T] = s.spectra.shape
  const u = readonlyData(s.spectra)!
  const half = T >> 1
  const out: Float64Array[] = []
  for (let k = 0; k < K; k++) {
    const full = new Float64Array(2 * T)
    for (let i = half; i < T; i++) {
      full[2 * i] = u[2 * (k * T + i)]
      full[2 * i + 1] = u[2 * (k * T + i) + 1]
    }
    // Mirror the positive half onto indices T/2 … 1 (conjugated), then index 0 from the last, as the reference code.
    for (let j = 0; j < half; j++) {
      const from = half + j
      const to = half - j
      full[2 * to] = u[2 * (k * T + from)]
      full[2 * to + 1] = -u[2 * (k * T + from) + 1]
    }
    full[0] = full[2 * (T - 1)]
    full[1] = -full[2 * (T - 1) + 1]
    const time = readonlyData(ifft(ifftshift(fromData(full, [T], 'complex128'))))!
    const n = T / 2
    const mode = new Float64Array(n)
    for (let i = 0; i < n; i++) mode[i] = time[2 * (T / 4 + i)]
    out.push(mode)
  }
  return out
}

/**
 * Variational mode decomposition of a real signal into K modes, as a `Decomposition` (`method: 'vmd'`), modes sorted
 * by centre frequency (slowest first). Each component carries `meta: { centreFrequency }` in the signal's units (Hz
 * for a `Signal`, cycles per sample for bare samples); `residual` is f − Σₖ uₖ (small when τ > 0 or α is small, not
 * zero: VMD trades reconstruction for narrow bands). An odd-length signal loses its last sample, as the reference
 * code. Stops at convergence or after `maxSteps` sweeps (default 500); `meta` on the result is not set, so read
 * `vmdSteps` for the convergence trace.
 */
export function vmd(x: SignalInput, options: VmdOptions & { maxSteps?: Size }): Decomposition {
  const input = readSamples(x, 'vmd')
  const alg = vmdSteps(x, options)
  // As the reference code: at most maxSteps − 1 sweeps.
  const s = run(alg, undefined, (options.maxSteps ?? 500) - 1)
  const modes = vmdModes(s)
  const n = modes[0].length
  const omega = readonlyData(s.omega)!
  const order = Array.from(omega, (_, k) => k).sort((a, b) => omega[a] - omega[b])
  const residual = Float64Array.from(input.values.subarray(0, n))
  for (const m of modes) for (let i = 0; i < n; i++) residual[i] -= m[i]
  const vec = (a: Float64Array) => fromData(a, [a.length])
  return {
    kind: 'decomposition',
    method: 'vmd',
    axis: vec(Float64Array.from({ length: n }, (_, k) => input.t0 + k / input.fs)),
    components: order.map((k, j) => ({
      name: `mode ${j + 1}`,
      values: vec(modes[k]),
      meta: { centreFrequency: omega[k] * input.fs, iterations: s.t, converged: s.converged },
    })),
    residual: vec(residual),
    original: vec(Float64Array.from(input.values.subarray(0, n))),
  }
}
