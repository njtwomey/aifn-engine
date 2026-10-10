/**
 * Variational mode decomposition (Dragomiretskiy and Zosso, 2014, "Variational mode decomposition", IEEE Trans. Signal
 * Process. 62(3)): $K$ band-limited modes $u_k$, each compact around a centre frequency $\omega_k$, found together by
 * ADMM in the Fourier domain. `vmdSteps` is one ADMM sweep per step (traceable); `vmd` runs it and returns a
 * `Decomposition`.
 *
 * The port follows the authors' reference code (as vmdpy does): the signal is trimmed to even length $n$ and mirrored
 * to $T = 2n$ samples, the modes live as one-sided spectra on the mirrored signal's frequency grid, and frequencies are
 * in cycles per sample, from 0 to $\tfrac{1}{2}$. Nothing is differentiable: the results are plain tensors.
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
  /** The number of modes $K$, a positive integer. */
  modes: Size
  /** The bandwidth penalty $\alpha > 0$ (larger: narrower modes). Default 2000. */
  alpha?: Scalar
  /** The dual ascent step $\tau$ (0 lets the modes absorb noise: no exact reconstruction is enforced). Default 0. */
  tau?: Scalar
  /** Hold the first mode at zero frequency (a trend). Default false. */
  dc?: boolean
  /**
   * The initial centre frequencies (cycles per sample, in $[0, \tfrac{1}{2}]$): `'zero'`, `'uniform'`
   * ($\omega_k = k / (2K)$ for $k = 0, \dots, K - 1$, the default), or a `Stream` to draw them log-uniformly on
   * $[1/n, \tfrac{1}{2}]$ and sort them, as the reference code's `init = 2`.
   */
  init?: 'zero' | 'uniform' | Stream
  /** Stop when $\sum_k \norm{\hat u_k^{n+1} - \hat u_k^n}^2 / T$ falls to this or below. Default 1e-7. */
  tolerance?: Scalar
}

/** A state of `vmdSteps`. */
export interface VmdState extends Status {
  /** ADMM sweeps made. */
  t: Size
  /**
   * The modes' one-sided spectra $\hat u_k$ on the mirrored signal: complex128, $K \times T$, fftshifted (index
   * $T/2$ is frequency 0, and the negative half is zero).
   */
  spectra: Tensor
  /** The centre frequencies $\omega_k$ in cycles per sample (length $K$), in the order of the modes. */
  omega: Vector
  /** The Lagrange multiplier's spectrum $\hat\lambda$ (complex128, length $T$); stays zero when $\tau = 0$. */
  multiplier: Tensor
  /**
   * The update size $\sum_k \norm{\hat u_k^{n+1} - \hat u_k^n}^2 / T$ of the last sweep, plus machine epsilon
   * ($\infty$ at the start).
   */
  change: Scalar
  /** The update size fell below the tolerance. */
  converged: boolean
}

/**
 * The mirrored signal (half of each end reflected, as the reference code) and its fftshifted spectrum, zero for
 * $f < 0$. An odd-length signal first loses its last sample.
 *
 * @param x The samples.
 * @returns `n`, the even length used; `T` $= 2n$, the mirrored length; `fPlus`, the one-sided spectrum $\hat f$ as
 *   $T$ interleaved complex values (re, im), fftshifted; and `freqs`, the frequency of each of its bins in cycles per
 *   sample, $i / T - \tfrac{1}{2}$ for bin $i$.
 */
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
 * VMD as a traceable algorithm. The signal (trimmed to even length $n$ and mirrored to $T = 2n$) is split into $K$
 * modes minimising $\sum_k \norm{\partial_t [(\delta(t) + \frac{j}{\pi t}) * u_k(t)] e^{-j\omega_k t}}_2^2$ (each
 * mode's analytic signal, shifted to baseband, has little bandwidth) subject to $\sum_k u_k = f$. Each step is one
 * ADMM sweep over the modes (Gauss–Seidel, so mode $k$ sees the new values of modes before it), in the Fourier domain:
 *
 * - $\hat u_k \leftarrow (\hat f - \sum_{i \ne k} \hat u_i - \hat\lambda / 2) / (1 + \alpha (\omega - \omega_k)^2)$,
 *   a Wiener filter centred on $\omega_k$;
 * - $\omega_k \leftarrow \int_0^\infty \omega \abs{\hat u_k}^2 \, d\omega / \int_0^\infty \abs{\hat u_k}^2 \, d\omega$,
 *   the centre of gravity of the mode's power (held at 0 for the first mode with `dc`);
 * - $\hat\lambda \leftarrow \hat\lambda + \tau (\sum_k \hat u_k - \hat f)$.
 *
 * The paper's denominator is $1 + 2\alpha(\omega - \omega_k)^2$; the authors' reference code (ported by vmdpy), which
 * this follows, has $1 + \alpha(\omega - \omega_k)^2$, so its $\alpha$ absorbs the 2. `init` takes no start.
 *
 * @param x The signal: a single-channel `Signal` or its samples, at least 4 of them.
 * @param options The number of modes, the penalties, the initial centre frequencies and the tolerance. A `modes` that
 *   is not a positive integer, an `alpha` that is not positive, or fewer than 4 samples throw `DomainError`.
 * @returns The algorithm, to run with `run` or step by hand; it is done when `converged`.
 *
 * @example The centre frequencies move onto two tones
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const alg = vmdSteps(x, { modes: 2 })
 * print('start:', run(alg, undefined, 0).omega)
 * print('after one sweep:', run(alg, undefined, 1).omega)
 * const last = run(alg, undefined, 200)
 * print('after', last.t, 'sweeps:', last.omega, 'converged:', last.converged)
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
 * and the middle $n$ samples (the unmirrored signal) kept.
 *
 * @param s A state of `vmdSteps`; only its `spectra` are read.
 * @returns $K$ arrays of length $n = T/2$, in the state's order of modes (not sorted by centre frequency).
 *
 * @example The modes of a converged state are the tones
 * const fast = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8))
 * const slow = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 32))
 * const state = run(vmdSteps(fast.map((v, t) => v + slow[t]), { modes: 2 }), undefined, 200)
 * const [first, second] = vmdModes(state)
 * print('centre frequencies:', state.omega)
 * const error = (mode, tone) => Math.max(...tone.map((v, t) => Math.abs(mode[t] - v)).slice(32, 224))
 * print('mode 1 against the slow tone, largest error away from the ends:', error(first, slow))
 * print('mode 2 against the fast tone, largest error away from the ends:', error(second, fast))
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
 * Variational mode decomposition of a real signal into $K$ modes, as a `Decomposition` (`method: 'vmd'`), modes sorted
 * by centre frequency (slowest first) and named `mode 1`, `mode 2`, and so on. Each component carries
 * `meta: { centreFrequency, iterations, converged }`: the centre frequency in the signal's units (Hz for a `Signal`,
 * cycles per sample for bare samples), and the sweeps made and whether they converged (the same for every mode).
 * `residual` is $f - \sum_k u_k$ (small when $\tau > 0$ or $\alpha$ is small, not zero: VMD trades reconstruction for
 * narrow bands). An odd-length signal loses its last sample, as the reference code. Stops at convergence or after
 * `maxSteps` $- 1$ sweeps (default 500, so 499), as the reference code; read `vmdSteps` for the convergence trace.
 * As `vmdpy.VMD`.
 *
 * @param x The signal: a single-channel `Signal` or its samples, at least 4 of them.
 * @param options The options of `vmdSteps`, and `maxSteps`, one more than the most sweeps made (default 500).
 * @returns The modes as components, slowest first, with the residual and the (trimmed) original.
 *
 * @example Two tones, with their centre frequencies
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const d = vmd(x, { modes: 2 })
 * for (const c of d.components) print(c.name, c.meta)
 *
 * @example The dual step enforces reconstruction
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * print('tau = 0, largest residual:', max(abs(vmd(x, { modes: 2 }).residual)))
 * print('tau = 0.1, largest residual:', max(abs(vmd(x, { modes: 2, tau: 0.1 }).residual)))
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
