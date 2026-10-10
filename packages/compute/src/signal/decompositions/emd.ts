/**
 * Empirical mode decomposition (Huang et al., 1998, Proc. R. Soc. Lond. A 454): a signal is split into intrinsic mode
 * functions (IMFs) by sifting, $x = \sum_k c_k + r$ with the fastest mode $c_1$ first, returned as a `Decomposition`.
 * `siftSteps` is the sifting of one IMF as a traceable algorithm; `eemd` is ensemble EMD (Wu and Huang, 2009, Adv.
 * Adapt. Data Anal. 1(1)); `ceemdan` is complete ensemble EMD with adaptive noise (Torres, Colominas, Schlotthauer and
 * Flandrin, 2011, ICASSP), all on the same sifting (`emd-core.ts`, after PyEMD).
 *
 * Signals are single-channel: a `Signal` (whose `fs` and `t0` set the time axis) or bare samples (time in samples).
 * The ensemble methods draw their noise from a `Stream`, trial $i$ from `child(s, 'trial', i)`, so a result is fixed by
 * the stream. Nothing is differentiable: the results are plain tensors.
 */

import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Decomposition, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { readSamples, type Samples, type SignalInput } from '../signal'
import { emd as emdCore, findExtrema, sift, siftStep, stopTest, type EmdArrays, type StopRule } from './emd-core'

export type { StopRule }

/**
 * Rilling's stopping rule for sifting one IMF with his default thresholds, $\theta_1 = 0.05$, $\theta_2 = 0.5$ and
 * $\alpha = 0.05$; see `StopRule`.
 */
export const RILLING_RULE: StopRule = { kind: 'rilling', theta1: 0.05, theta2: 0.5, alpha: 0.05 }

/** Options for `emd`. */
export interface EmdOptions {
  /** Most IMFs to extract ($-1$ for no limit). Default $-1$. */
  maxImfs?: Size
  /** When to stop sifting one IMF. Default PyEMD's rule, `{ kind: 'pyemd' }`. */
  rule?: StopRule
  /** Mirror extrema at the ends before fitting envelopes. Default true. */
  mirror?: boolean
}

/**
 * A `Decomposition` of the samples into IMFs (named `imf 1`, `imf 2`, and so on, fastest first) and the residue.
 *
 * @param method The method's name, recorded as `method`.
 * @param input The samples decomposed, with the sampling rate and start time that set the time axis
 *   $t_0 + k / f_s$.
 * @param imfs The IMFs, fastest first, each as long as the signal.
 * @param residue What the IMFs leave over, recorded as `residual`.
 * @returns The decomposition, with a copy of the samples as `original`.
 */
function decomposition(
  method: string,
  input: Samples,
  imfs: readonly Float64Array[],
  residue: Float64Array,
): Decomposition {
  const n = input.values.length
  const vec = (a: Float64Array): Tensor => fromData(a, [a.length])
  return {
    kind: 'decomposition',
    method,
    axis: fromData(
      Float64Array.from({ length: n }, (_, k) => input.t0 + k / input.fs),
      [n],
    ),
    components: imfs.map((imf, k) => ({ name: `imf ${k + 1}`, values: vec(imf) })),
    residual: vec(residue),
    original: vec(Float64Array.from(input.values)),
  }
}

/**
 * Empirical mode decomposition of a real signal, as a `Decomposition` (`method: 'emd'`) over the signal's time axis:
 * the IMFs are the components and the residue is `residual`, so the components and `residual` sum to `original`.
 * IMFs are sifted out until the residue has too few extrema, `maxImfs` are found, or the residue's range falls below
 * 0.001 or its $\ell_1$ norm below 0.005 (PyEMD's absolute thresholds). As `PyEMD.EMD`.
 *
 * @param x The signal: a single-channel `Signal` or its samples. A multichannel signal throws `ShapeError`.
 * @param options The most IMFs, the stopping rule for sifting and whether to mirror extrema at the ends.
 * @returns The IMFs as components `imf 1`, `imf 2`, and so on, fastest first, with the residue as `residual`.
 *
 * @example Two tones, of 1/8 and 1/32 cycles per sample, come out as the first two IMFs
 * const n = 256
 * const fast = Array.from({ length: n }, (_, t) => Math.cos((2 * Math.PI * t) / 8))
 * const slow = Array.from({ length: n }, (_, t) => Math.cos((2 * Math.PI * t) / 32))
 * const d = emd(fast.map((v, t) => v + slow[t]))
 * // Each component's frequency in cycles per sample, from its zero crossings.
 * for (const c of d.components) print(c.name, extrema(c.values).zeroCrossings / (2 * n))
 * // Away from the ends, where the envelopes are least sure.
 * const error = (c, tone) => Math.max(...tone.map((v, t) => Math.abs(c.values.data[t] - v)).slice(32, 224))
 * print('imf 1 against the fast tone, largest error:', error(d.components[0], fast))
 * print('imf 2 against the slow tone, largest error:', error(d.components[1], slow))
 *
 * @example A tone on a trend: the trend is left as the residual, and the parts sum to the signal
 * const x = Array.from({ length: 100 }, (_, t) => Math.sin(t) + 0.02 * t)
 * const d = emd(x)
 * print('IMFs:', d.components.length)
 * print('residual at the ends:', d.residual.data[0], d.residual.data[99], 'against the trend 0 and', 0.02 * 99)
 * const total = d.components.reduce((acc, c) => add(acc, c.values), d.residual)
 * print('largest reconstruction error:', max(abs(sub(total, d.original))))
 */
export function emd(x: SignalInput, options: EmdOptions = {}): Decomposition {
  const input = readSamples(x, 'emd')
  const d = emdCore(input.values, options)
  return decomposition('emd', input, d.imfs, d.residue)
}

/**
 * Local maxima and minima (indices, int32) and the number of zero crossings of a signal, as EMD's sifting sees them:
 * a flat plateau counts once, at its middle, the end samples are never extrema, and a run of exact zeros is one
 * crossing.
 *
 * @param x The signal: a single-channel `Signal` or its samples.
 * @returns The ascending indices of the maxima and of the minima, and the number of zero crossings.
 *
 * @example A plateau is one maximum
 * const e = extrema([1, 3, 2, 4, 4, 1, -1, 2])
 * print('maxima at', e.maxima)
 * print('minima at', e.minima)
 * print('zero crossings:', e.zeroCrossings)
 */
export function extrema(x: SignalInput): { maxima: Tensor; minima: Tensor; zeroCrossings: Size } {
  const e = findExtrema(readSamples(x, 'extrema').values)
  return {
    maxima: fromData(Int32Array.from(e.maxima), [e.maxima.length]),
    minima: fromData(Int32Array.from(e.minima), [e.minima.length]),
    zeroCrossings: e.zeroCrossings,
  }
}

/** Options for `siftSteps`. */
export interface SiftOptions {
  /** When to stop. Default PyEMD's rule. */
  rule?: StopRule
  /** Mirror extrema at the ends before fitting envelopes. Default true. */
  mirror?: boolean
}

/** A state of sifting: the candidate IMF $h$ and the envelopes of the step that produced it. */
export interface SiftState extends Status {
  /** Sifting steps made: the state holds $h_t$. */
  t: Size
  /** The candidate $h_t$ after $t$ sifting steps ($h_0 = x$). */
  h: Tensor
  /** The upper envelope $e_+$ of the last step, through the maxima of $h_{t-1}$ (empty at step 0). */
  upper: Tensor
  /** The lower envelope $e_-$ of the last step, through the minima of $h_{t-1}$ (empty at step 0). */
  lower: Tensor
  /** The envelope mean $(e_+ + e_-) / 2$ that the last step subtracted, $h_{t-1} - h_t$ (empty at step 0). */
  mean: Tensor
  /** Indices (int32) of the maxima of $h_{t-1}$ (empty at step 0). */
  maxima: Tensor
  /** Indices (int32) of the minima of $h_{t-1}$ (empty at step 0). */
  minima: Tensor
  /** Consecutive balanced steps (S-number rule). */
  balancedRun: Size
  /** The stopping rule holds: $h$ is an IMF. */
  converged: boolean
  /** Too few extrema to continue: $h$ is then a residue, not an IMF. */
  terminated: boolean
}

const empty = fromData(new Float64Array(0), [0])
const emptyInt = fromData(new Int32Array(0), [0])

/**
 * Sifting one IMF out of $x$ as a traceable algorithm (Rilling, Flandrin and Gonçalvès, 2003): each step fits
 * cubic-spline envelopes $e_+$ and $e_-$ through the maxima and minima of $h$ and subtracts their mean,
 * $h \leftarrow h - (e_+ + e_-) / 2$. Stops (`converged`) when the stopping rule holds, or (`terminated`, with $h$
 * left as it was) when $h$ has too few extrema. `init` takes no start.
 *
 * @param x The signal: a single-channel `Signal` or its samples; it is copied, so later changes to it are not seen.
 * @param options The stopping rule and whether to mirror extrema at the ends.
 * @returns The algorithm, to run with `run` or step by hand.
 *
 * @example Sift the faster tone out of two
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const first = run(siftSteps(x), undefined, 1)
 * print('step 1: maxima', first.maxima.shape[0], 'minima', first.minima.shape[0])
 * print('step 1: largest envelope mean', max(abs(first.mean)))
 * const last = run(siftSteps(x), undefined, 100)
 * print('steps until an IMF:', last.t, 'converged:', last.converged)
 * print('last step: largest envelope mean', max(abs(last.mean)))
 */
export function siftSteps(x: SignalInput, options: SiftOptions = {}): Algorithm<undefined, SiftState> {
  const { rule = { kind: 'pyemd' }, mirror = true } = options
  const values = readSamples(x, 'siftSteps').values
  return {
    name: 'sift',
    init: () => ({
      t: 0,
      h: fromData(Float64Array.from(values), [values.length]),
      upper: empty,
      lower: empty,
      mean: empty,
      maxima: emptyInt,
      minima: emptyInt,
      balancedRun: 0,
      converged: false,
      terminated: false,
    }),
    step: (s) => {
      const prev = dense.data(s.h)
      const step = siftStep(prev, mirror)
      if (!step) return { ...s, terminated: true }
      const t = s.t + 1
      const { converged, balancedRun } = stopTest(rule, prev, step, t, s.balancedRun, mirror)
      const n = step.next.length
      return {
        t,
        h: fromData(step.next, [n]),
        upper: fromData(step.upper, [n]),
        lower: fromData(step.lower, [n]),
        mean: fromData(step.mean, [n]),
        maxima: fromData(Int32Array.from(step.maxima), [step.maxima.length]),
        minima: fromData(Int32Array.from(step.minima), [step.minima.length]),
        balancedRun,
        converged,
        terminated: false,
      }
    },
  }
}

/**
 * Sift one IMF out of $x$: the result of `siftSteps` run to the end. The sifting stops after at most `maxSteps`
 * sifts (default 1000).
 *
 * @param x The signal: a single-channel `Signal` or its samples.
 * @param options The stopping rule, whether to mirror extrema at the ends, and `maxSteps`, the most sifts made
 *   (default 1000).
 * @returns The sifted `imf`, the number of `sifts` made, and `oscillating`, false when $x$ (or a sifted candidate) had
 *   too few extrema to sift, so that `imf` is a residue rather than an IMF.
 *
 * @example One IMF, and a ramp that has none
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const { sifts, oscillating } = siftImf(x)
 * print('two tones: sifts', sifts, 'oscillating', oscillating)
 * const ramp = siftImf([0, 1, 2, 3, 4])
 * print('ramp: sifts', ramp.sifts, 'oscillating', ramp.oscillating)
 *
 * @example Rules that stop sooner or later
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * print('PyEMD rule:', siftImf(x).sifts)
 * print('Rilling rule:', siftImf(x, { rule: RILLING_RULE }).sifts)
 * print('fixed, 10 sifts:', siftImf(x, { rule: { kind: 'fixed', sifts: 10 } }).sifts)
 */
export function siftImf(
  x: SignalInput,
  options: SiftOptions & { maxSteps?: Size } = {},
): { imf: Tensor; sifts: Size; oscillating: boolean } {
  const r = sift(readSamples(x, 'siftImf').values, options.rule, options.mirror, options.maxSteps)
  return { imf: fromData(r.imf, [r.imf.length]), sifts: r.steps.length, oscillating: r.oscillating }
}

/**
 * Ensemble EMD (Wu and Huang, 2009): the IMFs of $x + \varepsilon \sigma_x \wvec_i$ averaged over `trials` white-noise
 * realisations $\wvec_i$ (trial $i$ draws from `child(s, 'trial', i)`), slot by slot, with $\sigma_x$ the standard
 * deviation of $x$. Every trial is decomposed into at most `maxImfs` IMFs, and a trial that stops sooner adds zero to
 * the slots it lacks; the residues are averaged too. The sum of the averaged IMFs and residue differs from $x$ by the
 * mean added noise, of size $\varepsilon \sigma_x / \sqrt{I}$ for $I$ trials. Returns a `Decomposition`
 * (`method: 'eemd'`). As `PyEMD.EEMD`.
 *
 * @param s The random stream the noise is drawn from.
 * @param x The signal: a single-channel `Signal` or its samples.
 * @param options The ensemble and the decomposition of each trial.
 * @param options.trials The number of noise realisations $I$ (default 50).
 * @param options.epsilon The noise amplitude $\varepsilon$, relative to $\sigma_x$ (default 0.2).
 * @param options.maxImfs The most IMFs per trial, and the number of components returned (default −1: no limit, and
 *   as many components as the trial with the most IMFs).
 * @param options.rule When to stop sifting one IMF (default 10 fixed sifts, as Wu and Huang).
 * @returns The averaged IMFs as components `imf 1`, `imf 2`, and so on, with the averaged residue.
 *
 * @example Twenty noisy copies: the tones separate, and the sum carries the averaged noise
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const d = eemd(stream(0), x, { trials: 20, epsilon: 0.05, maxImfs: 3 })
 * for (const c of d.components) print(c.name, 'cycles per sample:', extrema(c.values).zeroCrossings / 512)
 * const total = d.components.reduce((acc, c) => add(acc, c.values), d.residual)
 * print('largest reconstruction error:', max(abs(sub(total, d.original))))
 */
export function eemd(
  s: Stream,
  x: SignalInput,
  options: { trials?: Size; epsilon?: Scalar; maxImfs?: Size; rule?: StopRule } = {},
): Decomposition {
  const { trials = 50, epsilon = 0.2, maxImfs = -1, rule = { kind: 'fixed', sifts: 10 } } = options
  const input = readSamples(x, 'eemd')
  const v = input.values
  const n = v.length
  let mean = 0
  for (const u of v) mean += u / n
  let sd = 0
  for (const u of v) sd += (u - mean) ** 2 / n
  const scale = epsilon * Math.sqrt(sd)
  const trialResults: EmdArrays[] = []
  let kMax = maxImfs >= 0 ? maxImfs : 0
  for (let trial = 0; trial < trials; trial++) {
    const w = dense.data(normals(child(s, 'trial', trial), [n]))
    const y = v.map((u, i) => u + scale * w[i])
    const d = emdCore(y, { maxImfs, rule })
    trialResults.push(d)
    if (d.imfs.length > kMax) kMax = d.imfs.length
  }
  const imfs = Array.from({ length: kMax }, () => new Float64Array(n))
  const residue = new Float64Array(n)
  for (const d of trialResults) {
    d.imfs.forEach((imf, j) => {
      for (let i = 0; i < n; i++) imfs[j][i] += imf[i] / trials
    })
    for (let i = 0; i < n; i++) residue[i] += d.residue[i] / trials
  }
  return decomposition('eemd', input, imfs, residue)
}

/** Options for `ceemdan`. */
export interface CeemdanOptions {
  /** Noise realisations $I$. Default 50. */
  trials?: Size
  /**
   * The noise amplitude $\varepsilon$ relative to the standard deviation of the signal (stage 1) or residue (later
   * stages). Default 0.2.
   */
  epsilon?: Scalar
  /** Most IMFs to extract ($-1$ for no limit). Default $-1$. */
  maxImfs?: Size
  /** When to stop sifting one IMF. Default 10 fixed sifts, as `eemd`. */
  rule?: StopRule
}

/**
 * Complete ensemble EMD with adaptive noise (Torres et al., 2011). EEMD averages whole decompositions of differently
 * noised copies, so its modes do not sum to the signal and the $k$-th averaged mode mixes different scales. CEEMDAN
 * extracts one mode at a time from a shared residue instead:
 *
 * $c_1 = \frac{1}{I} \sum_i E_1(x + \beta_0 \wvec_i)$, $r_1 = x - c_1$, and
 * $c_k = \frac{1}{I} \sum_i E_1(r_{k-1} + \beta_{k-1} E_{k-1}(\wvec_i))$, $r_k = r_{k-1} - c_k$,
 *
 * where $\wvec_i$ is unit white noise (trial $i$ draws from `child(s, 'trial', i)`), $E_j(\cdot)$ is the $j$-th mode
 * by EMD ($E_1$: the first IMF by sifting, shared with `emd`), $\beta_0 = \varepsilon \sigma(x)$ and
 * $\beta_k = \varepsilon \sigma(r_k)$. A noise realisation with no $k$-th mode adds no noise, and a noisy copy with
 * too few extrema to sift adds zero to the average. Stops when the residue has too few extrema to sift, or after
 * `maxImfs`. By construction the IMFs and residue sum to $x$ exactly (complete), unlike `eemd`. Returns a
 * `Decomposition` (`method: 'ceemdan'`). As `PyEMD.CEEMDAN`.
 *
 * @param s The random stream the noise is drawn from.
 * @param x The signal: a single-channel `Signal` or its samples.
 * @param options The ensemble, the noise amplitude, the most IMFs and the stopping rule.
 * @returns The IMFs as components `imf 1`, `imf 2`, and so on, fastest first, with the final residue.
 *
 * @example Complete: the modes and residue sum to the signal
 * const x = Array.from({ length: 256 }, (_, t) => Math.cos((2 * Math.PI * t) / 8) + Math.cos((2 * Math.PI * t) / 32))
 * const d = ceemdan(stream(0), x, { trials: 20, epsilon: 0.05, maxImfs: 3 })
 * // The small second mode is the averaged noise, a spurious mode of the original CEEMDAN.
 * for (const c of d.components)
 *   print(c.name, 'cycles per sample:', extrema(c.values).zeroCrossings / 512, 'peak:', max(abs(c.values)))
 * const total = d.components.reduce((acc, c) => add(acc, c.values), d.residual)
 * print('largest reconstruction error:', max(abs(sub(total, d.original))))
 */
export function ceemdan(s: Stream, x: SignalInput, options: CeemdanOptions = {}): Decomposition {
  const { trials = 50, epsilon = 0.2, maxImfs = -1, rule = { kind: 'fixed', sifts: 10 } } = options
  const input = readSamples(x, 'ceemdan')
  const v = input.values
  const n = v.length
  const sd = (a: ArrayLike<number>) => {
    let m = 0
    for (let i = 0; i < a.length; i++) m += a[i] / a.length
    let q = 0
    for (let i = 0; i < a.length; i++) q += (a[i] - m) ** 2 / a.length
    return Math.sqrt(q)
  }
  // The noise realisations and their EMD modes Eⱼ(wᵢ), computed once (a missing mode is zero).
  const noise = Array.from({ length: trials }, (_, i) => dense.data(normals(child(s, 'trial', i), [n])))
  const noiseModes = noise.map((w) => emdCore(w, { maxImfs, rule }).imfs)
  const first = (y: Float64Array) => {
    const r = sift(y, rule)
    return r.oscillating ? r.imf : null
  }
  const imfs: Float64Array[] = []
  const residue = Float64Array.from(v)
  while (maxImfs < 0 || imfs.length < maxImfs) {
    const k = imfs.length
    if (first(residue) === null) break
    const beta = epsilon * sd(residue)
    const imf = new Float64Array(n)
    for (let i = 0; i < trials; i++) {
      const add = k === 0 ? noise[i] : noiseModes[i][k - 1]
      const y = add ? residue.map((r, j) => r + beta * add[j]) : Float64Array.from(residue)
      const e = first(y)
      // A noisy copy with too few extrema to sift has no first mode: it contributes zero.
      if (e) for (let j = 0; j < n; j++) imf[j] += e[j]
    }
    for (let j = 0; j < n; j++) imf[j] /= trials
    imfs.push(imf)
    for (let j = 0; j < n; j++) residue[j] -= imf[j]
  }
  return decomposition('ceemdan', input, imfs, residue)
}
