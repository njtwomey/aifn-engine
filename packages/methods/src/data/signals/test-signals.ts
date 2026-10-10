/**
 * Deterministic test signals for filtering, denoising and pitch: the four test functions of Donoho and Johnstone
 * (1994, "Ideal spatial adaptation by wavelet shrinkage", Biometrika 81(3)), blocks, bumps, HeaviSine and Doppler, each
 * with a different kind of non-smoothness; and a synthetic voiced sound: a glottal pulse train at a fundamental
 * frequency $f_0$ (optionally with vibrato) through formant resonators, the source–filter model of speech (Fant,
 * 1960).
 *
 * Nothing here draws at random, so a signal is reproduced exactly from its arguments; add noise from a stream to make
 * a denoising problem. Both functions are registered (kind `function`) and collected in `testSignalFunctions`.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import { lfilter } from 'aifn-compute/signal/filters'

/** The names of the Donoho–Johnstone test functions, as `testFunction` takes them. */
export type TestFunctionName = 'blocks' | 'bumps' | 'heavisine' | 'doppler'

const BLOCK_T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
const BLOCK_H = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
const BUMP_H = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2]
const BUMP_W = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005]

/**
 * A Donoho–Johnstone test function sampled at $t_i = i / n$ for $i = 0, \dots, n - 1$, with the paper's constants,
 * then centred and rescaled to zero mean and unit (population) standard deviation, so that noise of standard deviation
 * $\sigma$ gives a signal-to-noise ratio of $1 / \sigma^2$ in power. A constant sample is centred but not rescaled.
 *
 * - `blocks`: a piecewise-constant sum $\sum_j h_j \indicator[t \ge t_j]$ of 11 steps;
 * - `bumps`: 11 sharp peaks $\sum_j h_j (1 + \lvert t - t_j \rvert / w_j)^{-4}$;
 * - `heavisine`: $4 \sin 4\pi t - \sgn(t - 0.3) - \sgn(0.72 - t)$, a sinusoid with two jumps;
 * - `doppler`: $\sqrt{t(1 - t)} \sin\left(2.1\pi / (t + 0.05)\right)$, oscillating ever faster towards $t = 0$.
 *
 * @param name Which of the four functions to sample.
 * @param n The number of samples, spread evenly over $[0, 1)$.
 * @returns The $n$ standardised samples, a float64 vector.
 *
 * @example The first samples of HeaviSine
 * const x = testFunction('heavisine', 256)
 * print('length:', x.shape[0])
 * print('first samples:', toArray(x).slice(0, 4))
 *
 * @example Every test function has zero mean and unit standard deviation
 * const v = toArray(testFunction('bumps', 512))
 * const mean = v.reduce((a, b) => a + b, 0) / v.length
 * print('mean:', mean, ' sd:', Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length))
 */
export function testFunction(name: TestFunctionName, n: number): Tensor {
  const f = (t: number): number => {
    switch (name) {
      case 'blocks':
        return BLOCK_T.reduce((s, tj, j) => s + BLOCK_H[j] * (t - tj >= 0 ? 1 : -1) * 0.5 + BLOCK_H[j] * 0.5, 0)
      case 'bumps':
        return BLOCK_T.reduce((s, tj, j) => s + BUMP_H[j] * (1 + Math.abs((t - tj) / BUMP_W[j])) ** -4, 0)
      case 'heavisine':
        return 4 * Math.sin(4 * Math.PI * t) - Math.sign(t - 0.3) - Math.sign(0.72 - t)
      case 'doppler':
        return Math.sqrt(t * (1 - t)) * Math.sin((2.1 * Math.PI) / (t + 0.05))
    }
  }
  const v = Float64Array.from({ length: n }, (_, i) => f(i / n))
  const mean = v.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1
  return fromData(
    v.map((u) => (u - mean) / sd),
    [n],
  )
}

/** One formant: a two-pole resonance at `frequency` Hz whose $-3$ dB bandwidth is `bandwidth` Hz. */
export type Formant = { frequency: number; bandwidth: number }

/** Options of `voicedSound`. */
export type VoicedOptions = {
  /** Sample rate, Hz (default 8000). */
  fs?: number
  /** Fundamental frequency, Hz (default 140). */
  f0?: number
  /**
   * Vibrato: a sinusoidal modulation $f_0(t) = f_0 (1 + d \sin 2\pi r t)$ of depth $d$ (`depth`, a fraction of $f_0$)
   * at rate $r$ (`rate`, Hz). Default none, a constant $f_0$.
   */
  vibrato?: { depth: number; rate: number }
  /** Formants (default an /a/: 730, 1090 and 2440 Hz, of bandwidths 90, 110 and 160 Hz). */
  formants?: readonly Formant[]
}

/**
 * A synthetic voiced sound: a unit pulse at every glottal closure (each sample where the phase $\int f_0(t)\,dt$
 * crosses an integer), smoothed by a one-pole glottal low-pass (pole at 0.95), then passed through a cascade of
 * two-pole resonators, one per formant, each with poles at $r e^{\pm i\theta}$ for $r = e^{-\pi B / f_s}$ and
 * $\theta = 2\pi F / f_s$ (formant frequency $F$, bandwidth $B$). Every filter has unit gain at 0 Hz, and the output is
 * scaled to a peak absolute value of 1. The true $f_0(t)$ is returned with the samples, so pitch estimators can be
 * scored.
 *
 * @param n The number of samples.
 * @param options The sample rate, the fundamental frequency, the vibrato and the formants.
 * @returns `x`, the $n$ samples of the sound, and `f0`, the true fundamental frequency in Hz at each sample.
 *
 * @example A tenth of a second at 8 kHz
 * const { x, f0 } = voicedSound(800)
 * print('samples:', x.shape[0], ' peak:', Math.max(...toArray(x).map(Math.abs)))
 * print('f0 (Hz) at the start:', toArray(f0)[0])
 *
 * @example Vibrato moves the pitch by its depth either side of f0
 * const { f0 } = voicedSound(8000, { f0: 200, vibrato: { depth: 0.05, rate: 5 } })
 * print('f0 from', Math.min(...toArray(f0)), 'to', Math.max(...toArray(f0)), 'Hz')
 */
export function voicedSound(n: number, options: VoicedOptions = {}): { x: Tensor; f0: Tensor } {
  const { fs = 8000, f0 = 140 } = options
  const formants = options.formants ?? [
    { frequency: 730, bandwidth: 90 },
    { frequency: 1090, bandwidth: 110 },
    { frequency: 2440, bandwidth: 160 },
  ]
  const pulses = new Float64Array(n)
  const pitch = new Float64Array(n)
  let phase = 0
  for (let i = 0; i < n; i++) {
    const v = options.vibrato
    pitch[i] = v ? f0 * (1 + v.depth * Math.sin((2 * Math.PI * v.rate * i) / fs)) : f0
    const next = phase + pitch[i] / fs
    if (Math.floor(next) > Math.floor(phase)) pulses[i] = 1
    phase = next
  }
  // Glottal low-pass (a pole at 0.95), then each resonator: poles at r e^{±iθ}, r = e^{−πB/fs}, θ = 2πF/fs, unit DC gain.
  let y = lfilter({ b: [1 - 0.95], a: [1, -0.95] }, fromData(pulses, [n])).y as Tensor
  for (const { frequency, bandwidth } of formants) {
    const r = Math.exp((-Math.PI * bandwidth) / fs)
    const a1 = -2 * r * Math.cos((2 * Math.PI * frequency) / fs)
    const a2 = r * r
    y = lfilter({ b: [1 + a1 + a2], a: [1, a1, a2] }, y).y as Tensor
  }
  const values = y.data as Float64Array
  const peak = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1
  return {
    x: fromData(
      Float64Array.from(values, (v) => v / peak),
      [n],
    ),
    f0: fromData(pitch, [n]),
  }
}

/** Registers a function of this file (kind `function`, area `data/signals`). */
const fn = definer<FunctionInfo>('function', 'data/signals')
fn(
  {
    key: 'testFunction',
    name: 'Donoho–Johnstone test functions',
    summary: 'Blocks, bumps, HeaviSine and Doppler: the standard test signals for denoising.',
    role: 'construction',
    notes: ['wavelet-denoising'],
    cite: ['donoho1994'],
  },
  testFunction,
)
fn(
  {
    key: 'voicedSound',
    name: 'Synthetic voiced sound',
    summary: 'A glottal pulse train through formant resonators, with its true f₀.',
    role: 'construction',
    notes: ['pitch-estimation', 'cepstrum'],
  },
  voicedSound,
)

/** The test-signal generators, keyed by name. */
export const testSignalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', { testFunction, voicedSound }) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
