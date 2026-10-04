/**
 * Deterministic test signals for filtering, denoising and pitch: the four test functions of Donoho and Johnstone
 * (1994, Biometrika 81(3)) — blocks, bumps, HeaviSine and Doppler, each with a different kind of non-smoothness — and a
 * synthetic voiced sound: a glottal pulse train at f₀ (optionally with vibrato) through formant resonators, the
 * source–filter model of speech (Fant, 1960).
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import { lfilter } from 'aifn-compute/signal/filters'

/** The names of the Donoho–Johnstone test functions. */
export type TestFunctionName = 'blocks' | 'bumps' | 'heavisine' | 'doppler'

const BLOCK_T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
const BLOCK_H = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
const BUMP_H = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2]
const BUMP_W = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005]

/**
 * A Donoho–Johnstone test function sampled at tᵢ = i/n, i = 0 … n − 1, with the paper's constants, rescaled to unit
 * standard deviation (so a noise sd σ gives an SNR of 1/σ² in power).
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

/** One formant: a two-pole resonance at `frequency` Hz with −3 dB `bandwidth` Hz. */
export type Formant = { frequency: number; bandwidth: number }

/** Options of `voicedSound`. */
export type VoicedOptions = {
  fs?: number
  /** Fundamental frequency, Hz (default 140). */
  f0?: number
  /** Vibrato: a sinusoidal f₀ modulation of `depth` (fraction of f₀) at `rate` Hz (default none). */
  vibrato?: { depth: number; rate: number }
  /** Formants (default an /a/: 730, 1090, 2440 Hz). */
  formants?: readonly Formant[]
}

/**
 * A synthetic voiced sound: a unit pulse at every glottal closure (the phase of ∫ f₀(t) dt crossing an integer), each
 * pulse smoothed by a one-pole glottal low-pass, then a cascade of two-pole resonators (one per formant). The true f₀(t)
 * is returned with the samples, so pitch estimators can be scored.
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
