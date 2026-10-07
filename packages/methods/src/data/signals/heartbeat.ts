/**
 * A synthetic electrocardiogram: beats drawn as sums of Gaussian bumps, a fraction of them ectopic, on a slow
 * sinusoidal baseline wander, with white noise, and each beat's start and kind as the truth.
 *
 * A normal beat has a P wave, a Q dip, a sharp R spike, an S dip and a broad T wave; an ectopic beat (a premature
 * ventricular contraction) is wide, with no P wave. Beats last 54 to 66 samples, so at the default rate of $60$ Hz they
 * last about a second, a resting heart rate; a recording such as `mitBihEcg` at $360$ Hz decimated by 4 has a similar
 * number of samples per beat. The trace is made for sparse coding and dictionary learning: the beats repeat a few
 * shapes at irregular positions, and the noise is not sparse in any of them.
 */

import type { DatasetInfo, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { signal } from 'aifn-compute/signal'
import { generatorRecipe, matrix, vector } from '../types'
import type { SignalDataset } from './processes'

/**
 * One synthetic heartbeat of `length` samples, a sum of Gaussian bumps in the beat's own time $u \in [0, 1]$: a P wave,
 * a Q dip, a sharp R spike, an S dip and a broad T wave; an ectopic beat is a wide spike, a dip and a low T wave, with
 * no P wave.
 *
 * @param length The number of samples, at least 2.
 * @param ectopic Draw the ectopic (premature ventricular) shape instead of a normal beat. Default false.
 * @returns The beat's samples; the R spike peaks near $1$.
 *
 * @example The ectopic beat is the wide one
 * // Samples above half the peak: a narrow R spike against a wide ectopic complex.
 * const wide = (b) => b.filter((v) => v > Math.max(...b) / 2).length
 * print('normal beat:', wide(syntheticBeat(60)), 'samples above half its peak')
 * print('ectopic beat:', wide(syntheticBeat(60, true)), 'samples above half its peak')
 */
export function syntheticBeat(length: Size, ectopic = false): number[] {
  if (!(Number.isInteger(length) && length >= 2))
    throw new DomainError('syntheticBeat', `syntheticBeat: length must be an integer of at least 2, got ${length}`)
  const g = (u: number, at: number, width: number) => Math.exp(-(((u - at) / width) ** 2))
  return Array.from({ length }, (_, k) => {
    const u = k / (length - 1)
    if (ectopic) return 0.9 * g(u, 0.38, 0.06) - 0.45 * g(u, 0.52, 0.07) + 0.25 * g(u, 0.75, 0.08)
    return (
      0.15 * g(u, 0.17, 0.05) -
      0.15 * g(u, 0.31, 0.02) +
      g(u, 0.35, 0.022) -
      0.3 * g(u, 0.4, 0.025) +
      0.35 * g(u, 0.66, 0.08)
    )
  })
}

/** Options of `syntheticEcg`. */
export interface SyntheticEcgOptions {
  /** The number of samples $n$ (default 900). */
  n?: Size
  /** The fraction of beats drawn with the ectopic shape, from 0 to 1 (default 0.25). */
  ectopic?: number
  /** The standard deviation of the white noise added to every sample (default 0.08). */
  noise?: number
  /** The amplitude of the baseline wander, a sinusoid of period 700 samples (default 0.2). */
  wander?: number
  /** The sample rate in Hz, which sets the times in `x` and the `signal`'s rate only (default 60). */
  fs?: number
}

/** A synthetic ECG: the noisy trace as a signal dataset, with the trace before noise and every beat. */
export type SyntheticEcg = SignalDataset & {
  /** The noisy trace, $n$ values. */
  readonly y: Tensor
  /** The trace without the noise: the beats plus the baseline wander, $n$ values. */
  readonly clean: Tensor
  /** Every beat in order: its first sample and whether it is ectopic. The last may run past the end of the trace. */
  readonly beats: readonly { readonly start: Size; readonly ectopic: boolean }[]
}

/**
 * A synthetic electrocardiogram of $n$ samples: beats of 54 to 66 samples one after another, each ectopic with
 * probability `ectopic`, plus a baseline wander $w \sin(2\pi t / 700 + \phi)$ and white noise of standard deviation
 * `noise`. The beat lengths and kinds come from `child(s, 'beats')` (two uniform draws per beat), the noise from
 * `child(s, 'noise')` and the wander's phase $\phi$ from `child(s, 'phase')`, so the same stream gives the same trace.
 *
 * @param s The stream the trace is drawn from.
 * @param options The length, the ectopic fraction, the noise, the wander and the sample rate.
 * @returns The times in `x` ($n \times 1$), the noisy trace in `y` and as a `Signal`, the trace without noise in
 *   `clean`, and every beat's start and kind in `beats`.
 *
 * @example A trace and its beats
 * const ecg = syntheticEcg(stream(1), { n: 600 })
 * print('samples:', ecg.y.shape[0], 'at', ecg.signal.fs, 'Hz')
 * print('beats:', ecg.beats.length, ' ectopic:', ecg.beats.filter((b) => b.ectopic).length)
 * print('first beats start at', ecg.beats.slice(0, 4).map((b) => b.start))
 *
 * @example The noise is the only difference between the trace and its clean form
 * const ecg = syntheticEcg(stream(2), { n: 600, noise: 0.08 })
 * const y = toArray(ecg.y)
 * const clean = toArray(ecg.clean)
 * const rms = Math.sqrt(y.reduce((a, v, t) => a + (v - clean[t]) ** 2, 0) / y.length)
 * print('RMS of the noise:', rms)
 */
export function syntheticEcg(s: Stream, options: SyntheticEcgOptions = {}): SyntheticEcg {
  const { n = 900, ectopic = 0.25, noise = 0.08, wander = 0.2, fs = 60 } = options
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('syntheticEcg', `syntheticEcg: n must be a positive integer, got ${n}`)
  if (!(ectopic >= 0 && ectopic <= 1))
    throw new DomainError('syntheticEcg', `syntheticEcg: ectopic must be a fraction from 0 to 1, got ${ectopic}`)
  if (!(fs > 0)) throw new DomainError('syntheticEcg', `syntheticEcg: fs must be positive, got ${fs}`)
  // The draws, their order and the constants are fixed: the same stream gives the same trace in every release.
  const draws = toFlat(uniform(child(s, 'beats'), 0, 1, { shape: [2 * Math.ceil(n / 50)] }) as Tensor)
  const beatsSum: number[] = []
  const beats: { start: Size; ectopic: boolean }[] = []
  for (let b = 0; beatsSum.length < n; b++) {
    const isEctopic = draws[2 * b + 1] < ectopic
    beats.push({ start: beatsSum.length, ectopic: isEctopic })
    beatsSum.push(...syntheticBeat(54 + Math.floor(draws[2 * b] * 13), isEctopic))
  }
  const white = toFlat(normals(child(s, 'noise'), n))
  const phase = 2 * Math.PI * toFlat(uniform(child(s, 'phase'), 0, 1, { shape: [1] }) as Tensor)[0]
  const clean = Float64Array.from(
    { length: n },
    (_, t) => beatsSum[t] + wander * Math.sin((2 * Math.PI * t) / 700 + phase),
  )
  const y = clean.map((v, t) => v + noise * white[t])
  const knobs = { n, ectopic, noise, wander, fs }
  return {
    kind: 'dataset',
    x: matrix(
      Float64Array.from({ length: n }, (_, t) => t / fs),
      n,
      1,
    ),
    y: vector(y),
    signal: signal(fromData(Float64Array.from(y), [n]), { fs }),
    clean: vector(clean),
    beats,
    meta: {
      name: 'synthetic ECG',
      description: `${n} samples at ${fs} Hz of synthetic heartbeats (54 to 66 samples each, ${Math.round(100 * ectopic)}% ectopic) on a baseline wander of amplitude ${wander}, with white noise of sd ${noise}.`,
      task: 'sequence',
      featureNames: ['t'],
      targetName: 'x',
      key: s.key,
      recipe: generatorRecipe('syntheticEcg', s.key, knobs),
    },
  }
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/signals')

dataset(
  {
    key: 'syntheticEcg',
    name: 'Synthetic ECG',
    summary:
      'Synthetic heartbeats, some ectopic, on a baseline wander with noise: a few shapes at irregular positions.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(60, 16384, { default: 900 }),
      ectopic: real(0, 1, { default: 0.25 }),
      noise: real(0, 1, { default: 0.08 }),
      wander: real(0, 2, { default: 0.2 }),
      fs: real(1, 1000, { default: 60, scale: 'log' }),
    }),
    truth: false,
    random: true,
    notes: ['dictionary-learning', 'convolutional-sparse-coding'],
  },
  syntheticEcg,
)
