/**
 * A real electrocardiogram: five minutes of lead MLII of record 208 of the MIT-BIH Arrhythmia Database (Moody and
 * Mark, 2001), sampled at $360$ Hz, as SciPy ships it (`scipy.datasets.electrocardiogram()`), with the database's
 * reference beat annotations for the same span.
 *
 * Record 208 is a hard one: besides normal beats it has frequent premature ventricular contractions (PVCs), often in
 * couplets and a bigeminal pattern, fusion beats, and stretches of noise. Each beat carries the cardiologists' MIT-BIH
 * symbol (`N` normal, `V` PVC, `F` fusion of ventricular and normal, `Q` unclassifiable) and its class in the AAMI
 * EC57 grouping used to evaluate beat classifiers: normal, supraventricular ectopic, ventricular ectopic, fusion and
 * unknown. The excerpt starts at sample $422820$ of the record (SciPy's documentation rounds it to 19:35); the
 * vendored file (`record.ts`, made by `scripts/ecg.py`) is checked against the record itself.
 *
 * The data are under the Open Data Commons Attribution License v1.0: cite Moody and Mark (2001) and Goldberger et
 * al. (2000) when you use them.
 */

import type { DatasetInfo, DatasetMeta, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { signal, type Signal } from 'aifn-compute/signal'
import { decimateSignal } from 'aifn-compute/signal/multirate'
import { labels, matrix, vector, type Dataset } from '../../types'
import { ADC_GAIN, BEAT_SAMPLES, BEAT_SYMBOLS, RECORD_START, SAMPLE_RATE, SAMPLES, SOURCE_SHA256 } from './record'

/** The AAMI EC57 beat classes, in the order of the labels `ecgBeats` gives. */
export const ECG_BEAT_CLASSES = [
  'normal',
  'supraventricular ectopic',
  'ventricular ectopic',
  'fusion',
  'unknown',
] as const

/** The AAMI class (an index into `ECG_BEAT_CLASSES`) of each MIT-BIH beat symbol. */
const AAMI: Readonly<Record<string, number>> = {
  N: 0, L: 0, R: 0, e: 0, j: 0,
  A: 1, a: 1, J: 1, S: 1,
  V: 2, E: 2,
  F: 3,
  '/': 4, f: 4, Q: 4,
} // prettier-ignore

/** An annotated beat of the record. */
export interface EcgBeat {
  /** The sample of the beat's fiducial point (near the R peak), within the excerpt. */
  readonly sample: Size
  /** The beat's time from the start of the excerpt, in seconds. */
  readonly time: number
  /** The MIT-BIH symbol: `N` normal, `V` premature ventricular contraction, `F` fusion, `Q` unclassifiable. */
  readonly symbol: string
  /** The AAMI class, an index into `ECG_BEAT_CLASSES`. */
  readonly label: number
}

/** The record: the signal, its annotated beats, and where it came from. */
export interface EcgRecord {
  /** Lead MLII in mV, $108000$ samples at `fs` $= 360$ Hz. */
  readonly signal: Signal
  /** Every annotated beat, in time order. */
  readonly beats: readonly EcgBeat[]
  /** Name, description, source, URL and class names, and `sha256`, the hash of SciPy's source file. */
  readonly meta: DatasetMeta & { readonly sha256: string }
}

/** Options of `ecgBeats`. */
export interface EcgBeatsOptions {
  /** Seconds of signal before each beat's fiducial point (default $0.25$). */
  before?: number
  /** Seconds of signal after it (default $0.4$). */
  after?: number
  /** Subtract each window's median, removing the baseline wander between beats (default false). */
  centre?: boolean
}

/** The citation recorded in each dataset's `meta.source`. */
const SOURCE =
  'Moody and Mark (2001), "The impact of the MIT-BIH Arrhythmia Database", IEEE Eng. in Medicine and Biology 20(3); ' +
  'Goldberger et al. (2000), "PhysioBank, PhysioToolkit, and PhysioNet", Circulation 101(23); via scipy.datasets ' +
  '(Open Data Commons Attribution License v1.0)'
/** The database's page on PhysioNet. */
const URL = 'https://physionet.org/content/mitdb/1.0.0/'

/** The decoded samples, filled on first use by `millivolts`. */
let samples: Float64Array | null = null

/**
 * The vendored samples in mV, decoded once.
 *
 * @returns The $108000$ samples of the excerpt, in mV.
 */
function millivolts(): Float64Array {
  if (samples) return samples
  const binary = atob(SAMPLES)
  const view = new DataView(Uint8Array.from(binary, (c) => c.charCodeAt(0)).buffer)
  samples = Float64Array.from({ length: binary.length / 2 }, (_, i) => view.getInt16(2 * i, true) / ADC_GAIN)
  return samples
}

/** Options of `mitBihEcg`. */
export interface MitBihEcgOptions {
  /**
   * Downsample by this integer factor $q$ with `decimateSignal` (an anti-aliasing lowpass, then every $q$th sample), so
   * the signal is at $360/q$ Hz and each beat's `sample` is its position at that rate. Default 1, the record as is.
   */
  decimate?: Size
}

/**
 * The five-minute MIT-BIH record 208 excerpt (lead MLII, $360$ Hz, in mV) as a `Signal`, ready for the spectral,
 * filtering and decomposition functions of `aifn-compute/signal`, with its $509$ annotated beats. At $360$ Hz a beat
 * spans about $210$ samples (the record's mean rate is about $102$ beats per minute); `decimate: 4` gives $90$ Hz and
 * about $53$ samples per beat.
 *
 * @param options The decimation factor.
 * @returns The signal, the beats (sample, time, MIT-BIH symbol and AAMI class) and the provenance.
 *
 * @example The record and its beats
 * const { signal, beats, meta } = mitBihEcg()
 * print('samples:', signal.data.shape[0], 'at', signal.fs, 'Hz')
 * print('beats:', beats.length, ' first:', beats[0])
 * print('mean heart rate:', (60 * beats.length) / (signal.data.shape[0] / signal.fs), 'beats per minute')
 *
 * @example How the beats divide
 * const counts = {}
 * for (const b of mitBihEcg().beats) counts[b.symbol] = (counts[b.symbol] ?? 0) + 1
 * print(counts)
 *
 * @example Decimated to 90 Hz
 * const { signal, beats } = mitBihEcg({ decimate: 4 })
 * print('samples:', signal.data.shape[0], 'at', signal.fs, 'Hz')
 * print('samples per beat:', (beats.at(-1).sample - beats[0].sample) / (beats.length - 1))
 */
export function mitBihEcg(options: MitBihEcgOptions = {}): EcgRecord {
  const { decimate = 1 } = options
  if (!(Number.isInteger(decimate) && decimate >= 1))
    throw new DomainError('mitBihEcg', `mitBihEcg: decimate must be a positive integer, got ${decimate}`)
  const beats = BEAT_SAMPLES.map((sample, i) => {
    const symbol = BEAT_SYMBOLS[i]
    return { sample: Math.round(sample / decimate), time: sample / SAMPLE_RATE, symbol, label: AAMI[symbol] ?? 4 }
  })
  const full = signal(vector(Float64Array.from(millivolts())), { fs: SAMPLE_RATE })
  return {
    signal: decimate === 1 ? full : decimateSignal(full, decimate),
    beats,
    meta: {
      name: 'MIT-BIH ECG (record 208)',
      description: `Five minutes of lead MLII of MIT-BIH Arrhythmia Database record 208 (from sample ${RECORD_START}), in mV at ${SAMPLE_RATE / decimate} Hz${decimate > 1 ? ` (decimated by ${decimate} from ${SAMPLE_RATE} Hz)` : ''}, with ${beats.length} reference beat annotations.`,
      task: 'sequence',
      labelNames: [...ECG_BEAT_CLASSES],
      source: SOURCE,
      url: URL,
      sha256: SOURCE_SHA256,
    },
  }
}

/**
 * Beat-aligned windows of the record as a labelled dataset: one row per annotated beat, the signal from `before`
 * seconds before its fiducial point to `after` seconds after (in mV), labelled by AAMI class. Beats whose window runs
 * off either end of the excerpt are left out. The rows suit beat classification, and, as signals of one length, the
 * sparse coding and dictionary learning of `aifn-compute/signal/sparse` (transposed, since those take one signal per
 * column).
 *
 * @param options The window around each beat, and whether to remove each window's baseline.
 * @returns The windows `x` (beats $\times$ window length), the AAMI class `y` of each, the beat's time in seconds as
 *   `t`, and the provenance; `meta.featureNames` gives each column's offset from the beat in milliseconds.
 *
 * @example Windows around the beats
 * const beats = ecgBeats()
 * print('x:', beats.x.shape, ' classes:', beats.meta.labelNames)
 * const y = toArray(beats.y)
 * print('ventricular ectopic beats:', y.filter((c) => c === 2).length, 'of', y.length)
 *
 * @example PVCs are wider than normal beats
 * // The time each beat stays above half its largest deflection, averaged over the beats of a class.
 * const beats = ecgBeats({ before: 0.1, after: 0.1, centre: true })
 * const rows = toArray(beats.x)
 * const y = toArray(beats.y)
 * const width = (row) => {
 *   const peak = Math.max(...row.map(Math.abs))
 *   return (1000 * row.filter((v) => Math.abs(v) > peak / 2).length) / 360
 * }
 * const meanWidth = (c) => {
 *   const w = rows.filter((_, i) => y[i] === c).map(width)
 *   return w.reduce((a, b) => a + b) / w.length
 * }
 * print('normal beats:', meanWidth(0), 'ms')
 * print('PVCs:', meanWidth(2), 'ms')
 */
export function ecgBeats(options: EcgBeatsOptions = {}): Dataset {
  const { before = 0.25, after = 0.4, centre = false } = options
  if (!(before >= 0 && after >= 0 && before + after > 0))
    throw new DomainError('ecgBeats', 'ecgBeats: before and after must be non-negative, and not both zero')
  const lead = Math.round(before * SAMPLE_RATE)
  const lag = Math.round(after * SAMPLE_RATE)
  const width = lead + lag + 1
  const all = millivolts()
  const record = mitBihEcg()
  const kept = record.beats.filter((b) => b.sample - lead >= 0 && b.sample + lag < all.length)
  const x = new Float64Array(kept.length * width)
  kept.forEach((b, r) => {
    const row = all.subarray(b.sample - lead, b.sample + lag + 1)
    const median = centre ? [...row].sort((p, q) => p - q)[width >> 1] : 0
    for (let c = 0; c < width; c++) x[r * width + c] = row[c] - median
  })
  return {
    kind: 'dataset',
    x: matrix(x, kept.length, width),
    y: labels(kept.map((b) => b.label)),
    t: vector(Float64Array.from(kept, (b) => b.time)),
    meta: {
      name: 'MIT-BIH ECG beats (record 208)',
      description: `${kept.length} annotated beats of MIT-BIH record 208, each a window from ${before} s before to ${after} s after the beat (${width} samples at ${SAMPLE_RATE} Hz, mV), labelled by AAMI class.`,
      task: 'classification',
      featureNames: Array.from({ length: width }, (_, c) => `${(((c - lead) * 1000) / SAMPLE_RATE).toFixed(1)} ms`),
      labelNames: [...ECG_BEAT_CLASSES],
      source: SOURCE,
      url: URL,
    },
  }
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/real/ecg')

dataset(
  {
    key: 'mitBihEcg',
    name: 'MIT-BIH ECG (record 208)',
    summary: 'Five minutes of a real ECG at 360 Hz with its reference beat annotations (normal, PVC, fusion).',
    task: 'sequence',
    output: 'series',
    knobs: space({ decimate: int(1, 12, { default: 1 }) }),
    truth: false,
    random: false,
  },
  mitBihEcg,
)

dataset(
  {
    key: 'ecgBeats',
    name: 'MIT-BIH ECG beats',
    summary: 'Beat-aligned windows of a real ECG, labelled normal, ventricular ectopic, fusion or unknown.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      before: real(0, 1, { default: 0.25 }),
      after: real(0, 1, { default: 0.4 }),
    }),
    truth: false,
    random: false,
  },
  ecgBeats,
)
