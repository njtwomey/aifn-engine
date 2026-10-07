/**
 * `aifn-methods/data/real/ecg`: a real electrocardiogram, five minutes of MIT-BIH Arrhythmia Database record 208 at
 * $360$ Hz with its reference beat annotations, as SciPy ships it.
 *
 * - The record: `mitBihEcg`, a `Signal` in mV with every annotated beat (sample, time, MIT-BIH symbol, AAMI class),
 *   at $360$ Hz or decimated (`decimate: 4` gives $90$ Hz, about $53$ samples per beat).
 * - Beats as a dataset: `ecgBeats`, one beat-aligned window per row, labelled by AAMI class (`ECG_BEAT_CLASSES`).
 *
 * A separate module so that the 294 KB record is loaded only by pages that use it (its own registry,
 * `ecgDatasetRegistry`, sits outside `aifn-methods/data`'s `datasetRegistry`). Open Data Commons Attribution License
 * v1.0: cite Moody and Mark (2001) and Goldberger et al. (2000).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { DatasetEntry } from '../../define'
import * as ecg from './ecg'

export {
  ECG_BEAT_CLASSES,
  ecgBeats,
  mitBihEcg,
  type EcgBeat,
  type EcgBeatsOptions,
  type EcgRecord,
  type MitBihEcgOptions,
} from './ecg'

/** The ECG dataset generators (kind `dataset`), keyed by `info.key`. */
export const ecgDatasetRegistry = entries('dataset', ecg) as Readonly<Record<string, DatasetEntry>>
