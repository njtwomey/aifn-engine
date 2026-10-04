/**
 * `aifn-methods/data/real/hyphenation`: common English words with dictionary hyphenation points from the Moby
 * Hyphenator II list (public domain), split by word into train and test, with the dictionary as truth. A separate
 * module so that the 79 KB word list is loaded only by pages that use it (its own registry,
 * `hyphenationDatasetRegistry`, sits outside `aifn-methods/data`'s `datasetRegistry`).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { DatasetEntry } from '../../define'
import * as hyphenation from './hyphenation'

export {
  hyphenLabels,
  MOBY_WORDS,
  mobyHyphenation,
  type DictionaryWord,
  type HyphenationData,
  type HyphenationPart,
  type HyphenationTruth,
  type MobyHyphenationOptions,
} from './hyphenation'

/** The hyphenation dataset generator (kind `dataset`), keyed by `info.key`. */
export const hyphenationDatasetRegistry = entries('dataset', hyphenation) as Readonly<Record<string, DatasetEntry>>
