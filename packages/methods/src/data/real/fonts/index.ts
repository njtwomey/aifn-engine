/**
 * `aifn-methods/data/real/fonts`: capital letters of 66 real font instances as outline vectors in dense
 * correspondence (the manifold-of-fonts data), as a dataset of fonts.
 *
 * - The dataset: `fonts`, one row per font, the outline samples of the chosen capitals as features and the design
 *   class (`FONT_CLASSES`) as label, with each font's family, style and licence (SIL OFL 1.1 or Apache 2.0).
 * - The whole table: `fontVectors` (decoded int16 values, glyph layouts and provenance) and `fontTable` (the same as
 *   a $66 \times 4479$ float64 tensor).
 * - Drawing: `glyphContours` cuts one glyph's contours out of a font vector, a data row or a model's prediction.
 *
 * A separate module so that the 800 KB table is loaded only by pages that use it (and so the dataset has its own
 * registry, `fontDatasetRegistry`, outside `aifn-methods/data`'s `datasetRegistry`; `fontFunctions` registers the
 * other functions).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { DatasetEntry } from '../../define'
import * as table from './fonts'

export {
  FONT_CLASSES,
  fontTable,
  fontVectors,
  fonts,
  glyphContours,
  type Contour,
  type FontClass,
  type FontDataset,
  type FontDatasetMeta,
  type FontInfo,
  type FontVectors,
  type FontsOptions,
  type GlyphLayout,
} from './fonts'

/** The font dataset generator (kind `dataset`), keyed by `info.key`. */
export const fontDatasetRegistry = entries('dataset', table) as Readonly<Record<string, DatasetEntry>>
export { fontFunctions } from './registry'
