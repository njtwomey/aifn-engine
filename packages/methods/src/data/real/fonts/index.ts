/**
 * `aifn-methods/data/real/fonts`: capital letters of 66 real font instances as outline vectors in dense
 * correspondence (the manifold-of-fonts data), as a dataset of fonts. A separate module so that the 800 KB table is
 * loaded only by pages that use it (and so the dataset has its own registry, `fontDatasetRegistry`, outside
 * `aifn-methods/data`'s `datasetRegistry`).
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
