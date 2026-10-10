/**
 * Row selection for whole datasets (private to `aifn-methods/data`): keeps `x`, `y`, `t`, `f` and the per-row metadata
 * aligned, for modifiers that resample, subsample or reorder rows.
 */

import { takeRows as take } from 'aifn-compute/learning/estimators'
import type { Dataset } from './types'

/**
 * The rows `index` of a dataset (any order, repeats allowed), with every per-row field selected alike: `x`, `y`, `t`
 * and `f`, and `meta.cleanLabels`, `meta.outliers`, `meta.missing` and `meta.complete`, each where present. Every
 * other field, `meta.truth` among them, is kept as it is.
 *
 * @param d The dataset to select from (not modified).
 * @param index The row indices to keep, in the order wanted; each must be a valid row of `d`.
 * @returns A dataset of `index.length` rows.
 */
export function selectRows(d: Dataset, index: readonly number[]): Dataset {
  const { cleanLabels, outliers, missing, complete } = d.meta
  return {
    ...d,
    x: take(d.x, index),
    y: d.y && take(d.y, index),
    t: d.t && take(d.t, index),
    f: d.f && take(d.f, index),
    meta: {
      ...d.meta,
      cleanLabels: cleanLabels && take(cleanLabels, index),
      outliers: outliers && take(outliers, index),
      missing: missing && take(missing, index),
      complete: complete && take(complete, index),
    },
  }
}
