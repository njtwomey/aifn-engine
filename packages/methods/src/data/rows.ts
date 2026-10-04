/** Row selection for whole datasets (private): keeps x, y, t, f and the per-row metadata aligned. */

import { takeRows as take } from 'aifn-compute/learning/estimators'
import type { Dataset } from './types'

/** The rows `index` of a dataset (any order, repeats allowed), with every per-row field selected alike. */
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
