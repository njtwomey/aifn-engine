/** Simple imputation of missing values (NaN), as scikit-learn's `SimpleImputer`. */

import { median } from 'aifn-compute/probability/stats'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { column, mapColumns, matrix, type FittedTransform, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { oneOf, real, space } from 'aifn-compute/foundation/space'

/** A fitted simple imputer. */
export interface SimpleImputer extends FittedTransform {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'simple-imputer'
  readonly strategy: 'mean' | 'median' | 'most-frequent' | 'constant'
  /** The value that replaces NaN in each column, [d]. NaN for a column with no observed values (reported in `empty`). */
  readonly statistics: Tensor
  /** Number of missing values in each training column. */
  readonly missing: readonly number[]
  /** Columns with no observed values; they stay NaN after `transform`. */
  readonly empty: readonly boolean[]
}

/**
 * Replace NaN in each column by the column's mean, median or most frequent observed value (the smallest of ties), or a
 * constant `fillValue`. The transform has no inverse: which values were missing is not kept.
 */
export function simpleImputer({
  strategy = 'mean',
  fillValue = 0,
}: { strategy?: SimpleImputer['strategy']; fillValue?: number } = {}): Transformer<Tensor, SimpleImputer> {
  return {
    name: 'simple-imputer',
    params: { strategy, fillValue },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'simpleImputer')
      const statistics = new Float64Array(d)
      const missing: number[] = []
      const empty: boolean[] = []
      for (let j = 0; j < d; j++) {
        const observed = column(v, n, d, j).filter((a) => !Number.isNaN(a))
        missing.push(n - observed.length)
        empty.push(observed.length === 0 && strategy !== 'constant')
        if (strategy === 'constant') statistics[j] = fillValue
        else if (observed.length === 0) statistics[j] = NaN
        else if (strategy === 'mean') statistics[j] = observed.reduce((a, b) => a + b, 0) / observed.length
        else if (strategy === 'median') statistics[j] = median(observed)
        else {
          const counts = new Map<number, number>()
          for (const a of observed) counts.set(a, (counts.get(a) ?? 0) + 1)
          let best = NaN
          let bestCount = 0
          for (const [a, c] of counts) if (c > bestCount || (c === bestCount && a < best)) [best, bestCount] = [a, c]
          statistics[j] = best
        }
      }
      return {
        kind: 'model',
        name: 'simple-imputer',
        strategy,
        statistics: fromData(statistics, [d]),
        missing,
        empty,
        transform: (input) => mapColumns(input, d, 'simpleImputer', (a, j) => (Number.isNaN(a) ? statistics[j] : a)),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'simpleImputer',
    module: 'learning/preprocessing',
    name: 'Simple imputer',
    summary: 'Fill missing values per feature with its mean, median, mode or a constant.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({
      strategy: oneOf(['mean', 'median', 'most-frequent', 'constant']),
      fillValue: real(-100, 100, { default: 0 }),
    }),
    notes: ['missing-data-and-imputation'],
    cite: ['little2019'],
  },
  simpleImputer,
)
