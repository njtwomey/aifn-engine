/**
 * Anomaly detector ensembles (Aggarwal and Sathe, 2017; Zhao, Nasrullah and Li, 2019): scores of different detectors
 * live on different scales, so each is first normalised (to ranks in [0, 1], or to z-scores) and the normalised
 * scores are then combined by their mean (averaging lowers variance) or their maximum (keeps any detector's alarm).
 */

import { toFlat } from 'aifn-compute/foundation/tensor'
import { ranks, standardise } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `combineScores`. */
export type EnsembleOptions = {
  /** `rank`: each score's rank divided by n − 1 (ties averaged); `zscore`: (s − mean)/sd. Default `rank`. */
  normalise?: 'rank' | 'zscore'
  /** Default `mean`. */
  combine?: 'mean' | 'max'
}

/** The scores of one detector normalised to ranks in [0, 1] (ties share their mean rank): (rank − 1)/(n − 1). */
export function rankNormalise(scores: ArrayLike<number>): Float64Array {
  const n = scores.length
  return Float64Array.from(toFlat(ranks(Float64Array.from(scores))), (r) => (r - 1) / Math.max(1, n - 1))
}

/** z-scores with the population standard deviation; a constant detector contributes zeros. */
function zNormalise(scores: ArrayLike<number>): Float64Array {
  const z = standardise(Float64Array.from(scores))
  return z.constant ? new Float64Array(scores.length) : Float64Array.from(toFlat(z.values))
}

/** Combine several detectors' scores of the same points (one array per detector) into one score per point. */
export function combineScores(scores: readonly ArrayLike<number>[], options: EnsembleOptions = {}): Float64Array {
  const { normalise = 'rank', combine = 'mean' } = options
  if (scores.length === 0) throw new DomainError('combineScores', 'combineScores: no detectors')
  const n = scores[0].length
  const norm = scores.map((s) => {
    if (s.length !== n)
      throw new ShapeError('combineScores', 'combineScores: detectors scored different numbers of points')
    return normalise === 'rank' ? rankNormalise(s) : zNormalise(s)
  })
  return Float64Array.from({ length: n }, (_, i) =>
    combine === 'max' ? Math.max(...norm.map((s) => s[i])) : norm.reduce((a, s) => a + s[i], 0) / norm.length,
  )
}
