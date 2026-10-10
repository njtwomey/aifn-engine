/**
 * Anomaly detector ensembles (Aggarwal and Sathe, 2017; Zhao, Nasrullah and Li, 2019): scores of different detectors
 * live on different scales, so each is first normalised (to ranks in $[0, 1]$, or to z-scores) and the normalised
 * scores are then combined by their mean (averaging lowers variance) or their maximum (keeps any detector's alarm).
 */

import { toFlat } from 'aifn-compute/foundation/tensor'
import { ranks, standardise } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `combineScores`. */
export type EnsembleOptions = {
  /**
   * `rank`: each score's rank, from 0, divided by $n - 1$ (ties averaged); `zscore`: $(s - \bar{s}) / \sigma$ with
   * the population standard deviation $\sigma$. Default `rank`.
   */
  normalise?: 'rank' | 'zscore'
  /** How the normalised scores of a point are combined: their `mean` (default) or their `max`. */
  combine?: 'mean' | 'max'
}

/**
 * The scores of one detector normalised to ranks in $[0, 1]$ (ties share their mean rank): $(r - 1)/(n - 1)$ for the
 * rank $r$ from 1. A single score maps to 0. NaN scores throw `DomainError`.
 *
 * @param scores One score per point, higher meaning more anomalous.
 * @returns The normalised rank of each score: 0 for the lowest, 1 for the highest.
 *
 * @example Ranks of four scores with a tie
 * print('ranks', rankNormalise([0.2, 5, 0.2, 1]))
 */
export function rankNormalise(scores: ArrayLike<number>): Float64Array {
  const n = scores.length
  return Float64Array.from(toFlat(ranks(Float64Array.from(scores))), (r) => (r - 1) / Math.max(1, n - 1))
}

/**
 * z-scores with the population standard deviation; a constant detector contributes zeros.
 *
 * @param scores One score per point.
 * @returns $(s_i - \bar{s}) / \sigma$ for each score, or all zeros when the scores are constant.
 */
function zNormalise(scores: ArrayLike<number>): Float64Array {
  const z = standardise(Float64Array.from(scores))
  return z.constant ? new Float64Array(scores.length) : Float64Array.from(toFlat(z.values))
}

/**
 * Combine several detectors' scores of the same points (one array per detector) into one score per point. Throws
 * `DomainError` when there are no detectors, and `ShapeError` when they scored different numbers of points.
 *
 * @param scores The scores, one array per detector, each with one score per point (higher meaning more anomalous).
 * @param options How each detector is normalised and how the normalised scores are combined.
 * @returns The combined score of each point.
 *
 * @example Two detectors on different scales agree on the third point
 * const a = [0.1, 0.2, 0.9, 0.3]
 * const b = [10, 40, 300, 20]
 * print('mean rank', combineScores([a, b]))
 * print('max z-score', combineScores([a, b], { normalise: 'zscore', combine: 'max' }))
 */
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
