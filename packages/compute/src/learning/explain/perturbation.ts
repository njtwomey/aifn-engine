/**
 * Perturbation curves: how fast a prediction decays as the inputs an explanation ranks first are removed. Given an
 * ordering O of an input's positions (most relevant first, MoRF) and a predictor that scores the input with some
 * positions removed, the curve is f(X), f(X \ O₁), f(X \ {O₁, O₂}), …; a faithful ordering makes it fall fast.
 *
 * - **AOPC** (area over the perturbation curve; Samek et al. 2017, "Evaluating the visualization of what a deep neural
 *   network has learned", IEEE TNNLS 28(11)): the mean drop (1/J) Σ_{j=1}^{J} [f(X) − f(MoRF_j)] over the J steps.
 * - **AOPCR** (AOPC relative to random; Early et al. 2021, used by MILLET, Early et al. 2024, App. D.1): AOPC of the
 *   ordering minus the mean AOPC of R random orderings, so 0 is no better than chance.
 *
 * Positions are removed in blocks (MILLET: blocks of 5% of the length, up to 50% removed, ten calls). How a removed
 * position is treated (dropped from a bag, set to a baseline, blurred) is the predictor's business: it receives the
 * mask of kept positions; `deletionCurve` is the batched special case that sets removed features to a baseline. The
 * MoRF order is `attributionOrder`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { attributionOrder } from './evaluation'

/** A prediction (a class score, a logit) of the input with only the positions where `kept[i] = 1` left in. */
export type MaskedPredictor = (kept: Uint8Array) => number

/** Options of the perturbation curve. */
export interface PerturbationOptions {
  /** Positions removed per step (default 1; MILLET uses 5% of the length). */
  block?: Size
  /** Largest share of positions removed (default: all but one, as in the unblocked AOPC; MILLET uses 0.5). */
  until?: number
}

/** A perturbation curve: positions removed at each step (0 first) and the prediction after removing them. */
export interface PerturbationCurve {
  readonly removed: readonly number[]
  readonly values: readonly number[]
}

/** The MoRF perturbation curve of `order` (a permutation of 0 … L − 1): remove positions block by block in that order. */
export function perturbationCurve(
  predict: MaskedPredictor,
  order: ArrayLike<number>,
  options: PerturbationOptions = {},
): PerturbationCurve {
  const L = order.length
  const { block = 1, until } = options
  if (!(Number.isInteger(block) && block >= 1))
    throw new DomainError('perturbationCurve', `perturbationCurve: block must be a positive integer, got ${block}`)
  const limit = until === undefined ? L - 1 : Math.floor(until * L)
  if (!(limit >= 0 && limit <= L - 1))
    throw new DomainError('perturbationCurve', 'perturbationCurve: `until` must leave at least one position')
  const kept = new Uint8Array(L).fill(1)
  const removed = [0]
  const values = [predict(kept)]
  for (let r = 0; r < limit;) {
    const next = Math.min(limit, r + block)
    for (; r < next; r++) kept[order[r]] = 0
    removed.push(r)
    values.push(predict(kept))
  }
  return { removed, values }
}

/** AOPC: the mean drop f(X) − f(MoRF_j) over the steps j ≥ 1 of a perturbation curve. */
export function aopc(curve: PerturbationCurve): number {
  const J = curve.values.length - 1
  if (J < 1) return 0
  let sum = 0
  for (let j = 1; j <= J; j++) sum += curve.values[0] - curve.values[j]
  return sum / J
}

/** The result of {@link aopcr}: the ordering's AOPC, the random orderings' and their difference. */
export interface Aopcr {
  readonly aopcr: number
  readonly aopc: number
  readonly randomAopc: readonly number[]
  readonly curve: PerturbationCurve
  readonly randomCurves: readonly PerturbationCurve[]
}

/**
 * AOPCR: AOPC of the MoRF order of `scores` minus the mean AOPC of `repeats` random orders (default 3, as MILLET),
 * each drawn from `child(s, 'random', r)`.
 */
export function aopcr(
  s: Stream,
  predict: MaskedPredictor,
  scores: ArrayLike<number>,
  options: PerturbationOptions & { repeats?: Size } = {},
): Aopcr {
  const { repeats = 3, ...curveOptions } = options
  const curve = perturbationCurve(predict, attributionOrder(scores), curveOptions)
  const randomCurves = Array.from({ length: repeats }, (_, r) =>
    perturbationCurve(predict, toFlat(permutation(child(s, 'random', r), scores.length)), curveOptions),
  )
  const randomAopc = randomCurves.map(aopc)
  const own = aopc(curve)
  const mean = randomAopc.reduce((a, v) => a + v, 0) / Math.max(1, repeats)
  return { aopcr: own - mean, aopc: own, randomAopc, curve, randomCurves }
}
