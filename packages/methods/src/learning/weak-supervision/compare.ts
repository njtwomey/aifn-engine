/**
 * `labelModelReport`: majority vote, Dawid–Skene EM (step by step) and the data-programming label model on the same
 * votes, with each method's estimate of every voter's accuracy and, when the true labels are given, the accuracy of the
 * labels each method assigns.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dawidSkeneSteps, labelModel, majorityVote, votesOf, type Votes } from './label-models'
import { stream, child } from 'aifn-compute/foundation/random'

/** One EM step of Dawid–Skene as a page draws it. */
export type DawidSkeneFrame = {
  step: Size
  logLikelihood: number
  /** Each voter's estimated accuracy Σ_k π_k θ_j[k][k]. */
  accuracy: number[]
  /** Accuracy of the argmax labels against the truth (NaN without it). */
  labelAccuracy: number
}

/** The report of `labelModelReport`. */
export type LabelModelReport = {
  /** Posteriors [n, K] of each method, row-major. */
  majority: Float64Array
  dawidSkene: Float64Array
  labelModel: Float64Array
  /** Every Dawid–Skene EM step, from the majority-vote start. */
  frames: DawidSkeneFrame[]
  /** The label model's accuracy and coverage per voter. */
  labelModelAccuracy: number[]
  labelModelCoverage: number[]
  /** Accuracy of each method's argmax labels against the truth (NaN without it); majority-vote ties count as half. */
  accuracy: { majority: number; dawidSkene: number; labelModel: number }
  /** The empirical accuracy and coverage of each voter against the truth (NaN without it). */
  empiricalAccuracy: number[]
  empiricalCoverage: number[]
}

const argmaxAccuracy = (post: ArrayLike<number>, K: Size, truth: ArrayLike<number> | undefined) => {
  if (!truth) return NaN
  let score = 0
  for (let i = 0; i < truth.length; i++) {
    let top = -Infinity
    for (let k = 0; k < K; k++) top = Math.max(top, post[i * K + k])
    let ties = 0
    for (let k = 0; k < K; k++) if (post[i * K + k] === top) ties++
    if (post[i * K + truth[i]] === top) score += 1 / ties
  }
  return score / truth.length
}

/** Run every label model on the votes (example × voter, −1 abstains) and report the estimates side by side. */
export function labelModelReport(
  votes: Tensor | Votes,
  classes: Size,
  truth?: ArrayLike<number>,
  options: { steps?: Size } = {},
): LabelModelReport {
  const v = 'votes' in votes && 'm' in votes ? (votes as Votes) : votesOf(votes as Tensor)
  const { n, m } = v
  const K = classes
  const mv = toFlat(majorityVote(v, K))
  const alg = dawidSkeneSteps(v, K)
  const s0 = stream('dawid-skene')
  let s = alg.init(undefined, s0)
  const frames: DawidSkeneFrame[] = []
  const frame = () => {
    const pri = toFlat(s.priors)
    const conf = toFlat(s.confusions)
    frames.push({
      step: s.t,
      logLikelihood: s.logLikelihood,
      accuracy: Array.from({ length: m }, (_, j) => {
        let a = 0
        for (let k = 0; k < K; k++) a += pri[k] * conf[(j * K + k) * K + k]
        return a
      }),
      labelAccuracy: argmaxAccuracy(toFlat(s.posteriors), K, truth),
    })
  }
  frame()
  for (let t = 0; t < (options.steps ?? 30); t++) {
    s = alg.step(s, { t, stream: child(s0, t) })
    frame()
  }
  const lm = labelModel(v, K)
  const lmPost = toFlat(lm.posteriors)
  const empiricalAccuracy: number[] = []
  const empiricalCoverage: number[] = []
  for (let j = 0; j < m; j++) {
    let cast = 0
    let right = 0
    for (let i = 0; i < n; i++) {
      const x = v.votes[i * m + j]
      if (x < 0) continue
      cast++
      if (truth && x === truth[i]) right++
    }
    empiricalCoverage.push(cast / n)
    empiricalAccuracy.push(truth && cast > 0 ? right / cast : NaN)
  }
  return {
    majority: Float64Array.from(mv),
    dawidSkene: Float64Array.from(toFlat(s.posteriors)),
    labelModel: Float64Array.from(lmPost),
    frames,
    labelModelAccuracy: Array.from(lm.accuracy),
    labelModelCoverage: Array.from(lm.coverage),
    accuracy: {
      majority: argmaxAccuracy(mv, K, truth),
      dawidSkene: argmaxAccuracy(toFlat(s.posteriors), K, truth),
      labelModel: argmaxAccuracy(lmPost, K, truth),
    },
    empiricalAccuracy,
    empiricalCoverage,
  }
}
