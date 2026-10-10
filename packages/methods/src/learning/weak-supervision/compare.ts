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
  /** The EM step (0 at the majority-vote start). */
  step: Size
  /** The log-likelihood of the votes at this step. */
  logLikelihood: number
  /** Each voter's estimated accuracy $\sum_k \pi_k \theta_{jkk}$. */
  accuracy: number[]
  /** Accuracy of the argmax labels against the truth (NaN without it). */
  labelAccuracy: number
}

/** The report of `labelModelReport`. */
export type LabelModelReport = {
  /** Majority vote's posteriors, $n \times K$, row-major. */
  majority: Float64Array
  /** Dawid–Skene's posteriors after the last step, $n \times K$, row-major. */
  dawidSkene: Float64Array
  /** The data-programming label model's posteriors, $n \times K$, row-major. */
  labelModel: Float64Array
  /** Every Dawid–Skene EM step, from the majority-vote start. */
  frames: DawidSkeneFrame[]
  /** The label model's accuracy per voter. */
  labelModelAccuracy: number[]
  /** The label model's coverage per voter. */
  labelModelCoverage: number[]
  /** Accuracy of each method's argmax labels against the truth (NaN without it); majority-vote ties count as half. */
  accuracy: { majority: number; dawidSkene: number; labelModel: number }
  /**
   * Each voter's empirical accuracy: the share of its votes that match the truth (NaN without the truth, or for a voter
   * that never voted).
   */
  empiricalAccuracy: number[]
  /** Each voter's empirical coverage: the share of examples it voted on (needs no truth). */
  empiricalCoverage: number[]
}

/**
 * The accuracy of the argmax labels of posteriors against the truth; an example whose top probability is tied between
 * $t$ classes scores $1/t$ when the true class is one of them.
 *
 * @param post The posteriors, $n \times K$, row-major.
 * @param K The number of classes.
 * @param truth The true class of each example, or undefined.
 * @returns The accuracy in $[0, 1]$, or NaN without the truth.
 */
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

/**
 * Run every label model on the votes and report the estimates side by side: majority vote, a fixed number of
 * Dawid–Skene EM steps from the majority-vote start (each recorded as a frame, without stopping at convergence) and the
 * data-programming label model with its default options.
 *
 * @param votes The votes, $n$ examples by $m$ voters, $-1$ for an abstention: a tensor of shape $[n, m]$ or `Votes`.
 * @param classes The number of classes $K$.
 * @param truth The true class of each example, when known; it adds the accuracies against the truth.
 * @param options `steps`, the number of Dawid–Skene EM steps (default 30).
 * @returns The posteriors, estimates and accuracies of the three methods.
 *
 * @example Majority vote, Dawid–Skene and the label model on three labelling functions
 * const s = stream(9)
 * const acc = [0.9, 0.7, 0.6]
 * const truth = Array.from({ length: 150 }, () => (uniform(s) < 0.5 ? 1 : 0))
 * const rows = truth.map((y) => acc.map((a) => (uniform(s) < 0.8 ? (uniform(s) < a ? y : 1 - y) : -1)))
 * const r = labelModelReport(votesOf(rows), 2, truth, { steps: 10 })
 * print('empirical accuracy per function:', r.empiricalAccuracy)
 * print('label model estimate:', r.labelModelAccuracy)
 * print('Dawid–Skene estimate:', r.frames.at(-1).accuracy)
 * print('accuracy of the labels:', r.accuracy)
 */
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
