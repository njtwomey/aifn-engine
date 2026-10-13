/**
 * The boosting attack of Blum and Hardt (2015, Section 5) on a leaderboard: submit $k$ random label vectors for a
 * holdout of $n$ random labels, keep those the leaderboard scores at loss at most $\tfrac12$, and submit their
 * coordinate-wise majority vote.
 *
 * Every submission's true loss is exactly $\tfrac12$, yet on a leaderboard that reports full-precision losses the
 * vote's holdout loss falls like $\tfrac12 - \Omega(\sqrt{k/n})$ (their Theorem 5.1). Against the Ladder the attacker
 * keeps, as Blum and Hardt do, only the submissions that moved the leaderboard, since the others reveal nothing; the
 * bias then grows only like $\sqrt{\log(k)/n}$. Against Thresholdout the attacker also holds a training set of $n$
 * random labels on which each random submission is scored, and keeps those whose answer is at most $\tfrac12$.
 *
 * The study is reported after every submission $t$, as if the attack had stopped there: the vote of the submissions
 * kept among the first $t$, its loss on the holdout, its loss on $n$ fresh labels (the truth, $\tfrac12$ in
 * expectation) and what the mechanism would report for it.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, randomBits, type Stream } from 'aifn-compute/foundation/random'
import { ladderBoard, type LadderBoard } from './ladder'
import { thresholdoutMechanism, type ThresholdoutMechanism, type ThresholdoutOptions } from './thresholdout'

/**
 * How the leaderboard answers: `exact` (the holdout loss), `rounded` (to `digits` decimals, as Kaggle), `ladder` (step
 * `eta`), `parameter-free` (the parameter-free Ladder) or `thresholdout` (against a training set).
 */
export type AttackMechanism = 'exact' | 'rounded' | 'ladder' | 'parameter-free' | 'thresholdout'

/** Options of `boostingAttack`. */
export interface BoostingAttackOptions extends ThresholdoutOptions {
  /** The holdout size $n$ (Blum and Hardt use 4000). */
  n: number
  /** The number of random submissions $k$. */
  k: number
  /** The leaderboard under attack (see `AttackMechanism`). */
  mechanism: AttackMechanism
  /** The Ladder's step $\eta$ (default 0.01). */
  eta?: number
  /** The decimals a `rounded` leaderboard reports (default 5, as Kaggle). */
  digits?: number
}

/** The attack after each submission $t = 1, \dots, k$. */
export interface BoostingAttackResult {
  /** The vote's loss on the holdout labels. */
  holdout: Float64Array
  /** The vote's loss on fresh labels: its true loss is $\tfrac12$. */
  fresh: Float64Array
  /** What the leaderboard would report for the vote (Thresholdout's peek spends no budget). */
  reported: Float64Array
  /** The leaderboard's answer to random submission $t$ (NaN once Thresholdout's budget is spent). */
  answers: Float64Array
  /** How many of the first $t$ submissions the attacker has kept. */
  kept: Int32Array
  /** Thresholdout's budget spent by the $k$ submissions (0 for the other mechanisms). */
  spent: number
}

/**
 * $n$ uniformly random bits, unpacked from 32-bit words.
 *
 * @param s The stream; advanced by $\lceil n/32 \rceil$ words.
 * @param n The number of bits.
 * @returns The bits, one per entry.
 */
function bitsOf(s: Stream, n: number): Uint8Array {
  const words = randomBits(s, Math.ceil(n / 32))
  return Uint8Array.from({ length: n }, (_, j) => (words[j >>> 5] >>> (j & 31)) & 1)
}

/**
 * The 0/1 losses of predictions against labels, and their mean.
 *
 * @param u The predicted bits.
 * @param y The label bits, as many as `u`.
 * @param out Overwritten with the per-example losses.
 * @returns The mean loss.
 */
function zeroOneLosses(u: Uint8Array, y: Uint8Array, out: Float64Array): number {
  let s = 0
  for (let j = 0; j < y.length; j++) {
    out[j] = u[j] === y[j] ? 0 : 1
    s += out[j]
  }
  return s / y.length
}

/**
 * The boosting attack (Blum and Hardt, 2015) against one leaderboard, reported after each submission (see the file's
 * introduction). The labels come from `child(s, 'labels')`, the fresh labels from `child(s, 'fresh')` and random
 * submission $i$ from `child(s, 'submission', i)`, so every mechanism faces the same submissions for the same `s`.
 * The vote breaks ties by bits from `child(s, 'ties')`. Thresholdout draws its noise from `child(s, 'mechanism')`, and
 * its answer for the vote after submission $t$ from `child(s, 'vote', t)`. Throws `DomainError` for $n < 2$, $k < 1$ or
 * a bad option.
 *
 * @param s The stream; not advanced.
 * @param options The holdout size, the number of submissions, the leaderboard and its parameters (see
 *   `BoostingAttackOptions`).
 * @returns The vote's holdout, fresh and reported loss after each submission, the answers and the number kept (see
 *   `BoostingAttackResult`).
 *
 * @example A full-precision leaderboard is fooled; the Ladder is not
 * for (const mechanism of ['exact', 'ladder']) {
 *   const r = boostingAttack(stream(0), { n: 1000, k: 200, mechanism, eta: 0.01 })
 *   print(mechanism, ' kept:', r.kept[199], ' holdout loss:', r.holdout[199], ' fresh loss:', r.fresh[199])
 * }
 */
export function boostingAttack(s: Stream, options: BoostingAttackOptions): BoostingAttackResult {
  const { n, k, mechanism, eta = 0.01, digits = 5 } = options
  const where = 'boostingAttack'
  if (!(Number.isInteger(n) && n >= 2)) throw new DomainError(where, `${where}: n must be an integer >= 2, got ${n}`)
  if (!(Number.isInteger(k) && k >= 1)) throw new DomainError(where, `${where}: k must be a positive integer, got ${k}`)
  if (!(Number.isInteger(digits) && digits >= 0))
    throw new DomainError(where, `${where}: digits must be a non-negative integer, got ${digits}`)
  if (!['exact', 'rounded', 'ladder', 'parameter-free', 'thresholdout'].includes(mechanism))
    throw new DomainError(where, `${where}: unknown mechanism '${mechanism}'`)
  const isLadder = mechanism === 'ladder' || mechanism === 'parameter-free'
  const withTrain = mechanism === 'thresholdout'
  const board: LadderBoard | undefined = isLadder
    ? ladderBoard(n, mechanism === 'ladder' ? eta : 'parameter-free')
    : undefined
  const guard: ThresholdoutMechanism | undefined = withTrain
    ? thresholdoutMechanism(child(s, 'mechanism'), options, where)
    : undefined
  const scale = 10 ** digits
  const rounded = (x: number) => Math.round(x * scale) / scale

  const y = bitsOf(child(s, 'labels'), n)
  const yFresh = bitsOf(child(s, 'fresh'), n)
  const yTrain = withTrain ? bitsOf(child(s, 'train-labels'), n) : new Uint8Array(0)
  const ties = bitsOf(child(s, 'ties'), 2 * n)
  // Vote counts (+1 per kept 1, -1 per kept 0) on the holdout and on the training set.
  const counts = new Int32Array(n)
  const countsTrain = new Int32Array(withTrain ? n : 0)
  const vote = new Uint8Array(n)
  const voteTrain = new Uint8Array(withTrain ? n : 0)
  const losses = new Float64Array(n)
  const voteLosses = new Float64Array(n)
  const scratch = new Float64Array(n)

  const out: BoostingAttackResult = {
    holdout: new Float64Array(k),
    fresh: new Float64Array(k),
    reported: new Float64Array(k),
    answers: new Float64Array(k),
    kept: new Int32Array(k),
    spent: 0,
  }
  let kept = 0
  let voteLoss = NaN
  let voteFresh = NaN
  let voteTrainLoss = NaN
  const recount = () => {
    for (let j = 0; j < n; j++) vote[j] = counts[j] > 0 ? 1 : counts[j] < 0 ? 0 : ties[j]
    voteLoss = zeroOneLosses(vote, y, voteLosses)
    voteFresh = zeroOneLosses(vote, yFresh, scratch)
    if (withTrain) {
      for (let j = 0; j < n; j++) voteTrain[j] = countsTrain[j] > 0 ? 1 : countsTrain[j] < 0 ? 0 : ties[n + j]
      voteTrainLoss = zeroOneLosses(voteTrain, yTrain, scratch)
    }
  }
  recount()

  for (let t = 0; t < k; t++) {
    const u = bitsOf(child(s, 'submission', t), n)
    const uTrain = withTrain ? bitsOf(child(s, 'submission-train', t), n) : u
    const loss = zeroOneLosses(u, y, losses)
    let answer: number
    let keep: boolean
    if (board) {
      const r = board.submit(losses)
      answer = r.reported
      keep = r.updated
    } else {
      answer =
        mechanism === 'exact'
          ? loss
          : mechanism === 'rounded'
            ? rounded(loss)
            : guard!.answer(zeroOneLosses(uTrain, yTrain, scratch), loss).value
      keep = answer <= 0.5
    }
    out.answers[t] = answer
    if (keep) {
      kept++
      for (let j = 0; j < n; j++) counts[j] += u[j] === 1 ? 1 : -1
      if (withTrain) for (let j = 0; j < n; j++) countsTrain[j] += uTrain[j] === 1 ? 1 : -1
      recount()
    }
    out.kept[t] = kept
    out.holdout[t] = voteLoss
    out.fresh[t] = voteFresh
    out.reported[t] = board
      ? board.peek(voteLosses)
      : mechanism === 'exact'
        ? voteLoss
        : mechanism === 'rounded'
          ? rounded(voteLoss)
          : guard!.peek(voteTrainLoss, voteLoss, child(s, 'vote', t))
  }
  out.spent = guard?.spent ?? 0
  return out
}
