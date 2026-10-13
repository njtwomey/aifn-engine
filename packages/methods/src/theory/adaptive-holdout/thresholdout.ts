/**
 * Thresholdout (Dwork, Feldman, Hardt, Pitassi, Reingold and Roth, 2015): a reusable holdout that answers a query from
 * the training set while training and holdout agree, and reveals a noisy holdout answer only when they do not.
 *
 * A query is a function $\phi$ of an example with values in $[0, 1]$, and $\expect_S[\phi]$ its mean over a sample
 * $S$. With threshold $T$, noise scale $\sigma$ and budget $B$, the mechanism (Figure 1 of the NeurIPS version, and the
 * Science paper) is:
 *
 * 1. draw $\gamma \sim \operatorname{Lap}(2\sigma)$ and set $\hat T = T + \gamma$;
 * 2. for each query: if $B < 1$, answer $\bot$; else draw $\eta \sim \operatorname{Lap}(4\sigma)$, and if
 *    $\lvert \expect_{S_h}[\phi] - \expect_{S_t}[\phi] \rvert > \hat T + \eta$, draw
 *    $\xi \sim \operatorname{Lap}(\sigma)$ and $\gamma \sim \operatorname{Lap}(2\sigma)$, set $B \gets B - 1$ and
 *    $\hat T = T + \gamma$, and answer $\expect_{S_h}[\phi] + \xi$; otherwise answer $\expect_{S_t}[\phi]$.
 *
 * $\operatorname{Lap}(b)$ is the Laplace distribution of scale $b$, drawn by inversion from one uniform. Every draw
 * comes from the one stream in the order above, so a stream fixes the whole transcript.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense } from 'aifn-compute/foundation/tensor'

/** Options of `thresholdout`. */
export interface ThresholdoutOptions {
  /** The threshold $T$ on the gap between the training and holdout means (default 0.04, as Dwork et al.). */
  threshold?: number
  /** The noise scale $\sigma$ (default 0.01, as Dwork et al.). */
  sigma?: number
  /** The budget $B$: how many holdout answers may be revealed (default unlimited). */
  budget?: number
}

/** A Thresholdout transcript. */
export interface ThresholdoutResult {
  /** The answer to each query; NaN ($\bot$) once the budget is spent. */
  answers: Float64Array
  /** Whether each query was answered from the holdout (the gap crossed the noisy threshold). */
  revealed: boolean[]
  /** The budget spent: the number of holdout answers. */
  spent: number
}

/**
 * A draw from $\operatorname{Lap}(b)$ by inversion: $-b \operatorname{sign}(u - \tfrac12)\log(1 - 2\lvert u - \tfrac12
 * \rvert)$ for $u$ uniform on $(0, 1)$.
 *
 * @param s The stream; advanced by one uniform.
 * @param b The scale $b$.
 * @returns The draw.
 */
export function laplaceDraw(s: Stream, b: number): number {
  const c = uniform(s) - 0.5
  return -b * Math.sign(c) * Math.log1p(-2 * Math.abs(c))
}

/** The state of a running Thresholdout, fed one query at a time. */
export interface ThresholdoutMechanism {
  /** Answer a query given its training and holdout means: the answer (NaN for $\bot$) and whether it is revealed. */
  answer(trainMean: number, holdoutMean: number): { value: number; revealed: boolean }
  /** What a query would be answered, with its noise from `s`, spending no budget and leaving the state unchanged. */
  peek(trainMean: number, holdoutMean: number, s: Stream): number
  /** The budget spent so far. */
  readonly spent: number
}

/**
 * A running Thresholdout (see the file's introduction), which draws $\hat T$ at once.
 *
 * @param s The stream every noise draw comes from, in the paper's order; advanced as the queries are answered.
 * @param options The threshold, noise scale and budget (see `ThresholdoutOptions`).
 * @param where The caller's name, for error messages.
 * @returns The mechanism.
 */
export function thresholdoutMechanism(
  s: Stream,
  options: ThresholdoutOptions = {},
  where = 'thresholdout',
): ThresholdoutMechanism {
  const { threshold = 0.04, sigma = 0.01, budget = Infinity } = options
  if (!(threshold >= 0)) throw new DomainError(where, `${where}: threshold must be non-negative, got ${threshold}`)
  if (!(sigma >= 0)) throw new DomainError(where, `${where}: sigma must be non-negative, got ${sigma}`)
  if (!(budget >= 0)) throw new DomainError(where, `${where}: budget must be non-negative, got ${budget}`)
  let spent = 0
  let noisy = threshold + laplaceDraw(s, 2 * sigma)
  return {
    answer(trainMean, holdoutMean) {
      if (budget - spent < 1) return { value: NaN, revealed: false }
      const eta = laplaceDraw(s, 4 * sigma)
      if (Math.abs(holdoutMean - trainMean) > noisy + eta) {
        const xi = laplaceDraw(s, sigma)
        const gamma = laplaceDraw(s, 2 * sigma)
        spent += 1
        noisy = threshold + gamma
        return { value: holdoutMean + xi, revealed: true }
      }
      return { value: trainMean, revealed: false }
    },
    peek(trainMean, holdoutMean, t) {
      if (budget - spent < 1) return NaN
      const eta = laplaceDraw(t, 4 * sigma)
      if (Math.abs(holdoutMean - trainMean) > noisy + eta) return holdoutMean + laplaceDraw(t, sigma)
      return trainMean
    },
    get spent() {
      return spent
    },
  }
}

/**
 * The row means of a matrix.
 *
 * @param x The matrix: row $j$ holds query $j$'s value on each example.
 * @param where The caller's name, for error messages.
 * @returns The mean of each row.
 */
function rowMeans(x: MatrixLike, where: string): Float64Array {
  const { data, m, n } = dense.toMatrixF64(x, where)
  if (n < 1) throw new ShapeError(where, `${where}: needs at least one example`)
  return Float64Array.from({ length: m }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += data[j * n + i]
    return s / n
  })
}

/**
 * Thresholdout (Dwork et al., 2015) on a sequence of queries given by their values on the training and holdout rows
 * (see the file's introduction). The queries are answered in order; the adaptivity the mechanism guards against is the
 * analyst's, who would choose query $j + 1$ after seeing answer $j$. Throws `ShapeError` when the two matrices hold
 * different numbers of queries, `DomainError` for a negative threshold, noise scale or budget.
 *
 * @param s The stream all noise is drawn from, in the paper's order (so the same stream gives the same transcript);
 *   advanced.
 * @param train The queries' values on the training set $S_t$: $m \times n_t$, row $j$ for query $j$.
 * @param holdout The queries' values on the holdout $S_h$: $m \times n_h$, row $j$ for query $j$.
 * @param options The threshold $T$, noise scale $\sigma$ and budget $B$ (see `ThresholdoutOptions`).
 * @returns The answers ($\bot$ as NaN), which of them came from the holdout, and the budget spent.
 *
 * @example A query that overfits the training set gets a noisy holdout answer
 * const train = [[1, 1, 0, 0], [1, 1, 1, 1]] // means 0.5 and 1
 * const holdout = [[1, 0, 1, 0], [0, 1, 0, 1]] // means 0.5 and 0.5
 * const r = thresholdout(stream(0), train, holdout, { threshold: 0.1, sigma: 0.01, budget: 5 })
 * print('answers:', r.answers)
 * print('from the holdout:', r.revealed, ' spent:', r.spent)
 */
export function thresholdout(
  s: Stream,
  train: MatrixLike,
  holdout: MatrixLike,
  options: ThresholdoutOptions = {},
): ThresholdoutResult {
  const t = rowMeans(train, 'thresholdout')
  const h = rowMeans(holdout, 'thresholdout')
  if (t.length !== h.length)
    throw new ShapeError('thresholdout', `thresholdout: ${t.length} training queries but ${h.length} holdout queries`)
  const mechanism = thresholdoutMechanism(s, options)
  const answers = new Float64Array(t.length)
  const revealed: boolean[] = []
  for (let j = 0; j < t.length; j++) {
    const a = mechanism.answer(t[j], h[j])
    answers[j] = a.value
    revealed.push(a.revealed)
  }
  return { answers, revealed, spent: mechanism.spent }
}
