/**
 * The Ladder of Blum and Hardt (2015): a leaderboard that reports a submission's holdout loss only when it beats the
 * best reported so far by a margin, and otherwise repeats the old number.
 *
 * With step $\eta$ (their Figure 1), $R_0 = \infty$ and, for submission $t$ with holdout loss $R_S(f_t)$,
 * $$
 * R_t = \begin{cases} [R_S(f_t)]_\eta & \text{if } R_S(f_t) < R_{t-1} - \eta, \\ R_{t-1} & \text{else,} \end{cases}
 * $$
 * where $[x]_\eta$ is $x$ rounded to the nearest multiple of $\eta$. The parameter-free Ladder (their Figure 2) keeps
 * the loss vector $\boldsymbol\ell$ of the best submission so far (all zeros at the start) and replaces $\eta$ by
 * $s/\sqrt n$, with $s$ the sample standard deviation (divisor $n - 1$) of $\boldsymbol\ell_t - \boldsymbol\ell$: a
 * one-sided paired $t$-test at about the 0.15 level. It then reports $[R_S(f_t)]_{1/n}$.
 *
 * The leaderboards here are step functions over a stream of loss vectors (`ladderBoard`), so the batch functions and
 * the boosting attack of this module share one implementation.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense } from 'aifn-compute/foundation/tensor'

/** What a leaderboard reports for a sequence of submissions. */
export interface LadderResult {
  /** The reported loss $R_t$ after each submission. */
  reported: Float64Array
  /** Whether the leaderboard moved at each submission (the first one always moves, from $R_0 = \infty$). */
  updated: boolean[]
}

/**
 * A leaderboard fed one loss vector at a time: the Ladder with a fixed step, or the parameter-free Ladder.
 * `submit` records a submission and returns what is reported; `peek` returns what would be reported for a submission
 * without recording it.
 */
export interface LadderBoard {
  /** Record a submission's per-example losses (length $n$, values in $[0, 1]$). */
  submit(losses: ArrayLike<number>): { reported: number; updated: boolean }
  /** What `submit` would report for these losses, leaving the leaderboard unchanged. */
  peek(losses: ArrayLike<number>): number
  /** The current reported loss ($\infty$ before the first submission). */
  readonly best: number
}

/**
 * $[x]_\eta$: $x$ rounded to the nearest integer multiple of $\eta$.
 *
 * @param x The value.
 * @param step The step $\eta > 0$. The result divides by $1/\eta$, which is exact when $1/\eta$ is an integer
 *   (such as $n$), so $[0.3]_{0.05}$ prints as 0.3.
 * @returns The rounded value.
 */
export function roundTo(x: number, step: number): number {
  return Math.round(x / step) / (1 / step)
}

/**
 * The mean of the first `n` values.
 *
 * @param v The values.
 * @param n How many of them to average.
 * @returns Their mean.
 */
function mean(v: ArrayLike<number>, n: number): number {
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]
  return s / n
}

/**
 * A leaderboard over holdout losses of $n$ examples: the Ladder with step `eta`, or the parameter-free Ladder when
 * `eta` is `'parameter-free'` (see the file's introduction).
 *
 * @param n The holdout size $n$: every loss vector has this length.
 * @param eta The step $\eta > 0$, or `'parameter-free'`.
 * @returns The leaderboard, empty ($R_0 = \infty$).
 */
export function ladderBoard(n: number, eta: number | 'parameter-free'): LadderBoard {
  const where = eta === 'parameter-free' ? 'ladderParameterFree' : 'ladder'
  if (!(Number.isInteger(n) && n >= 2)) throw new DomainError(where, `${where}: needs at least 2 examples, got ${n}`)
  if (eta !== 'parameter-free' && !(eta > 0)) throw new DomainError(where, `${where}: eta must be positive, got ${eta}`)
  let best = Infinity
  // The parameter-free Ladder's loss vector of the best submission, all zeros at the start.
  const previous = new Float64Array(n)
  const decide = (losses: ArrayLike<number>): number | undefined => {
    if (losses.length !== n) throw new DomainError(where, `${where}: expected ${n} losses, got ${losses.length}`)
    const r = mean(losses, n)
    if (eta !== 'parameter-free') return r < best - eta ? roundTo(r, eta) : undefined
    let d = 0
    for (let i = 0; i < n; i++) d += losses[i] - previous[i]
    d /= n
    let ss = 0
    for (let i = 0; i < n; i++) ss += (losses[i] - previous[i] - d) ** 2
    const s = Math.sqrt(ss / (n - 1))
    return r < best - s / Math.sqrt(n) ? roundTo(r, 1 / n) : undefined
  }
  return {
    submit(losses) {
      const r = decide(losses)
      if (r === undefined) return { reported: best, updated: false }
      best = r
      if (eta === 'parameter-free') for (let i = 0; i < n; i++) previous[i] = losses[i]
      return { reported: best, updated: true }
    },
    peek: (losses) => decide(losses) ?? best,
    get best() {
      return best
    },
  }
}

/**
 * Feed the rows of a $k \times n$ loss matrix to a leaderboard in order.
 *
 * @param losses The losses, row $t$ for submission $t$; every value must lie in $[0, 1]$.
 * @param eta The step $\eta$, or `'parameter-free'`.
 * @param where The caller's name, for error messages.
 * @returns What the leaderboard reports after each row, and whether it moved.
 */
function runBoard(losses: MatrixLike, eta: number | 'parameter-free', where: string): LadderResult {
  const { data, m: k, n } = dense.toMatrixF64(losses, where)
  for (const v of data)
    if (!(v >= 0 && v <= 1)) throw new DomainError(where, `${where}: losses must lie in [0, 1], got ${v}`)
  const board = ladderBoard(n, eta)
  const reported = new Float64Array(k)
  const updated: boolean[] = []
  for (let t = 0; t < k; t++) {
    const r = board.submit(data.subarray(t * n, (t + 1) * n))
    reported[t] = r.reported
    updated.push(r.updated)
  }
  return { reported, updated }
}

/**
 * The Ladder mechanism (Blum and Hardt, 2015, Figure 1): the leaderboard reports a submission's holdout loss, rounded
 * to a multiple of $\eta$, only when it is below the current value by more than $\eta$. With losses in $[0, 1]$ it
 * updates at most $1/\eta + 1$ times, whatever the analyst does, which bounds what the reports reveal about the
 * holdout. Throws `DomainError` for a loss outside $[0, 1]$ or $\eta \le 0$.
 *
 * @param losses The per-example holdout losses of the submissions, $k \times n$ in submission order: row $t$ holds
 *   submission $t$'s loss on each of the $n$ holdout examples.
 * @param options The step.
 * @param options.eta The step $\eta > 0$: the margin a submission must beat and the precision of what is reported.
 * @returns The reported loss $R_t$ after each submission, and whether it moved (see `LadderResult`).
 *
 * @example Small improvements are not reported
 * const losses = [
 *   [1, 1, 0, 0, 1, 0, 0, 0, 1, 0], // 0.4
 *   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0], // 0.3: better by 0.1
 *   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0], // 0.2: better by 0.1
 * ]
 * print('eta 0.05:', ladder(losses, { eta: 0.05 }).reported)
 * print('eta 0.15:', ladder(losses, { eta: 0.15 }).reported)
 */
export function ladder(losses: MatrixLike, options: { eta: number }): LadderResult {
  return runBoard(losses, options.eta, 'ladder')
}

/**
 * The parameter-free Ladder (Blum and Hardt, 2015, Figure 2): a submission updates the leaderboard when its mean loss
 * is below the current value by more than $s/\sqrt n$, $s$ the sample standard deviation of its loss differences from
 * the best submission so far; the update reports the mean to precision $1/n$ and makes the submission the new best.
 * The test is a heuristic: adaptively chosen submissions break its assumptions. Throws `DomainError` for a loss
 * outside $[0, 1]$.
 *
 * @param losses The per-example holdout losses of the submissions, $k \times n$ in submission order.
 * @returns The reported loss $R_t$ after each submission, and whether it moved (see `LadderResult`).
 *
 * @example A tiny, consistent improvement counts; a noisy one of the same size does not
 * const best = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] // 0.4
 * const consistent = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] // 0.2: two examples fixed
 * const noisy = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0] // 0.2: four fixed and two broken
 * print('consistent:', ladderParameterFree([best, consistent]).updated)
 * print('noisy:     ', ladderParameterFree([best, noisy]).updated)
 */
export function ladderParameterFree(losses: MatrixLike): LadderResult {
  return runBoard(losses, 'parameter-free', 'ladderParameterFree')
}
