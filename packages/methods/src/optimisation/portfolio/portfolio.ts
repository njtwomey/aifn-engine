/**
 * Portfolios of configurations chosen on a table of losses, one row per candidate configuration and one column per
 * task: the warm-start of a hyperparameter search from what worked on earlier tasks.
 *
 * The loss of a portfolio $\Pcal$ on task $t$ is that of its best member, and its loss over the tasks is the mean,
 *
 * $$ L(\Pcal) = \frac{1}{T} \sum_{t=1}^{T} \min\Big(w, \min_{c \in \Pcal} \ell_{ct}\Big), \qquad L(\emptyset) = w, $$
 *
 * with $w$ the loss of a task the portfolio has not yet covered (the worst value). The greedy portfolio adds, one at a
 * time, the candidate that lowers $L$ the most (Wistuba, Schilling and Schmidt-Thieme, 2015; Feurer, Eggensperger,
 * Falkner, Lindauer and Hutter, 2022). The reduction $w - L(\Pcal)$ is monotone and submodular in $\Pcal$, so the
 * greedy portfolio of size $k$ achieves at least $1 - (1 - 1/k)^k \ge 1 - 1/e$ of the best reduction any $k$
 * candidates achieve (Nemhauser, Wolsey and Fisher, 1978); for small tables the optimum is found by trying every
 * subset. The average rank of each candidate over the tasks is the simpler meta-learned ordering it is compared with.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'

const { toMatrixF64, vec } = dense

/** Options of `greedyPortfolio`, `portfolioLoss` and `leaveOneTaskOut`. */
export interface PortfolioOptions {
  /**
   * The loss $w$ of a task no member covers, and the cap on every loss (default the largest value of the table): an
   * error rate of 1, say, or the error of a default configuration.
   */
  worst?: number
  /**
   * The largest number of subsets of size $k$ that `greedyPortfolio` tries to find the optimum (default $10^5$); with
   * more, `optimal` is left out.
   */
  exhaustive?: number
}

/** The portfolio `greedyPortfolio` returns. */
export interface Portfolio {
  /** The candidates (rows of the table) in the order they were added. */
  order: number[]
  /** The reduction of the mean loss each addition made, $L(\Pcal_{s-1}) - L(\Pcal_s)$, non-increasing. */
  gain: number[]
  /** The mean loss $L(\Pcal_s)$ after each addition. */
  loss: number[]
  /** The loss on each task after each addition, $k \times T$. */
  perTask: Tensor
  /** The value $w$ used for uncovered tasks. */
  worst: number
  /** The best subset of size $k$ found by trying every one (in increasing order), when there are few enough. */
  optimal?: number[]
  /** The mean loss of `optimal`. */
  optimalLoss?: number
}

/**
 * The table's entries, checked: finite, at least one candidate and one task.
 *
 * @param table The candidate by task table.
 * @param where The caller's name, for error messages.
 * @returns The entries row-major, with `n` candidates and `T` tasks.
 */
function read(table: MatrixLike, where: string): { data: Float64Array; n: number; T: number } {
  const { data, m: n, n: T } = toMatrixF64(table, where)
  if (n === 0 || T === 0) throw new ShapeError(where, `${where}: the table needs at least one candidate and one task`)
  if (!data.every(Number.isFinite)) throw new DomainError(where, `${where}: every loss must be finite`)
  return { data, n, T }
}

/**
 * The loss of a task with no member, checked.
 *
 * @param data The table's entries.
 * @param worst The value given, if any.
 * @param where The caller's name, for error messages.
 * @returns $w$: `worst`, or the largest entry.
 */
function worstOf(data: Float64Array, worst: number | undefined, where: string): number {
  if (worst === undefined) return data.reduce((a, b) => Math.max(a, b), -Infinity)
  if (!Number.isFinite(worst)) throw new DomainError(where, `${where}: worst must be finite`)
  return worst
}

/**
 * The best loss per task over the given candidates, capped at $w$.
 *
 * @param data The table's entries, row-major.
 * @param T The number of tasks.
 * @param members The candidates.
 * @param worst $w$.
 * @returns The $T$ losses.
 */
function bestPerTask(
  data: Float64Array,
  T: number,
  members: readonly number[],
  worst: number,
): Float64Array<ArrayBuffer> {
  const best = new Float64Array(T).fill(worst)
  for (const c of members) for (let t = 0; t < T; t++) best[t] = Math.min(best[t], data[c * T + t])
  return best
}

/**
 * The mean of an array of values.
 *
 * @param a The values, at least one.
 * @returns Their mean.
 */
function mean(a: ArrayLike<number>): number {
  let total = 0
  for (let i = 0; i < a.length; i++) total += a[i]
  return total / a.length
}

/**
 * The greedy portfolio: starting empty, add the candidate that most lowers the mean over tasks of the best member's
 * loss, $k$ times (see the file comment; ties go to the lower index). When there are at most `exhaustive` subsets of
 * size $k$, the optimal subset is found too, to set the greedy one against. Throws `ShapeError` for an empty table
 * and `DomainError` for a loss that is not finite or $k$ outside $1, \dots, n$.
 *
 * @param table The losses, $n \times T$: one row per candidate configuration, one column per task (as the `table`
 *   of `tuningTables` in `aifn-methods/data/objectives`).
 * @param k The size of the portfolio.
 * @param options The loss of an uncovered task and the limit on the exhaustive search, as `PortfolioOptions`.
 * @returns The order of addition, the gains and losses, and the optimum when it was searched for, as `Portfolio`.
 *
 * @example Three tasks, four configurations: the greedy pair against the best pair
 * // Configuration 0 is good on average; 1, 2 and 3 are each the best on one task.
 * const table = [[0.35, 0.35, 0.35], [0.1, 0.6, 0.5], [0.5, 0.1, 0.6], [0.6, 0.5, 0.1]]
 * const p = greedyPortfolio(table, 2, { worst: 1 })
 * print('greedy order:', p.order, ' gains:', p.gain, ' loss:', p.loss)
 * print('optimal pair:', p.optimal, ' its loss:', p.optimalLoss)
 *
 * @example The first members of a portfolio on the SVM tuning tables
 * import { tuningTables } from 'aifn-methods/data/objectives'
 * const t = tuningTables()
 * const p = greedyPortfolio(t.table, 3, { exhaustive: 0 })
 * print('configurations (log10 C, log10 gamma):', p.order.map((c) => toArray(t.configurations)[c]))
 * print('mean best error:', p.loss)
 */
export function greedyPortfolio(table: MatrixLike, k: number, options: PortfolioOptions = {}): Portfolio {
  const where = 'greedyPortfolio'
  const { data, n, T } = read(table, where)
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new DomainError(where, `${where}: k must be an integer from 1 to ${n}`)
  const worst = worstOf(data, options.worst, where)
  const best = new Float64Array(T).fill(worst)
  const order: number[] = []
  const gain: number[] = []
  const loss: number[] = []
  const perTask = new Float64Array(k * T)
  let current = worst
  for (let s = 0; s < k; s++) {
    let pick = -1
    let pickLoss = Infinity
    for (let c = 0; c < n; c++) {
      if (order.includes(c)) continue
      let total = 0
      for (let t = 0; t < T; t++) total += Math.min(best[t], data[c * T + t])
      if (total / T < pickLoss) {
        pick = c
        pickLoss = total / T
      }
    }
    for (let t = 0; t < T; t++) best[t] = Math.min(best[t], data[pick * T + t])
    order.push(pick)
    gain.push(current - pickLoss)
    loss.push(pickLoss)
    perTask.set(best, s * T)
    current = pickLoss
  }
  const result: Portfolio = { order, gain, loss, perTask: fromData(perTask, [k, T]), worst }

  // The optimum by trying every subset of size k, when there are few enough of them.
  let subsets = 1
  for (let i = 0; i < k; i++) subsets = (subsets * (n - i)) / (i + 1)
  if (subsets <= (options.exhaustive ?? 1e5)) {
    let bestSet: number[] = []
    let bestLoss = Infinity
    const chosen: number[] = []
    const search = (from: number) => {
      if (chosen.length === k) {
        const l = mean(bestPerTask(data, T, chosen, worst))
        if (l < bestLoss) {
          bestLoss = l
          bestSet = [...chosen]
        }
        return
      }
      for (let c = from; c <= n - (k - chosen.length); c++) {
        chosen.push(c)
        search(c + 1)
        chosen.pop()
      }
    }
    search(0)
    result.optimal = bestSet
    result.optimalLoss = bestLoss
  }
  return result
}

/**
 * The loss of a given portfolio: on each task the best member's loss, capped at $w$, and their mean. Throws as
 * `greedyPortfolio` does, and `DomainError` for a member that is not a row of the table.
 *
 * @param table The losses, $n \times T$, one row per candidate and one column per task.
 * @param members The candidates in the portfolio (rows of the table); none gives $w$ on every task.
 * @param options The loss of an uncovered task, as `PortfolioOptions` (`exhaustive` is not used).
 * @returns `loss`, the mean over tasks, and `perTask`, the $T$ losses.
 *
 * @example The best single configuration on each task against a pair
 * const table = [[0.35, 0.35, 0.35], [0.1, 0.6, 0.5], [0.5, 0.1, 0.6], [0.6, 0.5, 0.1]]
 * print('{0}:', portfolioLoss(table, [0]).loss, ' {1, 2}:', portfolioLoss(table, [1, 2]).perTask)
 */
export function portfolioLoss(
  table: MatrixLike,
  members: readonly number[],
  options: PortfolioOptions = {},
): { loss: number; perTask: Tensor } {
  const where = 'portfolioLoss'
  const { data, n, T } = read(table, where)
  for (const c of members)
    if (!(Number.isInteger(c) && c >= 0 && c < n))
      throw new DomainError(where, `${where}: member ${c} is not a row of the table (0 to ${n - 1})`)
  const best = bestPerTask(data, T, members, worstOf(data, options.worst, where))
  return { loss: mean(best), perTask: vec(best) }
}

/**
 * Leave-one-task-out assessment of the greedy portfolio: for each task, build the portfolio of size $k$ on the other
 * tasks, then score its first $1, \dots, k$ members on the held-out task. This is how a portfolio is judged on a task
 * it has not seen. Needs at least two tasks; otherwise throws as `greedyPortfolio` does.
 *
 * @param table The losses, $n \times T$, one row per candidate and one column per task.
 * @param k The size of each portfolio.
 * @param options The loss of an uncovered task, as `PortfolioOptions`; $w$ is taken from the whole table when left
 *   out, so that every fold uses the same value.
 * @returns `order`, the portfolio built without each task ($T$ lists of $k$), `loss` ($T \times k$), the held-out
 *   task's best loss over the first $s$ members, and `mean` ($k$), its mean over the tasks.
 *
 * @example How two-member portfolios do on the task left out
 * const table = [[0.35, 0.35, 0.35], [0.1, 0.6, 0.5], [0.5, 0.1, 0.6], [0.6, 0.5, 0.1]]
 * const r = leaveOneTaskOut(table, 2, { worst: 1 })
 * print('portfolios:', r.order)
 * print('held-out loss after 1 and 2 members:', r.mean)
 */
export function leaveOneTaskOut(
  table: MatrixLike,
  k: number,
  options: PortfolioOptions = {},
): { order: number[][]; loss: Tensor; mean: Tensor } {
  const where = 'leaveOneTaskOut'
  const { data, n, T } = read(table, where)
  if (T < 2) throw new ShapeError(where, `${where}: needs at least two tasks`)
  const worst = worstOf(data, options.worst, where)
  const order: number[][] = []
  const loss = new Float64Array(T * k)
  for (let held = 0; held < T; held++) {
    const rest = Array.from({ length: n }, (_, c) =>
      Array.from({ length: T - 1 }, (_, t) => data[c * T + (t < held ? t : t + 1)]),
    )
    const p = greedyPortfolio(rest, k, { worst, exhaustive: 0 })
    order.push(p.order)
    let best = worst
    p.order.forEach((c, s) => {
      best = Math.min(best, data[c * T + held])
      loss[held * k + s] = best
    })
  }
  const means = Float64Array.from({ length: k }, (_, s) => {
    let total = 0
    for (let t = 0; t < T; t++) total += loss[t * k + s]
    return total / T
  })
  return { order, loss: fromData(loss, [T, k]), mean: vec(means) }
}

/**
 * The average rank of each candidate over the tasks: on each task the candidates are ranked by loss (1 the best, ties
 * sharing the mean of their ranks), and the ranks are averaged. Ordering by it is the simplest meta-learned ordering
 * of configurations, the baseline a greedy portfolio is set against. Throws as `greedyPortfolio` does.
 *
 * @param table The losses, $n \times T$, one row per candidate and one column per task.
 * @returns `rank`, the $n$ average ranks, and `order`, the candidates from the best average rank (ties by index).
 *
 * @example Configuration 0 is never the best but is ranked best on average
 * const table = [[0.35, 0.35, 0.35], [0.1, 0.6, 0.5], [0.5, 0.1, 0.6], [0.6, 0.5, 0.1]]
 * const r = averageRank(table)
 * print('average ranks:', r.rank, ' order:', r.order)
 */
export function averageRank(table: MatrixLike): { rank: Tensor; order: number[] } {
  const { data, n, T } = read(table, 'averageRank')
  const rank = new Float64Array(n)
  for (let t = 0; t < T; t++) {
    const sorted = Array.from({ length: n }, (_, c) => c).sort((a, b) => data[a * T + t] - data[b * T + t] || a - b)
    for (let i = 0; i < n;) {
      let j = i
      while (j + 1 < n && data[sorted[j + 1] * T + t] === data[sorted[i] * T + t]) j++
      for (let q = i; q <= j; q++) rank[sorted[q]] += (i + j) / 2 + 1
      i = j + 1
    }
  }
  for (let c = 0; c < n; c++) rank[c] /= T
  const order = Array.from({ length: n }, (_, c) => c).sort((a, b) => rank[a] - rank[b] || a - b)
  return { rank: vec(rank), order }
}
