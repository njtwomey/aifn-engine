/**
 * Ranking and retrieval metrics: precision and recall at k, R-precision, hit rate, average precision (MAP), reciprocal
 * rank (MRR), DCG and nDCG with an explicit gain, and expected reciprocal rank. The beyond-accuracy metrics of
 * recommendation lists (intra-list diversity, catalogue coverage, Gini and Herfindahl concentration, novelty) are in
 * `aifn-methods/evaluation`.
 *
 * Every ranking metric takes either
 * - `(grades, options)`: relevance grades already in rank order (rank 1 first), as in the notes; or
 * - `(relevance, scores, options)`: each item's relevance and the scores to rank it by (decreasing), as scikit-learn's
 *   `ndcg_score(y_true, y_score)`.
 * A matrix (rows as queries) gives the mean over queries. Binary metrics count a grade $g > 0$ as relevant. Ties in
 * scores are broken by input order, except in DCG and nDCG, which average over tied items (the expected value over
 * random tie orders, as scikit-learn).
 */

import {
  defineMetric,
  dense,
  divide,
  isArrayInput,
  isMatrixLike,
  orderDescending,
  values,
  type Data,
  type Rows,
} from './core'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Grades in rank order, or item relevance with scores, for one query (vectors) or several (matrices, rows as queries).
 */
export type RankingInput = Data | Rows

/**
 * One query's ranked grades: `grades`, the relevance grades in rank order; `ties`, when scores were given, the tie
 * group of each rank (ids counting up from 0 in rank order, equal for equal scores).
 */
type Query = { grades: Float64Array; ties?: Int32Array }

/**
 * Parse `(grades, options?)` or `(relevance, scores, options?)` into ranked queries and the options: the second
 * argument is scores when it is an array or tensor, else the options. Scores rank each query's items in decreasing
 * order, ties in input order. Throws `ShapeError` when relevance and scores differ in queries or in length.
 *
 * @param a The grades in rank order, or the items' relevance: a vector for one query, a matrix with a row per query.
 * @param b The scores to rank by, with the shape of `a`, or the options when there are no scores.
 * @param c The options, when `b` is scores.
 * @returns `queries`, one per row, with grades in rank order, and `options` (`{}` when none were given).
 */
function parse<O extends object>(a: RankingInput, b?: RankingInput | O, c?: O): { queries: Query[]; options: O } {
  const scored = b !== undefined && isArrayInput(b)
  const options = ((scored ? c : b) ?? {}) as O
  const rows = (x: RankingInput): Float64Array[] => {
    if (!isMatrixLike(x)) return [values(x as Data)]
    const d = dense(x as Rows, 'ranking')
    return Array.from({ length: d.rows }, (_, i) => d.data.slice(i * d.cols, (i + 1) * d.cols))
  }
  const relevance = rows(a)
  if (!scored) return { queries: relevance.map((grades) => ({ grades })), options }
  const scores = rows(b as RankingInput)
  if (scores.length !== relevance.length)
    throw new ShapeError('metrics', 'metrics: ranking: relevance and scores differ in queries')
  const queries = relevance.map((rel, q) => {
    const s = scores[q]
    if (s.length !== rel.length)
      throw new ShapeError('metrics', 'metrics: ranking: relevance and scores differ in length')
    const order = orderDescending(s)
    const ties = new Int32Array(order.length)
    for (let r = 1; r < order.length; r++) ties[r] = ties[r - 1] + (s[order[r]] === s[order[r - 1]] ? 0 : 1)
    return { grades: Float64Array.from(order, (i) => rel[i]), ties }
  })
  return { queries, options }
}

/**
 * The mean over queries of a per-query value; `empty` replaces undefined (NaN) per-query values when given.
 *
 * @param queries The ranked queries.
 * @param f The metric of one query.
 * @param empty The value of a query whose metric is NaN; left out, the NaN carries into the mean.
 * @returns The mean over queries.
 */
function meanOverQueries(queries: Query[], f: (q: Query) => number, empty?: number): number {
  let s = 0
  for (const q of queries) {
    const v = f(q)
    s += Number.isNaN(v) && empty !== undefined ? empty : v
  }
  return s / queries.length
}

/**
 * Whether a grade counts as relevant to the binary metrics: $g > 0$.
 *
 * @param g A relevance grade.
 * @returns True for a positive grade.
 */
const relevant = (g: number) => g > 0

/** Options of the cut-off metrics. */
export type AtKOptions = {
  /** Cut-off $k$, the number of top ranks that count; default the whole list. */
  k?: number
  /** The value of a query where the metric is undefined (for example no relevant items); default NaN. */
  empty?: number
}

/** Options of the metrics that need the number of relevant items in the whole collection. */
export type RecallOptions = AtKOptions & {
  /** $R$, the relevant items in the whole collection, including unretrieved ones; default those in the list. */
  totalRelevant?: number
}

/**
 * The number of relevant items in the top $k$ ranks.
 *
 * @param g The grades in rank order.
 * @param k The cut-off; more than the list's length counts the whole list.
 * @returns The count.
 */
function countRelevant(g: Float64Array, k: number): number {
  let c = 0
  for (let i = 0; i < Math.min(k, g.length); i++) if (relevant(g[i])) c++
  return c
}

/**
 * The registry metadata of a ranking metric: stable, read from a `ranking`, higher is better, range $[0, 1]$.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @param note The key of the note that explains it.
 * @returns The metadata, with its literal fields kept.
 */
const rankingInfo = (key: string, name: string, note: string) =>
  ({
    key,
    name,
    stability: 'stable',
    inputs: 'ranking',
    direction: 'higher',
    range: [0, 1],
    notes: [note],
    capability: 'score',
  }) as const

/**
 * Precision at $k$, $\frac{1}{k} \sum_{i \le k} \mathrm{rel}_i$; a list shorter than $k$ counts the missing
 * positions as not relevant.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, the cut-off (default the whole list), and `empty`, the value of a
 *   query where it is undefined.
 * @returns The precision at $k$, averaged over queries.
 *
 * @example From ranked grades, and from relevance with scores
 * print('P@2', precisionAtK([1, 0, 1, 0, 0], { k: 2 }))
 * print('P@1 by score', precisionAtK([0, 1, 1], [0.1, 0.9, 0.5], { k: 1 }))
 */
export const precisionAtK = defineMetric(
  rankingInfo('precisionAtK', 'Precision at k', 'precision-and-recall-at-k'),
  (a: RankingInput, b?: RankingInput | AtKOptions, c?: AtKOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(
      queries,
      ({ grades }) => {
        const k = options.k ?? grades.length
        return countRelevant(grades, k) / k
      },
      options.empty,
    )
  },
)

/**
 * Recall at $k$, $\frac{1}{R} \sum_{i \le k} \mathrm{rel}_i$, with $R$ the relevant items in the collection
 * (`totalRelevant`, default those in the list). NaN for a query with $R = 0$ unless `empty` is given.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, `totalRelevant` and `empty`.
 * @returns The recall at $k$, averaged over queries.
 *
 * @example Unretrieved relevant items lower the recall
 * print('R@2', recallAtK([1, 0, 1, 0, 0], { k: 2 }))
 * print('R@2 of 4 relevant', recallAtK([1, 0, 1, 0, 0], { k: 2, totalRelevant: 4 }))
 */
export const recallAtK = defineMetric(
  rankingInfo('recallAtK', 'Recall at k', 'precision-and-recall-at-k'),
  (a: RankingInput, b?: RankingInput | RecallOptions, c?: RecallOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(
      queries,
      ({ grades }) => {
        const R = options.totalRelevant ?? countRelevant(grades, grades.length)
        return divide(countRelevant(grades, options.k ?? grades.length), R)
      },
      options.empty,
    )
  },
)

/**
 * R-precision: precision at $R$, the number of relevant items of the query (hit-rate-and-r-precision). NaN for a
 * query with $R = 0$ unless `empty` is given.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `totalRelevant`, $R$ (default the relevant items in the list), and
 *   `empty`; `k` is not used.
 * @returns The R-precision, averaged over queries.
 *
 * @example Two relevant items, one in the top two
 * print('R-precision', rPrecision([1, 0, 1, 0, 0]))
 */
export const rPrecision = defineMetric(
  rankingInfo('rPrecision', 'R-precision', 'hit-rate-and-r-precision'),
  (a: RankingInput, b?: RankingInput | RecallOptions, c?: RecallOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(
      queries,
      ({ grades }) => {
        const R = options.totalRelevant ?? countRelevant(grades, grades.length)
        return divide(countRelevant(grades, R), R)
      },
      options.empty,
    )
  },
)

/**
 * Hit rate at $k$: the fraction of queries with at least one relevant item in the top $k$
 * (hit-rate-and-r-precision).
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, the cut-off (default the whole list); `empty` is not used.
 * @returns The fraction of queries with a hit.
 *
 * @example Two queries, one with its relevant item at rank 3
 * const grades = [
 *   [0, 0, 1],
 *   [0, 0, 0],
 * ]
 * print('k = 2', hitRate(grades, { k: 2 }))
 * print('k = 3', hitRate(grades, { k: 3 }))
 */
export const hitRate = defineMetric(
  rankingInfo('hitRate', 'Hit rate at k', 'hit-rate-and-r-precision'),
  (a: RankingInput, b?: RankingInput | AtKOptions, c?: AtKOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(queries, ({ grades }) => (countRelevant(grades, options.k ?? grades.length) > 0 ? 1 : 0))
  },
)

/** Options of `meanAveragePrecision`. */
export type AveragePrecisionOptions = RecallOptions & {
  /**
   * The divisor of AP@k: `relevant` (default) is $R$, every relevant item (TREC's trec_eval); `cutoff` is
   * $\min(R, k)$, so a perfect top $k$ scores 1 even when $R > k$ (the common AP@k of recommender and Kaggle
   * evaluations). `cutoff` without `k` is `relevant`.
   */
  normaliser?: 'relevant' | 'cutoff'
}

/**
 * Average precision of a ranking, $\frac{1}{R} \sum_k \mathrm{P@}k \cdot \mathrm{rel}_k$, averaged over queries (MAP)
 * (Manning et al. 2008; mean-average-precision-and-mean-reciprocal-rank). $R$ counts unretrieved relevant items when
 * `totalRelevant` is given; with `k`, only the top $k$ ranks contribute, still divided by $R$ unless `normaliser` is
 * `cutoff` (then $\min(R, k)$). NaN for a query with $R = 0$ unless `empty` is given.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, `totalRelevant`, `normaliser` and `empty`.
 * @returns The mean average precision.
 *
 * @example Hits at ranks 1 and 3 give AP (1 + 2/3)/2
 * print('AP', meanAveragePrecision([1, 0, 1, 0, 0]))
 * print('MAP of two queries', meanAveragePrecision([
 *   [1, 0, 1],
 *   [0, 1, 0],
 * ]))
 */
export const meanAveragePrecision = defineMetric(
  rankingInfo('meanAveragePrecision', 'Mean average precision', 'mean-average-precision-and-mean-reciprocal-rank'),
  (a: RankingInput, b?: RankingInput | AveragePrecisionOptions, c?: AveragePrecisionOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(
      queries,
      ({ grades }) => {
        const all = options.totalRelevant ?? countRelevant(grades, grades.length)
        const R = options.normaliser === 'cutoff' && options.k !== undefined ? Math.min(all, options.k) : all
        let hits = 0
        let s = 0
        for (let i = 0; i < Math.min(options.k ?? grades.length, grades.length); i++)
          if (relevant(grades[i])) {
            hits++
            s += hits / (i + 1)
          }
        return divide(s, R)
      },
      options.empty,
    )
  },
)

/**
 * Reciprocal rank $1/r$ of the first relevant item (0 if none in the top $k$), averaged over queries (MRR).
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, the cut-off (default the whole list); `empty` is not used.
 * @returns The mean reciprocal rank.
 *
 * @example First hits at ranks 2, 1 and never
 * print('MRR', meanReciprocalRank([
 *   [0, 1, 0],
 *   [1, 0, 0],
 *   [0, 0, 0],
 * ]))
 */
export const meanReciprocalRank = defineMetric(
  rankingInfo('meanReciprocalRank', 'Mean reciprocal rank', 'mean-average-precision-and-mean-reciprocal-rank'),
  (a: RankingInput, b?: RankingInput | AtKOptions, c?: AtKOptions): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(queries, ({ grades }) => {
      const k = Math.min(options.k ?? grades.length, grades.length)
      for (let i = 0; i < k; i++) if (relevant(grades[i])) return 1 / (i + 1)
      return 0
    })
  },
)

/**
 * The gain of a relevance grade: linear $g$, exponential $2^g - 1$ (the default, as the contract fixes), or a
 * function.
 */
export type Gain = 'linear' | 'exponential' | ((grade: number) => number)

/**
 * The gain function of a `Gain`: $g \mapsto g$ (`'linear'`), $g \mapsto 2^g - 1$ (`'exponential'`, the default), or
 * the function given. DCG, nDCG and ranking losses that weight pairs by $\lvert \Delta \mathrm{nDCG} \rvert$
 * (LambdaRank) share it.
 *
 * @param g The gain to use; left out, the exponential gain.
 * @returns The gain of a grade.
 *
 * @example The gain of a grade of 3
 * print('linear', gainFunction('linear')(3))
 * print('exponential', gainFunction(undefined)(3))
 * print('custom', gainFunction((g) => g * g)(3))
 */
export function gainFunction(g: Gain | undefined): (grade: number) => number {
  if (typeof g === 'function') return g
  return g === 'linear' ? (x) => x : (x) => 2 ** x - 1
}

/**
 * DCG's discount of a 0-based position (0 at the top): $1/\log_2(\mathit{position} + 2)$.
 *
 * @param position The 0-based rank (0 for the top item).
 * @returns The discount, 1 at the top.
 *
 * @example The first three discounts
 * print([0, 1, 2].map(positionDiscount))
 */
export const positionDiscount = (position: number): number => 1 / Math.log2(position + 2)

/** Options of DCG and nDCG. */
export type DcgOptions = AtKOptions & {
  /** Gain of a grade; default exponential, $2^g - 1$. The linear gain matches scikit-learn's `dcg_score`. */
  gain?: Gain
}

/**
 * DCG@k of one query: $\sum_{i \le k} \mathrm{gain}(g_i)/\log_2(i + 1)$ over 1-based ranks $i$. With tie groups, each
 * tied item gets the group's mean gain (over the whole group, even past $k$), which is the expected DCG over random
 * orders of the ties.
 *
 * @param options The query.
 * @param options.grades The grades in rank order.
 * @param options.ties The tie group of each rank, when the ranking came from scores; left out, no ties.
 * @param k The cut-off (`Infinity` for the whole list).
 * @param gain The gain function.
 * @returns The DCG@k.
 */
function dcgOf({ grades, ties }: Query, k: number, gain: (g: number) => number): number {
  const n = Math.min(k, grades.length)
  let s = 0
  if (!ties) {
    for (let i = 0; i < n; i++) s += gain(grades[i]) * positionDiscount(i)
    return s
  }
  let start = 0
  while (start < grades.length && start < k) {
    let end = start
    let groupGain = 0
    while (end < grades.length && ties[end] === ties[start]) groupGain += gain(grades[end++])
    groupGain /= end - start
    for (let i = start; i < Math.min(end, k); i++) s += groupGain * positionDiscount(i)
    start = end
  }
  return s
}

/**
 * Discounted cumulative gain at $k$ (Järvelin and Kekäläinen 2002; normalised-discounted-cumulative-gain):
 * $\mathrm{DCG@}k = \sum_{i \le k} \mathrm{gain}(g_i)/\log_2(i + 1)$, averaged over queries. State the gain: the
 * default is $2^g - 1$, and `gain: 'linear'` matches sklearn's `dcg_score`, including its averaging over tied scores.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, the cut-off (default the whole list), and `gain`; `empty` is not
 *   used.
 * @returns The DCG@k, averaged over queries.
 *
 * @example The scikit-learn example, with its linear gain
 * print('DCG', dcg([[10, 0, 0, 1, 5]], [[0.1, 0.2, 0.3, 4, 70]], { gain: 'linear' }))
 * print('exponential gain', dcg([3, 2, 0]), '= 7 + 3/log2(3) =', 7 + 3 / Math.log2(3))
 */
export const dcg = defineMetric(
  {
    ...rankingInfo('dcg', 'Discounted cumulative gain', 'normalised-discounted-cumulative-gain'),
    range: [0, Infinity],
  },
  (a: RankingInput, b?: RankingInput | DcgOptions, c?: DcgOptions): number => {
    const { queries, options } = parse(a, b, c)
    const gain = gainFunction(options.gain)
    return meanOverQueries(queries, (q) => dcgOf(q, options.k ?? Infinity, gain))
  },
)

/**
 * Normalised DCG at $k$: DCG@k divided by the DCG@k of the ideal ordering (normalised-discounted-cumulative-gain). The
 * ideal ordering sorts the list's own grades, or `ideal`: the grades of every judged item, including unretrieved ones,
 * which is stricter. NaN for a query with no relevant items unless `empty` is given. With `gain: 'linear'` it matches
 * sklearn's `ndcg_score`.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, `gain`, `empty`, and `ideal`, the grades the ideal ordering is made
 *   from (used for every query).
 * @returns The nDCG@k, in $[0, 1]$, averaged over queries.
 *
 * @example The scikit-learn examples: a ranking, and tied scores at k = 1
 * const relevance = [[10, 0, 0, 1, 5]]
 * print('nDCG', ndcg(relevance, [[0.1, 0.2, 0.3, 4, 70]], { gain: 'linear' }))
 * print('ties, k = 1', ndcg(relevance, [[1, 0, 0, 0, 1]], { gain: 'linear', k: 1 }))
 */
export const ndcg = defineMetric(
  rankingInfo('ndcg', 'Normalised DCG', 'normalised-discounted-cumulative-gain'),
  (a: RankingInput, b?: RankingInput | (DcgOptions & { ideal?: Data }), c?: DcgOptions & { ideal?: Data }): number => {
    const { queries, options } = parse(a, b, c)
    const gain = gainFunction(options.gain)
    const k = options.k ?? Infinity
    return meanOverQueries(
      queries,
      (q) => {
        const ideal = Float64Array.from(options.ideal ? values(options.ideal) : q.grades).sort((x, y) => y - x)
        return divide(dcgOf(q, k, gain), dcgOf({ grades: ideal }, k, gain))
      },
      options.empty,
    )
  },
)

/**
 * Expected reciprocal rank (Chapelle et al. 2009): a cascade user stops at rank $r$ with probability
 * $R_r \prod_{i<r}(1 - R_i)$, where $R_i = (2^{g_i} - 1)/2^{g_{\max}}$, and
 * $\mathrm{ERR} = \sum_r R_r \prod_{i<r}(1 - R_i)/r$. `maxGrade` defaults to the largest grade present.
 *
 * @param a The grades in rank order, or the items' relevance when `b` is scores; a matrix for several queries.
 * @param b The scores to rank the items by (decreasing), or the options.
 * @param c The options, when `b` is scores: `k`, the cut-off (default the whole list), and `maxGrade`, $g_{\max}$;
 *   `empty` is not used.
 * @returns The ERR, averaged over queries.
 *
 * @example A very relevant item first leaves little for the rest
 * print('ERR', expectedReciprocalRank([2, 0, 1]))
 * print('reversed', expectedReciprocalRank([1, 0, 2]))
 */
export const expectedReciprocalRank = defineMetric(
  rankingInfo('expectedReciprocalRank', 'Expected reciprocal rank', 'normalised-discounted-cumulative-gain'),
  (
    a: RankingInput,
    b?: RankingInput | (AtKOptions & { maxGrade?: number }),
    c?: AtKOptions & { maxGrade?: number },
  ): number => {
    const { queries, options } = parse(a, b, c)
    return meanOverQueries(queries, ({ grades }) => {
      const gMax = options.maxGrade ?? Math.max(0, ...grades)
      let notStopped = 1
      let s = 0
      for (let r = 0; r < Math.min(options.k ?? grades.length, grades.length); r++) {
        const R = (2 ** grades[r] - 1) / 2 ** gMax
        s += (notStopped * R) / (r + 1)
        notStopped *= 1 - R
      }
      return s
    })
  },
)
