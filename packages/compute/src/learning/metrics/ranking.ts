/**
 * Ranking and retrieval metrics: precision and recall at k, R-precision, hit rate, average precision (MAP), reciprocal
 * rank (MRR), DCG and nDCG with an explicit gain, and expected reciprocal rank; plus the beyond-accuracy metrics of
 * recommendation lists (intra-list diversity, catalogue coverage, Gini and Herfindahl concentration, novelty).
 *
 * Every ranking metric takes either
 * - `(grades, options)`: relevance grades already in rank order (rank 1 first), as in the notes; or
 * - `(relevance, scores, options)`: each item's relevance and the scores to rank it by (decreasing), as scikit-learn's
 *   `ndcg_score(y_true, y_score)`.
 * A matrix (rows as queries) gives the mean over queries. Binary metrics count a grade > 0 as relevant. Ties in scores
 * are broken by input order, except in DCG and nDCG, which average over tied items (the expected value over random tie
 * orders, as scikit-learn).
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

/** Grades in rank order, or item relevance with scores, for one query (vectors) or several (matrices, rows as queries). */
export type RankingInput = Data | Rows

/** One query's ranked grades, and the tie groups of its scores (group ids in rank order) when scores were given. */
type Query = { grades: Float64Array; ties?: Int32Array }

/** Parse `(grades, options?)` or `(relevance, scores, options?)` into ranked queries and the options. */
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

/** The mean over queries of a per-query value; `empty` replaces undefined (NaN) per-query values when given. */
function meanOverQueries(queries: Query[], f: (q: Query) => number, empty?: number): number {
  let s = 0
  for (const q of queries) {
    const v = f(q)
    s += Number.isNaN(v) && empty !== undefined ? empty : v
  }
  return s / queries.length
}

const relevant = (g: number) => g > 0

/** Options of the cut-off metrics. */
export type AtKOptions = {
  /** Cut-off k; default the whole list. */
  k?: number
  /** The value of a query where the metric is undefined (for example no relevant items); default NaN. */
  empty?: number
}

/** Options of the metrics that need the number of relevant items in the whole collection. */
export type RecallOptions = AtKOptions & {
  /** R, the relevant items in the whole collection, including unretrieved ones; default those in the list. */
  totalRelevant?: number
}

function countRelevant(g: Float64Array, k: number): number {
  let c = 0
  for (let i = 0; i < Math.min(k, g.length); i++) if (relevant(g[i])) c++
  return c
}

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

/** Precision at k, (1/k) Σ_{i ≤ k} relᵢ; a list shorter than k counts the missing positions as not relevant. */
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

/** Recall at k, (1/R) Σ_{i ≤ k} relᵢ, with R the relevant items in the collection (`totalRelevant`). */
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

/** R-precision: precision at R, the number of relevant items of the query (hit-rate-and-r-precision). */
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

/** Hit rate at k: the fraction of queries with at least one relevant item in the top k (hit-rate-and-r-precision). */
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
   * The divisor of AP@k: `relevant` (default) is R, every relevant item (TREC's trec_eval); `cutoff` is min(R, k), so a
   * perfect top k scores 1 even when R > k (the common AP@k of recommender and Kaggle evaluations).
   */
  normaliser?: 'relevant' | 'cutoff'
}

/**
 * Average precision of a ranking, (1/R) Σₖ P@k·relₖ, averaged over queries (MAP) (Manning et al. 2008;
 * mean-average-precision-and-mean-reciprocal-rank). R counts unretrieved relevant items when `totalRelevant` is given;
 * with `k`, only the top k ranks contribute, still divided by R unless `normaliser` is `cutoff` (then min(R, k)).
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

/** Reciprocal rank 1/r of the first relevant item (0 if none in the top k), averaged over queries (MRR). */
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

/** The gain of a relevance grade: linear g, exponential 2^g − 1 (the default, as the contract fixes), or a function. */
export type Gain = 'linear' | 'exponential' | ((grade: number) => number)

/**
 * The gain function of a `Gain`: g ↦ g ('linear'), g ↦ 2^g − 1 ('exponential', the default), or the function given.
 * DCG, nDCG and ranking losses that weight pairs by |Δ nDCG| (LambdaRank) share it.
 */
export function gainFunction(g: Gain | undefined): (grade: number) => number {
  if (typeof g === 'function') return g
  return g === 'linear' ? (x) => x : (x) => 2 ** x - 1
}

/** DCG's discount of a 0-based position (0 at the top): 1/log₂(position + 2). */
export const positionDiscount = (position: number): number => 1 / Math.log2(position + 2)

/** Options of DCG and nDCG. */
export type DcgOptions = AtKOptions & {
  /** Gain of a grade; default exponential, 2^g − 1. The linear gain matches scikit-learn's `dcg_score`. */
  gain?: Gain
}

/**
 * DCG@k of one query: Σ_{i ≤ k} gain(gᵢ)/log₂(i + 1). With tie groups, each tied item gets the group's mean gain, which
 * is the expected DCG over random orders of the ties.
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
 * Discounted cumulative gain at k (Järvelin and Kekäläinen 2002; normalised-discounted-cumulative-gain):
 * DCG@k = Σ_{i ≤ k} gain(gᵢ)/log₂(i + 1), averaged over queries. State the gain: the default is 2^g − 1.
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
 * Normalised DCG at k: DCG@k divided by the DCG@k of the ideal ordering (normalised-discounted-cumulative-gain). The
 * ideal ordering sorts the list's own grades, or `ideal`: the grades of every judged item, including unretrieved ones,
 * which is stricter. NaN for a query with no relevant items unless `empty` is given.
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
 * Expected reciprocal rank (Chapelle et al. 2009): a cascade user stops at rank r with probability
 * Rᵣ Π_{i<r}(1 − Rᵢ), where Rᵢ = (2^{gᵢ} − 1)/2^{g_max}, and ERR = Σᵣ Rᵣ Π_{i<r}(1 − Rᵢ)/r. `maxGrade` defaults to the
 * largest grade present.
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
