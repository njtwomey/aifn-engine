/**
 * Hyperparameter search and nested cross-validation (plan §5.5). A search scores every candidate by cross-validation,
 * keeps the full results table, and refits the best candidate on all the data; as an estimator it can itself be
 * cross-validated, which is nested cross-validation.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import { grid, sample, type Space, type ValuesOf } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Features, FitOptions, ServedMetric } from 'aifn-compute/learning/estimators'
import { ranks as rankData } from 'aifn-compute/probability/stats'
import { crossValidate, type CrossValidation, type CrossValidationData, type Fittable } from './cross'
import type { Splitter } from './splitters'
import { DomainError } from 'aifn-compute/foundation/errors'

/** One row of a search's results table. */
export interface SearchRow<P> {
  index: number
  params: P
  /** The selection metric on each fold, [folds]. */
  scores: Tensor
  mean: number
  std: number
  /** 1 for the best mean (by the metric's direction); ties share the lower rank. */
  rank: number
  /** Total fit time over the folds, in milliseconds. */
  fitMs: number
  /** Every metric's mean over folds, by `info.key`. */
  means: Record<string, number>
}

/** The full result of a search. */
export interface SearchResult<P, M> {
  /** The `info.key` of the selection metric. */
  metric: string
  direction: 'higher' | 'lower'
  /** One row per candidate, in the order tried. */
  rows: SearchRow<P>[]
  /** The selection metric per candidate and fold, [candidates, folds]. */
  scores: Tensor
  best: SearchRow<P>
  /** Each candidate's full cross-validation (fold models, predictions, traces). */
  validations: CrossValidation<M>[]
  /** The best candidate refitted on all the data (absent when `refit` is false). */
  model?: M
}

/** A search: how to make an estimator from parameters, the candidates, and the metric that selects. */
export interface Search<P, M> {
  readonly name: string
  readonly make: (params: P) => Fittable<Features, M>
  /** The candidates to try; random searches draw them from the stream. */
  candidates(s?: Stream): P[]
  /** The metric that selects the best candidate (the first metric). */
  readonly metric: ServedMetric
  /** Every metric recorded for each candidate (the selection metric first). */
  readonly metrics: readonly ServedMetric[]
  /** Refit the best candidate on all the data (default true). */
  readonly refit: boolean
  /** Score every candidate by cross-validation with `splitter`, and refit the best. */
  run(data: CrossValidationData<Features>, splitter: Splitter, s?: Stream): SearchResult<P, M>
  /** The search as an estimator whose fitted model is the refitted best model plus the search results. */
  estimator(splitter: Splitter): Fittable<Features, SearchModel<P, M>>
}

/** A fitted search: the best model's capabilities, the results table and the chosen parameters. */
export type SearchModel<P, M> = M & { readonly search: SearchResult<P, M>; readonly params: P }

/** Competition ranks of candidate means, 1 for the best by `direction`; NaN means rank last. */
function ranks(means: number[], direction: 'higher' | 'lower'): number[] {
  const sign = direction === 'higher' ? -1 : 1
  return toFlat(
    rankData(
      means.map((m) => (Number.isNaN(m) ? Infinity : sign * m)),
      'min',
    ),
  )
}

function makeSearch<P, M>(
  name: string,
  make: (params: P) => Fittable<Features, M>,
  candidates: (s?: Stream) => P[],
  metrics: readonly ServedMetric[],
  refit: boolean,
): Search<P, M> {
  if (metrics.length === 0) throw new DomainError(name, `${name}: needs at least one metric`)
  const metric = metrics[0]
  const search: Search<P, M> = {
    name,
    make,
    candidates,
    metric,
    metrics,
    refit,
    run(data, splitter, s) {
      const list = candidates(s && child(s, 'candidates'))
      // Every candidate sees the same splits and fold streams (common random numbers).
      const validations = list.map((params) =>
        crossValidate(make(params) as Fittable<Features, never>, data, splitter, metrics, {
          stream: s && child(s, 'cv'),
        }),
      ) as unknown as CrossValidation<M>[]
      const key = metric.info.key
      const means = validations.map((v) => v.mean[key])
      const rank = ranks(means, metric.info.direction)
      const rows: SearchRow<P>[] = list.map((params, k) => ({
        index: k,
        params,
        scores: validations[k].scores[key],
        mean: means[k],
        std: validations[k].std[key],
        rank: rank[k],
        fitMs: validations[k].folds.reduce((a, f) => a + f.fitMs, 0),
        means: validations[k].mean,
      }))
      const best = rows.reduce((a, b) => (b.rank < a.rank ? b : a))
      const folds = validations[0]?.folds.length ?? 0
      const scores = new Float64Array(rows.length * folds)
      rows.forEach((r, k) => r.scores.data.forEach((v, f) => (scores[k * folds + f] = v)))
      const result: SearchResult<P, M> = {
        metric: key,
        direction: metric.info.direction,
        rows,
        scores: fromData(scores, [rows.length, folds]),
        best,
        validations,
      }
      if (refit) result.model = make(best.params).fit(data, { stream: s && child(s, 'refit') })
      return result
    },
    estimator(splitter) {
      return {
        name: `${name} with ${splitter.name}`,
        fit(data, options: FitOptions = {}) {
          const result = search.run(data, splitter, options.stream)
          if (!result.model) throw new DomainError(name, `${name}: an estimator needs refit`)
          return { ...(result.model as object), search: result, params: result.best.params } as SearchModel<P, M>
        },
      }
    },
  }
  return search
}

/**
 * Exhaustive grid search over a `Space` (from `aifn-compute/foundation/space`): every combination of `grid(space, { points })`
 * (the last dimension varying fastest; conditional dimensions only where they apply), each scored by
 * cross-validation on the first of `metrics`; the best mean wins. The search keeps the full results table.
 *
 * @example
 * const search = gridSearch((p) => logisticRegression({ l2: p.l2 }), space({ l2: real(0.01, 1, { scale: 'log' }) }), { metrics: [logLoss], points: 3 })
 * search.run(data, kFold({ k: 5 })).best.params // { l2: … }
 */
export function gridSearch<const S extends Space, M>(
  make: (params: ValuesOf<S>) => Fittable<never, M>,
  space: S,
  { metrics, points = 5, refit = true }: { metrics: readonly ServedMetric[]; points?: number; refit?: boolean },
): Search<ValuesOf<S>, M> {
  return makeSearch(
    'grid-search',
    make as (p: ValuesOf<S>) => Fittable<Features, M>,
    () => grid(space, { points }) as ValuesOf<S>[],
    metrics,
    refit,
  )
}

/**
 * Random search (Bergstra and Bengio, 2012, "Random search for hyper-parameter optimization", JMLR 13): `iterations`
 * candidates drawn uniformly from a `Space` (log-uniform on log scales), candidate k from `child(s, 'candidate', k)`,
 * scored as in `gridSearch`. Needs a stream.
 */
export function randomSearch<const S extends Space, M>(
  make: (params: ValuesOf<S>) => Fittable<never, M>,
  space: S,
  {
    metrics,
    iterations = 10,
    refit = true,
  }: { metrics: readonly ServedMetric[]; iterations?: number; refit?: boolean },
): Search<ValuesOf<S>, M> {
  const candidates = (s?: Stream) => {
    if (!s) throw new DomainError('randomSearch', 'randomSearch: needs a stream')
    return Array.from({ length: iterations }, (_, k) => sample(child(s, 'candidate', k), space as Space) as ValuesOf<S>)
  }
  return makeSearch('random-search', make as (p: ValuesOf<S>) => Fittable<Features, M>, candidates, metrics, refit)
}

/** The result of nested cross-validation. */
export interface NestedCrossValidation<P, M> {
  /** The outer cross-validation of the whole search; each fold's model carries its inner search. */
  outer: CrossValidation<SearchModel<P, M>>
  /** Per outer fold: the chosen parameters, the inner best mean and the outer test score of the selection metric. */
  perFold: { params: P; innerScore: number; outerScore: number }[]
  metric: string
  direction: 'higher' | 'lower'
  /** The nested estimate: the mean outer test score. An honest estimate of the tuned procedure's performance. */
  nestedScore: number
  nestedStd: number
  /** The search run on all the data with the inner splitter. */
  unnested: SearchResult<P, M>
  /** The best inner mean of `unnested`: the usual, optimistic, estimate. */
  unnestedScore: number
  /** How much better the unnested estimate looks: unnested − nested for a higher-is-better metric, else the reverse. */
  optimism: number
}

/**
 * Nested cross-validation (Varma and Simon, 2006, "Bias in error estimation when using cross-validation for model
 * selection", BMC Bioinformatics 7; Cawley and Talbot, 2010, JMLR 11): the `inner` splitter selects hyperparameters
 * with `search` on each `outer` training set, and the outer test sets score the selected, refitted models. Both
 * levels are kept, and the search is also run once on all the data so that the optimism of its best inner score, the
 * unnested estimate, can be shown.
 */
export function nested<P, M>(
  outer: Splitter,
  inner: Splitter,
  search: Search<P, M>,
  data: CrossValidationData<Features>,
  { stream }: { stream?: Stream } = {},
): NestedCrossValidation<P, M> {
  const cv = crossValidate(search.estimator(inner) as Fittable<Features, never>, data, outer, search.metrics, {
    stream: stream && child(stream, 'outer'),
  }) as unknown as CrossValidation<SearchModel<P, M>>
  const name = search.metric.info.key
  const perFold = cv.folds.map((f) => ({
    params: f.model.params,
    innerScore: f.model.search.best.mean,
    outerScore: f.metrics[name],
  }))
  const unnested = search.run(data, inner, stream && child(stream, 'unnested'))
  const nestedScore = cv.mean[name]
  const sign = search.metric.info.direction === 'higher' ? 1 : -1
  return {
    outer: cv,
    perFold,
    metric: name,
    direction: search.metric.info.direction,
    nestedScore,
    nestedStd: cv.std[name],
    unnested,
    unnestedScore: unnested.best.mean,
    optimism: sign * (unnested.best.mean - nestedScore),
  }
}
