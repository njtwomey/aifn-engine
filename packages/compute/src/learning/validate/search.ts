/**
 * Hyperparameter search and nested cross-validation (plan §5.5). A search scores every candidate by cross-validation,
 * keeps the full results table, and refits the best candidate on all the data; as an estimator it can itself be
 * cross-validated, which is nested cross-validation.
 *
 * Candidates are points of a `Space` (`aifn-compute/foundation/space`), turned into estimators by a `make` function.
 * The first metric selects; every candidate is cross-validated on the same splits with the same fold streams (common
 * random numbers), so differences between candidates are not differences of luck in the splits.
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
  /** The candidate's position in the order tried, from 0. */
  index: number
  /** The candidate's hyperparameters. */
  params: P
  /** The selection metric on each fold, a vector of $k$ values for $k$ folds. */
  scores: Tensor
  /** The selection metric's mean over folds. */
  mean: number
  /** The selection metric's sample standard deviation over folds. */
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
  /** Whether higher or lower values of the selection metric are better. */
  direction: 'higher' | 'lower'
  /** One row per candidate, in the order tried. */
  rows: SearchRow<P>[]
  /** The selection metric per candidate and fold, $c \times k$ for $c$ candidates and $k$ folds. */
  scores: Tensor
  /** The row of rank 1 (the first such row when several tie). */
  best: SearchRow<P>
  /** Each candidate's full cross-validation (fold models, predictions, traces). */
  validations: CrossValidation<M>[]
  /** The best candidate refitted on all the data (absent when `refit` is false). */
  model?: M
}

/** A search: how to make an estimator from parameters, the candidates, and the metric that selects. */
export interface Search<P, M> {
  /** `'grid-search'` or `'random-search'`. */
  readonly name: string
  /** Makes the estimator of a candidate from its hyperparameters. */
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

/**
 * Competition ranks of candidate means, 1 for the best by `direction`; tied means share the lower rank, and a NaN mean
 * ranks last.
 *
 * @param means Each candidate's mean of the selection metric.
 * @param direction Whether higher or lower means are better.
 * @returns One rank per candidate, from 1.
 */
function ranks(means: number[], direction: 'higher' | 'lower'): number[] {
  const sign = direction === 'higher' ? -1 : 1
  return toFlat(
    rankData(
      means.map((m) => (Number.isNaN(m) ? Infinity : sign * m)),
      'min',
    ),
  )
}

/**
 * The search object shared by `gridSearch` and `randomSearch`: `run` cross-validates every candidate and refits the
 * best, and `estimator` wraps `run` as an estimator. Throws `DomainError` when `metrics` is empty.
 *
 * @param name The search's name, also used in error messages.
 * @param make Makes the estimator of a candidate from its hyperparameters.
 * @param candidates The candidates to try, given the stream `child(s, 'candidates')` of `run` (undefined without one).
 * @param metrics The metrics to record; the first selects.
 * @param refit Whether `run` refits the best candidate on all the data.
 * @returns The search.
 */
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
 * Exhaustive grid search over a `Space` (from `aifn-compute/foundation/space`): every combination of
 * `grid(space, { points })` (the last dimension varying fastest; conditional dimensions only where they apply), each
 * scored by cross-validation on the first of `metrics`; the best mean wins, the first candidate among ties. The search
 * keeps the full results table. Nothing runs until `run` (or a fit of `estimator(splitter)`).
 *
 * @param make Makes the estimator of a candidate from its hyperparameters (a point of `space`).
 * @param space The hyperparameter space.
 * @param options The metrics, the grid resolution and whether to refit.
 * @param options.metrics The metrics recorded for every candidate; the first selects. Throws `DomainError` when
 *   empty.
 * @param options.points The number of grid points along each real or integer dimension (choices list every option).
 * @param options.refit Refit the best candidate on all the data, as the result's `model`.
 * @returns The search: call `run(data, splitter, s)` for the results, or `estimator(splitter)` to fit it as a model.
 *
 * @example Three shrinkages of a mean predictor, each scored by 3-fold cross-validation
 * // Predict (1 - shrink) times the training mean, scored by mean squared error.
 * const shrunkMean = ({ shrink }) => ({
 *   name: 'shrunk mean',
 *   fit: (d) => {
 *     const m = (1 - shrink) * mean(d.y)
 *     return { decide: (x) => full([x.shape[0]], m) }
 *   },
 * })
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const shrinks = { dims: { shrink: { type: 'real', min: 0, max: 1, default: 0 } } }
 * const data = { x: tensor([[0], [1], [2], [3], [4], [5]]), y: tensor([1, 2, 3, 4, 5, 6]) }
 * const result = gridSearch(shrunkMean, shrinks, { metrics: [mse], points: 3 }).run(data, kFold({ k: 3 }))
 * print('candidates:', result.rows.map((r) => r.params.shrink))
 * print('mean mse:', result.rows.map((r) => r.mean))
 * print('ranks:', result.rows.map((r) => r.rank))
 * print('best:', result.best.params)
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
 * candidates drawn uniformly from a `Space` (log-uniform on log scales), candidate $k$ from
 * `child(c, 'candidate', k)` where $c$ is the candidates' stream `child(s, 'candidates')` of `run`, scored as in
 * `gridSearch`. Needs a stream: `run` throws `DomainError` without one.
 *
 * @param make Makes the estimator of a candidate from its hyperparameters (a point of `space`).
 * @param space The hyperparameter space the candidates are drawn from.
 * @param options The metrics, the number of candidates and whether to refit.
 * @param options.metrics The metrics recorded for every candidate; the first selects. Throws `DomainError` when
 *   empty.
 * @param options.iterations The number of candidates drawn.
 * @param options.refit Refit the best candidate on all the data, as the result's `model`.
 * @returns The search: call `run(data, splitter, s)` with a stream for the results.
 *
 * @example Three random shrinkages, reproducible from the seed
 * // Predict (1 - shrink) times the training mean, scored by mean squared error.
 * const shrunkMean = ({ shrink }) => ({
 *   name: 'shrunk mean',
 *   fit: (d) => {
 *     const m = (1 - shrink) * mean(d.y)
 *     return { decide: (x) => full([x.shape[0]], m) }
 *   },
 * })
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const shrinks = { dims: { shrink: { type: 'real', min: 0, max: 1, default: 0 } } }
 * const data = { x: tensor([[0], [1], [2], [3], [4], [5]]), y: tensor([1, 2, 3, 4, 5, 6]) }
 * const search = randomSearch(shrunkMean, shrinks, { metrics: [mse], iterations: 3 })
 * const result = search.run(data, kFold({ k: 3 }), stream(0))
 * print('candidates:', result.rows.map((r) => r.params.shrink))
 * print('mean mse:', result.rows.map((r) => r.mean))
 * print('best:', result.best.params)
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
  /** The `info.key` of the selection metric. */
  metric: string
  /** Whether higher or lower values of the selection metric are better. */
  direction: 'higher' | 'lower'
  /** The nested estimate: the mean outer test score. An honest estimate of the tuned procedure's performance. */
  nestedScore: number
  /** The sample standard deviation of the outer test scores. */
  nestedStd: number
  /** The search run on all the data with the inner splitter. */
  unnested: SearchResult<P, M>
  /** The best inner mean of `unnested`: the usual, optimistic, estimate. */
  unnestedScore: number
  /**
   * How much better the unnested estimate looks: unnested minus nested for a higher-is-better metric, else nested
   * minus unnested.
   */
  optimism: number
}

/**
 * Nested cross-validation (Varma and Simon, 2006, "Bias in error estimation when using cross-validation for model
 * selection", BMC Bioinformatics 7; Cawley and Talbot, 2010, JMLR 11): the `inner` splitter selects hyperparameters
 * with `search` on each `outer` training set, and the outer test sets score the selected, refitted models. Both
 * levels are kept, and the search is also run once on all the data so that the optimism of its best inner score, the
 * unnested estimate, can be shown. The search must refit (each outer fold's model is the refitted best), or the fit
 * throws `DomainError`.
 *
 * @param outer The splitter whose test sets score the tuned procedure.
 * @param inner The splitter the search cross-validates with, inside each outer training set and on all the data.
 * @param search The hyperparameter search to tune with, from `gridSearch` or `randomSearch`; its metrics are scored
 *   on the outer test sets too.
 * @param data The rows: inputs `x`, targets `y` and, for a grouped splitter, `groups`.
 * @param options The stream: the outer cross-validation draws from `child(stream, 'outer')` and the unnested search
 *   from `child(stream, 'unnested')`. Needed by randomised splitters and by `randomSearch`.
 * @param options.stream The randomness of both levels.
 * @returns Both levels, the parameters chosen per outer fold, the nested and unnested estimates and the optimism.
 *
 * @example The unnested estimate looks better than the nested one
 * // Predict (1 - shrink) times the training mean, scored by mean squared error.
 * const shrunkMean = ({ shrink }) => ({
 *   name: 'shrunk mean',
 *   fit: (d) => {
 *     const m = (1 - shrink) * mean(d.y)
 *     return { decide: (x) => full([x.shape[0]], m) }
 *   },
 * })
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const shrinks = { dims: { shrink: { type: 'real', min: 0, max: 0.5, default: 0 } } }
 * const data = { x: tensor([[0], [1], [2], [3], [4], [5]]), y: tensor([1, 2, 3, 4, 5, 6]) }
 * const search = gridSearch(shrunkMean, shrinks, { metrics: [mse], points: 3 })
 * const result = nested(kFold({ k: 3 }), kFold({ k: 2 }), search, data)
 * print('chosen shrink per outer fold:', result.perFold.map((f) => f.params.shrink))
 * print('outer test mse per fold:', result.perFold.map((f) => f.outerScore))
 * print('nested mse:', result.nestedScore)
 * print('unnested mse:', result.unnestedScore)
 * print('optimism:', result.optimism)
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
