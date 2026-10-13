/**
 * The one-standard-error rule (Breiman, Friedman, Olshen and Stone, 1984, §3.4.3; Hastie, Tibshirani and Friedman,
 * 2009, §7.10): among candidates scored by $K$-fold cross-validation, choose the simplest whose mean score is within
 * one standard error of the best mean.
 *
 * For candidate $j$ with fold scores $s_{j1}, \dots, s_{jK}$, the mean is $\bar s_j$ and its standard error is
 * $\mathrm{SE}_j = \operatorname{sd}(s_{j\cdot}) / \sqrt K$, the sample standard deviation over folds (divided by
 * $K - 1$) over $\sqrt K$, as in glmnet's `cvsd`. With $j^\star$ the best mean, a lower-is-better score makes candidate
 * $j$ eligible when $\bar s_j \le \bar s_{j^\star} + \mathrm{SE}_{j^\star}$ (glmnet's `lambda.1se`), a higher-is-better
 * one when $\bar s_j \ge \bar s_{j^\star} - \mathrm{SE}_{j^\star}$; the rule returns the simplest eligible candidate.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import type { CrossValidation } from 'aifn-compute/learning/validate'

/** Fold scores with their direction: what a `gridSearch` or `randomSearch` result (`SearchResult`) carries. */
export interface ScoredCandidates {
  /** The score of each candidate on each fold, $c \times K$ for $c$ candidates and $K$ folds. */
  readonly scores: Tensor
  /** Whether higher or lower scores are better. */
  readonly direction: 'higher' | 'lower'
}

/** Options of `oneStandardError`. */
export interface OneStandardErrorOptions {
  /**
   * How simple each candidate is, lower being simpler: $c$ values, or a function of the candidate's index. Default:
   * the candidates' order, the first the simplest. For a ridge grid in increasing penalty, pass the negated penalties.
   */
  simplicity?: VectorLike | ((index: number) => number)
  /** Whether higher or lower scores are better; overrides the input's own direction (default `'lower'` for a matrix). */
  direction?: 'higher' | 'lower'
  /** For a list of cross-validations, the `info.key` of the metric to read (default: the first metric of the first). */
  metric?: string
}

/** The one-standard-error rule's choice, from `oneStandardError`. */
export interface OneStandardError {
  /** Each candidate's mean score over folds, $c$ values. */
  mean: Tensor
  /** Each candidate's standard error, the sample standard deviation over folds over $\sqrt K$, $c$ values. */
  se: Tensor
  /** The index of the best mean (the first among ties). */
  best: number
  /** The best mean plus (lower is better) or minus (higher is better) its standard error. */
  threshold: number
  /** 1 for each candidate whose mean is within one standard error of the best, else 0 (bool, $c$ values). */
  eligible: Tensor
  /** The index of the simplest eligible candidate (the first among equally simple ones). */
  chosen: number
  /** Whether higher or lower scores were taken as better. */
  direction: 'higher' | 'lower'
}

/**
 * The fold-score matrix and direction of whichever input `oneStandardError` was given.
 *
 * @param input A matrix, a `ScoredCandidates`, or a list of cross-validations.
 * @param options The direction override and metric key.
 * @returns The scores as a row-major array, their shape and the direction.
 */
function foldScores(
  input: Tensor | ScoredCandidates | readonly CrossValidation<unknown>[],
  options: OneStandardErrorOptions,
): { s: Float64Array; c: number; k: number; direction: 'higher' | 'lower' } {
  if (isTensor(input)) {
    const { data, m, n } = dense.toMatrixF64(input, 'oneStandardError')
    return { s: data, c: m, k: n, direction: options.direction ?? 'lower' }
  }
  if (!Array.isArray(input)) {
    const { scores, direction } = input as ScoredCandidates
    const { data, m, n } = dense.toMatrixF64(scores, 'oneStandardError')
    return { s: data, c: m, k: n, direction: options.direction ?? direction }
  }
  const list = input as readonly CrossValidation<unknown>[]
  if (list.length === 0) throw new DomainError('oneStandardError', 'oneStandardError: no candidates')
  const key = options.metric ?? Object.keys(list[0].scores)[0]
  const rows = list.map((cv) => {
    const v = cv.scores[key]
    if (!v) throw new DomainError('oneStandardError', `oneStandardError: no metric '${key}' in a cross-validation`)
    return dense.data(v)
  })
  const k = rows[0].length
  if (rows.some((r) => r.length !== k))
    throw new ShapeError('oneStandardError', 'oneStandardError: every candidate needs the same number of folds')
  const s = new Float64Array(rows.length * k)
  rows.forEach((r, j) => s.set(r, j * k))
  return { s, c: rows.length, k, direction: options.direction ?? list[0].directions[key] }
}

/**
 * The one-standard-error rule over candidates scored by $K$-fold cross-validation (Breiman et al., 1984; Hastie,
 * Tibshirani and Friedman, 2009, §7.10): the best mean, each candidate's standard error (sample standard deviation
 * over folds over $\sqrt K$), and the simplest candidate whose mean is within one standard error of the best, as
 * glmnet's `lambda.1se`. Reads a `gridSearch` or `randomSearch` result (its `scores` and `direction`), a list of
 * `crossValidate` results (one per candidate, with the same folds), or a $c \times K$ matrix. Throws `DomainError` for
 * fewer than two folds or no candidates, and `ShapeError` for a `simplicity` of the wrong length.
 *
 * @param input The fold scores: a search result, a list of cross-validations, or a matrix with a row per candidate.
 * @param options The order of simplicity, the direction, and the metric to read from cross-validations.
 * @returns Each candidate's mean and standard error, the best, the threshold, the eligible candidates and the chosen
 *   one.
 *
 * @example Four candidates, simplest first: the second is within one SE of the best and is chosen
 * const scores = tensor([
 *   [9.0, 8.0, 10.0, 9.0, 9.0],
 *   [5.0, 4.6, 5.9, 5.3, 4.2],
 *   [5.0, 4.0, 5.5, 4.5, 5.0],
 *   [5.1, 4.2, 5.9, 4.6, 5.3],
 * ])
 * const rule = oneStandardError(scores)
 * print('mean:', rule.mean)
 * print('se:', rule.se)
 * print('best, threshold:', rule.best, rule.threshold)
 * print('chosen:', rule.chosen)
 *
 * @example From a grid search: shrinking a mean predictor, larger shrinkage simpler
 * import { gridSearch, kFold } from 'aifn-compute/learning/validate'
 * const shrunkMean = ({ shrink }) => ({
 *   name: 'shrunk mean',
 *   fit: (d) => ({ decide: (x) => full([x.shape[0]], (1 - shrink) * mean(d.y)) }),
 * })
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const shrinks = { dims: { shrink: { type: 'real', min: 0, max: 0.3, default: 0 } } }
 * const data = { x: zeros([12, 1]), y: tensor([3, 5, 4, 6, 2, 5, 4, 3, 6, 5, 4, 4]) }
 * const search = gridSearch(shrunkMean, shrinks, { metrics: [mse], points: 4, refit: false }).run(data, kFold({ k: 4 }))
 * const rule = oneStandardError(search, { simplicity: (i) => -search.rows[i].params.shrink })
 * print('shrink:', search.rows.map((r) => r.params.shrink))
 * print('mean mse:', rule.mean)
 * print('best, chosen:', search.rows[rule.best].params, search.rows[rule.chosen].params)
 */
export function oneStandardError(
  input: Tensor | ScoredCandidates | readonly CrossValidation<unknown>[],
  options: OneStandardErrorOptions = {},
): OneStandardError {
  const { s, c, k, direction } = foldScores(input, options)
  if (c === 0) throw new DomainError('oneStandardError', 'oneStandardError: no candidates')
  if (k < 2) throw new DomainError('oneStandardError', 'oneStandardError: needs at least two folds')
  const mean = new Float64Array(c)
  const se = new Float64Array(c)
  for (let j = 0; j < c; j++) {
    let m = 0
    for (let f = 0; f < k; f++) m += s[j * k + f] / k
    let ss = 0
    for (let f = 0; f < k; f++) ss += (s[j * k + f] - m) ** 2
    mean[j] = m
    se[j] = Math.sqrt(ss / (k - 1) / k)
  }
  const sign = direction === 'higher' ? -1 : 1
  // The best by direction, the first among ties; a NaN mean never wins.
  let best = 0
  for (let j = 1; j < c; j++) if (sign * mean[j] < sign * mean[best] || Number.isNaN(mean[best])) best = j
  const threshold = mean[best] + sign * se[best]
  const { simplicity } = options
  let order: ArrayLike<number>
  if (simplicity === undefined) order = Float64Array.from({ length: c }, (_, j) => j)
  else if (typeof simplicity === 'function') order = Float64Array.from({ length: c }, (_, j) => simplicity(j))
  else order = dense.toF64(simplicity, 'oneStandardError')
  if (order.length !== c)
    throw new ShapeError('oneStandardError', `oneStandardError: ${c} candidates but ${order.length} simplicities`)
  const eligible = new Uint8Array(c)
  let chosen = best
  for (let j = 0; j < c; j++) {
    eligible[j] = sign * mean[j] <= sign * threshold ? 1 : 0
    if (eligible[j] && order[j] < order[chosen]) chosen = j
  }
  // Among equally simple eligible candidates, the first.
  for (let j = 0; j < chosen; j++) if (eligible[j] && order[j] === order[chosen]) chosen = j
  return {
    mean: fromData(mean, [c]),
    se: fromData(se, [c]),
    best,
    threshold,
    eligible: fromData(eligible, [c]),
    chosen,
    direction,
  }
}
