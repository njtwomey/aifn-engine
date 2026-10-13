/**
 * Subset selection for least squares (Hastie, Tibshirani and Friedman, 2009, §3.3): the best subset of each size by
 * exhaustive search, forward and backward stepwise selection with a stopping criterion, and incremental forward
 * stagewise regression, the greedy methods to set against the lasso and least-angle paths.
 *
 * Every fit is ordinary least squares on a subset of the columns, with an unpenalised intercept when the data are
 * centred (`intercept`, the default). The fits work on the Gram matrix $\Xmat^\top\Xmat$ and $\Xmat^\top\yvec$: the
 * residual sum of squares of a subset $S$ is $\yvec^\top\yvec - \zvec^\top\zvec$ with $\Lmat\zvec = \Xmat_S^\top\yvec$
 * and $\Lmat\Lmat^\top = \Xmat_S^\top\Xmat_S$, so each costs $\lvert S \rvert^3$ operations whatever $n$. A column that
 * is (numerically) a combination of the earlier columns of the subset is skipped, so a collinear subset gets the
 * residual of the projection onto its span and a weight of 0 on that column, and its rank, not its size, counts as
 * its parameters. The information criteria are those of R's `step` for a linear model,
 * $\mathrm{AIC} = n\ln(\mathrm{RSS}/n) + 2k$ and $\mathrm{BIC} = n\ln(\mathrm{RSS}/n) + k\ln n$, with $k$ the rank of
 * the subset plus the intercept.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { centre, gram, type Gram } from './elasticNet'

/** A least-squares fit on a subset of the columns. */
interface SubsetFit {
  /** The residual sum of squares. */
  readonly rss: number
  /** The number of columns not skipped as collinear. */
  readonly rank: number
  /** The weights, $d$ values, zero off the subset (only when asked for). */
  readonly coef: Float64Array | null
}

/**
 * Least squares on the columns `subset` of a problem in Gram form, by a Cholesky factorisation that skips a column
 * whose pivot falls below $10^{-12}$ of its diagonal entry (a column in the span of the earlier ones).
 *
 * @param g The centred problem in Gram form.
 * @param subset The columns, in any order (a skipped column is the later of a dependent set).
 * @param weights Whether to solve for the weights as well as the residual sum of squares.
 * @returns The residual sum of squares, the rank and, with `weights`, the $d$ weights.
 */
function subsetFit(g: Gram, subset: readonly number[], weights: boolean): SubsetFit {
  const { d, H, q, yy } = g
  const k = subset.length
  const L = new Float64Array(k * k)
  const z = new Float64Array(k)
  const kept: boolean[] = []
  let explained = 0
  for (let a = 0; a < k; a++) {
    const ja = subset[a]
    for (let b = 0; b <= a; b++) {
      if (!kept[b] && b < a) continue
      let s = H[ja * d + subset[b]]
      for (let c = 0; c < b; c++) if (kept[c]) s -= L[a * k + c] * L[b * k + c]
      if (b < a) L[a * k + b] = s / L[b * k + b]
      else {
        const ok = s > 1e-12 * H[ja * d + ja] && s > 0
        kept[a] = ok
        L[a * k + a] = ok ? Math.sqrt(s) : 0
      }
    }
    if (!kept[a]) continue
    let s = q[ja]
    for (let c = 0; c < a; c++) if (kept[c]) s -= L[a * k + c] * z[c]
    z[a] = s / L[a * k + a]
    explained += z[a] * z[a]
  }
  const rank = kept.filter(Boolean).length
  let coef: Float64Array | null = null
  if (weights) {
    coef = new Float64Array(d)
    const u = new Float64Array(k)
    for (let a = k - 1; a >= 0; a--) {
      if (!kept[a]) continue
      let s = z[a]
      for (let c = a + 1; c < k; c++) if (kept[c]) s -= L[c * k + a] * u[c]
      u[a] = s / L[a * k + a]
      coef[subset[a]] = u[a]
    }
  }
  return { rss: Math.max(0, yy - explained), rank, coef }
}

/**
 * The intercept that goes with weights fitted on centred data.
 *
 * @param yMean The mean of the target.
 * @param xMean The column means.
 * @param w The weights.
 * @returns $\bar y - \bar{\xvec}^\top\wvec$.
 */
function interceptOf(yMean: number, xMean: Float64Array, w: Float64Array): number {
  let b = yMean
  for (let j = 0; j < w.length; j++) b -= xMean[j] * w[j]
  return b
}

/**
 * The number of subsets of at most `most` of `d` columns.
 *
 * @param d The columns.
 * @param most The largest subset size.
 * @returns $\sum_{k \le \text{most}} \binom{d}{k}$.
 */
function subsetCount(d: number, most: number): number {
  let total = 0
  let choose = 1
  for (let k = 0; k <= most; k++) {
    total += choose
    choose = (choose * (d - k)) / (k + 1)
  }
  return total
}

/** Options of `bestSubset`. */
export interface BestSubsetOptions {
  /** The largest subset size searched (default every column, $d$). */
  maxSize?: number
  /** Centre the data, for an unpenalised intercept (default true). */
  intercept?: boolean
}

/** The best subset of each size, and every subset searched. */
export interface BestSubsets {
  /** The sizes $k = 0, 1, \dots$, up to `maxSize`. */
  readonly sizes: number[]
  /** The smallest residual sum of squares of each size. */
  readonly rss: number[]
  /** The best subset of each size, its columns in increasing order. */
  readonly subsets: number[][]
  /** The weights of each best subset, $d \times (K + 1)$: column $k$ is the fit of size $k$, zero off its subset. */
  readonly coef: Tensor
  /** The intercept of each best subset (0 without an intercept). */
  readonly intercept: number[]
  /** Every subset searched, by size and then lexicographically: its size, columns and residual sum of squares. */
  readonly all: { readonly sizes: number[]; readonly subsets: number[][]; readonly rss: number[] }
}

/**
 * Best-subset selection by exhaustive search (Hastie, Tibshirani and Friedman, 2009, §3.3.1): for each size $k$ up
 * to `maxSize`, the subset of $k$ columns whose least-squares fit has the smallest residual sum of squares, and the
 * residual sum of squares of every subset searched (the cloud of points under the lower envelope in a plot against
 * $k$). The search visits $\sum_k \binom{d}{k}$ subsets, so it is for small $d$: it throws `DomainError` when that
 * count passes $2^{20}$ (lower `maxSize`, or use `stepwise`). Ties go to the lexicographically first subset. Throws
 * `DomainError` for a `maxSize` outside $[0, d]$, and `ShapeError` for mismatched shapes.
 *
 * @param x The design, $n \times d$.
 * @param y The target, $n$ values.
 * @param options The largest size and whether to centre, as `BestSubsetOptions`.
 * @returns The best subset, residual sum of squares, weights and intercept of each size, and every subset searched.
 *
 * @example The best subsets of three columns are nested here, but need not be
 * const x = normals(stream(0), [50, 3])
 * const y = tensor(toArray(x).map((r, i) => 2 * r[0] - r[1] + 0.3 * Math.sin(i)))
 * const best = bestSubset(x, y)
 * print('best subsets:', best.subsets)
 * print('rss by size:', best.rss.map((v) => v.toFixed(3)))
 * print('every subset:', best.all.subsets.map((s, i) => `{${s}}: ${best.all.rss[i].toFixed(2)}`))
 */
export function bestSubset(x: Tensor, y: Tensor, options: BestSubsetOptions = {}): BestSubsets {
  const { intercept = true } = options
  const c = centre(x, y, intercept, 'bestSubset')
  const g = gram(c)
  const { d } = c
  const most = options.maxSize ?? d
  if (!(Number.isInteger(most) && most >= 0 && most <= d))
    throw new DomainError('bestSubset', `bestSubset: maxSize must be an integer from 0 to ${d}`)
  if (subsetCount(d, most) > 2 ** 20)
    throw new DomainError('bestSubset', 'bestSubset: more than 2^20 subsets; lower maxSize or use stepwise')
  const all = { sizes: [] as number[], subsets: [] as number[][], rss: [] as number[] }
  const sizes: number[] = []
  const rss: number[] = []
  const subsets: number[][] = []
  for (let k = 0; k <= most; k++) {
    let best = Infinity
    let bestSet: number[] = []
    // The k-subsets in lexicographic order.
    const idx = Array.from({ length: k }, (_, i) => i)
    for (;;) {
      const r = subsetFit(g, idx, false).rss
      all.sizes.push(k)
      all.subsets.push([...idx])
      all.rss.push(r)
      if (r < best) {
        best = r
        bestSet = [...idx]
      }
      let i = k - 1
      while (i >= 0 && idx[i] === d - k + i) i--
      if (i < 0) break
      idx[i]++
      for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1
    }
    sizes.push(k)
    rss.push(best)
    subsets.push(bestSet)
  }
  const K = sizes.length
  const coef = new Float64Array(d * K)
  const intercepts: number[] = []
  subsets.forEach((s, t) => {
    const w = subsetFit(g, s, true).coef!
    for (let j = 0; j < d; j++) coef[j * K + t] = w[j]
    intercepts.push(interceptOf(c.yMean, c.xMean, w))
  })
  return { sizes, rss, subsets, coef: fromData(coef, [d, K]), intercept: intercepts, all }
}

/** Options of `stepwise`. */
export interface StepwiseOptions {
  /**
   * `'forward'` (default): from the empty model, add the column that most lowers the criterion. `'backward'`: from
   * the full model, remove the column whose removal most lowers it. `'both'`: from `start`, make whichever addition
   * or removal most lowers it, as R's `step`.
   */
  direction?: 'forward' | 'backward' | 'both'
  /**
   * What a step minimises and when the run stops: `'rss'` (default for forward and backward) runs to the end, the
   * full or the empty model, so every size is visited; `'aic'` (default for `'both'`) and `'bic'` stop when no move
   * lowers the criterion. `'both'` needs `'aic'` or `'bic'`.
   */
  criterion?: 'rss' | 'aic' | 'bic'
  /** The columns of the starting model for `'both'` (default none). */
  start?: readonly number[]
  /** The most moves (default no limit beyond the criterion's own stop). */
  maxSteps?: number
  /** Centre the data, for an unpenalised intercept (default true). */
  intercept?: boolean
}

/** One model of a stepwise run: the move that reached it and its fit. */
export interface StepwiseStep {
  /** `'start'` for the first model, then `'add'` or `'remove'`. */
  readonly action: 'start' | 'add' | 'remove'
  /** The column added or removed ($-1$ at the start). */
  readonly feature: number
  /** The columns of the model, in increasing order. */
  readonly active: number[]
  /** Its residual sum of squares. */
  readonly rss: number
  /** Its criterion (the residual sum of squares for `'rss'`). */
  readonly criterion: number
}

/** A stepwise run: the sequence of models, with their fits. */
export interface Stepwise {
  /** The models visited, the starting one first. */
  readonly steps: StepwiseStep[]
  /** The weights of each model, $d \times k$: column $t$ is `steps[t]`. */
  readonly coef: Tensor
  /** The intercept of each model (0 without an intercept). */
  readonly intercept: number[]
}

/**
 * Forward, backward or bidirectional stepwise selection for least squares (Hastie, Tibshirani and Friedman, 2009,
 * §3.3.2): each move adds or removes the one column that most lowers the criterion, which for a move of a given kind
 * is the one giving the smallest residual sum of squares. With `'rss'` forward selection visits one model of every
 * size, to set against `bestSubset` and the lasso path; with `'aic'` or `'bic'` the run stops at a local minimum of
 * the criterion. Ties go to the lowest column index. Unlike scikit-learn's `SequentialFeatureSelector`, which scores
 * by cross-validation, this is the classical greedy search on the training fit. Throws `DomainError` for an unknown
 * direction or criterion, for `'both'` with `'rss'`, and for a `start` column out of range; `ShapeError` for
 * mismatched shapes.
 *
 * @param x The design, $n \times d$.
 * @param y The target, $n$ values.
 * @param options The direction, criterion, starting model, most moves and centring, as `StepwiseOptions`.
 * @returns The models visited, with the move, columns, residual sum of squares and criterion of each, and their
 *   weights and intercepts.
 *
 * @example Forward selection on four columns, and where AIC stops it
 * // Only the first two columns carry signal.
 * const x = normals(stream(0), [60, 4])
 * const noise = toFlat(normals(stream(10), [60]))
 * const y = tensor(toArray(x).map((r, i) => 2 * r[0] - r[1] + noise[i]))
 * const all = stepwise(x, y)
 * print('forward:', all.steps.map((s) => `${s.action} ${s.feature}: rss ${s.rss.toFixed(2)}`))
 * const aic = stepwise(x, y, { criterion: 'aic' })
 * print('with AIC:', aic.steps.map((s) => `${s.action} ${s.feature}: aic ${s.criterion.toFixed(2)}`))
 */
export function stepwise(x: Tensor, y: Tensor, options: StepwiseOptions = {}): Stepwise {
  const { direction = 'forward', intercept = true, maxSteps = Infinity } = options
  const criterion = options.criterion ?? (direction === 'both' ? 'aic' : 'rss')
  if (!['forward', 'backward', 'both'].includes(direction))
    throw new DomainError('stepwise', `stepwise: unknown direction '${String(direction)}'`)
  if (!['rss', 'aic', 'bic'].includes(criterion))
    throw new DomainError('stepwise', `stepwise: unknown criterion '${String(criterion)}'`)
  if (direction === 'both' && criterion === 'rss')
    throw new DomainError('stepwise', "stepwise: direction 'both' needs criterion 'aic' or 'bic'")
  const c = centre(x, y, intercept, 'stepwise')
  const g = gram(c)
  const { n, d } = c
  const score = (fit: SubsetFit): number => {
    if (criterion === 'rss') return fit.rss
    const k = fit.rank + (intercept ? 1 : 0)
    return n * Math.log(fit.rss / n) + (criterion === 'aic' ? 2 : Math.log(n)) * k
  }
  let active: number[]
  if (direction === 'forward') active = []
  else if (direction === 'backward') active = Array.from({ length: d }, (_, j) => j)
  else {
    active = [...new Set(options.start ?? [])].sort((a, b) => a - b)
    if (active.some((j) => !(Number.isInteger(j) && j >= 0 && j < d)))
      throw new DomainError('stepwise', `stepwise: start columns must be integers from 0 to ${d - 1}`)
  }
  const first = subsetFit(g, active, false)
  const steps: StepwiseStep[] = [
    { action: 'start', feature: -1, active: [...active], rss: first.rss, criterion: score(first) },
  ]
  while (steps.length - 1 < maxSteps) {
    const current = steps[steps.length - 1]
    let best: StepwiseStep | null = null
    const consider = (action: 'add' | 'remove', feature: number): void => {
      const next = action === 'add' ? [...active, feature].sort((a, b) => a - b) : active.filter((j) => j !== feature)
      const fit = subsetFit(g, next, false)
      const value = score(fit)
      if (best === null || value < best.criterion)
        best = { action, feature, active: next, rss: fit.rss, criterion: value }
    }
    if (direction !== 'backward') for (let j = 0; j < d; j++) if (!active.includes(j)) consider('add', j)
    if (direction !== 'forward') for (const j of active) consider('remove', j)
    const chosen = best as StepwiseStep | null
    if (chosen === null) break
    if (criterion !== 'rss' && !(chosen.criterion < current.criterion)) break
    steps.push(chosen)
    active = chosen.active
  }
  const k = steps.length
  const coef = new Float64Array(d * k)
  const intercepts: number[] = []
  steps.forEach((s, t) => {
    const w = subsetFit(g, s.active, true).coef!
    for (let j = 0; j < d; j++) coef[j * k + t] = w[j]
    intercepts.push(interceptOf(c.yMean, c.xMean, w))
  })
  return { steps, coef: fromData(coef, [d, k]), intercept: intercepts }
}

/** Options of `forwardStagewise`. */
export interface ForwardStagewiseOptions {
  /** The step $\varepsilon > 0$ added to one weight at a time (default 0.01). */
  epsilon?: number
  /** The most steps (default 1000). */
  steps?: number
  /** Centre the data, for an unpenalised intercept (default true). */
  intercept?: boolean
}

/** The incremental forward stagewise path, one point per step. */
export interface ForwardStagewisePath {
  /** The weights after each step, $d \times (k + 1)$: column 0 is zero, column $t$ after $t$ steps. */
  readonly coef: Tensor
  /** $\norm{\wvec}_1$ after each step, the horizontal axis on which to set the path against the lasso's. */
  readonly l1Norm: number[]
  /** The column moved at each step, $k$ values. */
  readonly features: number[]
  /** The residual sum of squares after each step, $k + 1$ values. */
  readonly rss: number[]
  /** The intercept after each step (0 without an intercept). */
  readonly intercept: number[]
  /** Whether the run stopped because no step of size $\varepsilon$ lowers the residual sum of squares. */
  readonly converged: boolean
}

/**
 * Incremental forward stagewise regression, $\mathrm{FS}_\varepsilon$ (Hastie, Tibshirani and Friedman, 2009,
 * §3.8.1; Efron, Hastie, Johnstone and Tibshirani, 2004): from $\wvec = \zeros$, find the column $j$ most correlated
 * with the residual, $\lvert \xvec_j^\top\rvec \rvert$ largest, and move its weight by
 * $\varepsilon\operatorname{sign}(\xvec_j^\top\rvec)$; repeat. As $\varepsilon \to 0$ the path tends to the
 * monotone lasso path, which is the lasso path itself wherever no lasso weight crosses back towards zero. The
 * columns should be standardised, as the method compares raw correlations. The run stops at `steps`, or when the
 * best step would raise the residual sum of squares
 * ($\lvert \xvec_j^\top\rvec \rvert < \varepsilon\norm{\xvec_j}^2/2$), flagged `converged`. Ties go to the lowest
 * column index. Throws `DomainError` for $\varepsilon \le 0$ or a `steps` that is not a non-negative integer, and
 * `ShapeError` for mismatched shapes.
 *
 * @param x The design, $n \times d$, standardised.
 * @param y The target, $n$ values.
 * @param options The step size, the most steps and centring, as `ForwardStagewiseOptions`.
 * @returns The path: weights, $\ell_1$ norms, residual sums of squares and intercepts after each step, and the
 *   column moved at each.
 *
 * @example Small steps trace the lasso path
 * const z = normals(stream(0), [100, 3])
 * const y = tensor(toArray(z).map((r) => 3 * r[0] + 1.5 * r[1] + 0.5 * r[2]))
 * const path = forwardStagewise(z, y, { epsilon: 0.01, steps: 2000 })
 * print('steps =', path.features.length, ' converged =', path.converged)
 * print('final weights =', toArray(path.coef).map((row) => row.at(-1).toFixed(3)))
 * print('step at which each column first moves:', [0, 1, 2].map((j) => path.features.indexOf(j)))
 */
export function forwardStagewise(x: Tensor, y: Tensor, options: ForwardStagewiseOptions = {}): ForwardStagewisePath {
  const { epsilon = 0.01, steps = 1000, intercept = true } = options
  if (!(epsilon > 0) || !Number.isFinite(epsilon))
    throw new DomainError('forwardStagewise', 'forwardStagewise: epsilon must be positive and finite')
  if (!(Number.isInteger(steps) && steps >= 0))
    throw new DomainError('forwardStagewise', 'forwardStagewise: steps must be a non-negative integer')
  const c = centre(x, y, intercept, 'forwardStagewise')
  const { d, H, q, yy } = gram(c)
  const w = new Float64Array(d)
  // The correlations Xᵀr with the current residual.
  const corr = Float64Array.from(q)
  let rss = yy
  const snapshots: Float64Array[] = [w.slice()]
  const rssPath = [rss]
  const features: number[] = []
  let converged = false
  for (let t = 0; t < steps; t++) {
    let j = -1
    let top = 0
    for (let k = 0; k < d; k++) {
      if (Math.abs(corr[k]) > top) {
        top = Math.abs(corr[k])
        j = k
      }
    }
    if (j < 0 || top < (epsilon * H[j * d + j]) / 2) {
      converged = true
      break
    }
    const delta = epsilon * Math.sign(corr[j])
    w[j] += delta
    rss += -2 * delta * corr[j] + delta * delta * H[j * d + j]
    for (let k = 0; k < d; k++) corr[k] -= delta * H[k * d + j]
    snapshots.push(w.slice())
    rssPath.push(rss)
    features.push(j)
  }
  const k = snapshots.length
  const coef = new Float64Array(d * k)
  const l1Norm: number[] = []
  const intercepts: number[] = []
  snapshots.forEach((wt, t) => {
    let l1 = 0
    for (let j = 0; j < d; j++) {
      coef[j * k + t] = wt[j]
      l1 += Math.abs(wt[j])
    }
    l1Norm.push(l1)
    intercepts.push(interceptOf(c.yMean, c.xMean, wt))
  })
  return { coef: fromData(coef, [d, k]), l1Norm, features, rss: rssPath, intercept: intercepts, converged }
}
