/**
 * Selecting and assessing a model over many data sets (Varma and Simon, 2006; Cawley and Talbot, 2010; Krstajic et
 * al., 2014; Hastie, Tibshirani and Friedman, 2009, §7.10): on each data set a penalty is tuned by cross-validation
 * over a grid, and every estimate of the tuned model's error is set beside its true error.
 *
 * Each data set has `rows` points of a regression problem of `aifn-methods/theory` (`drawTrainingSet`): the first
 * `rows - test` form the development set and the last `test` a held-out test set. On the development set
 * `nested` (`aifn-compute/learning/validate`) runs the `inner` grid search on every `outer` training set and also on
 * all the rows. The unnested search gives the minimum cross-validation score, its minimiser $\hat\lambda$ and
 * `oneStandardError`'s choice (`aifn-compute/learning/selection`). Each development fit's true error
 * $\mathrm{Err}(\lambda) = \sigma^2 + \expect_x(f(x) - \hat f_\lambda(x))^2$ is measured on one large fresh sample.
 * The estimates are the best inner score (optimistic: it chose $\hat\lambda$), the training error, the nested
 * cross-validation score and the test-set score. Each is compared with $\mathrm{Err}(\hat\lambda)$.
 *
 * The learners are penalised linear smoothers on a fixed basis. A `pspline` is cubic B-splines on equal intervals of
 * $[-1, 1]$ with a second-order difference penalty (Eilers and Marx, 1996), as `pspline` of
 * `aifn-compute/numerics/interpolate`. A `ridge-polynomial` is ridge regression on $x, \dots, x^d$ with an unpenalised
 * intercept, as `polynomialFeatures` followed by `linearRegression({ l2 })`. The basis of every row is computed once
 * per data set. The searches see the row indices as their inputs, so a fit only forms and solves the
 * penalised normal equations, $(\Bmat^\top\Bmat + \lambda\Pmat)\betavec = \Bmat^\top\yvec$, with
 * $\Bmat^\top\Bmat$ cached per training set. The splitters, searches and rules are the engine's own.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError } from 'aifn-compute/learning/selection'
import { gridSearch, kFold, nested, type Splitter } from 'aifn-compute/learning/validate'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { bsplineBasis, differencePenalty, uniformKnots } from 'aifn-compute/numerics/interpolate'
import { drawTrainingSet, type TargetName } from '../regression'

/** The learners of `selectionStudy`, each tuned by its penalty $\lambda$. */
export type SelectionLearnerName = 'pspline' | 'ridge-polynomial'

/**
 * A learner of `selectionStudy`: its name, or its name with its basis size.
 *
 * - `pspline`: cubic B-splines on `segments` equal intervals of $[-1, 1]$ (default 15, so 18 basis functions) with the
 *   penalty $\lambda\lVert \Delta^2\betavec \rVert^2$; no separate intercept.
 * - `ridge-polynomial`: $x, \dots, x^d$ for `degree` $d$ (default 10), centred, with the penalty
 *   $\lambda\lVert \wvec \rVert^2$ and an unpenalised intercept.
 */
export type SelectionLearner =
  SelectionLearnerName | { kind: 'pspline'; segments?: number } | { kind: 'ridge-polynomial'; degree?: number }

/** Options of `selectionStudy`. Every field is plain data or a splitter, so a compute worker can build them. */
export interface SelectionStudyOptions {
  /** The regression problem: `target` (default `sine`) and noise standard deviation `noise` $\sigma$ (default 0.3). */
  generator?: { target?: TargetName; noise?: number }
  /** Points per data set (default 100). */
  rows?: number
  /** Of these, the last `test` are the held-out test set (default 25). */
  test?: number
  /** Points of the fresh sample that measures true errors (default 5000), one sample for all data sets. */
  fresh?: number
  /** The learner (default `pspline`). */
  learner?: SelectionLearner
  /** The penalties $\lambda$ searched, larger meaning simpler (default 25 values from $10^{-4}$ to $10^4$). */
  grid?: readonly number[]
  /** The splitter of the search (default 10-fold, shuffled). */
  inner?: Splitter
  /** The splitter of nested cross-validation's outer level (default 10-fold, shuffled). */
  outer?: Splitter
  /** Data sets (default 200). */
  repeats?: number
  /** The index of the first data set (default 0), so that a long study can run in parts with the same streams. */
  first?: number
}

/** One estimate's agreement with the true error over the data sets. */
export interface EstimateSummary {
  /** The estimate's mean over data sets. */
  mean: number
  /** The mean of the estimate minus the true error $\mathrm{Err}(\hat\lambda)$: negative is optimistic. */
  bias: number
  /** The standard deviation of the estimate minus the true error. */
  sd: number
}

/** The per-data-set values of `selectionStudy`, one entry per data set. */
export interface SelectionRepeats {
  /** The minimum cross-validation score over the grid, at $\hat\lambda$. */
  bestScore: Float64Array
  /** The true error $\mathrm{Err}(\hat\lambda)$ of the development-set fit at $\hat\lambda$. */
  trueError: Float64Array
  /** The true error of the development-set fit at the one-standard-error choice. */
  oneSeTrueError: Float64Array
  /** The smallest true error over the grid (the oracle's choice). */
  oracleTrueError: Float64Array
  /** The development-set training error of the fit at $\hat\lambda$. */
  trainingError: Float64Array
  /** The nested cross-validation estimate, the mean outer-fold score. */
  nestedScore: Float64Array
  /** Its standard error: the outer scores' standard deviation over the square root of the outer folds. */
  nestedSe: Float64Array
  /** The mean squared error of the fit at $\hat\lambda$ on the test set. */
  testScore: Float64Array
  /** Its standard error: the squared errors' standard deviation over the square root of `test`. */
  testSe: Float64Array
  /** The grid index of $\hat\lambda$. */
  bestIndex: Int32Array
  /** The grid index of the one-standard-error choice. */
  oneSeIndex: Int32Array
  /** The grid index of the oracle's choice. */
  oracleIndex: Int32Array
}

/** The result of `selectionStudy`. */
export interface SelectionStudy {
  /** The penalties searched. */
  grid: number[]
  /** Every value per data set. */
  perRepeat: SelectionRepeats
  /**
   * For each estimate (and each true error), its mean, and the mean and standard deviation of its difference from
   * `trueError`.
   */
  summary: Record<
    'bestScore' | 'trueError' | 'oneSeTrueError' | 'oracleTrueError' | 'trainingError' | 'nestedScore' | 'testScore',
    EstimateSummary
  >
}

/** The default penalties: 25 values evenly spaced in $\log_{10}\lambda$ from $-4$ to 4, to three digits. */
const DEFAULT_GRID = Array.from({ length: 25 }, (_, i) => Number((10 ** (-4 + (i * 8) / 24)).toPrecision(3)))

/** A penalised basis: its rows at given points, the penalty matrix, and whether an intercept is centred out. */
interface Basis {
  /** The number of basis functions $p$. */
  p: number
  /** The basis at points, row-major $[m, p]$. */
  at: (x: Float64Array) => Float64Array
  /** The penalty matrix $\Pmat$, row-major $[p, p]$. */
  penalty: Float64Array
  /** Fit an unpenalised intercept by centring. */
  centre: boolean
}

/**
 * The basis and penalty of a learner.
 *
 * @param learner The learner.
 * @returns Its basis.
 */
function basisOf(learner: SelectionLearner): Basis {
  const spec = typeof learner === 'string' ? { kind: learner } : learner
  if (spec.kind === 'pspline') {
    const segments = (spec as { segments?: number }).segments ?? 15
    const knots = uniformKnots(-1, 1, segments, 3)
    const p = segments + 3
    return {
      p,
      at: (x) => Float64Array.from(toFlat(bsplineBasis(fromData(x, [x.length]), knots, 3))),
      penalty: Float64Array.from(toFlat(differencePenalty(p, 2))),
      centre: false,
    }
  }
  if (spec.kind === 'ridge-polynomial') {
    const p = (spec as { degree?: number }).degree ?? 10
    const penalty = new Float64Array(p * p)
    for (let j = 0; j < p; j++) penalty[j * p + j] = 1
    const at = (x: Float64Array) => {
      const out = new Float64Array(x.length * p)
      for (let i = 0; i < x.length; i++) {
        let v = x[i]
        for (let j = 0; j < p; j++) {
          out[i * p + j] = v
          v *= x[i]
        }
      }
      return out
    }
    return { p, at, penalty, centre: true }
  }
  throw new DomainError('selectionStudy', `selectionStudy: unknown learner '${String(spec.kind)}'`)
}

/** A fitted penalised smoother: coefficients and intercept. */
interface SmootherFit {
  /** The coefficients $\betavec$, $p$ values. */
  beta: Float64Array
  /** The intercept (0 without centring). */
  intercept: number
}

/**
 * The predictions $b + \Bmat\betavec$ of a fit on basis rows.
 *
 * @param fit The fit.
 * @param B The basis rows, row-major $[m, p]$.
 * @param p The number of basis functions.
 * @returns The $m$ predictions.
 */
function predict(fit: SmootherFit, B: Float64Array, p: number): Float64Array {
  const m = B.length / p
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let v = fit.intercept
    for (let j = 0; j < p; j++) v += B[i * p + j] * fit.beta[j]
    out[i] = v
  }
  return out
}

/**
 * Solve $\Amat\betavec = \bvec$ for a symmetric positive definite $p \times p$ matrix by its Cholesky factor, in place
 * on copies; the small systems of the study are solved without tensor overhead.
 *
 * @param A The matrix, row-major $[p, p]$; not modified.
 * @param b The right-hand side, $p$ values; not modified.
 * @param p The size.
 * @returns The solution, or undefined when a pivot is not positive (the matrix is not numerically positive definite).
 */
function solvePositive(A: Float64Array, b: Float64Array, p: number): Float64Array | undefined {
  const L = new Float64Array(p * p)
  for (let j = 0; j < p; j++) {
    let d = A[j * p + j]
    for (let k = 0; k < j; k++) d -= L[j * p + k] ** 2
    if (!(d > 0)) return undefined
    const ljj = Math.sqrt(d)
    L[j * p + j] = ljj
    for (let i = j + 1; i < p; i++) {
      let v = A[i * p + j]
      for (let k = 0; k < j; k++) v -= L[i * p + k] * L[j * p + k]
      L[i * p + j] = v / ljj
    }
  }
  const z = new Float64Array(p)
  for (let i = 0; i < p; i++) {
    let v = b[i]
    for (let k = 0; k < i; k++) v -= L[i * p + k] * z[k]
    z[i] = v / L[i * p + i]
  }
  for (let i = p - 1; i >= 0; i--) {
    let v = z[i]
    for (let k = i + 1; k < p; k++) v -= L[k * p + i] * z[k]
    z[i] = v / L[i * p + i]
  }
  return z
}

/**
 * The learner of one data set as estimators over row indices: `make(lambda)` fits rows given by index (the inputs
 * `x` of the data it is given, $m \times 1$) from the precomputed basis rows `B` and targets `y`, and its model's
 * `decide` predicts at rows given by index. The cross-products of each training set are cached, so the grid shares
 * them.
 *
 * @param basis The basis.
 * @param B The basis at every development row, row-major $[n, p]$.
 * @param y The development targets.
 * @returns `make`.
 */
function indexLearner(basis: Basis, B: Float64Array, y: Float64Array) {
  const { p, penalty, centre } = basis
  const cache = new Map<string, { G: Float64Array; b: Float64Array; xm: Float64Array; ym: number }>()
  const crossProducts = (rows: ArrayLike<number>) => {
    const key = Array.prototype.join.call(rows, ',')
    const hit = cache.get(key)
    if (hit) return hit
    const m = rows.length
    const xm = new Float64Array(p)
    let ym = 0
    if (centre) {
      for (const r of Array.from(rows)) {
        ym += y[r] / m
        for (let j = 0; j < p; j++) xm[j] += B[r * p + j] / m
      }
    }
    const G = new Float64Array(p * p)
    const b = new Float64Array(p)
    const row = new Float64Array(p)
    for (const r of Array.from(rows)) {
      for (let j = 0; j < p; j++) row[j] = B[r * p + j] - xm[j]
      const t = y[r] - ym
      for (let j = 0; j < p; j++) {
        const v = row[j]
        if (v === 0) continue
        b[j] += v * t
        for (let k = 0; k < p; k++) G[j * p + k] += v * row[k]
      }
    }
    const out = { G, b, xm, ym }
    cache.set(key, out)
    return out
  }
  const fitRows = (rows: ArrayLike<number>, lambda: number): SmootherFit => {
    const { G, b, xm, ym } = crossProducts(rows)
    const A = new Float64Array(p * p)
    for (let k = 0; k < p * p; k++) A[k] = G[k] + lambda * penalty[k]
    const beta = solvePositive(A, b, p) ?? Float64Array.from(dense.data(lstsq(fromData(A, [p, p]), fromData(b, [p])).x))
    let intercept = ym
    for (let j = 0; j < p; j++) intercept -= xm[j] * beta[j]
    return { beta, intercept }
  }
  const rowsOf = (x: Tensor) => dense.data(x)
  return (lambda: number) => ({
    name: 'penalised-smoother',
    fit: (d: { x: Tensor; y: Tensor }) => {
      const fit = fitRows(rowsOf(d.x), lambda)
      return {
        ...fit,
        decide: (x: Tensor) => {
          const rows = rowsOf(x)
          const Bx = new Float64Array(rows.length * p)
          for (let i = 0; i < rows.length; i++) Bx.set(B.subarray(rows[i] * p, (rows[i] + 1) * p), i * p)
          return fromData(predict(fit, Bx, p), [rows.length])
        },
      }
    },
  })
}

/**
 * The mean squared error of predictions.
 *
 * @param y The targets.
 * @param yHat The predictions.
 * @returns The mean of $(y_i - \hat y_i)^2$.
 */
function mse(y: ArrayLike<number>, yHat: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < y.length; i++) s += (y[i] - yHat[i]) ** 2
  return s / y.length
}

/**
 * Model selection and assessment over many data sets (see the file's introduction). Per data set: the tuned penalty
 * $\hat\lambda$ minimising the inner cross-validation score; the one-standard-error choice (larger $\lambda$ simpler)
 * and the oracle's choice; the true errors of the development-set fits at all three; and four estimates of
 * $\mathrm{Err}(\hat\lambda)$: the best inner score, the training error, nested cross-validation and the test set,
 * the last two with standard errors. Costs about $(K_\text{outer} + 1) K_\text{inner} c$ solves of a $p \times p$
 * system per data set for $c$ penalties. Throws `DomainError` for an unknown learner, a `test` that leaves no
 * development rows, or an empty grid.
 *
 * @param s The stream: the fresh sample comes from `child(s, 'fresh')`; data set $r$ from `child(s, 'repeat', r)`,
 *   whose child `data` draws the points and whose child `cv` is `nested`'s stream. `s` is not advanced.
 * @param options The problem, sizes, learner, grid, splitters and number of data sets (see `SelectionStudyOptions`).
 * @returns The grid, every value per data set, and each estimate's mean, bias and spread against the true error (see
 *   `SelectionStudy`).
 *
 * @example Five small data sets: each estimate's mean beside the mean true error
 * import { kFold } from 'aifn-compute/learning/validate'
 * const folds = kFold({ k: 5, shuffle: true })
 * const grid = [0.01, 0.1, 1, 10, 100]
 * const r = selectionStudy(stream(0), { repeats: 5, fresh: 1000, grid, inner: folds, outer: folds })
 * print('true error     ', r.summary.trueError.mean)
 * print('best CV score  ', r.summary.bestScore.mean)
 * print('nested CV      ', r.summary.nestedScore.mean)
 * print('test set       ', r.summary.testScore.mean)
 * print('training error ', r.summary.trainingError.mean)
 * print('chosen indices ', r.perRepeat.bestIndex)
 */
export function selectionStudy(s: Stream, options: SelectionStudyOptions = {}): SelectionStudy {
  const {
    generator = {},
    rows = 100,
    test = 25,
    fresh = 5000,
    learner = 'pspline',
    grid = DEFAULT_GRID,
    inner = kFold({ k: 10, shuffle: true }),
    outer = kFold({ k: 10, shuffle: true }),
    repeats = 200,
    first = 0,
  } = options
  const { target = 'sine', noise = 0.3 } = generator
  const nDev = rows - test
  if (!(test >= 1 && nDev >= 2))
    throw new DomainError('selectionStudy', 'selectionStudy: test must leave at least two development rows')
  if (grid.length === 0) throw new DomainError('selectionStudy', 'selectionStudy: the grid is empty')
  const basis = basisOf(learner)
  const { p } = basis
  const c = grid.length
  const freshSet = drawTrainingSet(child(s, 'fresh'), target, fresh, noise)
  const freshB = basis.at(freshSet.x)
  const lambdas = space({ lambda: oneOf([...grid]) })
  const index = fromData(
    Float64Array.from({ length: nDev }, (_, i) => i),
    [nDev, 1],
  )

  const column = () => new Float64Array(repeats)
  const out: SelectionRepeats = {
    bestScore: column(),
    trueError: column(),
    oneSeTrueError: column(),
    oracleTrueError: column(),
    trainingError: column(),
    nestedScore: column(),
    nestedSe: column(),
    testScore: column(),
    testSe: column(),
    bestIndex: new Int32Array(repeats),
    oneSeIndex: new Int32Array(repeats),
    oracleIndex: new Int32Array(repeats),
  }
  for (let r = 0; r < repeats; r++) {
    const rs = child(s, 'repeat', first + r)
    const all = drawTrainingSet(child(rs, 'data'), target, rows, noise)
    const yDev = all.y.slice(0, nDev)
    const yTest = all.y.slice(nDev)
    const Bdev = basis.at(all.x.slice(0, nDev))
    const Btest = basis.at(all.x.slice(nDev))
    const make = indexLearner(basis, Bdev, yDev)
    const search = gridSearch(({ lambda }) => make(lambda), lambdas, { metrics: [meanSquaredError] })
    const result = nested(outer, inner, search, dataset(index, fromData(yDev)), { stream: child(rs, 'cv') })
    const rule = oneStandardError(result.unnested, { simplicity: (i) => -grid[i] })
    const data = { x: index, y: fromData(yDev) }
    const fits = grid.map((lambda) => make(lambda).fit(data))
    const truth = fits.map((f) => mse(freshSet.y, predict(f, freshB, p)))
    let oracle = 0
    for (let i = 1; i < c; i++) if (truth[i] < truth[oracle]) oracle = i
    const best = rule.best
    const testPred = predict(fits[best], Btest, p)
    const sq = Float64Array.from(yTest, (v, i) => (v - testPred[i]) ** 2)
    const testMean = sq.reduce((a, b) => a + b, 0) / test
    const testSd = test > 1 ? Math.sqrt(sq.reduce((a, b) => a + (b - testMean) ** 2, 0) / (test - 1)) : NaN
    out.bestScore[r] = toFlat(rule.mean)[best]
    out.trueError[r] = truth[best]
    out.oneSeTrueError[r] = truth[rule.chosen]
    out.oracleTrueError[r] = truth[oracle]
    out.trainingError[r] = mse(yDev, predict(fits[best], Bdev, p))
    out.nestedScore[r] = result.nestedScore
    out.nestedSe[r] = result.nestedStd / Math.sqrt(result.perFold.length)
    out.testScore[r] = testMean
    out.testSe[r] = testSd / Math.sqrt(test)
    out.bestIndex[r] = best
    out.oneSeIndex[r] = rule.chosen
    out.oracleIndex[r] = oracle
  }
  const summarise = (v: Float64Array): EstimateSummary => {
    const n = v.length
    let mean = 0
    let bias = 0
    for (let i = 0; i < n; i++) {
      mean += v[i] / n
      bias += (v[i] - out.trueError[i]) / n
    }
    let ss = 0
    for (let i = 0; i < n; i++) ss += (v[i] - out.trueError[i] - bias) ** 2
    return { mean, bias, sd: n > 1 ? Math.sqrt(ss / (n - 1)) : NaN }
  }
  return {
    grid: [...grid],
    perRepeat: out,
    summary: {
      bestScore: summarise(out.bestScore),
      trueError: summarise(out.trueError),
      oneSeTrueError: summarise(out.oneSeTrueError),
      oracleTrueError: summarise(out.oracleTrueError),
      trainingError: summarise(out.trainingError),
      nestedScore: summarise(out.nestedScore),
      testScore: summarise(out.testScore),
    },
  }
}
