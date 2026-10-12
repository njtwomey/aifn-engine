/**
 * The elastic net and the lasso by cyclic coordinate descent with soft-thresholding (Friedman, Hastie and Tibshirani,
 * 2010), one fit at a time or along a warm-started path of penalties, in scikit-learn's scaling.
 *
 * Both minimise
 *
 * $$
 * \frac{1}{2n}\norm{\yvec - \Xmat\wvec - b\ones}^2 + \alpha\rho\norm{\wvec}_1
 *   + \frac{\alpha(1 - \rho)}{2}\norm{\wvec}^2
 * $$
 *
 * with $\rho$ the `l1Ratio` ($\rho = 1$ is the lasso of Tibshirani, 1996), as scikit-learn's `ElasticNet`, `Lasso`
 * and `enet_path`. With an intercept the data are centred first, so $b$ is never penalised. The sweeps work on the
 * Gram matrix $\Xmat^\top\Xmat$ and $\Xmat^\top\yvec$, so one sweep costs $d^2$ operations whatever $n$; a coordinate
 * set to zero by the threshold is exactly zero. A run stops when the largest change of a sweep is below `tolerance`
 * times the largest weight and the duality gap is below `tolerance` times $\yvec^\top\yvec$ (scikit-learn's test,
 * on the centred target).
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Decides, Estimator, Fitted, Supervised } from 'aifn-compute/learning/estimators'
import { defineModel, matrixShape, targetValues } from 'aifn-compute/learning/estimators'
import { bool, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A design and target, centred when an intercept is fitted. */
export interface Centred {
  /** Rows. */
  readonly n: number
  /** Columns. */
  readonly d: number
  /** The design, centred by column when `intercept` holds; $n \times d$ row-major. */
  readonly x: Float64Array
  /** The target, centred when `intercept` holds; $n$ values. */
  readonly y: Float64Array
  /** The column means of the design (zeros without an intercept). */
  readonly xMean: Float64Array
  /** The mean of the target (0 without an intercept). */
  readonly yMean: number
}

/**
 * The design and target as float64 copies, centred when an intercept is to be fitted. Throws `ShapeError` when `x` is
 * not a matrix or `y` does not have one value per row.
 *
 * @param x The design, $n \times d$.
 * @param y The target, $n$ values.
 * @param intercept Whether to centre (an intercept is fitted, and so not penalised).
 * @param where The caller's name, for error messages.
 * @returns The copies, with the means removed.
 */
export function centre(x: Tensor, y: Tensor, intercept: boolean, where: string): Centred {
  const [n, d] = matrixShape(x, where)
  const target = targetValues(y, where)
  if (target.length !== n) throw new ShapeError(where, `${where}: ${n} rows of x but ${target.length} targets`)
  const X = Float64Array.from(dense.data(x))
  const Y = Float64Array.from(target)
  const xMean = new Float64Array(d)
  let yMean = 0
  if (intercept) {
    for (let i = 0; i < n; i++) {
      yMean += Y[i] / n
      for (let j = 0; j < d; j++) xMean[j] += X[i * d + j] / n
    }
    for (let i = 0; i < n; i++) {
      Y[i] -= yMean
      for (let j = 0; j < d; j++) X[i * d + j] -= xMean[j]
    }
  }
  return { n, d, x: X, y: Y, xMean, yMean }
}

/** The Gram form of a least-squares problem: $\Xmat^\top\Xmat$, $\Xmat^\top\yvec$ and $\yvec^\top\yvec$. */
export interface Gram {
  /** Rows of the design. */
  readonly n: number
  /** Columns of the design. */
  readonly d: number
  /** $\Xmat^\top\Xmat$, $d \times d$ row-major. */
  readonly H: Float64Array
  /** $\Xmat^\top\yvec$, $d$ values. */
  readonly q: Float64Array
  /** $\yvec^\top\yvec$. */
  readonly yy: number
}

/**
 * The Gram form of a centred problem.
 *
 * @param c The design and target, as `centre` returns them.
 * @returns $\Xmat^\top\Xmat$, $\Xmat^\top\yvec$ and $\yvec^\top\yvec$.
 */
export function gram(c: Centred): Gram {
  const { n, d, x, y } = c
  const H = new Float64Array(d * d)
  const q = new Float64Array(d)
  let yy = 0
  for (let i = 0; i < n; i++) {
    yy += y[i] * y[i]
    for (let j = 0; j < d; j++) {
      const v = x[i * d + j]
      q[j] += v * y[i]
      for (let k = 0; k <= j; k++) H[j * d + k] += v * x[i * d + k]
    }
  }
  for (let j = 0; j < d; j++) for (let k = 0; k < j; k++) H[k * d + j] = H[j * d + k]
  return { n, d, H, q, yy }
}

/** The outcome of a coordinate-descent run. */
export interface Sweeps {
  /** Sweeps made (each visits every coordinate once). */
  readonly steps: number
  /**
   * The duality gap at the end, of the objective scaled by $n$:
   * $\frac12\norm{\rvec}^2 + n\alpha\rho\norm{\wvec}_1 + \frac{n\alpha(1 - \rho)}{2}\norm{\wvec}^2$.
   */
  readonly gap: number
  /** Whether the gap fell below the tolerance before `maxSteps`. */
  readonly converged: boolean
}

/**
 * The duality gap of the elastic net in Gram form (scikit-learn's `enet_coordinate_descent_gram`): with
 * $\rvec = \yvec - \Xmat\wvec$ and $\avec = \Xmat^\top\rvec - \beta\wvec$, the residual is rescaled into the dual
 * feasible set by $c = \min(1, \lambda / \norm{\avec}_\infty)$.
 *
 * @param g The problem in Gram form.
 * @param w The weights, $d$ values.
 * @param l1 The $\ell_1$ weight $\lambda = n\alpha\rho$.
 * @param l2 The $\ell_2$ weight $\beta = n\alpha(1 - \rho)$.
 * @returns The gap, non-negative up to rounding.
 */
export function dualityGap(g: Gram, w: Float64Array, l1: number, l2: number): number {
  const { d, H, q, yy } = g
  let dual = 0
  let wHw = 0
  let qw = 0
  let ww = 0
  let l1norm = 0
  for (let j = 0; j < d; j++) {
    let hw = 0
    for (let k = 0; k < d; k++) hw += H[j * d + k] * w[k]
    dual = Math.max(dual, Math.abs(q[j] - hw - l2 * w[j]))
    wHw += w[j] * hw
    qw += q[j] * w[j]
    ww += w[j] * w[j]
    l1norm += Math.abs(w[j])
  }
  const rr = Math.max(0, yy - 2 * qw + wHw)
  const c = dual > l1 ? l1 / dual : 1
  const gap = dual > l1 ? 0.5 * rr * (1 + c * c) : rr
  return gap + l1 * l1norm - c * (yy - qw) + 0.5 * l2 * (1 + c * c) * ww
}

/**
 * Cyclic coordinate descent on the elastic net in Gram form, in place: each coordinate is set to
 * $S(q_j - (\Hmat\wvec)_j + H_{jj}w_j, \lambda) / (H_{jj} + \beta)$, $S$ the soft threshold. Stops as scikit-learn
 * does: when a sweep's largest change is below `tolerance` times the largest weight (or every weight is zero), the
 * duality gap is checked against `tolerance` times $\yvec^\top\yvec$.
 *
 * @param g The problem in Gram form.
 * @param w The starting weights, $d$ values; overwritten with the solution.
 * @param l1 The $\ell_1$ weight $\lambda = n\alpha\rho$.
 * @param l2 The $\ell_2$ weight $\beta = n\alpha(1 - \rho)$.
 * @param tolerance The relative tolerance of both tests.
 * @param maxSteps The most sweeps.
 * @returns The sweeps made, the final gap and whether it converged.
 */
export function coordinateSweeps(
  g: Gram,
  w: Float64Array,
  l1: number,
  l2: number,
  tolerance: number,
  maxSteps: number,
): Sweeps {
  const { d, H, q, yy } = g
  const Hw = new Float64Array(d)
  for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) Hw[j] += H[j * d + k] * w[k]
  const gapTolerance = tolerance * yy
  let gap = Infinity
  for (let step = 1; step <= maxSteps; step++) {
    let wMax = 0
    let dMax = 0
    for (let j = 0; j < d; j++) {
      const hjj = H[j * d + j]
      if (hjj === 0) continue
      const old = w[j]
      const tmp = q[j] - Hw[j] + hjj * old
      const next = Math.abs(tmp) > l1 ? (Math.sign(tmp) * (Math.abs(tmp) - l1)) / (hjj + l2) : 0
      if (next !== old) {
        const delta = next - old
        for (let k = 0; k < d; k++) Hw[k] += H[k * d + j] * delta
        w[j] = next
      }
      dMax = Math.max(dMax, Math.abs(next - old))
      wMax = Math.max(wMax, Math.abs(next))
    }
    if (wMax === 0 || dMax / wMax < tolerance || step === maxSteps) {
      gap = dualityGap(g, w, l1, l2)
      if (gap <= gapTolerance) return { steps: step, gap, converged: true }
      // Refresh Hw from scratch, so that rounding in the running updates does not accumulate over a long run.
      Hw.fill(0)
      for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) Hw[j] += H[j * d + k] * w[k]
    }
  }
  return { steps: maxSteps, gap, converged: false }
}

/** Hyperparameters of `elasticNet`. */
export interface ElasticNetParams {
  /** The penalty strength $\alpha > 0$, as scikit-learn's `alpha`. */
  alpha: number
  /** The mixing $\rho \in [0, 1]$ of the $\ell_1$ and squared $\ell_2$ penalties; default 1, the lasso. */
  l1Ratio?: number
  /** Fit an unpenalised intercept (default true). */
  intercept?: boolean
  /**
   * The relative tolerance of the stopping tests (default $10^{-10}$, tighter than scikit-learn's $10^{-4}$: the
   * sweeps cost $d^2$ each, so a tight fit is cheap).
   */
  tolerance?: number
  /** The most sweeps (default 10000). */
  maxSteps?: number
  /** Starting weights, $d$ values (default zeros): a nearby solution, such as the fit at a neighbouring $\alpha$. */
  warmStart?: Tensor
}

/** A fitted elastic net (or lasso) $y = \xvec^\top\wvec + b$. */
export interface ElasticNetModel extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'elastic-net'
  /** The weights $\wvec$, $d$ values; those the $\ell_1$ penalty removed are exactly 0. */
  readonly weights: Tensor
  /** The intercept $b$ (0 without an intercept). */
  readonly intercept: number
  /** The penalty strength $\alpha$. */
  readonly alpha: number
  /** The mixing $\rho$. */
  readonly l1Ratio: number
  /** The objective at the solution, in the $\frac{1}{2n}$ scaling above. */
  readonly objective: number
  /** The number of weights that are not zero. */
  readonly nonzero: number
  /** The sweeps of coordinate descent made. */
  readonly steps: number
  /** The duality gap at the end, in the scaling of `objective` (so an upper bound on its distance to the optimum). */
  readonly gap: number
  /** Whether the stopping tests passed before `maxSteps`. */
  readonly converged: boolean
}

/**
 * Check the penalty parameters shared by the elastic-net functions.
 *
 * @param alpha The penalty strength; must be positive.
 * @param l1Ratio The mixing; must lie in $[0, 1]$.
 * @param where The caller's name, for error messages.
 */
function checkPenalty(alpha: number, l1Ratio: number, where: string): void {
  if (!(alpha > 0) || !Number.isFinite(alpha))
    throw new DomainError(where, `${where}: alpha must be positive and finite (use linearRegression for alpha = 0)`)
  if (!(l1Ratio >= 0 && l1Ratio <= 1)) throw new DomainError(where, `${where}: l1Ratio must lie in [0, 1]`)
}

/**
 * The elastic net by cyclic coordinate descent, as scikit-learn's `ElasticNet(alpha, l1_ratio)` and, with
 * $\rho = 1$, `Lasso(alpha)`: $\wvec$ and $b$ minimise the objective above,
 * $\frac{1}{2n}\norm{\yvec - \Xmat\wvec - b\ones}^2 + \alpha\rho\norm{\wvec}_1
 * + \frac12\alpha(1 - \rho)\norm{\wvec}^2$.
 * The $\ell_1$ part sets weights exactly to zero; the squared $\ell_2$ part shares weight among correlated features
 * that the lasso would choose between (Zou and Hastie, 2005). The features are not rescaled: standardise them first
 * (`standardScaler`) for the penalty to treat them alike. Throws `DomainError` at once for $\alpha \le 0$ or $\rho$
 * outside $[0, 1]$, and `ShapeError` from `fit` for mismatched shapes or a `warmStart` of the wrong length. A run that
 * reaches `maxSteps` returns its last weights with `converged` false.
 *
 * Capabilities: `forward` and `decide` (the prediction $\xvec^\top\wvec + b$, $m$ values).
 *
 * @param params The penalty `alpha` and mixing `l1Ratio`, the intercept, the stopping rule and a warm start, as
 *   `ElasticNetParams`.
 * @returns The estimator: `fit({ x, y })`, with `x` $n \times d$ and `y` $n$ values, returns an `ElasticNetModel`.
 *
 * @example The lasso keeps two of four features
 * // y depends on the first two columns only.
 * const x = normals(stream(0), [100, 4])
 * const cols = toArray(x)
 * const y = tensor(cols.map((r, i) => 3 * r[0] - 2 * r[1] + 0.1 * Math.sin(i)))
 * const lasso = elasticNet({ alpha: 0.1 }).fit({ x, y })
 * print('weights =', lasso.weights, ' nonzero =', lasso.nonzero)
 * print('sweeps =', lasso.steps, ' converged =', lasso.converged)
 *
 * @example From the lasso to ridge-like shrinkage as the mixing falls
 * const x = normals(stream(0), [100, 4])
 * const y = tensor(toArray(x).map((r) => 3 * r[0] - 2 * r[1]))
 * for (const l1Ratio of [1, 0.5, 0.1]) {
 *   const m = elasticNet({ alpha: 0.5, l1Ratio }).fit({ x, y })
 *   print('l1Ratio =', l1Ratio, ' weights =', m.weights, ' objective =', m.objective)
 * }
 */
export function elasticNet(params: ElasticNetParams): Estimator<Supervised<Tensor, Tensor>, ElasticNetModel> {
  const { alpha, l1Ratio = 1, intercept = true, tolerance = 1e-10, maxSteps = 10000, warmStart } = params
  checkPenalty(alpha, l1Ratio, 'elasticNet')
  return {
    name: 'elastic-net',
    params: { alpha, l1Ratio, intercept, tolerance, maxSteps },
    fit({ x, y }) {
      const c = centre(x, y, intercept, 'elasticNet')
      const { n, d } = c
      const w = new Float64Array(d)
      if (warmStart) {
        const start = dense.data(warmStart)
        if (start.length !== d)
          throw new ShapeError('elasticNet', `elasticNet: warmStart has ${start.length} values for ${d} features`)
        w.set(start)
      }
      const run = coordinateSweeps(gram(c), w, n * alpha * l1Ratio, n * alpha * (1 - l1Ratio), tolerance, maxSteps)
      let b = c.yMean
      for (let j = 0; j < d; j++) b -= c.xMean[j] * w[j]
      let rss = 0
      for (let i = 0; i < n; i++) {
        let r = c.y[i]
        for (let j = 0; j < d; j++) r -= c.x[i * d + j] * w[j]
        rss += r * r
      }
      let l1norm = 0
      let ww = 0
      let nonzero = 0
      for (const v of w) {
        l1norm += Math.abs(v)
        ww += v * v
        if (v !== 0) nonzero++
      }
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'elasticNet.forward')
        if (cols !== d) throw new ShapeError('elasticNet', `elasticNet: fitted on ${d} features, given ${cols}`)
        const Z = dense.data(input)
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let f = b
          for (let j = 0; j < d; j++) f += Z[i * d + j] * w[j]
          out[i] = f
        }
        return fromData(out, [m])
      }
      return {
        kind: 'model',
        name: 'elastic-net',
        weights: fromData(w, [d]),
        intercept: b,
        alpha,
        l1Ratio,
        objective: rss / (2 * n) + alpha * l1Ratio * l1norm + 0.5 * alpha * (1 - l1Ratio) * ww,
        nonzero,
        steps: run.steps,
        gap: run.gap / n,
        converged: run.converged,
        forward,
        decide: forward,
      }
    },
  }
}

/** Options of `elasticNetPath`. */
export interface ElasticNetPathOptions {
  /** The mixing $\rho \in [0, 1]$ (default 1, the lasso path). */
  l1Ratio?: number
  /** The penalties, any order (they are fitted from largest to smallest); default the log-spaced grid below. */
  alphas?: readonly number[]
  /** The number of penalties of the default grid (default 81). */
  count?: number
  /** The decades the default grid spans below $\alpha_{\max}$ (default 4). */
  span?: number
  /** Fit an unpenalised intercept, centring the data (default true). */
  intercept?: boolean
  /** The relative tolerance of each fit's stopping tests (default $10^{-10}$). */
  tolerance?: number
  /** The most sweeps of each fit (default 10000). */
  maxSteps?: number
}

/** A regularisation path: one fit per penalty, the largest penalty first. */
export interface ElasticNetPath {
  /** The penalties $\alpha$, decreasing. */
  readonly alphas: number[]
  /** The weights, $d \times k$: column $t$ is the fit at `alphas[t]`. */
  readonly coef: Tensor
  /** The intercept of each fit (0 without an intercept). */
  readonly intercept: number[]
  /** The number of nonzero weights of each fit. */
  readonly nonzero: number[]
  /** The sweeps each fit took from the previous one's weights. */
  readonly steps: number[]
  /** Whether every fit met its stopping tests. */
  readonly converged: boolean
}

/**
 * The elastic-net (or lasso) regularisation path, as scikit-learn's `enet_path` and `lasso_path`: the fits of
 * `elasticNet` at a decreasing sequence of penalties, each started from the previous one's weights. The default grid
 * runs from $\alpha_{\max} = \max_j \lvert \xvec_j^\top\yvec \rvert / (n\rho)$ (on the centred data), the smallest
 * penalty at which every weight is zero, down `span` decades in `count` log-spaced steps. With $\rho = 0$ there is
 * no such $\alpha_{\max}$, so `alphas` must be given. Throws `DomainError` for $\rho$ outside $[0, 1]$, a penalty
 * that is not positive, or $\rho = 0$ without `alphas`; `ShapeError` for mismatched shapes.
 *
 * @param x The design, $n \times d$; standardise it first for the penalty to treat the features alike.
 * @param y The target, $n$ values.
 * @param options The mixing, the grid (or explicit penalties), the intercept and the stopping rule, as
 *   `ElasticNetPathOptions`.
 * @returns The path: the penalties, the weights ($d \times k$), intercepts, nonzero counts and sweeps per fit.
 *
 * @example Features join the lasso path one by one as the penalty falls
 * const x = normals(stream(0), [100, 3])
 * const y = tensor(toArray(x).map((r) => 3 * r[0] - 1 * r[1] + 0.3 * r[2]))
 * const path = elasticNetPath(x, y, { count: 9, span: 2 })
 * print('alphas:', path.alphas.map((a) => a.toFixed(3)))
 * print('nonzero:', path.nonzero)
 * print('last weights:', toArray(path.coef).map((row) => row.at(-1)))
 */
export function elasticNetPath(x: Tensor, y: Tensor, options: ElasticNetPathOptions = {}): ElasticNetPath {
  const { l1Ratio = 1, count = 81, span = 4, intercept = true, tolerance = 1e-10, maxSteps = 10000 } = options
  if (!(l1Ratio >= 0 && l1Ratio <= 1))
    throw new DomainError('elasticNetPath', 'elasticNetPath: l1Ratio must lie in [0, 1]')
  const c = centre(x, y, intercept, 'elasticNetPath')
  const g = gram(c)
  const { n, d } = c
  let alphas: number[]
  if (options.alphas) alphas = [...options.alphas].sort((a, b) => b - a)
  else {
    if (l1Ratio === 0)
      throw new DomainError('elasticNetPath', 'elasticNetPath: give alphas when l1Ratio is 0 (there is no alpha max)')
    if (!(count >= 1 && Number.isInteger(count)))
      throw new DomainError('elasticNetPath', 'elasticNetPath: count must be a positive integer')
    let top = 0
    for (let j = 0; j < d; j++) top = Math.max(top, Math.abs(g.q[j]))
    // scikit-learn's _alpha_grid: a target with no correlation gives a grid at the float resolution.
    const max = Math.max(top / (n * l1Ratio), 1e-15)
    alphas = Array.from({ length: count }, (_, t) => max * 10 ** (count === 1 ? 0 : (-span * t) / (count - 1)))
  }
  for (const a of alphas) checkPenalty(a, l1Ratio, 'elasticNetPath')
  const k = alphas.length
  const coef = new Float64Array(d * k)
  const w = new Float64Array(d)
  const intercepts: number[] = []
  const nonzero: number[] = []
  const steps: number[] = []
  let converged = true
  alphas.forEach((alpha, t) => {
    const run = coordinateSweeps(g, w, n * alpha * l1Ratio, n * alpha * (1 - l1Ratio), tolerance, maxSteps)
    converged &&= run.converged
    steps.push(run.steps)
    let b = c.yMean
    let nz = 0
    for (let j = 0; j < d; j++) {
      coef[j * k + t] = w[j]
      b -= c.xMean[j] * w[j]
      if (w[j] !== 0) nz++
    }
    intercepts.push(b)
    nonzero.push(nz)
  })
  return { alphas, coef: fromData(coef, [d, k]), intercept: intercepts, nonzero, steps, converged }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'elasticNet',
    module: 'learning/linear',
    name: 'Elastic net',
    summary: 'Least squares with an L1 and a squared L2 penalty (the lasso when l1Ratio is 1), by coordinate descent.',
    task: 'regression',
    capabilities: ['forward', 'decide'],
    hyper: space({
      alpha: real(1e-4, 100, { default: 1, scale: 'log', label: '\\alpha' }),
      l1Ratio: real(0, 1, { default: 1, label: '\\rho' }),
      intercept: bool({ default: true }),
    }),
    cite: ['zou2005', 'tibshirani1996', 'friedman2010'],
  },
  elasticNet,
)
