/**
 * The optimism of the training error by simulation (Efron, 1986; Hastie, Tibshirani and Friedman, 2009, §7.4): fit a
 * model to $n$ noisy targets many times and compare its training error $\overline{\mathrm{err}}$ with its in-sample
 * error $\mathrm{Err}_\text{in}$, the error on fresh targets at the same inputs. For squared error their expected
 * difference, the optimism, is the covariance penalty
 *
 * $$\omega = \frac{2}{n}\sum_{i=1}^n \cov(\hat y_i, y_i) = \frac{2\sigma^2}{n}\,\mathrm{df},$$
 *
 * which defines the effective degrees of freedom $\mathrm{df}$ of any fitting procedure (Efron, 2004; Ye, 1998). For a
 * linear smoother $\hat\yvec = \Smat\yvec$ it is $\trace\Smat$, so $p$ for least squares with $p$ coefficients.
 *
 * Models: polynomials of a degree (least squares or ridge) and $k$-nearest-neighbour regression, which are linear
 * smoothers, and the best of $M$ pure-noise features, an adaptive procedure whose $\mathrm{df}$ exceeds its one fitted
 * coefficient. The problems are the shared layer of `aifn-methods/theory` (`targetFunction`, `unitGrid`), plus the
 * target `zero` ($f = 0$, pure noise).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import { erf } from 'aifn-compute/numerics/special'
import { targetFunction, unitGrid, type TargetName } from '../regression'

/** A regression function of `aifn-methods/theory` (see `targetFunction`), or `zero`: $f = 0$, so $y$ is pure noise. */
export type OptimismTarget = TargetName | 'zero'

/**
 * A fitting procedure for the optimism study:
 *
 * - `polynomial`: least squares on $1, x, \dots, x^d$ (minimum-norm beyond $n$ coefficients), ridge-penalised on
 *   every coefficient when `ridge` ($\lambda$) is above its default 0; a linear smoother with
 *   $\trace\Smat = \sum_i s_i^2 / (s_i^2 + \lambda)$ over the design's singular values $s_i$.
 * - `knn`: the mean target of the $k$ nearest training inputs (ties to the earlier point); $\trace\Smat = n/k$.
 * - `best-of-noise-features`: $M$ = `candidates` features of pure noise, orthogonal with $\lVert \zvec_j \rVert^2 = n$;
 *   the procedure keeps the one most correlated with $\yvec$ and fits its coefficient by least squares (no
 *   intercept). Needs $M \le n$.
 */
export type OptimismModel =
  | { kind: 'polynomial'; degree: number; ridge?: number }
  | { kind: 'knn'; k: number }
  | { kind: 'best-of-noise-features'; candidates: number }

/** Options of `optimismStudy`. */
export interface OptimismOptions {
  /** The regression function (default `sine`). */
  target?: OptimismTarget
  /** The noise standard deviation $\sigma$ (default 0.3). */
  noise?: number
  /** Training points $n$ (default 50). */
  n?: number
  /**
   * `fixed` (default): the inputs (and noise features) are drawn once and every repeat draws fresh targets at them, the
   * setting of the covariance formula. `random`: the inputs are drawn afresh in every repeat.
   */
  design?: 'fixed' | 'random'
  /** The fitting procedure. */
  model: OptimismModel
  /** Training sets drawn (default 500). */
  repeats?: number
  /** Points of the grid on $[-1, 1]$ that averages the extra-sample error over $x$ (default 201). */
  grid?: number
}

/** The Monte Carlo averages of `optimismStudy`. */
export interface OptimismResult {
  /** The mean training error $\overline{\mathrm{err}} = \frac1n\lVert \yvec - \hat\yvec \rVert^2$. */
  trainError: number
  /** The mean in-sample error $\sigma^2 + \frac1n\lVert \fvec - \hat\yvec \rVert^2$ (fresh targets, same inputs). */
  inSampleError: number
  /** The mean extra-sample error $\sigma^2 + \expect_x(f(x) - \hat f(x))^2$, a fresh input as well as a target. */
  extraSampleError: number
  /** `inSampleError` minus `trainError`: the optimism measured directly. */
  optimism: number
  /** The covariance penalty $\frac2n\sum_i \widehat{\cov}(\hat y_i, y_i)$: the optimism through the covariances. */
  covariancePenalty: number
  /** The effective degrees of freedom $\sum_i \widehat{\cov}(\hat y_i, y_i) / \sigma^2$. */
  df: number
  /** The Monte Carlo standard error of `optimism`. */
  optimismSe: number
  /** The Monte Carlo standard error of `df`. */
  dfSe: number
  /**
   * The degrees of freedom theory gives: $\trace\Smat$ for a linear smoother (averaged over the designs of a `random`
   * design); for `best-of-noise-features` with the target `zero`, $\expect[\max_j \chi^2_{1,j}]$ over the $M$
   * candidates; NaN for `best-of-noise-features` with a nonzero target.
   */
  referenceDf: number
}

/**
 * The regression function of a target, with `zero` the function 0.
 *
 * @param target The target.
 * @returns $f$.
 */
function regressionFunction(target: OptimismTarget): (x: number) => number {
  return target === 'zero' ? () => 0 : targetFunction(target)
}

/**
 * Polynomial features $1, x, \dots, x^d$, row-major $[m, d + 1]$.
 *
 * @param x The points.
 * @param degree The degree $d$.
 * @returns The design.
 */
function powers(x: ArrayLike<number>, degree: number): Float64Array {
  const p = degree + 1
  const out = new Float64Array(x.length * p)
  for (let i = 0; i < x.length; i++) {
    let v = 1
    for (let j = 0; j < p; j++) {
      out[i * p + j] = v
      v *= x[i]
    }
  }
  return out
}

/** A linear smoother on one design: the hat matrix at the training inputs, its rows at the grid, and its trace. */
interface Smoother {
  /** $\Smat$, row-major $[n, n]$. */
  S: Float64Array
  /** The smoother's rows at the grid points, row-major $[m, n]$. */
  G: Float64Array
  /** $\trace\Smat$. */
  trace: number
}

/**
 * The smoother of a polynomial fit, from one SVD $\Phimat = \Umat\,\mathrm{diag}(s)\Vmat^\top$ of the design: the
 * weights are $\Wmat\yvec$ with $\Wmat = \Vmat\,\mathrm{diag}(s_i/(s_i^2 + \lambda))\Umat^\top$, or the
 * pseudo-inverse (singular values below $\max(n, p)\,\epsilon\,s_1$ dropped) when $\lambda = 0$, as `lstsq`.
 *
 * @param x The training inputs, $n$ values.
 * @param at The grid points, $m$ values.
 * @param degree The degree $d$.
 * @param ridge The penalty $\lambda \ge 0$.
 * @returns The smoother.
 */
function polynomialSmoother(x: Float64Array, at: Float64Array, degree: number, ridge: number): Smoother {
  const n = x.length
  const m = at.length
  const p = degree + 1
  const { U, S, V } = svd(fromData(powers(x, degree), [n, p]))
  const u = toFlat(U)
  const s = toFlat(S)
  const v = toFlat(V)
  const r = s.length
  const tol = Math.max(n, p) * Number.EPSILON * (s[0] ?? 0)
  const scale = s.map((d) => (ridge > 0 ? d / (d * d + ridge) : d > tol ? 1 / d : 0))
  // W = V diag(scale) Uᵀ, row-major [p, n].
  const W = new Float64Array(p * n)
  for (let a = 0; a < p; a++)
    for (let j = 0; j < n; j++) {
      let w = 0
      for (let k = 0; k < r; k++) w += v[a * r + k] * scale[k] * u[j * r + k]
      W[a * n + j] = w
    }
  const apply = (phi: Float64Array, rows: number) => {
    const out = new Float64Array(rows * n)
    for (let i = 0; i < rows; i++)
      for (let a = 0; a < p; a++) {
        const f = phi[i * p + a]
        if (f === 0) continue
        for (let j = 0; j < n; j++) out[i * n + j] += f * W[a * n + j]
      }
    return out
  }
  const Smat = apply(powers(x, degree), n)
  let trace = 0
  for (let i = 0; i < n; i++) trace += Smat[i * n + i]
  return { S: Smat, G: apply(powers(at, degree), m), trace }
}

/**
 * The smoother of $k$-nearest-neighbour regression: row $i$ puts $1/k$ on the $k$ training inputs nearest to point
 * $i$ (ties to the earlier point; $k$ capped at $n$).
 *
 * @param x The training inputs, $n$ values.
 * @param at The grid points, $m$ values.
 * @param k The number of neighbours.
 * @returns The smoother.
 */
function knnSmoother(x: Float64Array, at: Float64Array, k: number): Smoother {
  const n = x.length
  const kk = Math.min(k, n)
  const sorted = Array.from(x.keys()).sort((a, b) => x[a] - x[b] || a - b)
  const rows = (points: ArrayLike<number>) => {
    const out = new Float64Array(points.length * n)
    for (let i = 0; i < points.length; i++) {
      const q = points[i]
      // Binary search for the first sorted input at or above q, then take the nearer side k times.
      let lo = 0
      let hi = n
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (x[sorted[mid]] < q) lo = mid + 1
        else hi = mid
      }
      let left = lo - 1
      let right = lo
      for (let j = 0; j < kk; j++) {
        const l = left >= 0 ? sorted[left] : -1
        const r = right < n ? sorted[right] : -1
        const dl = l >= 0 ? q - x[l] : Infinity
        const dr = r >= 0 ? x[r] - q : Infinity
        const takeLeft = dl < dr || (dl === dr && l >= 0 && l < r)
        out[i * n + (takeLeft ? l : r)] = 1 / kk
        if (takeLeft) left--
        else right++
      }
    }
    return out
  }
  const S = rows(x)
  let trace = 0
  for (let i = 0; i < n; i++) trace += S[i * n + i]
  return { S, G: rows(at), trace }
}

/**
 * $M$ orthogonal noise features of squared norm $n$: standard normal columns orthogonalised by modified Gram–Schmidt
 * and rescaled.
 *
 * @param s The stream the normals come from.
 * @param n The rows.
 * @param M The columns; at most $n$.
 * @returns The features, row-major $[n, M]$.
 */
function noiseFeatures(s: Stream, n: number, M: number): Float64Array {
  const z = standardNormals(s, n * M)
  for (let j = 0; j < M; j++) {
    for (let l = 0; l < j; l++) {
      let d = 0
      for (let i = 0; i < n; i++) d += z[i * M + j] * z[i * M + l]
      for (let i = 0; i < n; i++) z[i * M + j] -= (d / n) * z[i * M + l]
    }
    let norm = 0
    for (let i = 0; i < n; i++) norm += z[i * M + j] ** 2
    const c = Math.sqrt(n / norm)
    for (let i = 0; i < n; i++) z[i * M + j] *= c
  }
  return z
}

/**
 * The best-of-$M$ fit: the feature $\zvec_j$ with the largest $(\zvec_j^\top\yvec)^2$, and its least-squares
 * coefficient $b = \zvec_j^\top\yvec / n$.
 *
 * @param Z The features, row-major $[n, M]$, each of squared norm $n$.
 * @param n The rows.
 * @param M The columns.
 * @param y The targets.
 * @returns The fitted values $b\zvec_j$ and the coefficient $b$.
 */
function bestOfFeatures(
  Z: Float64Array,
  n: number,
  M: number,
  y: ArrayLike<number>,
): { fitted: Float64Array; b: number } {
  let best = 0
  let bestZ = -1
  for (let j = 0; j < M; j++) {
    let d = 0
    for (let i = 0; i < n; i++) d += Z[i * M + j] * y[i]
    if (d * d > bestZ) [best, bestZ] = [j, d * d]
  }
  let d = 0
  for (let i = 0; i < n; i++) d += Z[i * M + best] * y[i]
  const b = d / n
  return { fitted: Float64Array.from({ length: n }, (_, i) => b * Z[i * M + best]), b }
}

/**
 * $\expect[\max_{j \le M} \chi^2_{1,j}]$ for $M$ independent chi-squared variables of one degree of freedom:
 * $\int_0^\infty 1 - \mathrm{erf}(\sqrt{t/2})^M\,dt$, by Simpson's rule after $t = u^2$.
 *
 * @param M The number of variables.
 * @returns The expected maximum (1 for $M = 1$).
 */
function expectedMaxChiSquare(M: number): number {
  const steps = 4000
  const hi = 14
  const h = hi / steps
  const g = (u: number) => 2 * u * (1 - erf(u / Math.SQRT2) ** M)
  let total = g(0) + g(hi)
  for (let i = 1; i < steps; i++) total += (i % 2 ? 4 : 2) * g(i * h)
  return (total * h) / 3
}

/**
 * Inputs on $[-1, 1]$ (and the noise features of `best-of-noise-features`) for one design.
 *
 * @param s The design's stream: inputs from `child(s, 'x')`, features from `child(s, 'features')`.
 * @param n The number of inputs.
 * @param model The model, for its features.
 * @returns The inputs and, for the best-of model, the features.
 */
function drawDesign(s: Stream, n: number, model: OptimismModel): { x: Float64Array; Z?: Float64Array } {
  const x = Float64Array.from(units(child(s, 'x'), n), (u) => 2 * u - 1)
  if (model.kind !== 'best-of-noise-features') return { x }
  return { x, Z: noiseFeatures(child(s, 'features'), n, model.candidates) }
}

/**
 * The optimism of one fitting procedure by simulation (see the file's introduction). Each repeat draws targets
 * $y_i = f(x_i) + \sigma\varepsilon_i$, fits, and records the training, in-sample and extra-sample errors. The
 * covariances are estimated by $\sum_i (\hat y_i - \hat y_i^0)(y_i - f_i)$, averaged over repeats, where $\hat\yvec^0$
 * is the same procedure fitted to the noiseless targets $\fvec$: $\hat\yvec^0$ does not depend on the noise, so the
 * estimate is unbiased for $\sum_i \cov(\hat y_i, y_i)$ under either design, and the subtraction removes the part of
 * $\hat\yvec$ that does not move with the noise. Throws `DomainError` for `best-of-noise-features` with more
 * candidates than points, or for fewer than two repeats.
 *
 * @param s The stream; the fixed design comes from `child(s, 'design')` (repeat $r$'s from `child(s, 'design', r)`
 *   for a random design), and repeat $r$'s noise from `child(s, 'noise', r)`. `s` is not advanced, so the same `s`
 *   gives every model the same designs and noise.
 * @param options The problem, the design, the model and the number of repeats (see `OptimismOptions`).
 * @returns The mean errors, the optimism two ways, the degrees of freedom, their standard errors and the degrees of
 *   freedom from theory (see `OptimismResult`).
 *
 * @example Least squares with $p = 4$ coefficients: optimism near $2p\sigma^2/n$ and df near 4
 * const r = optimismStudy(stream(1), { model: { kind: 'polynomial', degree: 3 }, n: 40, noise: 1, repeats: 400 })
 * print('optimism =', r.optimism, ' covariance penalty =', r.covariancePenalty, ' 2p/n =', 8 / 40)
 * print('df =', r.df, '+/-', r.dfSe, ' trace S =', r.referenceDf)
 *
 * @example Choosing the best of 20 noise features costs far more than one degree of freedom
 * const model = { kind: 'best-of-noise-features', candidates: 20 }
 * const r = optimismStudy(stream(1), { target: 'zero', noise: 1, n: 40, model, repeats: 200 })
 * print('df =', r.df, ' E max chi2 =', r.referenceDf)
 */
export function optimismStudy(s: Stream, options: OptimismOptions): OptimismResult {
  const { target = 'sine', noise = 0.3, n = 50, design = 'fixed', model, repeats = 500, grid = 201 } = options
  if (!(repeats >= 2)) throw new DomainError('optimismStudy', 'optimismStudy: needs at least two repeats')
  if (model.kind === 'best-of-noise-features' && !(model.candidates >= 1 && model.candidates <= n))
    throw new DomainError('optimismStudy', 'optimismStudy: needs between 1 and n noise features')
  const f = regressionFunction(target)
  const at = unitGrid(grid)
  const fAt = Float64Array.from(at, f)
  const meanF2 = fAt.reduce((a, v) => a + v * v, 0) / grid
  const sigma2 = noise * noise

  // One design: its inputs, true values and, for a linear smoother, its smoother.
  const setUp = (ds: Stream) => {
    const { x, Z } = drawDesign(ds, n, model)
    const fx = Float64Array.from(x, f)
    const smoother =
      model.kind === 'polynomial'
        ? polynomialSmoother(x, at, model.degree, model.ridge ?? 0)
        : model.kind === 'knn'
          ? knnSmoother(x, at, model.k)
          : undefined
    return { x, Z, fx, smoother }
  }
  const fixed = design === 'fixed' ? setUp(child(s, 'design')) : undefined

  const train = new Float64Array(repeats)
  const inSample = new Float64Array(repeats)
  const extra = new Float64Array(repeats)
  const cov = new Float64Array(repeats)
  let traceSum = 0
  for (let r = 0; r < repeats; r++) {
    const d = fixed ?? setUp(child(s, 'design', r))
    const e = standardNormals(child(s, 'noise', r), n).map((v) => noise * v)
    const y = Float64Array.from(d.fx, (v, i) => v + e[i])
    let yHat: Float64Array
    let yHat0: Float64Array
    let extraBias: number
    if (d.smoother) {
      const { S, G } = d.smoother
      yHat = new Float64Array(n)
      yHat0 = new Float64Array(n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          yHat[i] += S[i * n + j] * y[j]
          yHat0[i] += S[i * n + j] * d.fx[j]
        }
      extraBias = 0
      for (let g = 0; g < grid; g++) {
        let v = 0
        for (let j = 0; j < n; j++) v += G[g * n + j] * y[j]
        extraBias += (fAt[g] - v) ** 2 / grid
      }
      traceSum += d.smoother.trace
    } else {
      const M = (model as { candidates: number }).candidates
      const fit = bestOfFeatures(d.Z!, n, M, y)
      yHat = fit.fitted
      yHat0 = bestOfFeatures(d.Z!, n, M, d.fx).fitted
      // A fresh feature value is independent of x with mean 0 and variance 1.
      extraBias = meanF2 + fit.b * fit.b
    }
    let tr = 0
    let ins = 0
    let c = 0
    for (let i = 0; i < n; i++) {
      tr += (y[i] - yHat[i]) ** 2 / n
      ins += (d.fx[i] - yHat[i]) ** 2 / n
      c += (yHat[i] - yHat0[i]) * e[i]
    }
    train[r] = tr
    inSample[r] = sigma2 + ins
    extra[r] = sigma2 + extraBias
    cov[r] = c
  }
  const mean = (a: Float64Array) => a.reduce((p, q) => p + q, 0) / a.length
  const se = (a: Float64Array) => {
    const m = mean(a)
    return Math.sqrt(a.reduce((p, q) => p + (q - m) ** 2, 0) / (a.length - 1) / a.length)
  }
  const gap = Float64Array.from(inSample, (v, i) => v - train[i])
  const sumCov = mean(cov)
  const referenceDf =
    model.kind === 'best-of-noise-features'
      ? target === 'zero'
        ? expectedMaxChiSquare(model.candidates)
        : NaN
      : traceSum / repeats
  return {
    trainError: mean(train),
    inSampleError: mean(inSample),
    extraSampleError: mean(extra),
    optimism: mean(gap),
    covariancePenalty: (2 * sumCov) / n,
    df: sumCov / sigma2,
    optimismSe: se(gap),
    dfSe: se(cov) / sigma2,
    referenceDf,
  }
}

/** The averages of `optimismStudy` for each value of a sweep, one entry per value. */
export interface OptimismSweep {
  /** The swept values: degrees, $k$, penalties $\lambda$ or candidate counts $M$. */
  values: number[]
  /** Mean training error per value. */
  trainError: Float64Array
  /** Mean in-sample error per value. */
  inSampleError: Float64Array
  /** Mean extra-sample error per value. */
  extraSampleError: Float64Array
  /** Optimism measured directly per value. */
  optimism: Float64Array
  /** Covariance penalty per value. */
  covariancePenalty: Float64Array
  /** Effective degrees of freedom per value. */
  df: Float64Array
  /** Monte Carlo standard error of the optimism per value. */
  optimismSe: Float64Array
  /** Monte Carlo standard error of the degrees of freedom per value. */
  dfSe: Float64Array
  /** The degrees of freedom from theory per value (see `OptimismResult`). */
  referenceDf: Float64Array
}

/**
 * `optimismStudy` across a family of models, one study per value: polynomial degrees, $k$ for nearest neighbours,
 * ridge penalties $\lambda$ on a polynomial of fixed degree, or candidate counts $M$ for the best of $M$ noise
 * features. Every study draws from `child(s, 'sweep')`, so all values see the same designs and noise.
 *
 * @param s The stream; not advanced.
 * @param options The problem and study size (as `OptimismOptions`, without `model`), and the family.
 * @param options.family `polynomial` sweeps the degree, `knn` sweeps $k$, `ridge` sweeps $\lambda$ at `degree`, and
 *   `best-of-noise-features` sweeps $M$.
 * @param options.values The degrees, $k$ values, penalties or candidate counts, one study each.
 * @param options.degree The polynomial degree of the `ridge` family (default 9).
 * @param options.ridge The ridge penalty of every fit of the `polynomial` family (default 0).
 * @returns The `values`, and per value every average of `OptimismResult`.
 *
 * @example Optimism grows with the degree as $2(d + 1)\sigma^2/n$
 * const r = optimismSweep(stream(0), { family: 'polynomial', values: [1, 3, 5], noise: 1, n: 40, repeats: 100 })
 * print('degree       ', r.values)
 * print('df           ', r.df)
 * print('trace S      ', r.referenceDf)
 * print('optimism     ', r.optimism)
 */
export function optimismSweep(
  s: Stream,
  options: Omit<OptimismOptions, 'model'> & {
    family: 'polynomial' | 'knn' | 'ridge' | 'best-of-noise-features'
    values: readonly number[]
    degree?: number
    ridge?: number
  },
): OptimismSweep {
  const { family, values, degree = 9, ridge = 0, ...rest } = options
  const modelOf = (v: number): OptimismModel =>
    family === 'polynomial'
      ? { kind: 'polynomial', degree: v, ridge }
      : family === 'ridge'
        ? { kind: 'polynomial', degree, ridge: v }
        : family === 'knn'
          ? { kind: 'knn', k: v }
          : { kind: 'best-of-noise-features', candidates: v }
  const out = values.map((v) => optimismStudy(child(s, 'sweep'), { ...rest, model: modelOf(v) }))
  const column = (key: keyof OptimismResult) => Float64Array.from(out, (r) => r[key])
  return {
    values: [...values],
    trainError: column('trainError'),
    inSampleError: column('inSampleError'),
    extraSampleError: column('extraSampleError'),
    optimism: column('optimism'),
    covariancePenalty: column('covariancePenalty'),
    df: column('df'),
    optimismSe: column('optimismSe'),
    dfSe: column('dfSe'),
    referenceDf: column('referenceDf'),
  }
}
