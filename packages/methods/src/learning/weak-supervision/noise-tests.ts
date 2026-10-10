/**
 * Hypothesis tests for class-conditional label noise from anchor points (Poyiadzi, Yang, Twomey and Santos-Rodriguez
 * 2022, ECML-PKDD, arXiv 2103.02630; Yang, Poyiadzi, Twomey and Santos-Rodriguez 2024, AAAI). Labels $y \in \{0, 1\}$
 * (the papers' $\pm 1$) are flipped with $\alpha = p(\tilde{y} = 0 \mid y = 1)$ and
 * $\beta = p(\tilde{y} = 1 \mid y = 0)$; the noisy posterior is
 * $\tilde{\eta}(\xvec) = (1 - \alpha - \beta) \eta(\xvec) + \beta$. An anchor point is an $\xvec$ whose clean posterior
 * is $\eta(\xvec) = \tfrac{1}{2}$, so $\tilde{\eta}(\xvec) = (1 - \alpha + \beta)/2$, which is $\tfrac{1}{2}$ exactly
 * when $\alpha = \beta$. The tests are two-sided $z$-tests of
 *
 * $H_0: \alpha = \beta$ (uniform noise, or none) against $H_1: \alpha \ne \beta$ (class-conditional noise)
 *
 * on $\bar{\eta} = \frac{1}{k} \sum_i \hat{\eta}(\xvec_i)$ over $k$ anchors, with $\hat{\eta}$ a model fitted to the
 * noisy labels:
 *
 * - **parametric** (2022): logistic regression by maximum likelihood on $(1, \xvec)$; by the asymptotic normality of
 *   the MLE and the delta method, $\bar{\eta} \sim \Gauss(\tfrac{1}{2}, v)$ under $H_0$ with
 *   $v = \tfrac{1}{16} \bar{\xvec}^\top \hat{\Hmat} \bar{\xvec}$, $\hat{\Hmat} = (\Xmat^\top \Dmat \Xmat)^{-1}$,
 *   $\Dmat = \diag(\hat{\eta}_i(1 - \hat{\eta}_i))$, $\bar{\xvec}$ the mean of the augmented anchors (Eqs. 4, 6 and
 *   §3.2);
 * - **local** (2024): local likelihood logistic regression at each anchor (`localLogistic`), whose log-odds have the
 *   sandwich variance; $v = \frac{1}{16k^2} \sum_{a,b} \cov(\hat{\beta}_0(\xvec_a), \hat{\beta}_0(\xvec_b))$ (Eqs. 7–9
 *   and the multiple-anchor variance).
 *
 * $z = (\bar{\eta} - \tfrac{1}{2})/\sqrt{v}$, and the p-value is $2(1 - \Phi(\lvert z \rvert))$. The power at level $a$
 * is $1 - b$ with $b = \Phi(u_+) - \Phi(u_-)$, $u_\pm = (\pm z_{1-a/2}\sqrt{v} + h)/\sqrt{\tilde{v}}$,
 * $h = (\beta - \alpha)/2$ (Prop. 3.1, Eq. 11); under $H_1$ the delta-method factor is
 * $[\tilde{\eta}(1 - \tilde{\eta})]^2$ with $\tilde{\eta} = (1 - \alpha + \beta)/2$, so
 * $\tilde{v} = 16[\tilde{\eta}(1 - \tilde{\eta})]^2 v$. (The paper prints the factor as
 * $[(1 - \alpha + \beta)(\beta - \alpha)]^2/16$; the reading here, logged in the progress file, is the delta method's.)
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, stream as makeStream, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { inverse } from 'aifn-compute/numerics/linalg'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import { Normal } from 'aifn-compute/probability/distributions'
import type { TestResult } from 'aifn-compute/probability/tests'
import { localLogistic, localLogisticBandwidth, localLogisticCovariance, weightedLogistic } from './local-likelihood'

/**
 * Which model estimates the posterior at the anchors: logistic regression by maximum likelihood, or local likelihood.
 */
export type NoiseTestModel = 'parametric' | 'local'

/** Options of {@link classConditionalNoiseTest}. */
export interface NoiseTestOptions {
  /** `parametric` (logistic regression MLE, default) or `local` (local likelihood). */
  model?: NoiseTestModel
  /** The local model's kernel bandwidth (default 1); unused by the parametric model. */
  bandwidth?: number
  /** The local model's polynomial order (default 1); unused by the parametric model. */
  degree?: 0 | 1 | 2
}

/** A noise test's result: the z-test, with the posterior estimates at the anchors. */
export type NoiseTestResult = TestResult & {
  /** $\hat{\eta}$ at each anchor. */
  readonly anchorEstimates: readonly number[]
  /** Their mean $\bar{\eta}$ (also the result's `estimate`). */
  readonly meanEstimate: number
  /** $v$, the variance of $\bar{\eta}$ under $H_0$. */
  readonly variance: number
}

/**
 * The logistic-regression MLE on $(1, \xvec)$ and $\hat{\Hmat} = (\Xmat^\top \Dmat \Xmat)^{-1}$ at it, with $\Xmat$
 * the design of an intercept and the features.
 *
 * @param x The points, $n \times d$.
 * @param y The noisy labels, $\tilde{y}_i \in \{0, 1\}$, one per point.
 * @returns `theta`, the $q = d + 1$ coefficients (intercept first), `H`, $\hat{\Hmat}$ ($q \times q$, row-major),
 *   `q` and `d`.
 */
function logisticMle(x: MatrixLike, y: ArrayLike<number>) {
  const m = dense.toMatrixF64(x, 'classConditionalNoiseTest')
  const { m: n, n: d } = m
  const q = d + 1
  const design = new Float64Array(n * q)
  for (let i = 0; i < n; i++) {
    design[i * q] = 1
    for (let j = 0; j < d; j++) design[i * q + 1 + j] = m.data[i * d + j]
  }
  const { beta: theta, p: mu } = weightedLogistic(design, y, new Float64Array(n).fill(1), q)
  const info = new Float64Array(q * q)
  for (let i = 0; i < n; i++) {
    const w = mu[i] * (1 - mu[i])
    for (let j = 0; j < q; j++) for (let k = 0; k < q; k++) info[j * q + k] += w * design[i * q + j] * design[i * q + k]
  }
  return { theta, H: dense.data(inverse(fromData(info, [q, q])) as Tensor), q, d }
}

/**
 * The anchor-point test for class-conditional label noise (see the file's notes): a two-sided $z$-test of
 * $H_0: \alpha = \beta$ on the mean noisy posterior at the anchors. Throws `DomainError` for no anchors and
 * `ShapeError` when the anchors and the data differ in dimension.
 *
 * @param x The points, $n \times d$.
 * @param y The noisy labels, $\tilde{y}_i \in \{0, 1\}$, one per point.
 * @param anchors The anchor points, $k \times d$: points whose clean posterior is (about) $\tfrac{1}{2}$.
 * @param options The model (default `parametric`), and the local model's bandwidth and order.
 * @returns The test result ($z$, p-value, null law $\Gauss(0, 1)$, estimate $\bar{\eta}$ against $\tfrac{1}{2}$), with
 *   the estimate at each anchor and the variance $v$.
 *
 * @example Uniform noise passes, class-conditional noise is caught
 * // The clean posterior is sigmoid(2x), so x = 0 is an anchor.
 * const noisy = (seed, alpha, beta) => {
 *   const s = stream(seed)
 *   const x = Array.from({ length: 400 }, () => [uniform(s, -3, 3)])
 *   const y = x.map(([v]) => {
 *     const clean = uniform(s) < 1 / (1 + Math.exp(-2 * v)) ? 1 : 0
 *     return uniform(s) < (clean === 1 ? alpha : beta) ? 1 - clean : clean
 *   })
 *   return { x, y }
 * }
 * const uniformNoise = noisy(1, 0.2, 0.2)
 * const classNoise = noisy(1, 0.3, 0.05)
 * print('alpha = beta = 0.2, p =', classConditionalNoiseTest(uniformNoise.x, uniformNoise.y, [[0]]).pValue)
 * const t = classConditionalNoiseTest(classNoise.x, classNoise.y, [[0]])
 * print('alpha = 0.3, beta = 0.05, p =', t.pValue, ' mean noisy posterior:', t.meanEstimate)
 * const local = classConditionalNoiseTest(classNoise.x, classNoise.y, [[-0.2], [0], [0.2]], { model: 'local' })
 * print('the same with the local model at three anchors, p =', local.pValue)
 */
export function classConditionalNoiseTest(
  x: MatrixLike,
  y: ArrayLike<number>,
  anchors: MatrixLike,
  options: NoiseTestOptions = {},
): NoiseTestResult {
  const { model = 'parametric', bandwidth = 1, degree = 1 } = options
  const A = dense.toMatrixF64(anchors, 'classConditionalNoiseTest')
  const k = A.m
  if (k < 1) throw new DomainError('classConditionalNoiseTest', 'classConditionalNoiseTest: give at least one anchor')
  let estimates: number[]
  let v: number
  if (model === 'parametric') {
    const { theta, H, q, d } = logisticMle(x, y)
    if (A.n !== d)
      throw new ShapeError(
        'classConditionalNoiseTest',
        'classConditionalNoiseTest: anchors and data differ in dimension',
      )
    const bar = new Float64Array(q)
    estimates = Array.from({ length: k }, (_, a) => {
      let z = theta[0]
      bar[0] += 1 / k
      for (let j = 0; j < d; j++) {
        z += theta[1 + j] * A.data[a * d + j]
        bar[1 + j] += A.data[a * d + j] / k
      }
      return 1 / (1 + Math.exp(-z))
    })
    let quad = 0
    for (let j = 0; j < q; j++) for (let l = 0; l < q; l++) quad += bar[j] * H[j * q + l] * bar[l]
    v = quad / 16
  } else {
    const fits = Array.from({ length: k }, (_, a) =>
      localLogistic(x, y, A.data.subarray(a * A.n, (a + 1) * A.n), { bandwidth, degree }),
    )
    estimates = fits.map((f) => f.estimate)
    let total = 0
    for (let a = 0; a < k; a++)
      for (let b = 0; b < k; b++) total += a === b ? fits[a].logitVariance : localLogisticCovariance(fits[a], fits[b])
    v = total / (16 * k * k)
  }
  const mean = estimates.reduce((s, e) => s + e, 0) / k
  const z = (mean - 0.5) / Math.sqrt(v)
  const p = 2 * (1 - (normalCdf(Math.abs(z)) as number))
  return {
    kind: 'test-result',
    test: 'classConditionalNoiseTest',
    method: `${model === 'parametric' ? 'Parametric (logistic MLE)' : 'Local likelihood'} anchor test for class-conditional noise, k = ${k}`,
    statistic: z,
    symbol: 'z',
    pValue: Math.min(1, p),
    alternative: 'two-sided',
    tail: 'both',
    null: Normal(0, 1),
    estimand: 'mean noisy posterior at the anchors, η̄',
    estimate: mean,
    nullValue: 0.5,
    n: y.length,
    anchorEstimates: estimates,
    meanEstimate: mean,
    variance: v,
  }
}

/**
 * The power $1 - b$ of the two-sided test at level `level` (Prop. 3.1, Eq. 11; Eq. 12 for $k$ anchors chosen at
 * random): $v$ is the variance of $\bar{\eta}$ under $H_0$ (already divided by $k$ for several anchors), and
 * $\tilde{v} = 16[\tilde{\eta}(1 - \tilde{\eta})]^2 v$ with $\tilde{\eta} = (1 - \alpha + \beta)/2$ its variance under
 * $H_1$ (see the file's notes). Throws `DomainError` unless the variance is positive.
 *
 * @param options `variance`, $v$; `alpha` and `beta`, the noise rates $\alpha = p(\tilde{y} = 0 \mid y = 1)$ and
 *   $\beta = p(\tilde{y} = 1 \mid y = 0)$; and `level`, the test's level $a$ (default 0.05).
 * @returns The power, the probability that the test rejects $H_0$ at these noise rates.
 *
 * @example Power grows with the gap between the noise rates
 * for (const alpha of [0.1, 0.15, 0.2, 0.3]) {
 *   print('alpha =', alpha, ' beta = 0.1, power:', noiseTestPower({ variance: 0.001, alpha, beta: 0.1 }))
 * }
 */
export function noiseTestPower(options: { variance: number; alpha: number; beta: number; level?: number }): number {
  const { variance: v, alpha, beta, level = 0.05 } = options
  if (!(v > 0)) throw new DomainError('noiseTestPower', 'noiseTestPower: the variance must be positive')
  const z = normalQuantile(1 - level / 2) as number
  const eta = (1 - alpha + beta) / 2
  const vt = 16 * (eta * (1 - eta)) ** 2 * v
  const h = (beta - alpha) / 2
  const b =
    (normalCdf((z * Math.sqrt(v) + h) / Math.sqrt(vt)) as number) -
    (normalCdf((-z * Math.sqrt(v) + h) / Math.sqrt(vt)) as number)
  return 1 - b
}

// ── Simulation ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** One cell of a simulation: noisy datasets with their anchor sets, tested the same way. */
export interface NoiseTestCell {
  /** The cell's name, as a figure shows it. */
  readonly label: string
  /** The datasets: features `x` ($n \times d$) and noisy labels `y` ($n$ values in $\{0, 1\}$). */
  readonly datasets: readonly { x: Tensor; y: Tensor }[]
  /** One anchor set ($k \times d$) per dataset, in the same order. */
  readonly anchors: readonly MatrixLike[]
}

/** The p-values of every cell so far, and the share rejected at each level. */
export interface NoiseTestSimulation {
  /** The cells' labels. */
  readonly labels: readonly string[]
  /** The p-values so far, one list per cell. */
  readonly pValues: number[][]
  /**
   * The share of each cell's p-values below 0.05 (the size under $H_0$, the power under $H_1$; NaN for a cell not yet
   * started).
   */
  readonly rejected05: number[]
  /** The same at level 0.10. */
  readonly rejected10: number[]
  /** The datasets tested so far. */
  readonly done: number
  /** The datasets in all. */
  readonly total: number
}

/**
 * Run the test on every dataset of every cell, as a generator that yields after each dataset, so a figure can fill its
 * box plots while the worker computes. With the local model and `bandwidths`, the bandwidth is chosen per dataset by
 * leave-one-out over a subsample of 40 points (`localLogisticBandwidth`). The yielded snapshots share their `pValues`
 * lists, which grow as the run goes on.
 *
 * @param cells The cells, each a set of datasets with their anchors.
 * @param options The test's options, with `bandwidths`, the grid the local model's bandwidth is chosen from, and
 *   `seed`, the seed of the streams that draw the leave-one-out subsamples (default 0).
 * @returns A generator of snapshots, one after each dataset; it returns the final one.
 *
 * @example The rejection rate under uniform and under class-conditional noise
 * const noisy = (s, alpha, beta) => {
 *   const x = Array.from({ length: 150 }, () => [uniform(s, -3, 3)])
 *   const y = x.map(([v]) => {
 *     const clean = uniform(s) < 1 / (1 + Math.exp(-2 * v)) ? 1 : 0
 *     return uniform(s) < (clean === 1 ? alpha : beta) ? 1 - clean : clean
 *   })
 *   return { x: tensor(x), y: tensor(y) }
 * }
 * const s = stream(2)
 * const cell = (label, alpha, beta) => ({
 *   label,
 *   datasets: Array.from({ length: 5 }, () => noisy(s, alpha, beta)),
 *   anchors: Array.from({ length: 5 }, () => [[0]]),
 * })
 * const cells = [cell('uniform', 0.2, 0.2), cell('class-conditional', 0.35, 0.05)]
 * let last
 * for (const snapshot of noiseTestSimulation(cells)) last = snapshot
 * print(last.labels, ' rejected at 0.05:', last.rejected05, ' datasets:', last.done, 'of', last.total)
 */
export function* noiseTestSimulation(
  cells: readonly NoiseTestCell[],
  options: NoiseTestOptions & { bandwidths?: readonly number[]; seed?: number | string } = {},
): Generator<NoiseTestSimulation, NoiseTestSimulation> {
  const root: Stream = makeStream(`noiseTestSimulation/${options.seed ?? 0}`)
  const pValues = cells.map(() => [] as number[])
  const total = cells.reduce((s, c) => s + c.datasets.length, 0)
  let done = 0
  const rate = (ps: number[], a: number) => (ps.length ? ps.filter((p) => p < a).length / ps.length : NaN)
  const snapshot = (): NoiseTestSimulation => ({
    labels: cells.map((c) => c.label),
    pValues,
    rejected05: pValues.map((ps) => rate(ps, 0.05)),
    rejected10: pValues.map((ps) => rate(ps, 0.1)),
    done,
    total,
  })
  for (let c = 0; c < cells.length; c++)
    for (let r = 0; r < cells[c].datasets.length; r++) {
      const { x, y } = cells[c].datasets[r]
      const labels = toFlat(y)
      const bandwidth =
        options.model === 'local' && options.bandwidths
          ? localLogisticBandwidth(child(root, c, r), x, labels, options.bandwidths, {
              degree: options.degree,
              subsample: 40,
            }).bandwidth
          : options.bandwidth
      pValues[c].push(classConditionalNoiseTest(x, labels, cells[c].anchors[r], { ...options, bandwidth }).pValue)
      done++
      yield snapshot()
    }
  return snapshot()
}

/**
 * The number of anchors $k$ that reaches power `target` at level `level` for given noise rates and single-anchor
 * variance $v$, taking the variance with $k$ anchors as $v/k$. Only powers of two are tried, so the result is the
 * smallest power of two that is enough, or `Infinity` when none up to `max` is.
 *
 * @param options `variance`, $v$ for one anchor; `alpha` and `beta`, the noise rates; `target`, the power wanted
 *   (default 0.8); `level`, the test's level (default 0.05); and `max`, the most anchors tried (default 4096).
 * @returns The number of anchors, a power of two, or `Infinity`.
 *
 * @example Smaller gaps between the noise rates need more anchors
 * for (const alpha of [0.3, 0.2, 0.15]) {
 *   print('alpha =', alpha, ' beta = 0.1, anchors:', anchorsForPower({ variance: 0.01, alpha, beta: 0.1 }))
 * }
 */
export function anchorsForPower(options: {
  variance: number
  alpha: number
  beta: number
  target?: number
  level?: number
  max?: Size
}): number {
  const { target = 0.8, max = 4096 } = options
  for (let k = 1; k <= max; k *= 2)
    if (noiseTestPower({ ...options, variance: options.variance / k }) >= target) return k
  return Infinity
}
