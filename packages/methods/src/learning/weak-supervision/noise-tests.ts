/**
 * Hypothesis tests for class-conditional label noise from anchor points (Poyiadzi, Yang, Twomey and Santos-Rodriguez
 * 2022, ECML-PKDD, arXiv 2103.02630; Yang, Poyiadzi, Twomey and Santos-Rodriguez 2024, AAAI). Labels y ∈ {0, 1} (the
 * papers' ±1) are flipped with α = P(ỹ = 0 | y = 1) and β = P(ỹ = 1 | y = 0); the noisy posterior is
 * η̃(x) = (1 − α − β) η(x) + β. An anchor point is an x whose clean posterior is η(x) = ½, so η̃(x) = (1 − α + β)/2,
 * which is ½ exactly when α = β. The tests are two-sided z-tests of
 *
 *   H₀: α = β (uniform noise, or none)   against   H₁: α ≠ β (class-conditional noise)
 *
 * on η̄ = (1/k) Σ η̂(x_i) over k anchors, with η̂ a model fitted to the noisy labels:
 *
 * - **parametric** (2022): logistic regression by maximum likelihood on (1, x); by the asymptotic normality of the MLE
 *   and the delta method, η̄ ~ N(½, v) under H₀ with v = (1/16) x̄ᵀĤx̄, Ĥ = (XᵀDX)⁻¹, D = diag(η̂ᵢ(1 − η̂ᵢ)), x̄ the mean
 *   of the augmented anchors (Eqs. 4, 6 and §3.2);
 * - **local** (2024): local likelihood logistic regression at each anchor (`localLogistic`), whose log-odds have the
 *   sandwich variance; v = (1/(16k²)) Σ_{k,j} cov(β̂₀(x_k), β̂₀(x_j)) (Eqs. 7–9 and the multiple-anchor variance).
 *
 * z = (η̄ − ½)/√v, and the p-value is 2(1 − Φ(|z|)). The power at level a is 1 − b with
 * b = Φ((z_{1−a/2}√v + h)/√ṽ) − Φ((−z_{1−a/2}√v + h)/√ṽ), h = (β − α)/2 (Prop. 3.1, Eq. 11); under H₁ the delta-method
 * factor is [η̃(1 − η̃)]² with η̃ = (1 − α + β)/2, so ṽ = 16[η̃(1 − η̃)]² v. (The paper prints the factor as
 * [(1 − α + β)(β − α)]²/16; the reading here, logged in the progress file, is the delta method's.)
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

/** Which model estimates the posterior at the anchors. */
export type NoiseTestModel = 'parametric' | 'local'

/** Options of {@link classConditionalNoiseTest}. */
export interface NoiseTestOptions {
  /** `parametric` (logistic regression MLE, default) or `local` (local likelihood). */
  model?: NoiseTestModel
  /** The local model's bandwidth (default 1) and polynomial order (default 1). */
  bandwidth?: number
  degree?: 0 | 1 | 2
}

/** A noise test's result: the z-test, with the posterior estimates at the anchors. */
export type NoiseTestResult = TestResult & {
  /** η̂ at each anchor and their mean η̄. */
  readonly anchorEstimates: readonly number[]
  readonly meanEstimate: number
  /** v, the variance of η̄ under H₀. */
  readonly variance: number
}

/** The logistic-regression MLE on (1, x) and Ĥ = (XᵀDX)⁻¹ at it. */
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
 * The anchor-point test for class-conditional label noise (module notes): `x` [n, d] and the noisy labels `y` ∈ {0, 1}
 * of the data, and `anchors` [k, d], points whose clean posterior is (about) ½.
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
 * The power 1 − b of the two-sided test at level `level` (Prop. 3.1, Eq. 11; Eq. 12 for k anchors chosen at random):
 * v is the variance of η̄ under H₀ (already divided by k for several anchors), and ṽ = 16[η̃(1 − η̃)]² v with
 * η̃ = (1 − α + β)/2 its variance under H₁ (module notes).
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
  readonly label: string
  /** The datasets' features and noisy labels, and one anchor set [k, d] per dataset. */
  readonly datasets: readonly { x: Tensor; y: Tensor }[]
  readonly anchors: readonly MatrixLike[]
}

/** The p-values of every cell so far, and the share rejected at each level. */
export interface NoiseTestSimulation {
  readonly labels: readonly string[]
  readonly pValues: number[][]
  /** Rejection rates at 0.05 and 0.10 per cell (the size under H₀, the power under H₁). */
  readonly rejected05: number[]
  readonly rejected10: number[]
  readonly done: number
  readonly total: number
}

/**
 * Run the test on every dataset of every cell, as a generator that yields after each dataset, so a figure can fill its
 * box plots while the worker computes. With `bandwidths`, the local model's bandwidth is chosen per dataset by
 * leave-one-out (`localLogisticBandwidth`).
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

/** The number of anchors k that reaches power `target` at level `level` for given noise rates and single-anchor v. */
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
