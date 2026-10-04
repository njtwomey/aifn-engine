/**
 * Split conformal prediction: score n held-out cases with a nonconformity score, take the ⌈(n + 1)(1 − α)⌉-th
 * smallest score q̂, and include in each prediction set every answer whose score is at most q̂. If the calibration and
 * test cases are exchangeable, the set covers the truth with probability at least 1 − α (and at most
 * 1 − α + 1/(n + 1) when the scores have no ties). Regression intervals from absolute residuals, conformalised
 * quantile regression (CQR), classification sets by the LAC, APS and RAPS scores, and Mondrian (group-conditional)
 * quantiles; coverage and set-size summaries.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

type F64 = dense.F64

function checkAlpha(alpha: number, where: string) {
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(where, `${where}: α must be in (0, 1)`)
}

/**
 * The conformal quantile of calibration scores: the ⌈(n + 1)(1 − α)⌉-th smallest, or +∞ when that rank exceeds n
 * (too few calibration cases for level α: every answer is included).
 */
export function conformalQuantile(scores: VectorLike, alpha: number): number {
  const where = 'conformalQuantile'
  checkAlpha(alpha, where)
  const s = Float64Array.from(dense.toF64(scores, where)).sort()
  const n = s.length
  if (n === 0) throw new DomainError(where, `${where}: no calibration scores`)
  if (s.some((v) => Number.isNaN(v))) throw new DomainError(where, `${where}: a score is NaN`)
  // (n + 1)(1 − α) is an integer more often than its floating-point value shows ((1 − 0.7) · 10 = 3.0000000000000004),
  // so a rounding error above an integer must not push the rank up by one.
  const x = (n + 1) * (1 - alpha)
  const k = Math.ceil(x - 1e-9 * Math.max(1, x))
  return k > n ? Infinity : s[k - 1]
}

// ── Regression ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Prediction intervals with the conformal quantile that made them. */
export interface ConformalIntervals {
  readonly lower: Tensor
  readonly upper: Tensor
  /** The conformal quantile q̂ of the calibration scores. */
  readonly quantile: number
  /** The calibration scores [n]. */
  readonly scores: Tensor
}

/**
 * Split conformal regression with the absolute residual score |y − ŷ| (Lei et al., 2018): intervals ŷ ± q̂ of the
 * same width everywhere. `calibration` holds the held-out targets and predictions; `predictions` the test points'.
 */
export function splitConformalRegression(
  calibration: { targets: VectorLike; predictions: VectorLike },
  predictions: VectorLike,
  alpha: number,
): ConformalIntervals {
  const where = 'splitConformalRegression'
  const y = dense.toF64(calibration.targets, where)
  const f = dense.toF64(calibration.predictions, where)
  if (y.length !== f.length) throw new DomainError(where, `${where}: ${y.length} targets and ${f.length} predictions`)
  const scores = y.map((v, i) => Math.abs(v - f[i]))
  const q = conformalQuantile(scores, alpha)
  const p = dense.toF64(predictions, where)
  return {
    lower: dense.vec(p.map((v) => v - q)),
    upper: dense.vec(p.map((v) => v + q)),
    quantile: q,
    scores: dense.vec(scores),
  }
}

/**
 * Conformalised quantile regression (Romano, Patterson and Candès, 2019): a model's lower and upper quantile
 * predictions are widened (or narrowed) by q̂ of the score max(l(x) − y, y − u(x)), giving [l(x) − q̂, u(x) + q̂]. The
 * intervals keep the model's adaptive shape and gain the finite-sample coverage guarantee.
 */
export function conformalisedQuantileRegression(
  calibration: { targets: VectorLike; lower: VectorLike; upper: VectorLike },
  test: { lower: VectorLike; upper: VectorLike },
  alpha: number,
): ConformalIntervals {
  const where = 'conformalisedQuantileRegression'
  const y = dense.toF64(calibration.targets, where)
  const lo = dense.toF64(calibration.lower, where)
  const hi = dense.toF64(calibration.upper, where)
  if (lo.length !== y.length || hi.length !== y.length)
    throw new DomainError(where, `${where}: the calibration targets and quantiles differ in length`)
  const scores = y.map((v, i) => Math.max(lo[i] - v, v - hi[i]))
  const q = conformalQuantile(scores, alpha)
  const tl = dense.toF64(test.lower, where)
  const tu = dense.toF64(test.upper, where)
  return {
    lower: dense.vec(tl.map((v) => v - q)),
    upper: dense.vec(tu.map((v) => v + q)),
    quantile: q,
    scores: dense.vec(scores),
  }
}

// ── Classification ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A classification nonconformity score: `lac` (least ambiguous set-valued classifier; Sadinle, Lei and Wasserman,
 * 2019), 1 − p_y, which gives the smallest sets on average; `aps` (adaptive prediction sets; Romano, Sesia and Candès,
 * 2020), the total probability of the classes ranked at or above y, randomised by u·p_y, which adapts the set size to
 * the case's difficulty; `raps` (regularised APS; Angelopoulos et al., 2021), APS plus λ(rank − k_reg)₊, which
 * penalises long tails of unlikely classes.
 */
export type ClassificationScore = 'lac' | 'aps' | 'raps'

/** Options of the classification scores and sets. */
export type ClassificationOptions = {
  score?: ClassificationScore
  /** Randomise APS and RAPS by u ~ U(0, 1) (default true when a stream is given). */
  stream?: Stream
  /** The RAPS penalty λ (default 0.01) and the free rank k_reg (default 1). */
  lambda?: number
  kReg?: number
}

/** Rows of probabilities [n, K], checked. */
function readProbabilities(probabilities: MatrixLike, where: string): { p: F64; n: number; K: number } {
  const { data, m: n, n: K } = dense.toMatrixF64(probabilities, where)
  for (const v of data)
    if (!(v >= 0 && v <= 1)) throw new DomainError(where, `${where}: probability ${v} is not in [0, 1]`)
  return { p: data, n, K }
}

/** The score of class k for one row, with u the randomisation (1 for the non-randomised score). */
function classScore(row: F64, k: number, score: ClassificationScore, u: number, lambda: number, kReg: number): number {
  if (score === 'lac') return 1 - row[k]
  let above = 0
  let rank = 1
  for (let j = 0; j < row.length; j++)
    if (row[j] > row[k] || (row[j] === row[k] && j < k)) {
      above += row[j]
      rank++
    }
  const s = above + u * row[k]
  return score === 'raps' ? s + lambda * Math.max(0, rank - kReg) : s
}

/** The nonconformity score of each case's true label [n]. */
export function classificationScores(
  probabilities: MatrixLike,
  labels: VectorLike,
  options: ClassificationOptions = {},
): Tensor {
  const where = 'classificationScores'
  const { p, n, K } = readProbabilities(probabilities, where)
  const y = dense.toF64(labels, where)
  if (y.length !== n) throw new DomainError(where, `${where}: ${n} rows and ${y.length} labels`)
  const { score = 'lac', lambda = 0.01, kReg = 1, stream } = options
  return dense.vec(
    Float64Array.from({ length: n }, (_, i) => {
      if (!Number.isInteger(y[i]) || y[i] < 0 || y[i] >= K) throw new DomainError(where, `${where}: label ${y[i]}`)
      const u = stream && score !== 'lac' ? uniform(stream) : 1
      return classScore(p.subarray(i * K, (i + 1) * K), y[i], score, u, lambda, kReg)
    }),
  )
}

/** Conformal prediction sets for classification. */
export interface ConformalSets {
  /** 1 where a class is in the case's set [m, K]. */
  readonly sets: Tensor
  /** The size of each set [m]. */
  readonly sizes: Tensor
  readonly quantile: number
  /** The calibration scores [n]. */
  readonly scores: Tensor
}

/**
 * Split conformal classification sets: q̂ from the calibration scores of the true labels, then every class whose score
 * on a test case is at most q̂ (LAC: p_k ≥ 1 − q̂; APS and RAPS: the top classes until their cumulative probability
 * reaches q̂). Coverage is at least 1 − α marginally, over calibration and test draws.
 */
export function conformalClassification(
  calibration: { probabilities: MatrixLike; labels: VectorLike },
  probabilities: MatrixLike,
  alpha: number,
  options: ClassificationOptions = {},
): ConformalSets {
  const where = 'conformalClassification'
  const scores = classificationScores(calibration.probabilities, calibration.labels, options)
  const q = conformalQuantile(scores, alpha)
  const { p, n, K } = readProbabilities(probabilities, where)
  const { score = 'lac', lambda = 0.01, kReg = 1, stream } = options
  const sets = new Float64Array(n * K)
  const sizes = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const row = p.subarray(i * K, (i + 1) * K)
    const u = stream && score !== 'lac' ? uniform(stream) : 1
    for (let k = 0; k < K; k++)
      if (classScore(row, k, score, u, lambda, kReg) <= q) {
        sets[i * K + k] = 1
        sizes[i]++
      }
  }
  return { sets: dense.mat(sets, n, K), sizes: dense.vec(sizes), quantile: q, scores }
}

// ── Mondrian ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Mondrian (group-conditional) conformal quantiles (Vovk, Gammerman and Shafer, 2005): one q̂ per group, from the
 * calibration scores of that group alone, so coverage holds within every group, not only on average. Groups are
 * integer labels (a class, a region, a protected attribute); a group with too few cases gets +∞.
 */
export function mondrianQuantiles(scores: VectorLike, groups: VectorLike, alpha: number): Tensor {
  const where = 'mondrianQuantiles'
  const s = dense.toF64(scores, where)
  const g = dense.toF64(groups, where)
  if (g.length !== s.length) throw new DomainError(where, `${where}: ${s.length} scores and ${g.length} groups`)
  let G = 0
  for (const v of g) {
    if (!Number.isInteger(v) || v < 0) throw new DomainError(where, `${where}: group ${v} is not an index`)
    G = Math.max(G, v + 1)
  }
  const members: number[][] = Array.from({ length: G }, () => [])
  g.forEach((v, i) => members[v].push(s[i]))
  return dense.vec(Float64Array.from(members, (m) => (m.length === 0 ? Infinity : conformalQuantile(m, alpha))))
}

// ── Summaries ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Coverage and size of prediction sets or intervals against the truth. */
export interface CoverageSummary {
  /** The share of cases whose set or interval contains the truth. */
  readonly coverage: number
  /** The mean set size (classification) or interval width (regression). */
  readonly meanSize: number
  /** 1 where each case is covered [m]. */
  readonly covered: Tensor
}

/** The empirical coverage and mean width of intervals [lower, upper] against targets. */
export function intervalCoverage(lower: VectorLike, upper: VectorLike, targets: VectorLike): CoverageSummary {
  const where = 'intervalCoverage'
  const lo = dense.toF64(lower, where)
  const hi = dense.toF64(upper, where)
  const y = dense.toF64(targets, where)
  if (lo.length !== y.length || hi.length !== y.length)
    throw new DomainError(where, `${where}: intervals and targets differ in length`)
  const covered = y.map((v, i) => (v >= lo[i] && v <= hi[i] ? 1 : 0))
  const n = y.length
  return {
    coverage: covered.reduce((a, b) => a + b, 0) / n,
    meanSize: hi.reduce((a, v, i) => a + (v - lo[i]), 0) / n,
    covered: dense.vec(covered),
  }
}

/** The empirical coverage and mean size of classification sets [m, K] (0/1) against labels. */
export function setCoverage(sets: MatrixLike, labels: VectorLike): CoverageSummary {
  const where = 'setCoverage'
  const { data, m: n, n: K } = dense.toMatrixF64(sets, where)
  const y = dense.toF64(labels, where)
  if (y.length !== n) throw new DomainError(where, `${where}: ${n} sets and ${y.length} labels`)
  const covered = Float64Array.from({ length: n }, (_, i) => (data[i * K + y[i]] > 0 ? 1 : 0))
  let size = 0
  for (const v of data) size += v > 0 ? 1 : 0
  return { coverage: covered.reduce((a, b) => a + b, 0) / n, meanSize: size / n, covered: dense.vec(covered) }
}
