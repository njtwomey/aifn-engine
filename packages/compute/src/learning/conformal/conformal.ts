/**
 * Split conformal prediction: score $n$ held-out cases with a nonconformity score, take the
 * $\lceil (n + 1)(1 - \alpha) \rceil$-th smallest score $\hat q$, and include in each prediction set every answer whose
 * score is at most $\hat q$. If the calibration and test cases are exchangeable, the set covers the truth with
 * probability at least $1 - \alpha$ (and at most $1 - \alpha + 1/(n + 1)$ when the scores have no ties). Regression
 * intervals from absolute residuals, conformalised quantile regression (CQR), classification sets by the LAC, APS and
 * RAPS scores, and Mondrian (group-conditional) quantiles; coverage and set-size summaries.
 *
 * The model is fitted elsewhere: these functions take its predictions on the calibration and test cases. A level
 * $\alpha$ outside $(0, 1)$, inputs of different lengths and labels or probabilities out of range throw `DomainError`.
 * Everything is deterministic except APS and RAPS given a stream, which draw one uniform per case from it.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

type F64 = dense.F64

/**
 * Throws `DomainError` unless $0 < \alpha < 1$ (NaN included).
 *
 * @param alpha The miscoverage level $\alpha$ to check.
 * @param where The caller's name, for the error message.
 */
function checkAlpha(alpha: number, where: string) {
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(where, `${where}: α must be in (0, 1)`)
}

/**
 * The conformal quantile of calibration scores: the $\lceil (n + 1)(1 - \alpha) \rceil$-th smallest of the $n$, or
 * $+\infty$ when that rank exceeds $n$ (too few calibration cases for level $\alpha$: every answer is included). A
 * rank that is an integer up to rounding ($(1 - 0.7) \cdot 10$) is not pushed up by one. Throws `DomainError` for
 * $\alpha$ outside $(0, 1)$, no scores, or a NaN score.
 *
 * @param scores The nonconformity scores of the $n$ calibration cases, in any order (not modified).
 * @param alpha The miscoverage level $\alpha$ in $(0, 1)$: a set built with the quantile covers the truth with
 *   probability at least $1 - \alpha$.
 * @returns $\hat q$, the score at that rank, or `Infinity`.
 *
 * @example Nine scores: the 5th smallest at alpha = 0.5, the 8th at alpha = 0.2
 * const scores = [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6]
 * print('alpha = 0.5: q =', conformalQuantile(scores, 0.5))
 * print('alpha = 0.2: q =', conformalQuantile(scores, 0.2))
 *
 * @example Too few cases for the level: the quantile is infinite
 * // (9 + 1)(1 - 0.05) = 9.5 rounds up to rank 10, beyond the 9 scores.
 * print('q =', conformalQuantile([0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6], 0.05))
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
  /** The lower end of each test case's interval ($m$ values). */
  readonly lower: Tensor
  /** The upper end of each test case's interval ($m$ values). */
  readonly upper: Tensor
  /** The conformal quantile $\hat q$ of the calibration scores (`Infinity` when there are too few). */
  readonly quantile: number
  /** The calibration scores ($n$ values, in the order of the calibration cases). */
  readonly scores: Tensor
}

/**
 * Split conformal regression with the absolute residual score $\lvert y - \hat y \rvert$ (Lei et al., 2018): intervals
 * $\hat y \pm \hat q$ of the same width everywhere. Throws `DomainError` when the calibration targets and predictions
 * differ in length, or for $\alpha$ outside $(0, 1)$.
 *
 * @param calibration The held-out cases, not used to fit the model: their `targets` $y_i$ and the model's
 *   `predictions` $\hat y_i$, $n$ of each.
 * @param predictions The model's predictions $\hat y$ at the $m$ test points: the centres of the intervals.
 * @param alpha The miscoverage level $\alpha$ in $(0, 1)$; the intervals cover with probability at least
 *   $1 - \alpha$.
 * @returns The intervals $[\hat y - \hat q, \hat y + \hat q]$ of the test points, with $\hat q$ and the calibration
 *   scores $\lvert y_i - \hat y_i \rvert$.
 *
 * @example Residuals 0.1 to 0.9: at alpha = 0.2 the half-width is the 8th smallest
 * const targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
 * const residuals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
 * const calibration = { targets, predictions: targets.map((y, i) => y + residuals[i]) }
 * const { lower, upper, quantile } = splitConformalRegression(calibration, [2, 5], 0.2)
 * print('q =', quantile)
 * print('lower =', lower)
 * print('upper =', upper)
 *
 * @example Coverage on fresh data is close to 1 - alpha
 * // y = 2x + N(0, 1) noise, and a model that predicts 2x: 500 calibration and 2000 test cases, alpha = 0.1.
 * // The guarantee is on average over calibration draws: one draw's q can fall a little either side.
 * const s = stream(7)
 * const draw = (n) => {
 *   const x = toArray(uniform(s, 0, 10, { shape: [n] }))
 *   const noise = toArray(normals(s, n))
 *   return { predictions: x.map((v) => 2 * v), targets: x.map((v, i) => 2 * v + noise[i]) }
 * }
 * const calibration = draw(500)
 * const test = draw(2000)
 * const { lower, upper, quantile } = splitConformalRegression(calibration, test.predictions, 0.1)
 * print('q =', quantile, '(the 0.9 quantile of |N(0, 1)| is 1.645)')
 * print('coverage =', intervalCoverage(lower, upper, test.targets).coverage)
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
 * predictions are widened (or narrowed, when $\hat q < 0$) by $\hat q$ of the score $\max(l(x) - y, y - u(x))$,
 * giving $[l(x) - \hat q, u(x) + \hat q]$. The intervals keep the model's adaptive shape and gain the finite-sample
 * coverage guarantee. Throws `DomainError` when the calibration targets and quantiles differ in length, or for $\alpha$
 * outside $(0, 1)$.
 *
 * @param calibration The held-out cases: their `targets` $y_i$ and the model's `lower` and `upper` quantile
 *   predictions $l(x_i)$ and $u(x_i)$, $n$ of each.
 * @param test The model's `lower` and `upper` quantile predictions at the $m$ test points, equal in length.
 * @param alpha The miscoverage level $\alpha$ in $(0, 1)$; the intervals cover with probability at least
 *   $1 - \alpha$.
 * @returns The adjusted test intervals, with $\hat q$ and the calibration scores $\max(l(x_i) - y_i, y_i - u(x_i))$
 *   (negative for a target inside its interval).
 *
 * @example A quantile model's intervals widened or narrowed to the level
 * // Only the third target lies outside its interval (6 is below 6.5), so only its score is positive.
 * const calibration = {
 *   targets: [2, 4, 6, 8, 10],
 *   lower: [1, 3.5, 6.5, 7, 9],
 *   upper: [3, 4.5, 7, 9, 10.5],
 * }
 * const test = { lower: [0, 5], upper: [2, 8] }
 * const wide = conformalisedQuantileRegression(calibration, test, 0.2)
 * print('scores =', wide.scores)
 * print('alpha = 0.2: q =', wide.quantile, 'lower =', wide.lower, 'upper =', wide.upper)
 * const narrow = conformalisedQuantileRegression(calibration, test, 0.4)
 * print('alpha = 0.4: q =', narrow.quantile, 'lower =', narrow.lower, 'upper =', narrow.upper)
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
 * A classification nonconformity score of class $y$ with probabilities $p_1, \dots, p_K$: `lac` (least ambiguous
 * set-valued classifier; Sadinle, Lei and Wasserman, 2019), $1 - p_y$, which gives the smallest sets on average; `aps`
 * (adaptive prediction sets; Romano, Sesia and Candès, 2020), the total probability of the classes ranked above $y$
 * plus $u\,p_y$ (ties ranked by index), which adapts the set size to the case's difficulty; `raps` (regularised APS;
 * Angelopoulos et al., 2021), APS plus $\lambda (r_y - k_{\text{reg}})_+$ with $r_y$ the rank of $y$ (1 for the most
 * probable class), which penalises long tails of unlikely classes.
 */
export type ClassificationScore = 'lac' | 'aps' | 'raps'

/** Options of `classificationScores` and `conformalClassification`. */
export type ClassificationOptions = {
  /** The nonconformity score (default `lac`). */
  score?: ClassificationScore
  /**
   * Where APS and RAPS draw $u \sim \Unif(0, 1)$, one per case, to randomise the score. Without a stream $u = 1$:
   * the score counts all of $p_y$, which gives larger sets. LAC ignores it.
   */
  stream?: Stream
  /** The RAPS penalty $\lambda$ per rank beyond $k_{\text{reg}}$ (default 0.01). */
  lambda?: number
  /** The RAPS free rank $k_{\text{reg}}$: ranks up to it are not penalised (default 1). */
  kReg?: number
}

/**
 * Rows of probabilities, checked: throws `DomainError` for an entry outside $[0, 1]$ (rows need not sum to 1).
 *
 * @param probabilities One row of $K$ class probabilities per case ($n \times K$).
 * @param where The caller's name, for the error message.
 * @returns The entries `p`, row-major, with the number of rows `n` and of classes `K`.
 */
function readProbabilities(probabilities: MatrixLike, where: string): { p: F64; n: number; K: number } {
  const { data, m: n, n: K } = dense.toMatrixF64(probabilities, where)
  for (const v of data)
    if (!(v >= 0 && v <= 1)) throw new DomainError(where, `${where}: probability ${v} is not in [0, 1]`)
  return { p: data, n, K }
}

/**
 * The nonconformity score of class $k$ for one case (see `ClassificationScore`).
 *
 * @param row The case's $K$ class probabilities.
 * @param k The class scored, an index in $0, \dots, K - 1$.
 * @param score Which score: `lac`, `aps` or `raps`.
 * @param u The randomisation $u$ in $[0, 1]$ of APS and RAPS (1 for the non-randomised score; unused by LAC).
 * @param lambda The RAPS penalty $\lambda$ per rank beyond `kReg`.
 * @param kReg The RAPS free rank $k_{\text{reg}}$.
 * @returns The score: $1 - p_k$ for LAC; for APS the probability of the classes ranked above $k$ (larger, or equal
 *   with a smaller index) plus $u\,p_k$, and for RAPS that plus $\lambda (r_k - k_{\text{reg}})_+$.
 */
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

/**
 * The nonconformity score of each case's true label: the calibration scores of `conformalClassification`. Throws
 * `DomainError` for a probability outside $[0, 1]$, a label that is not a class index, or labels and rows that differ
 * in number.
 *
 * @param probabilities One row of $K$ class probabilities per case ($n \times K$).
 * @param labels The true class of each case, an index in $0, \dots, K - 1$ ($n$ values).
 * @param options The score (`lac` by default), the stream that randomises APS and RAPS (one uniform drawn per case),
 *   and the RAPS penalty.
 * @returns The $n$ scores.
 *
 * @example The three scores of the same cases
 * // Case 2's true class is ranked second, behind a class of probability 0.6.
 * const probabilities = [
 *   [0.7, 0.2, 0.1],
 *   [0.3, 0.6, 0.1],
 *   [0.2, 0.3, 0.5],
 * ]
 * const labels = [0, 0, 2]
 * print('lac  =', classificationScores(probabilities, labels))
 * print('aps  =', classificationScores(probabilities, labels, { score: 'aps' }))
 * print('raps =', classificationScores(probabilities, labels, { score: 'raps', lambda: 0.1 }))
 */
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
  /** The sets as an $m \times K$ matrix: 1 where a class is in the case's set, else 0. */
  readonly sets: Tensor
  /** The size of each set ($m$ values; 0 for an empty set). */
  readonly sizes: Tensor
  /** The conformal quantile $\hat q$ of the calibration scores (`Infinity` when there are too few). */
  readonly quantile: number
  /** The calibration scores of the true labels ($n$ values). */
  readonly scores: Tensor
}

/**
 * Split conformal classification sets: $\hat q$ from the calibration scores of the true labels, then every class whose
 * score on a test case is at most $\hat q$ (LAC: $p_k \ge 1 - \hat q$; APS without a stream: the top classes whose
 * cumulative probability stays at most $\hat q$). Coverage is at least $1 - \alpha$ marginally, over calibration and
 * test draws, not for each case. A set can be empty (LAC when no probability reaches $1 - \hat q$). Throws
 * `DomainError` as `classificationScores` does, or for $\alpha$ outside $(0, 1)$.
 *
 * @param calibration The held-out cases: the model's `probabilities` ($n \times K$) and the true `labels`.
 * @param probabilities The model's probabilities at the $m$ test cases ($m \times K$, the same $K$ classes).
 * @param alpha The miscoverage level $\alpha$ in $(0, 1)$.
 * @param options The score and its settings, the same for calibration and test; a stream is drawn from for the
 *   calibration cases first, then one uniform per test case.
 * @returns The sets and their sizes, with $\hat q$ and the calibration scores.
 *
 * @example LAC sets: every class with probability at least 1 - q
 * const calibration = {
 *   probabilities: [
 *     [0.8, 0.1, 0.1],
 *     [0.1, 0.7, 0.2],
 *     [0.2, 0.2, 0.6],
 *     [0.5, 0.4, 0.1],
 *   ],
 *   labels: [0, 1, 2, 1],
 * }
 * const test = [
 *   [0.9, 0.05, 0.05],
 *   [0.45, 0.45, 0.1],
 *   [0.34, 0.33, 0.33],
 * ]
 * const { sets, sizes, quantile, scores } = conformalClassification(calibration, test, 0.2)
 * print('calibration scores =', scores)
 * print('q =', quantile)
 * print('sets =', sets)
 * print('sizes =', sizes)
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
 * Mondrian (group-conditional) conformal quantiles (Vovk, Gammerman and Shafer, 2005): one $\hat q$ per group, from
 * the calibration scores of that group alone, so coverage holds within every group, not only on average. Groups are
 * integer labels (a class, a region, a protected attribute); a group with too few cases, or none, gets $+\infty$.
 * Throws `DomainError` for a group that is not a non-negative integer, groups and scores that differ in number, or
 * $\alpha$ outside $(0, 1)$.
 *
 * @param scores The nonconformity scores of the $n$ calibration cases.
 * @param groups The group of each case, an integer from 0 ($n$ values); the groups are $0, \dots, G - 1$ with $G$ one
 *   more than the largest.
 * @param alpha The miscoverage level $\alpha$ in $(0, 1)$, the same in every group.
 * @returns The $G$ quantiles, one per group: test cases of group $g$ use the $g$-th.
 *
 * @example Two groups with different noise: one quantile each
 * const scores = [0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4]
 * const groups = [0, 0, 0, 0, 1, 1, 1, 1]
 * print('per group =', mondrianQuantiles(scores, groups, 0.2))
 * print('pooled =', conformalQuantile(scores, 0.2))
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

/** Coverage and size of prediction sets or intervals against the truth, from `intervalCoverage` or `setCoverage`. */
export interface CoverageSummary {
  /** The share of cases whose set or interval contains the truth. */
  readonly coverage: number
  /** The mean set size (classification) or interval width (regression). */
  readonly meanSize: number
  /** 1 where a case is covered, else 0 ($m$ values). */
  readonly covered: Tensor
}

/**
 * The empirical coverage and mean width of intervals $[l_i, u_i]$ against targets $y_i$: a target on an end counts as
 * covered. Throws `DomainError` when the three differ in length; no cases give NaN.
 *
 * @param lower The lower end $l_i$ of each of the $m$ intervals.
 * @param upper The upper end $u_i$ of each interval.
 * @param targets The true values $y_i$.
 * @returns The share of $y_i \in [l_i, u_i]$, the mean of $u_i - l_i$, and which cases are covered.
 *
 * @example Two of three targets inside their intervals
 * const { coverage, meanSize, covered } = intervalCoverage([0, 1, 2], [2, 3, 3], [1, 3.5, 2])
 * print('covered =', covered)
 * print('coverage =', coverage)
 * print('mean width =', meanSize)
 */
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

/**
 * The empirical coverage and mean size of classification sets against labels. Throws `DomainError` when there are not
 * as many labels as sets. Labels are not checked: one outside $0, \dots, K - 1$ counts as not covered.
 *
 * @param sets The sets as an $m \times K$ matrix, a positive entry where a class is in the case's set (the `sets` of
 *   `conformalClassification`).
 * @param labels The true class of each case, an index in $0, \dots, K - 1$ ($m$ values).
 * @returns The share of cases whose set holds the label, the mean set size, and which cases are covered.
 *
 * @example Sets of sizes 1, 2 and 0 against their labels
 * const sets = [
 *   [1, 0, 0],
 *   [1, 1, 0],
 *   [0, 0, 0],
 * ]
 * const { coverage, meanSize, covered } = setCoverage(sets, [0, 1, 2])
 * print('covered =', covered)
 * print('coverage =', coverage)
 * print('mean size =', meanSize)
 */
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
