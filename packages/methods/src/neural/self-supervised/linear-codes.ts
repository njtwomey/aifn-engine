/**
 * The self-supervised objectives of the three families evaluated for the simplest encoder there is, a linear map to
 * one number, $z = \wvec^\top \xvec$ with $\wvec \in \reals^d$, so their landscapes can be drawn over the encoder.
 *
 * Two augmented views $\avec_i, \bvec_i$ of each of $n$ inputs $\xvec_i$ are given. The objectives, all to be
 * minimised:
 *
 * - `reconstruction` (the masked and generative family in its linear form): the mean squared error of the best linear
 *   decoder $\hat\xvec = \vvec z$ of the centred clean inputs,
 *   $\operatorname{tr}\Sigmamat - \wvec^\top \Sigmamat^2 \wvec / \wvec^\top \Sigmamat \wvec$, which is minimal on the
 *   leading principal direction and does not depend on the scale of $\wvec$.
 * - `contrastive`: InfoNCE over the batch with critic $s_{ij} = -(z^a_i - z^b_j)^2$, so the scale of $\wvec$ acts as
 *   an inverse temperature; $\log n$ at $\wvec = \zeros$ (van den Oord et al., 2018; Chen et al., 2020).
 * - `agreement`: the mean squared distance between the two views' codes, $\frac1n \sum_i (z^a_i - z^b_i)^2$; zero at
 *   $\wvec = \zeros$, the collapsed encoder.
 * - `variance`: `agreement` plus a hinge on the standard deviation of each view's codes over the batch,
 *   $\mu \max(0, \gamma - \operatorname{sd}(z^a)) + \mu \max(0, \gamma - \operatorname{sd}(z^b))$, VICReg's variance
 *   term for one dimension with $\gamma = 1$ (Bardes et al., 2022, eq. 1).
 *
 * Plain loops over row-major arrays (row $i$ at entries $id$ to $id + d - 1$); light enough to evaluate on a grid of
 * encoders.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { covariance } from 'aifn-compute/probability/stats'

/** Which objective; see the file comment. */
export type LinearCodeObjective = 'reconstruction' | 'contrastive' | 'agreement' | 'variance'

/** Inputs and their two views, row-major $n \times d$. */
export type LinearCodeProblem = {
  /** The clean inputs $\xvec_i$. */
  x: Float64Array
  /** The first views $\avec_i$. */
  a: Float64Array
  /** The second views $\bvec_i$. */
  b: Float64Array
  /** The number of inputs $n$. */
  n: number
  /** The input dimension $d$. */
  d: number
}

/**
 * The codes $z_i = \wvec^\top \rvec_i$ of the rows of a row-major matrix.
 *
 * @param rows The rows, row-major $n \times d$.
 * @param w The encoder, $d$ values.
 * @param n The number of rows.
 * @param d The row length.
 * @returns $n$ codes.
 */
const project = (rows: Float64Array, w: ArrayLike<number>, n: number, d: number) => {
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += rows[i * d + j] * w[j]
    z[i] = s
  }
  return z
}

/**
 * The sample standard deviation of codes (divisor $n - 1$, at least 1).
 *
 * @param z The codes.
 * @returns Their standard deviation.
 */
const sd = (z: Float64Array) => {
  let m = 0
  for (const v of z) m += v / z.length
  let s = 0
  for (const v of z) s += (v - m) ** 2 / Math.max(1, z.length - 1)
  return Math.sqrt(s)
}

/**
 * The covariance $\Sigmamat$ of the clean inputs, with divisor $n$: entry $(j, k)$ is `covariance` of columns $j$
 * and $k$ (`aifn-compute/probability/stats`). Throws `ShapeError` when `x` does not hold $nd$ values.
 *
 * @param p The problem; only `x`, `n` and `d` are read.
 * @returns Row-major $d \times d$ values.
 *
 * @example The covariance of three points on a line
 * const p = { x: Float64Array.from([0, 0, 1, 2, 2, 4]), a: new Float64Array(6), b: new Float64Array(6), n: 3, d: 2 }
 * print(inputCovariance(p))
 */
export function inputCovariance(p: LinearCodeProblem): Float64Array {
  const { x, n, d } = p
  if (x.length !== n * d)
    throw new ShapeError('inputCovariance', `inputCovariance: x has ${x.length} values, not n × d = ${n * d}`)
  const columns = Array.from({ length: d }, (_, j) => Float64Array.from({ length: n }, (_, i) => x[i * d + j]))
  const S = new Float64Array(d * d)
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) S[j * d + k] = S[k * d + j] = covariance(columns[j], columns[k])
  return S
}

/**
 * One objective at encoder $\wvec$ (see the file comment). NaN for `reconstruction` where $\wvec^\top \Sigmamat \wvec
 * \le 10^{-12}$, where the code is constant.
 *
 * @param p The inputs and views.
 * @param w The encoder's weights ($d$ values).
 * @param objective Which objective.
 * @param options `mu`, the weight $\mu$ of the variance hinge (default 1), and `sigma`, the inputs' covariance when
 *   already computed (for `reconstruction`).
 * @returns The objective's value.
 *
 * @example The objectives at the collapsed encoder and at a good one
 * const x = Float64Array.from([-2, 0.1, -1, -0.2, 1, 0.1, 2, -0.1])
 * const p = { x, a: x.map((v) => v + 0.05), b: x.map((v) => v - 0.05), n: 4, d: 2 }
 * for (const objective of ['contrastive', 'agreement', 'variance']) {
 *   const [collapsed, good] = [[0, 0], [1, 0]].map((w) => linearCodeObjective(p, w, objective))
 *   print(objective, ' w = 0:', collapsed, ' w = (1, 0):', good)
 * }
 * print('reconstruction along x1:', linearCodeObjective(p, [1, 0], 'reconstruction'))
 */
export function linearCodeObjective(
  p: LinearCodeProblem,
  w: ArrayLike<number>,
  objective: LinearCodeObjective,
  options: { mu?: number; sigma?: Float64Array } = {},
): number {
  const { n, d } = p
  if (objective === 'reconstruction') {
    const S = options.sigma ?? inputCovariance(p)
    const Sw = new Float64Array(d)
    for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) Sw[j] += S[j * d + k] * w[k]
    let wSw = 0
    let SwSw = 0
    let tr = 0
    for (let j = 0; j < d; j++) {
      wSw += w[j] * Sw[j]
      SwSw += Sw[j] * Sw[j]
      tr += S[j * d + j]
    }
    return wSw > 1e-12 ? tr - SwSw / wSw : NaN
  }
  const za = project(p.a, w, n, d)
  const zb = project(p.b, w, n, d)
  if (objective === 'contrastive') {
    let loss = 0
    for (let i = 0; i < n; i++) {
      let top = -Infinity
      for (let j = 0; j < n; j++) top = Math.max(top, -((za[i] - zb[j]) ** 2))
      let z = 0
      for (let j = 0; j < n; j++) z += Math.exp(-((za[i] - zb[j]) ** 2) - top)
      loss += (za[i] - zb[i]) ** 2 + top + Math.log(z)
    }
    return loss / n
  }
  let align = 0
  for (let i = 0; i < n; i++) align += (za[i] - zb[i]) ** 2 / n
  if (objective === 'agreement') return align
  const mu = options.mu ?? 1
  return align + mu * Math.max(0, 1 - sd(za)) + mu * Math.max(0, 1 - sd(zb))
}

/**
 * The objective on a square grid of two-dimensional encoders $\wvec = (w_1, w_2)$, and its smallest value (the first
 * on a tie, NaN values skipped). Throws `DomainError` unless $d = 2$ and there are at least two points per side.
 *
 * @param p The inputs and views ($d = 2$).
 * @param objective Which objective.
 * @param range The grid spans $[-r, r]^2$.
 * @param count Points per side.
 * @param options `mu`, the weight of the variance hinge (default 1).
 * @returns `axis`, the grid's coordinates; `z[i][j]`, the value at $(w_1, w_2) = $ (`axis[j]`, `axis[i]`); `best`,
 *   the grid point with the smallest value, and `bestValue`, that value.
 *
 * @example The InfoNCE landscape of a few encoders
 * const x = Float64Array.from([-2, 0.1, -1, -0.2, 1, 0.1, 2, -0.1])
 * const p = { x, a: x.map((v) => v + 0.05), b: x.map((v) => v - 0.05), n: 4, d: 2 }
 * const { axis, best, bestValue } = linearCodeLandscape(p, 'contrastive', 2, 5)
 * print('axis:', axis, ' best encoder:', best, ' value:', bestValue)
 */
export function linearCodeLandscape(
  p: LinearCodeProblem,
  objective: LinearCodeObjective,
  range: number,
  count: number,
  options: { mu?: number } = {},
): { axis: number[]; z: number[][]; best: [number, number]; bestValue: number } {
  if (p.d !== 2) throw new DomainError('linearCodeLandscape', 'linearCodeLandscape: needs two-dimensional inputs')
  if (!(count >= 2)) throw new DomainError('linearCodeLandscape', 'linearCodeLandscape: needs two points per side')
  const axis = Array.from({ length: count }, (_, i) => -range + (2 * range * i) / (count - 1))
  const sigma = inputCovariance(p)
  let best: [number, number] = [0, 0]
  let bestValue = Infinity
  const z = axis.map((w2) =>
    axis.map((w1) => {
      const v = linearCodeObjective(p, [w1, w2], objective, { ...options, sigma })
      if (v < bestValue) {
        bestValue = v
        best = [w1, w2]
      }
      return v
    }),
  )
  return { axis, z, best, bestValue }
}

/**
 * The accuracy of the best threshold on the code $z = \wvec^\top \xvec$ for two classes: a linear probe on one
 * feature. Thresholds fall between distinct code values only, so points with equal codes always get the same
 * prediction (ties cannot be split to inflate the accuracy).
 *
 * @param x The inputs, row-major $n \times d$.
 * @param y The classes, 0 or 1.
 * @param w The encoder.
 * @param d The input dimension.
 * @returns The best accuracy over thresholds and both orientations, in $[0.5, 1]$.
 *
 * @example A code that separates, and one that ties every point
 * const x = Float64Array.from([0, 1, 1, 1, 2, 0, 3, 0])
 * print('along x1:', thresholdProbeAccuracy(x, [0, 0, 1, 1], [1, 0], 2))
 * print('w = 0 (all codes equal):', thresholdProbeAccuracy(x, [0, 0, 1, 1], [0, 0], 2))
 */
export function thresholdProbeAccuracy(x: Float64Array, y: ArrayLike<number>, w: ArrayLike<number>, d: number): number {
  const n = y.length
  const z = project(x, w, n, d)
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => z[i] - z[j])
  let ones = 0
  for (let i = 0; i < n; i++) ones += y[i] === 1 ? 1 : 0
  // Predict 1 above the threshold: correct = zeros below + ones above.
  let zerosBelow = 0
  let onesBelow = 0
  let best = Math.max(ones, n - ones)
  for (let r = 0; r < n; r++) {
    const i = order[r]
    if (y[i] === 1) onesBelow++
    else zerosBelow++
    // A threshold here only if the next code differs: equal codes fall on the same side.
    if (r + 1 < n && z[order[r + 1]] === z[i]) continue
    const correct = zerosBelow + (ones - onesBelow)
    best = Math.max(best, correct, n - correct)
  }
  return best / n
}
