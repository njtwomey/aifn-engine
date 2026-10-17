/**
 * A novelty score for open-set semi-supervised learning from the labelled points alone: the shared-covariance Gaussian
 * class model of linear discriminant analysis (Hastie, Tibshirani and Friedman, 2009, §4.3), with a spherical pooled
 * variance, used as a filter on unlabelled data as in the out-of-distribution filtering of open-set semi-supervised
 * learning (Chen et al., 2020; Yu et al., 2020).
 *
 * Each known class $k$ gets the mean $\hat\muvec_k$ of its labelled points, and all classes share one spherical
 * variance $\hat\sigma^2$, pooled over the labelled points with $d$ degrees of freedom removed per class mean. A
 * point's score is its smallest squared standardised distance
 * $s(\xvec) = \min_k \lVert \xvec - \hat\muvec_k \rVert^2 / \hat\sigma^2$, and its p-value is
 * $P(\chi^2_d \ge s(\xvec))$, the chance that a point of the nearest known class lies at least that far out if the
 * classes were Gaussian. Unlabelled points with a small p-value are likely from a class no label covers and can be left
 * out of semi-supervised training. The spherical variance stays estimable with one or two labels per class, where the
 * full covariance of a Mahalanobis score (`mahalanobisModel` in `aifn-methods/unsupervised/anomaly`) does not.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { chiSquareSf } from 'aifn-compute/numerics/special'

/** Novelty scores of every row. */
export interface GaussianNovelty {
  /** Class means from the labelled rows, $K \times d$. */
  means: Tensor
  /** The pooled spherical variance $\hat\sigma^2$. */
  variance: number
  /** $s(\xvec)$ of every row: the smallest squared standardised distance to a class mean. */
  score: Float64Array
  /** $P(\chi^2_d \ge s(\xvec))$ of every row. */
  pValue: Float64Array
  /** The nearest class of every row. */
  nearest: Int32Array
}

/**
 * Novelty scores from labelled class means and a pooled spherical variance; see the file comment. Throws `ShapeError`
 * when `x` and `labels` differ in rows, and `DomainError` when a class has no labelled row or there are no more
 * labelled rows than classes.
 *
 * @param x All rows, $n \times d$.
 * @param labels A class $0, \dots, K - 1$ for each labelled row, $-1$ for unlabelled rows. Every class needs a label,
 *   and there must be more labelled rows than classes.
 * @param classes The number of known classes $K$.
 * @param varianceFloor The smallest variance used (default $10^{-6}$).
 * @returns The class means, the pooled variance, and each row's score, p-value and nearest class.
 *
 * @example The open-set note's worked example: 3.5 is far from both known classes
 * const x = [[-1.6], [-0.8], [0.8], [1.6], [3.5], [2.5], [0], [1.9]]
 * const r = gaussianNoveltyScores(x, [0, 0, 1, 1, -1, -1, -1, -1], 2)
 * print('means:', r.means, ' variance:', r.variance)
 * print('scores:', r.score.slice(4), ' p-values:', r.pValue.slice(4))
 */
export function gaussianNoveltyScores(
  x: MatrixLike,
  labels: ArrayLike<number>,
  classes: number,
  varianceFloor = 1e-6,
): GaussianNovelty {
  const where = 'gaussianNoveltyScores'
  const { data: X, m: n, n: d } = dense.toMatrixF64(x, where)
  if (labels.length !== n) throw new ShapeError(where, `${where}: x and labels differ in rows`)
  const sums = new Float64Array(classes * d)
  const counts = new Array<number>(classes).fill(0)
  for (let i = 0; i < n; i++) {
    const k = labels[i]
    if (k < 0) continue
    counts[k]++
    for (let j = 0; j < d; j++) sums[k * d + j] += X[i * d + j]
  }
  if (counts.some((c) => c === 0)) throw new DomainError(where, `${where}: every class needs a labelled row`)
  const labelled = counts.reduce((a, c) => a + c, 0)
  if (labelled <= classes) throw new DomainError(where, `${where}: need more labelled rows than classes`)
  const means = Float64Array.from(sums, (v, idx) => v / counts[Math.floor(idx / d)])
  let ss = 0
  for (let i = 0; i < n; i++) {
    const k = labels[i]
    if (k < 0) continue
    for (let j = 0; j < d; j++) ss += (X[i * d + j] - means[k * d + j]) ** 2
  }
  const variance = Math.max(varianceFloor, ss / (d * (labelled - classes)))
  const score = new Float64Array(n)
  const pValue = new Float64Array(n)
  const nearest = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    let best = Infinity
    for (let k = 0; k < classes; k++) {
      let s = 0
      for (let j = 0; j < d; j++) s += (X[i * d + j] - means[k * d + j]) ** 2
      s /= variance
      if (s < best) {
        best = s
        nearest[i] = k
      }
    }
    score[i] = best
    pValue[i] = chiSquareSf(best, d)
  }
  return { means: fromData(means, [classes, d]), variance, score, pValue, nearest }
}
