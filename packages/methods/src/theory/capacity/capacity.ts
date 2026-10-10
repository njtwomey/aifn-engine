/**
 * Capacity of hypothesis classes on a finite sample. A class shatters points when it realises every one of their $2^n$
 * labellings; the VC dimension is the size of the largest set it shatters (Vapnik and Chervonenkis, 1971). Here the
 * labellings a class realises are enumerated exactly for three classes on the plane or the line:
 *
 * - half-planes $\sgn(\wvec^\top\xvec + b)$: a labelling is realisable when the linear program that seeks $\wvec$
 *   and $b$ with $y_i(\wvec^\top\xvec_i + b) \ge 1$ for every $i$ is feasible (VC dimension 3 in the plane);
 * - axis-aligned rectangles labelling their inside positive: realisable when the bounding box of the positives holds no
 *   negative (VC dimension 4);
 * - intervals on the line (first coordinate), positive inside: the same test in one dimension (VC dimension 2).
 *
 * The empirical Rademacher complexity
 * $\hat{\Rcal}(\Hcal) = \expect_{\sigmavec} \sup_h \frac{1}{n} \sum_i \sigma_i h(\xvec_i)$ of the
 * realised labellings (as $\pm 1$) is estimated by Monte Carlo over random signs $\sigmavec$, against Massart's
 * finite-class bound $\sqrt{2 \ln \lvert \Hcal \rvert / n}$.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, type MatrixLike } from 'aifn-compute/foundation/tensor'
import { linprog } from 'aifn-compute/optim/programming'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A hypothesis class of the demonstrations: half-planes or axis-aligned rectangles in the plane, or intervals on the
 * line.
 */
export type ShatterClass = 'half-planes' | 'rectangles' | 'intervals'

/**
 * Whether a $\pm 1$ labelling of points $[n, 2]$ is realisable by the class: by a feasibility linear program for
 * half-planes, and by the bounding box (closed, so a negative on its edge is inside) of the positives for rectangles
 * and intervals. A labelling with no positives (or, for half-planes, no negatives) is always realisable.
 *
 * @param points The points, an $n \times 2$ matrix with one point per row. Intervals read the first column only, but
 *   the rows are still read as pairs.
 * @param labels One label per point: positive when above 0, negative otherwise.
 * @param family The hypothesis class.
 * @returns Whether some hypothesis of the class labels the points this way.
 *
 * @example XOR defeats half-planes and rectangles; a diagonal pair does not defeat half-planes
 * const square = [[0, 0], [1, 1], [1, 0], [0, 1]]
 * print('XOR by half-planes:', realisable(square, [1, 1, -1, -1], 'half-planes'))
 * print('XOR by rectangles:', realisable(square, [1, 1, -1, -1], 'rectangles'))
 * print('one corner by half-planes:', realisable(square, [1, -1, -1, -1], 'half-planes'))
 */
export function realisable(points: MatrixLike, labels: ArrayLike<number>, family: ShatterClass): boolean {
  const { data: X, m: n, n: d } = dense.toMatrixF64(points, 'realisable')
  const pos = Array.from({ length: n }, (_, i) => i).filter((i) => labels[i] > 0)
  const neg = Array.from({ length: n }, (_, i) => i).filter((i) => labels[i] <= 0)
  if (family === 'half-planes') {
    if (pos.length === 0 || neg.length === 0) return true
    // Variables (w₁, w₂, b), free; constraints −yᵢ(w·xᵢ + b) ≤ −1; objective 0 (feasibility).
    const A = Array.from({ length: n }, (_, i) => {
      const y = labels[i] > 0 ? 1 : -1
      return [-y * X[d * i], -y * X[d * i + 1], -y]
    })
    const r = linprog({ c: [0, 0, 0], A_ub: A, b_ub: new Array<number>(n).fill(-1), bounds: [null, null] })
    return r.status === 'optimal'
  }
  if (pos.length === 0) return true
  const dims = family === 'rectangles' ? 2 : 1
  const lo = [Infinity, Infinity]
  const hi = [-Infinity, -Infinity]
  for (const i of pos)
    for (let k = 0; k < dims; k++) {
      lo[k] = Math.min(lo[k], X[d * i + k])
      hi[k] = Math.max(hi[k], X[d * i + k])
    }
  return !neg.some((i) => {
    for (let k = 0; k < dims; k++) if (X[d * i + k] < lo[k] || X[d * i + k] > hi[k]) return false
    return true
  })
}

/**
 * Every labelling of the points (bit $i$ of the index is point $i$ positive), with whether the class realises it, by
 * `realisable` on each. More than 14 points throws `DomainError`.
 *
 * @param points The points, an $n \times 2$ matrix with one point per row ($n \le 14$).
 * @param family The hypothesis class.
 * @returns The $2^n$ `labellings` ($\pm 1$ per point), whether each is `realised`, the `count` realised, and whether
 *   the points are `shattered` (every labelling realised).
 *
 * @example Half-planes shatter three points but not four, so their VC dimension is 3
 * const three = shatteringTable([[0, 0], [1, 0], [0, 1]], 'half-planes')
 * print('3 points:', three.count, 'of 8 realised, shattered:', three.shattered)
 * const four = shatteringTable([[0, 0], [1, 0], [0, 1], [1, 1]], 'half-planes')
 * print('4 points:', four.count, 'of 16 realised, shattered:', four.shattered)
 *
 * @example Rectangles shatter four points in a diamond
 * const diamond = shatteringTable([[0, 1], [1, 0], [0, -1], [-1, 0]], 'rectangles')
 * print('diamond:', diamond.count, 'of 16 realised, shattered:', diamond.shattered)
 */
export function shatteringTable(
  points: MatrixLike,
  family: ShatterClass,
): { labellings: Int8Array[]; realised: boolean[]; count: number; shattered: boolean } {
  const n = dense.toMatrixF64(points, 'shatteringTable').m
  if (n > 14) throw new DomainError('shatteringTable', 'shatteringTable: at most 14 points (2ⁿ labellings)')
  const labellings: Int8Array[] = []
  const realised: boolean[] = []
  for (let mask = 0; mask < 1 << n; mask++) {
    const y = Int8Array.from({ length: n }, (_, i) => ((mask >> i) & 1 ? 1 : -1))
    labellings.push(y)
    realised.push(realisable(points, y, family))
  }
  const count = realised.filter(Boolean).length
  return { labellings, realised, count, shattered: count === 1 << n }
}

/**
 * The empirical Rademacher complexity of a finite set of $\pm 1$ labellings on $n$ points,
 * $\expect_{\sigmavec} \max_h \frac{1}{n} \sum_i \sigma_i h_i$, by `draws` Monte Carlo sign vectors; with Massart's
 * bound $\sqrt{2 \ln \lvert \Hcal \rvert / n}$ and the running estimate after each draw. An empty set of
 * labellings throws `DomainError`.
 *
 * @param s The stream; draw $d$ uses `child(s, 'signs', d)`, so `s` is not advanced.
 * @param labellings The hypotheses $\Hcal$, each its $\pm 1$ labels $h_i$ of the same $n$ points.
 * @param draws The number of random sign vectors $\sigmavec$.
 * @returns The `estimate`, its Monte Carlo `standardError`, the `massart` bound, and the `running` estimate after each
 *   draw.
 *
 * @example Half-planes on six random points, against Massart's bound
 * const points = uniform(stream(1), 0, 1, { shape: [6, 2] })
 * const { labellings, realised } = shatteringTable(points, 'half-planes')
 * const H = labellings.filter((_, i) => realised[i])
 * const r = empiricalRademacher(stream(2), H, 500)
 * print('|H| =', H.length, ' of', labellings.length)
 * print('Rademacher =', r.estimate, '+/-', r.standardError, ' Massart bound =', r.massart)
 *
 * @example One hypothesis has complexity near 0; every labelling gives complexity 1
 * print('one:', empiricalRademacher(stream(0), [[1, 1, -1, -1]], 200).estimate)
 * const all = shatteringTable([[0, 1], [1, 0], [0, -1], [-1, 0]], 'rectangles').labellings
 * print('all 16:', empiricalRademacher(stream(0), all, 200).estimate)
 */
export function empiricalRademacher(
  s: Stream,
  labellings: readonly ArrayLike<number>[],
  draws = 2000,
): { estimate: number; standardError: number; massart: number; running: Float64Array } {
  if (labellings.length === 0)
    throw new DomainError('empiricalRademacher', 'empiricalRademacher: needs at least one hypothesis')
  const n = labellings[0].length
  const running = new Float64Array(draws)
  let sum = 0
  let sumSq = 0
  for (let d = 0; d < draws; d++) {
    const u = units(child(s, 'signs', d), n)
    let best = -Infinity
    for (const h of labellings) {
      let v = 0
      for (let i = 0; i < n; i++) v += (u[i] < 0.5 ? -1 : 1) * h[i]
      if (v > best) best = v
    }
    sum += best / n
    sumSq += (best / n) ** 2
    running[d] = sum / (d + 1)
  }
  const estimate = sum / draws
  const variance = Math.max(0, sumSq / draws - estimate * estimate)
  return {
    estimate,
    standardError: Math.sqrt(variance / draws),
    massart: Math.sqrt((2 * Math.log(labellings.length)) / n),
    running,
  }
}
