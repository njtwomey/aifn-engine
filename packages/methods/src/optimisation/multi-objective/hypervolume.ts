/**
 * The hypervolume indicator of a front, the improvement a new point would make to it, and the expected improvement
 * under a Gaussian posterior: the quantities multi-objective Bayesian optimisation maximises.
 *
 * With every objective minimised and a reference point $\rvec$ worse than every point of interest, the hypervolume
 * of a set $\Pcal$ is the volume of the region it dominates inside the box below $\rvec$ (Zitzler and Thiele, 1999):
 *
 * $$ \mathrm{HV}(\Pcal) = \operatorname{vol} \bigcup_{\pvec \in \Pcal} [\pvec, \rvec], \qquad
 * \mathrm{HVI}(\yvec) = \mathrm{HV}(\Pcal \cup \{\yvec\}) - \mathrm{HV}(\Pcal). $$
 *
 * The volume is computed exactly: by a sweep along the first objective for $M = 2$, and for $M \ge 3$ by slicing along
 * the last objective into slabs, each the $(M-1)$-dimensional volume of the points below it times its thickness (the
 * hypervolume by slicing objectives of While, Hingston, Barone and Huband, 2006). The cost grows as $n^{M-1}$, which
 * is fine for the fronts of a figure. The improvement of a point is the volume of its own box less the part of it the
 * front already dominates, which is the hypervolume of the front clipped to that box (the limit set of While,
 * Bradstreet and Barone, 2012).
 *
 * The expected improvement (EHVI; Emmerich, Giannakoglou and Naujoks, 2006) takes independent Gaussian marginals
 * $y_j \sim \Gauss(\mu_j, \sigma_j^2)$ per objective. For $M = 2$ it is exact: with the front sorted by the first
 * objective, $p_1 < \dots < p_k$ in it and descending second objectives $q_1 > \dots > q_k$, the region the front does
 * not dominate splits into vertical strips $[p_i, p_{i+1}) \times (-\infty, q_i)$ (with $p_0 = -\infty$, $q_0 = r_2$
 * and $p_{k+1} = r_1$), and the improvement is a sum over strips of a product of one function of $y_1$ and one of
 * $y_2$. Its expectation is then
 *
 * $$ \mathrm{EHVI} = \sum_{i=0}^{k} \big[\Psi_1(p_{i+1}) - \Psi_1(p_i)\big]\,\Psi_2(q_i), \qquad
 * \Psi_j(c) = \expect[(c - y_j)^+] = (c - \mu_j)\,\Phi(z) + \sigma_j\,\phi(z), \quad z = \frac{c - \mu_j}{\sigma_j}, $$
 *
 * with $\Psi_1(-\infty) = 0$: the strip form of the cell decompositions of Emmerich, Deutz and Klinkenberg (2011) and
 * Yang, Emmerich, Deutz and Bäck (2019). For $M = 1$ it is the expected improvement on $\min(r, \min \Pcal)$. For
 * $M \ge 3$ it is a Monte Carlo average of $\mathrm{HVI}$ over draws from the posterior.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import { dense, isTensor, toFlat, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
import { dominates, lexicographic, nonDominated, rowsOf, signs, type Directions } from './pareto'

const { toF64, vec } = dense

/** Options of `hypervolume` and `hypervolumeImprovement`. */
export interface HypervolumeOptions {
  /** Minimise every objective (default true), maximise every one (false), or one flag per objective. */
  minimise?: Directions
}

/** Options of `expectedHypervolumeImprovement`. */
export interface ExpectedHypervolumeOptions extends HypervolumeOptions {
  /**
   * `exact` (the default for one or two objectives) or `monte-carlo` (the only choice for three or more, and
   * available for any number, to compare with the exact value).
   */
  method?: 'exact' | 'monte-carlo'
  /** The number of posterior draws of the Monte Carlo estimate (default 4096). */
  samples?: number
  /** The stream the Monte Carlo draws come from; required by the Monte Carlo method. */
  stream?: Stream
}

/**
 * The reference point in minimisation form, checked against the number of objectives.
 *
 * @param reference The reference point, in the units given.
 * @param sign The sign of each objective, from `signs`.
 * @param where The caller's name, for error messages.
 * @returns The reference point multiplied by the signs.
 */
function referenceOf(reference: readonly number[] | Tensor, sign: Float64Array, where: string): Float64Array {
  const r = toF64(reference as never, where)
  if (r.length !== sign.length)
    throw new ShapeError(where, `${where}: a reference point of length ${r.length} for ${sign.length} objectives`)
  if (!r.every((v) => !Number.isNaN(v))) throw new DomainError(where, `${where}: the reference point is NaN`)
  return r.map((v, j) => v * sign[j])
}

/**
 * The number of objectives of a call: the reference point's length (the points must match it).
 *
 * @param reference The reference point.
 * @param where The caller's name, for error messages.
 * @returns $M$.
 */
function objectivesOf(reference: readonly number[] | Tensor, where: string): number {
  const M = isTensor(reference) ? (toFlat(reference) as ArrayLike<number>).length : reference.length
  if (M === 0) throw new ShapeError(where, `${where}: the reference point is empty`)
  return M
}

/**
 * The points strictly inside the box below the reference point that no other such point dominates, without
 * repeats: the only points that contribute volume.
 *
 * @param rows The points, every objective minimised.
 * @param r The reference point, minimised.
 * @returns The contributing points.
 */
function contributing(rows: readonly Float64Array[], r: Float64Array): Float64Array[] {
  const inside = rows.filter((p) => p.every((v, j) => v < r[j]))
  const kept = nonDominated(inside).map((i) => inside[i])
  kept.sort(lexicographic)
  return kept.filter((p, i) => i === 0 || lexicographic(kept[i - 1], p) !== 0)
}

/**
 * The exact hypervolume of points that lie strictly below the reference point, in minimisation form: a sweep for
 * one and two objectives, slicing along the last objective for more.
 *
 * @param points The points, each of the reference point's length and below it in every coordinate.
 * @param r The reference point.
 * @returns The volume of the union of the boxes $[\pvec, \rvec]$.
 */
export function volume(points: readonly ArrayLike<number>[], r: ArrayLike<number>): number {
  const n = points.length
  const d = r.length
  if (n === 0) return 0
  if (d === 1) return r[0] - Math.min(...points.map((p) => p[0]))
  if (d === 2) {
    const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
    let area = 0
    let lowest = Infinity
    for (let i = 0; i < n; i++) {
      lowest = Math.min(lowest, sorted[i][1])
      const right = i + 1 < n ? sorted[i + 1][0] : r[0]
      area += (right - sorted[i][0]) * (r[1] - lowest)
    }
    return area
  }
  // Slabs between consecutive values of the last objective: in each, the points at or below it, projected.
  const sorted = [...points].sort((a, b) => a[d - 1] - b[d - 1])
  const below = Array.from({ length: d - 1 }, (_, j) => r[j])
  let total = 0
  const slice: ArrayLike<number>[] = []
  for (let i = 0; i < n; i++) {
    const projected = Array.from({ length: d - 1 }, (_, j) => sorted[i][j])
    if (!slice.some((q) => dominates(q, projected) || lexicographic(q, projected) === 0)) {
      for (let k = slice.length - 1; k >= 0; k--) if (dominates(projected, slice[k])) slice.splice(k, 1)
      slice.push(projected)
    }
    const top = i + 1 < n ? sorted[i + 1][d - 1] : r[d - 1]
    if (top > sorted[i][d - 1]) total += (top - sorted[i][d - 1]) * volume(slice, below)
  }
  return total
}

/**
 * The hypervolume improvement of one point over a contributing front: its box's volume less the front's volume
 * clipped to the box (the limit set). Zero when the point is not strictly below the reference point.
 *
 * @param front The contributing front, minimised.
 * @param y The new point, minimised.
 * @param r The reference point.
 * @returns $\mathrm{HVI}(\yvec)$.
 */
function improvementOf(front: readonly Float64Array[], y: ArrayLike<number>, r: Float64Array): number {
  let box = 1
  for (let j = 0; j < r.length; j++) {
    if (!(y[j] < r[j])) return 0
    box *= r[j] - y[j]
  }
  const limited = front.map((p) => p.map((v, j) => Math.max(v, y[j])))
  return Math.max(0, box - volume(contributing(limited, r), r))
}

/**
 * The hypervolume of a set of points against a reference point (Zitzler and Thiele, 1999): the volume of the region
 * that the points dominate and that dominates the reference point, computed exactly (see the file comment). The
 * points need not be a front: dominated points add nothing, and so do points that are not strictly better than the
 * reference point in every objective. Throws `ShapeError` when the reference point and the points differ in length,
 * and `DomainError` for a NaN value.
 *
 * @param front The points, $n \times M$, one row per point (`paretoFront` gives a front, but any set will do).
 * @param reference The reference point $\rvec$, $M$ values: a point worse than all those of interest, often a little
 *   beyond the worst value of each objective.
 * @param options The direction of each objective, as `HypervolumeOptions`.
 * @returns The hypervolume, in the product of the objectives' units.
 *
 * @example The staircase of three points in two objectives
 * const front = [[1, 4], [2, 2], [3, 1]]
 * print('HV =', hypervolume(front, [4, 5]))
 * print('adding a dominated point changes nothing:', hypervolume([...front, [3, 3]], [4, 5]))
 *
 * @example Three objectives: two boxes that overlap
 * print('HV =', hypervolume([[0, 0, 1], [1, 1, 0]], [2, 2, 2]), ' (4 + 2 - 1)')
 */
export function hypervolume(
  front: MatrixLike,
  reference: readonly number[] | Tensor,
  options: HypervolumeOptions = {},
): number {
  const sign = signs(options.minimise ?? true, objectivesOf(reference, 'hypervolume'), 'hypervolume')
  const r = referenceOf(reference, sign, 'hypervolume')
  return volume(contributing(rowsOf(front, sign, 'hypervolume'), r), r)
}

/**
 * The hypervolume improvement of each of several points over a front: $\mathrm{HV}(\Pcal \cup \{\yvec\}) -
 * \mathrm{HV}(\Pcal)$ for each $\yvec$ on its own, exact, computed as the volume of the box $[\yvec, \rvec]$ less the
 * front's volume inside it. It is zero for a point that the front dominates or that is not strictly better than the
 * reference point. Throws `ShapeError` when the lengths differ, and `DomainError` for a NaN value.
 *
 * @param front The current front $\Pcal$, $n \times M$ (dominated points are allowed and ignored).
 * @param reference The reference point $\rvec$, $M$ values.
 * @param y The candidate points, $m \times M$.
 * @param options The direction of each objective, as `HypervolumeOptions`.
 * @returns The $m$ improvements, a vector.
 *
 * @example A point that fills a step, one that is dominated, one beyond the reference
 * const front = [[1, 4], [2, 2], [3, 1]]
 * print('HVI =', hypervolumeImprovement(front, [4, 5], [[1.5, 3], [2.5, 2.5], [5, 0]]))
 */
export function hypervolumeImprovement(
  front: MatrixLike,
  reference: readonly number[] | Tensor,
  y: MatrixLike,
  options: HypervolumeOptions = {},
): Tensor {
  const where = 'hypervolumeImprovement'
  const sign = signs(options.minimise ?? true, objectivesOf(reference, where), where)
  const r = referenceOf(reference, sign, where)
  const P = contributing(rowsOf(front, sign, where), r)
  return vec(Float64Array.from(rowsOf(y, sign, where), (p) => improvementOf(P, p, r)))
}

/**
 * $\Psi(c) = \expect[(c - y)^+]$ for $y \sim \Gauss(\mu, \sigma^2)$: the expected amount by which $y$ falls below $c$,
 * $\max(c - \mu, 0)$ when $\sigma = 0$ and 0 at $c = -\infty$.
 *
 * @param c The level.
 * @param mu The mean $\mu$.
 * @param sd The standard deviation $\sigma \ge 0$.
 * @returns $\Psi(c)$.
 */
function shortfall(c: number, mu: number, sd: number): number {
  if (c === -Infinity) return 0
  if (sd === 0) return Math.max(c - mu, 0)
  const z = (c - mu) / sd
  return Math.max(0, (c - mu) * normalCdf(z) + sd * normalPdf(z))
}

/**
 * The expected hypervolume improvement of candidates whose objectives have independent Gaussian posteriors (Emmerich,
 * Giannakoglou and Naujoks, 2006): $\expect[\mathrm{HVI}(\yvec)]$ with $y_j \sim \Gauss(\mu_j, \sigma_j^2)$. Exact for
 * one and two objectives (the strip decomposition of the file comment, as Emmerich et al., 2011, and Yang et al.,
 * 2019), and a Monte Carlo average for three or more, or when asked. The Monte Carlo estimate uses the same standard
 * normal draws for every candidate (common random numbers), so it varies smoothly from one candidate to the next.
 * Throws `ShapeError` when the shapes differ, and `DomainError` for a negative standard deviation, a NaN value, fewer
 * than one sample, or the Monte Carlo method without a stream.
 *
 * @param mean The posterior means, $m \times M$: one row per candidate, one column per objective.
 * @param sd The posterior standard deviations: an $m \times M$ matrix like `mean`, or one number for all.
 * @param front The current front $\Pcal$, $n \times M$ (dominated points are allowed and ignored).
 * @param reference The reference point $\rvec$, $M$ values.
 * @param options The directions, the method, and the samples and stream of the Monte Carlo method, as
 *   `ExpectedHypervolumeOptions`.
 * @returns The $m$ expected improvements, a vector.
 *
 * @example Exact for two objectives, against a Monte Carlo estimate
 * const front = [[1, 4], [2, 2], [3, 1]]
 * const mean = [[1.5, 3], [2.5, 2.5], [0.5, 0.5]]
 * print('exact EHVI:', expectedHypervolumeImprovement(mean, 0.5, front, [4, 5]))
 * const mc = expectedHypervolumeImprovement(mean, 0.5, front, [4, 5], { method: 'monte-carlo', stream: stream(1) })
 * print('Monte Carlo:', mc)
 * print('with sd = 0 it is the HVI of the mean:', expectedHypervolumeImprovement(mean, 0, front, [4, 5]))
 *
 * @example Three objectives, by Monte Carlo
 * const front = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
 * const mean = [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
 * print('EHVI:', expectedHypervolumeImprovement(mean, 0.3, front, [2, 2, 2], { samples: 2000, stream: stream(0) }))
 */
export function expectedHypervolumeImprovement(
  mean: MatrixLike,
  sd: MatrixLike | number,
  front: MatrixLike,
  reference: readonly number[] | Tensor,
  options: ExpectedHypervolumeOptions = {},
): Tensor {
  const where = 'expectedHypervolumeImprovement'
  const M = objectivesOf(reference, where)
  const sign = signs(options.minimise ?? true, M, where)
  const r = referenceOf(reference, sign, where)
  const mu = rowsOf(mean, sign, where)
  const s = typeof sd === 'number' ? mu.map(() => new Float64Array(M).fill(sd)) : rowsOf(sd, null, where)
  if (s.length !== mu.length || s.some((row) => row.length !== M))
    throw new ShapeError(where, `${where}: the standard deviations must be a number or of the shape of the means`)
  if (s.some((row) => row.some((v) => !(v >= 0))))
    throw new DomainError(where, `${where}: standard deviations must be non-negative`)
  const P = contributing(rowsOf(front, sign, where), r)
  const method = options.method ?? (M <= 2 ? 'exact' : 'monte-carlo')
  if (method === 'exact' && M > 2)
    throw new DomainError(where, `${where}: the exact method is for one or two objectives; use 'monte-carlo'`)

  if (method === 'exact') {
    if (M === 1) {
      const level = P.length > 0 ? Math.min(r[0], P[0][0]) : r[0]
      return vec(Float64Array.from(mu, (m, i) => shortfall(level, m[0], s[i][0])))
    }
    // The strips: corners p_0 = -inf < p_1 < ... < p_k < p_{k+1} = r_1, heights q_0 = r_2 > q_1 > ... > q_k.
    const p = [-Infinity, ...P.map((v) => v[0]), r[0]]
    const q = [r[1], ...P.map((v) => v[1])]
    return vec(
      Float64Array.from(mu, (m, c) => {
        let total = 0
        for (let i = 0; i < q.length; i++) {
          const width = shortfall(p[i + 1], m[0], s[c][0]) - shortfall(p[i], m[0], s[c][0])
          total += width * shortfall(q[i], m[1], s[c][1])
        }
        return Math.max(0, total)
      }),
    )
  }

  const samples = options.samples ?? 4096
  if (!(Number.isInteger(samples) && samples >= 1))
    throw new DomainError(where, `${where}: samples must be a positive integer`)
  if (!options.stream) throw new DomainError(where, `${where}: the Monte Carlo method needs a stream`)
  const z = toFlat(normals(options.stream, [samples, M])) as ArrayLike<number>
  const y = new Float64Array(M)
  return vec(
    Float64Array.from(mu, (m, c) => {
      let total = 0
      for (let k = 0; k < samples; k++) {
        for (let j = 0; j < M; j++) y[j] = m[j] + s[c][j] * z[k * M + j]
        total += improvementOf(P, y, r)
      }
      return total / samples
    }),
  )
}
