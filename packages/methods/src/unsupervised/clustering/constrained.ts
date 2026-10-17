/**
 * Semi-supervised clustering with pairwise side information: must-link and cannot-link constraints, labelled seeds,
 * and a metric learned from must-link groups.
 *
 * `constraintClosure` joins must-linked points into groups (the transitive closure), lifts cannot-links to groups and
 * reports a contradiction (a cannot-link inside a group). `constrainedKMeansSteps` is Lloyd's algorithm with a
 * constrained assignment step that visits the points in a fixed order. With `rule: 'cop'` (COP-k-means; Wagstaff,
 * Cardie, Rogers and Schroedl, 2001, Table 1, closure in §3.1) a point goes to the nearest centroid whose cluster
 * breaks no constraint with the points already assigned in this pass, and the pass fails when no cluster is legal.
 * With `rule: 'pck'` (PCK-means; Basu, Banerjee and Mooney, 2004, §3, uniform weights) a point goes to the cluster
 * that minimises $\tfrac12\lVert \xvec_i - \muvec_h \rVert^2$ plus `weight` for every must-link partner assigned
 * elsewhere and every cannot-link partner in $h$; the objective it lowers is half the inertia plus the cost of the
 * violated constraints. Points with a `fixed` cluster (labelled seeds, the constrained k-means of Basu, Banerjee and
 * Mooney, 2002) always go to it, whatever the constraints say. `seededCentroids` gives the starting centroids of
 * seeded k-means, the means of the labelled seeds of each cluster. `relevantComponentMetric` learns a Mahalanobis
 * metric from must-link groups (relevant component analysis; Bar-Hillel, Hertz, Shental and Weinshall, 2005): the
 * within-group covariance $\hat\Cmat$ and the whitening map $\xvec \mapsto \hat\Cmat^{-1/2}\xvec$, after which plain
 * k-means measures distance in units of the within-group spread.
 *
 * Differences from the papers: COP-k-means here keeps the partial assignment of a failed pass (points with no legal
 * cluster get label $-1$) instead of returning the empty partition; PCK-means starts from given centroids rather than
 * from the must-link neighbourhoods. Both are order-dependent by design.
 */

import type { MatrixLike, Status, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { unionFind, unionFindRoot, unite } from 'aifn-compute/graph'
import { eigh, symmetricInverseSqrt } from 'aifn-compute/numerics/linalg'

/** A pair of point indices, a must-link or a cannot-link constraint. */
export type ConstraintPair = readonly [number, number]

/**
 * The rows of a matrix as plain arrays.
 *
 * @param m The matrix, $n \times d$: a tensor or nested arrays.
 * @param where The caller's name, for error messages.
 * @returns $n$ arrays of $d$ numbers.
 */
function rowsOf(m: MatrixLike, where: string): number[][] {
  const { data, m: n, n: d } = dense.toMatrixF64(m, where)
  return Array.from({ length: n }, (_, i) => Array.from(data.subarray(i * d, i * d + d)))
}

/** The transitive closure of must-links and the cannot-links between the resulting groups. */
export interface ConstraintClosure {
  /** The group of each point (points in no must-link form a group of their own), numbered by first member. */
  group: Int32Array
  /** The members of each group, in increasing order. */
  members: number[][]
  /** Must-link partners of each point after closure (every other member of its group). */
  mustLink: number[][]
  /** Cannot-link partners of each point after closure (every member of a group it has a cannot-link with). */
  cannotLink: number[][]
  /** False when a cannot-link joins two points of one must-link group. */
  consistent: boolean
}

/**
 * Join must-linked points into groups by union-find and lift cannot-links to whole groups (Wagstaff et al., 2001,
 * §3.1). A cannot-link inside a group makes the constraints inconsistent, which is reported, not thrown. Throws
 * `DomainError` for an index outside $0, \dots, n - 1$.
 *
 * @param n The number of points.
 * @param mustLink Pairs that must share a cluster.
 * @param cannotLink Pairs that must not.
 * @returns The groups, the closed constraint lists of each point, and whether the constraints are consistent.
 *
 * @example Must-link 2–3 and cannot-link 1–3 imply cannot-link 1–2
 * const c = constraintClosure(4, [[2, 3]], [[1, 3]])
 * print('groups:', c.members)
 * print('cannot-link partners of 2:', c.cannotLink[2], ' consistent:', c.consistent)
 * print('a contradiction:', constraintClosure(2, [[0, 1]], [[0, 1]]).consistent)
 */
export function constraintClosure(
  n: number,
  mustLink: readonly ConstraintPair[],
  cannotLink: readonly ConstraintPair[],
): ConstraintClosure {
  const inRange = (a: number) => Number.isInteger(a) && a >= 0 && a < n
  const uf = unionFind(n)
  for (const [a, b] of mustLink) {
    if (!inRange(a) || !inRange(b)) throw new DomainError('constraintClosure', 'constraintClosure: index out of range')
    unite(uf, a, b)
  }
  const roots = new Map<number, number>()
  const group = new Int32Array(n)
  const members: number[][] = []
  for (let i = 0; i < n; i++) {
    const r = unionFindRoot(uf, i)
    if (!roots.has(r)) {
      roots.set(r, members.length)
      members.push([])
    }
    group[i] = roots.get(r)!
    members[group[i]].push(i)
  }
  const apart = new Set<string>()
  let consistent = true
  for (const [a, b] of cannotLink) {
    if (!inRange(a) || !inRange(b)) throw new DomainError('constraintClosure', 'constraintClosure: index out of range')
    const ga = group[a]
    const gb = group[b]
    if (ga === gb) consistent = false
    else apart.add(ga < gb ? `${ga},${gb}` : `${gb},${ga}`)
  }
  const mustLinkOf = Array.from({ length: n }, (_, i) => members[group[i]].filter((j) => j !== i))
  const apartOf: number[][] = members.map(() => [])
  for (const key of apart) {
    const [ga, gb] = key.split(',').map(Number)
    apartOf[ga].push(gb)
    apartOf[gb].push(ga)
  }
  const cannotLinkOf = Array.from({ length: n }, (_, i) => apartOf[group[i]].flatMap((g) => members[g]))
  return { group, members, mustLink: mustLinkOf, cannotLink: cannotLinkOf, consistent }
}

/** Options of `constrainedKMeansSteps`. */
export interface ConstrainedKMeansOptions {
  /** The number of clusters $k$. */
  k: number
  /** Pairs that must share a cluster. */
  mustLink?: readonly ConstraintPair[]
  /** Pairs that must not. */
  cannotLink?: readonly ConstraintPair[]
  /** `cop`: constraints are hard (COP-k-means); `pck`: each violation costs `weight` (PCK-means). */
  rule: 'cop' | 'pck'
  /** PCK-means' cost per violated constraint, in units of squared distance (default 1). */
  weight?: number
  /**
   * A cluster for each point, or $-1$ for none: labelled seeds that always go to their cluster, even against a
   * constraint.
   */
  fixed?: ArrayLike<number>
  /** The order in which the assignment step visits the points (default $0, 1, \dots, n - 1$). */
  order?: readonly number[]
}

/** How `constrainedKMeansSteps` starts: the centroids. */
export interface ConstrainedKMeansStart {
  /** The starting centroids, $k \times d$, as a tensor or nested arrays (e.g. from `seededCentroids`). */
  centroids: MatrixLike
}

/** One state of constrained k-means. */
export interface ConstrainedKMeansState extends Status {
  /** Lloyd iterations done; state 0 is the assignment to the starting centroids. */
  t: number
  /** Centroids, $k \times d$; an empty cluster keeps its centroid. */
  centroids: Tensor
  /** The cluster of each point; $-1$ for a point COP-k-means could not place. */
  labels: Int32Array
  /** $\sum_i \lVert \xvec_i - \muvec_{z_i} \rVert^2$ over the placed points. */
  inertia: number
  /** Given constraints (before closure) broken by the assignment. */
  violations: number
  /**
   * PCK-means' objective, $\tfrac12$ the inertia plus `weight` per broken constraint after closure (each pair once);
   * it never rises from one state to the next under `rule: 'pck'`.
   */
  objective: number
  /** COP-k-means found a point with no legal cluster (the pass stopped there). */
  failed: boolean
  /** The assignment did not change, or COP-k-means failed. */
  converged: boolean
}

/**
 * Lloyd's algorithm with a constrained assignment step (see the file comment): COP-k-means or PCK-means, with labelled
 * seeds. `init` takes the starting centroids and draws nothing. Under `rule: 'pck'` a point not yet visited in a pass
 * keeps its label from the previous pass when its partners are checked, so every move lowers the objective. Throws
 * `DomainError` for a bad $k$, rule or visiting order, and `ShapeError` for starting centroids that are not
 * $k \times d$.
 *
 * @param x The data, $n \times d$, as a tensor or nested arrays.
 * @param options $k$, the constraints, the rule, PCK-means' weight, fixed seeds and the visiting order.
 * @returns The algorithm; its states carry the centroids, labels, inertia, violations and failure flag.
 *
 * @example The note's six points: a must-link moves 6 to the left cluster
 * const x = [[0], [1], [4], [6], [9], [10]]
 * const start = { centroids: [[0], [10]] }
 * const plain = run(constrainedKMeansSteps(x, { k: 2, rule: 'pck', weight: 0 }), start, 20)
 * const cop = run(constrainedKMeansSteps(x, { k: 2, rule: 'cop', mustLink: [[2, 3]] }), start, 20)
 * print('k-means:', plain.labels, ' inertia:', plain.inertia)
 * print('COP, must-link 4–6:', cop.labels, ' inertia:', cop.inertia)
 */
export function constrainedKMeansSteps(
  x: MatrixLike,
  options: ConstrainedKMeansOptions,
): Algorithm<ConstrainedKMeansStart, ConstrainedKMeansState> {
  const OP = 'constrainedKMeansSteps'
  const rows = rowsOf(x, OP)
  const n = rows.length
  const d = rows[0]?.length ?? 0
  const { k, rule } = options
  if (!(Number.isInteger(k) && k >= 1)) throw new DomainError(OP, `${OP}: k must be a positive integer`)
  if (rule !== 'cop' && rule !== 'pck') throw new DomainError(OP, `${OP}: rule must be 'cop' or 'pck'`)
  const weight = options.weight ?? 1
  const closure = constraintClosure(n, options.mustLink ?? [], options.cannotLink ?? [])
  const order = options.order ?? Array.from({ length: n }, (_, i) => i)
  const fixed = options.fixed
  if (order.length !== n) throw new DomainError(OP, `${OP}: order must visit every point once`)

  const dist2 = (i: number, c: readonly number[]) => {
    let s = 0
    for (let j = 0; j < d; j++) s += (rows[i][j] - c[j]) ** 2
    return s
  }

  // COP-k-means checks the points already placed in this pass (Wagstaff et al.); PCK-means is a greedy pass in
  // which a point not yet visited keeps its label from the previous pass, so every move lowers the objective.
  const assign = (centroids: readonly (readonly number[])[], previous: Int32Array | null) => {
    const labels = new Int32Array(n).fill(-2) // -2: not yet visited in this pass
    const partner = (j: number) => (labels[j] !== -2 ? labels[j] : rule === 'pck' && previous ? previous[j] : -2)
    let failed = false
    for (const i of order) {
      if (failed) {
        labels[i] = -1
        continue
      }
      const seed = fixed ? fixed[i] : -1
      if (seed >= 0) {
        labels[i] = seed
        continue
      }
      let best = -1
      let bestCost = Infinity
      for (let h = 0; h < k; h++) {
        let broken = 0
        for (const j of closure.mustLink[i]) {
          const l = partner(j)
          if (l >= 0 && l !== h) broken++
        }
        for (const j of closure.cannotLink[i]) if (partner(j) === h) broken++
        const cost =
          rule === 'pck'
            ? 0.5 * dist2(i, centroids[h]) + weight * broken
            : broken > 0
              ? Infinity
              : dist2(i, centroids[h])
        if (cost < bestCost) {
          best = h
          bestCost = cost
        }
      }
      if (best < 0) {
        failed = true
        labels[i] = -1
      } else labels[i] = best
    }
    return { labels, failed }
  }

  const given = { mustLink: options.mustLink ?? [], cannotLink: options.cannotLink ?? [] }
  const describe = (
    t: number,
    centroids: number[][],
    labels: Int32Array,
    failed: boolean,
    previous: Int32Array | null,
  ): ConstrainedKMeansState => {
    let inertia = 0
    for (let i = 0; i < n; i++) if (labels[i] >= 0) inertia += dist2(i, centroids[labels[i]])
    let violations = 0
    for (const [a, b] of given.mustLink) if (labels[a] >= 0 && labels[b] >= 0 && labels[a] !== labels[b]) violations++
    for (const [a, b] of given.cannotLink) if (labels[a] >= 0 && labels[a] === labels[b]) violations++
    let closed = 0
    for (let i = 0; i < n; i++) {
      for (const j of closure.mustLink[i])
        if (j > i && labels[i] >= 0 && labels[j] >= 0 && labels[i] !== labels[j]) closed++
      for (const j of closure.cannotLink[i]) if (j > i && labels[i] >= 0 && labels[i] === labels[j]) closed++
    }
    let same = previous !== null
    if (previous) for (let i = 0; i < n; i++) if (previous[i] !== labels[i]) same = false
    return {
      t,
      centroids: fromData(Float64Array.from(centroids.flat()), [k, d]),
      labels,
      inertia,
      violations,
      objective: 0.5 * inertia + weight * closed,
      failed,
      converged: failed || same,
    }
  }

  const update = (centroids: readonly (readonly number[])[], labels: Int32Array) =>
    centroids.map((c, h) => {
      const sum = new Array<number>(d).fill(0)
      let count = 0
      for (let i = 0; i < n; i++)
        if (labels[i] === h) {
          count++
          for (let j = 0; j < d; j++) sum[j] += rows[i][j]
        }
      return count > 0 ? sum.map((v) => v / count) : [...c] // an empty cluster keeps its centroid
    })

  return {
    name: rule === 'cop' ? 'cop-k-means' : 'pck-means',
    init(start, _s: Stream) {
      const c = dense.toMatrixF64(start.centroids, OP)
      if (c.m !== k || c.n !== d)
        throw new ShapeError(OP, `${OP}: k = ${k} starting centroids of dimension ${d} are needed`)
      const centroids = rowsOf(start.centroids, OP)
      const { labels, failed } = assign(centroids, null)
      return describe(0, centroids, labels, failed, null)
    },
    step(state) {
      const centroids = update(rowsOf(state.centroids, OP), state.labels)
      const { labels, failed } = assign(centroids, state.labels)
      return describe(state.t + 1, centroids, labels, failed, state.labels)
    },
  }
}

/**
 * Seeded k-means' starting centroids: the mean of the labelled seeds of each cluster (Basu, Banerjee and Mooney,
 * 2002). A cluster with no seed takes `fallback[h]`.
 *
 * @param x The data, $n \times d$.
 * @param seeds A cluster $0, \dots, k - 1$ for each seed point, $-1$ for the others.
 * @param k The number of clusters.
 * @param fallback Centroids for clusters without seeds, $k \times d$.
 * @returns The $k$ starting centroids, one array of $d$ numbers each.
 *
 * @example Two seeded clusters and one from the fallback
 * const x = [[0], [1], [9], [10], [5]]
 * print(seededCentroids(x, [0, 0, 1, 1, -1], 3, [[0], [0], [5]]))
 */
export function seededCentroids(
  x: MatrixLike,
  seeds: ArrayLike<number>,
  k: number,
  fallback: readonly (readonly number[])[],
): number[][] {
  const rows = rowsOf(x, 'seededCentroids')
  const d = rows[0]?.length ?? 0
  return Array.from({ length: k }, (_, h) => {
    const sum = new Array<number>(d).fill(0)
    let count = 0
    for (let i = 0; i < rows.length; i++)
      if (seeds[i] === h) {
        count++
        for (let j = 0; j < d; j++) sum[j] += rows[i][j]
      }
    return count > 0 ? sum.map((v) => v / count) : [...fallback[h]]
  })
}

/** A Mahalanobis metric learned from must-link groups. */
export interface RelevantComponentMetric {
  /** The within-group covariance $\hat\Cmat$, $d \times d$. */
  covariance: number[][]
  /** The whitening matrix $\Wmat = \hat\Cmat^{-1/2}$, $d \times d$ (symmetric). */
  whitening: number[][]
  /** The rows mapped by $\Wmat$, $n \times d$. */
  transformed: number[][]
}

/**
 * Relevant component analysis (Bar-Hillel et al., 2005): with chunklets $G_1, \dots, G_m$ (groups of points known to
 * share a class, of two or more points) and their means $\hat\muvec_j$, the within-chunklet covariance is
 * $\hat\Cmat = \frac{1}{p}\sum_j\sum_{i \in G_j}(\xvec_i - \hat\muvec_j)(\xvec_i - \hat\muvec_j)^\top$, $p$ the number
 * of points in chunklets, and the learned distance is $(\xvec - \yvec)^\top\hat\Cmat^{-1}(\xvec - \yvec)$, the
 * Euclidean distance after $\xvec \mapsto \hat\Cmat^{-1/2}\xvec$. The whitening is `symmetricInverseSqrt` of
 * `aifn-compute/numerics/linalg` with its eigenvalues floored at `ridge` times the largest. Throws `DomainError` when
 * no chunklet has two or more points, and `NumericalError` ('degenerate') when every chunklet's points coincide.
 *
 * @param x The data, $n \times d$.
 * @param groups The chunklets (index lists, e.g. the `members` of `constraintClosure`); groups of one point are
 *   ignored.
 * @param ridge The relative floor on the eigenvalues of $\hat\Cmat$ (default $10^{-6}$).
 * @returns The covariance, the whitening matrix and the whitened rows.
 *
 * @example Whitening twice leaves the identity
 * const x = [[0, 0], [2, 0.1], [0, 1], [2, 1.2], [5, 0], [7, 0.3]]
 * const groups = [[0, 1], [2, 3], [4, 5]]
 * const m = relevantComponentMetric(x, groups)
 * print('chunklet covariance:', m.covariance)
 * print('after whitening:', relevantComponentMetric(m.transformed, groups).covariance)
 */
export function relevantComponentMetric(
  x: MatrixLike,
  groups: readonly (readonly number[])[],
  ridge = 1e-6,
): RelevantComponentMetric {
  const OP = 'relevantComponentMetric'
  const rows = rowsOf(x, OP)
  const d = rows[0]?.length ?? 0
  const C = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  let p = 0
  for (const g of groups) {
    if (g.length < 2) continue
    const mean = new Array<number>(d).fill(0)
    for (const i of g) for (let j = 0; j < d; j++) mean[j] += rows[i][j] / g.length
    for (const i of g) {
      p++
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += (rows[i][a] - mean[a]) * (rows[i][b] - mean[b])
    }
  }
  if (p === 0) throw new DomainError(OP, `${OP}: no chunklet of two or more points`)
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= p
  const top = Math.max(...dense.data(eigh(fromData(Float64Array.from(C.flat()), [d, d])).values))
  if (!(top > 0)) throw new NumericalError(OP, `${OP}: the chunklet covariance is zero`, 'degenerate')
  const W = rowsOf(symmetricInverseSqrt(C, { floor: ridge * top }), OP)
  const transformed = rows.map((r) => W.map((w) => w.reduce((s, v, j) => s + v * r[j], 0)))
  return { covariance: C, whitening: W, transformed }
}
