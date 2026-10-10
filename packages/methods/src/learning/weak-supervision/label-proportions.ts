/**
 * Label propagation for learning with label proportions (LP-LLP; Poyiadzi, Santos-Rodriguez and Twomey 2018, MLSP).
 * Points $\Xmat = \{\xvec_1, \dots, \xvec_n\}$ are split into disjoint bags $\Bcal_k$, and only each bag's class
 * proportions $\pivec_k = (\pi_{k,1}, \dots, \pi_{k,c})$ are known. Each point starts from its bag's proportions as a
 * soft label, $\hat{\yvec}_i = \pivec_k$ for $\xvec_i \in \Bcal_k$; the scores then spread over a similarity graph
 * and are pulled back onto the bags' class masses, in turn:
 *
 * 1. $W_{ij} = \exp(-\gamma \lVert \xvec_i - \xvec_j \rVert^2)$, $W_{ii} = 0$, and $\Smat = \Dmat^{-1}\Wmat$
 *    (Algorithm 1 of the paper);
 * 2. propagate: $\Fmat \leftarrow (1 - \alpha)(\Imat - \alpha\Smat)^{-1}\Fmat$ (the limit of label propagation
 *    started from $\Fmat$; Algorithm 2 step 2);
 * 3. project: alternating projections (Boyd and Dattorro 2003) between the rows on the probability simplex and the
 *    bags' class masses $\sum_{i \in \Bcal_k} F_{ic} = n_k \pi_{k,c}$ (the relaxed constraint $\Amat\fvec = \bvec$ of
 *    Eq. 2);
 * 4. repeat 2–3 until $\Fmat$ stops changing; label each point by its largest score ($\sgn(f^* - 0.5)$ for two
 *    classes).
 *
 * Readings of the paper (logged in the progress file): Algorithm 2 writes $(\Imat - \alpha\Smat)^{-1}$ without the
 * $(1 - \alpha)$ of Algorithm 1; with it the propagation is an average (row-stochastic), so the scores stay on the
 * simplex scale. `normalise: false` uses the unscaled form. The two-class problem with scores $\fvec \in [0, 1]^n$ is
 * the case $c = 2$ of the multiclass form (rows $(1 - f, f)$): the simplex projection of such a row clips $f$ to
 * $[0, 1]$, and the bag constraint shifts both columns by opposite amounts.
 *
 * Also the paper's baselines and two classical ones, each transductive (trained on every point, labels for every
 * point): inverse calibration (InvCal; Rüping 2010), alternating $\propto$SVM (Yu et al. 2013), MeanMap (Quadrianto
 * et al. 2009); the proportion-loss classifier is `proportionClassifier`. A bag index of $-1$ marks a point in no bag.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  child,
  integers,
  normal,
  permutation,
  stream as makeStream,
  uniform,
  type Stream,
} from 'aifn-compute/foundation/random'
import {
  dense,
  fromData,
  logsumexp,
  matmul,
  mul,
  reshape,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { pointAffinity, randomWalkMatrix, spreadingResolvent } from 'aifn-compute/graph/propagation'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { alternatingProjections, projectGroupSums, projectSimplexRows } from 'aifn-compute/optim/proximal'
import { boxQuadprog, quadprog } from 'aifn-compute/optim/programming'
import { lbfgs } from 'aifn-compute/optim/second-order'
import { trace } from 'aifn-compute/foundation/trace'
import { proportionClassifier } from './classifiers'

// ── Bags ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Bags and their class proportions as the algorithms read them. */
export interface BagProportions {
  /** Each point's bag ($0, \dots, B - 1$), or $-1$ for a point in no bag. */
  readonly bags: Int32Array
  /** The proportions $\pi_{k,c}$, $B \times C$ row-major, each row summing to 1. */
  readonly proportions: Float64Array
  /** The number of bags $B$. */
  readonly bagCount: Size
  /** The number of classes $C$. */
  readonly classes: Size
  /** Points per bag, $n_k$. */
  readonly sizes: Float64Array
  /** The class masses the bags fix, $n_k \pi_{k,c}$, $B \times C$ row-major. */
  readonly targets: Float64Array
}

/**
 * Read bags and proportions, checking them: each row of proportions is renormalised to sum to one (a row of zeros
 * becomes uniform). Throws `DomainError` for a bag index that is neither $-1$ nor in $0, \dots, B - 1$, or a negative
 * proportion.
 *
 * @param bags Each point's bag index in $0, \dots, B - 1$, or $-1$ for a point in no bag.
 * @param proportions The bags' class proportions, $B \times C$; $B$ and $C$ are read from its shape.
 * @param where The caller's name, for error messages.
 * @returns The bags with their normalised proportions, sizes and class masses.
 *
 * @example Two bags, a point in none, and a row of proportions that does not sum to one
 * const info = readBags([0, 0, 1, 1, -1], [[1, 1], [0.25, 0.75]])
 * print('proportions:', info.proportions)
 * print('sizes:', info.sizes, ' class masses:', info.targets)
 */
export function readBags(bags: ArrayLike<number>, proportions: MatrixLike, where = 'readBags'): BagProportions {
  const P = dense.toMatrixF64(proportions, where)
  const B = P.m
  const C = P.n
  const b = Int32Array.from(bags)
  const sizes = new Float64Array(B)
  for (let i = 0; i < b.length; i++) {
    if (b[i] === -1) continue
    if (!(b[i] >= 0 && b[i] < B))
      throw new DomainError(where, `${where}: bag ${b[i]} of point ${i} is not −1 or in 0 … ${B - 1}`)
    sizes[b[i]]++
  }
  const pi = new Float64Array(B * C)
  for (let k = 0; k < B; k++) {
    let total = 0
    for (let c = 0; c < C; c++) {
      const v = P.data[k * C + c]
      if (!(v >= 0)) throw new DomainError(where, `${where}: proportions must be non-negative`)
      total += v
    }
    for (let c = 0; c < C; c++) pi[k * C + c] = total > 0 ? P.data[k * C + c] / total : 1 / C
  }
  const targets = Float64Array.from(pi, (v, idx) => v * sizes[Math.floor(idx / C)])
  return { bags: b, proportions: pi, bagCount: B, classes: C, sizes, targets }
}

/**
 * The class proportions of bags given each point's true label: what an LLP oracle reveals. Points in no bag are
 * skipped, and an empty bag gets the uniform distribution.
 *
 * @param bags Each point's bag index in $0, \dots, B - 1$, or a negative value for none.
 * @param labels Each point's class in $0, \dots, C - 1$.
 * @param bagCount The number of bags $B$.
 * @param classes The number of classes $C$.
 * @returns The proportions, $B \times C$.
 *
 * @example Two bags of labelled points
 * print(bagProportionsOf([0, 0, 0, 1, 1, -1], [0, 1, 1, 1, 1, 0], 2, 2))
 */
export function bagProportionsOf(
  bags: ArrayLike<number>,
  labels: ArrayLike<number>,
  bagCount: Size,
  classes: Size,
): Tensor {
  const out = new Float64Array(bagCount * classes)
  const size = new Float64Array(bagCount)
  for (let i = 0; i < bags.length; i++) {
    if (bags[i] < 0) continue
    out[bags[i] * classes + labels[i]]++
    size[bags[i]]++
  }
  return fromData(
    out.map((v, idx) => (size[Math.floor(idx / classes)] > 0 ? v / size[Math.floor(idx / classes)] : 1 / classes)),
    [bagCount, classes],
  )
}

/**
 * Split labelled points into bags with given class proportions, as the paper's experiments do ("the data is first
 * generated … and then separated into the bags, respecting the desired bag proportions"): bag $k$ gets `sizes[k]`
 * points (default: $n$ split evenly, as scikit-learn's splits), of which `sizes[k]` times $\pi_{k,c}$ of class $c$,
 * rounded by largest remainders, drawn without replacement from each class in a random order. When a class runs out,
 * the bag is filled from the class with the most points left, so the realised proportions (returned) can differ from
 * the requested ones. Points not placed (when the sizes sum to less than $n$) get bag $-1$. Throws `ShapeError` when
 * there is not one size per bag and `DomainError` when the sizes add up to more than $n$.
 *
 * @param s The stream that orders each class's points (class $c$ uses `child(s, 'class', c)`).
 * @param labels Each point's class in $0, \dots, C - 1$.
 * @param proportions The requested proportions, $B \times C$ (rows renormalised; a row of zeros is uniform).
 * @param options `sizes`, the number of points in each bag.
 * @returns Each point's bag, and the proportions the bags realise ($B \times C$).
 *
 * @example Twelve points of two classes into a mostly-0 and a mostly-1 bag
 * const labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
 * const { bags, proportions } = bagsByProportion(stream(0), labels, [[0.75, 0.25], [0.25, 0.75]])
 * print('bags:', bags)
 * print('realised proportions:', proportions)
 */
export function bagsByProportion(
  s: Stream,
  labels: ArrayLike<number>,
  proportions: MatrixLike,
  options: { sizes?: readonly number[] } = {},
): { bags: Int32Array; proportions: Tensor } {
  const P = dense.toMatrixF64(proportions, 'bagsByProportion')
  const B = P.m
  const K = P.n
  const n = labels.length
  const sizes = options.sizes ?? Array.from({ length: B }, (_, k) => Math.floor(n / B) + (k < n % B ? 1 : 0))
  if (sizes.length !== B)
    throw new ShapeError('bagsByProportion', `bagsByProportion: ${sizes.length} sizes for ${B} bags`)
  if (sizes.reduce((a, v) => a + v, 0) > n)
    throw new DomainError('bagsByProportion', 'bagsByProportion: the bags hold more points than there are')
  const pools = Array.from({ length: K }, (_, c) => {
    const members = Array.from({ length: n }, (_, i) => i).filter((i) => labels[i] === c)
    const order = toFlat(permutation(child(s, 'class', c), members.length))
    return Array.from(order, (o) => members[o])
  })
  const bags = new Int32Array(n).fill(-1)
  for (let k = 0; k < B; k++) {
    const m = sizes[k]
    let total = 0
    for (let c = 0; c < K; c++) total += P.data[k * K + c]
    const want = Array.from({ length: K }, (_, c) => (total > 0 ? (m * P.data[k * K + c]) / total : m / K))
    const counts = want.map(Math.floor)
    const order = want.map((w, c) => [w - Math.floor(w), c]).sort((a, b) => b[0] - a[0] || a[1] - b[1])
    for (let r = 0; r < m - counts.reduce((a, v) => a + v, 0); r++) counts[order[r][1]]++
    for (let c = 0; c < K; c++)
      for (let r = 0; r < counts[c]; r++) {
        let from = c
        if (pools[from].length === 0) from = pools.reduce((best, p, j) => (p.length > pools[best].length ? j : best), 0)
        const i = pools[from].pop()
        if (i !== undefined) bags[i] = k
      }
  }
  return { bags, proportions: bagProportionsOf(bags, labels, B, K) }
}

// ── LP-LLP ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of LP-LLP. */
export interface LpLlpOptions {
  /** $\alpha \in (0, 1)$: the weight of the neighbours against the point's own score (default 0.5, as the paper). */
  alpha?: number
  /** $\gamma$ of the similarity $\exp(-\gamma \lVert \xvec_i - \xvec_j \rVert^2)$ (default 1). */
  gamma?: number
  /** Keep only each point's $k$ nearest neighbours (symmetrised); default 0, the full graph as in the paper. */
  neighbours?: Size
  /**
   * Use $(1 - \alpha)(\Imat - \alpha\Smat)^{-1}$ (default true); false uses $(\Imat - \alpha\Smat)^{-1}$ as
   * Algorithm 2 prints it.
   */
  normalise?: boolean
  /** Stop when an outer step moves $\Fmat$ by less than this in the max norm (default 1e-4). */
  tolerance?: number
  /** Tolerance of the alternating projections (default 1e-9). */
  projectionTolerance?: number
  /** The most cycles of the alternating projections per step (default 2000). */
  maxProjections?: Size
}

/** One state of LP-LLP: the scores after t propagate-and-project steps. */
export interface LpLlpState extends Status {
  /**
   * $\Fmat$, $n \times C$, after the projection: rows on the simplex, bag masses fixed. At $t = 0$, the bags'
   * proportions (uniform for a point in no bag).
   */
  readonly scores: Tensor
  /**
   * $\Fmat$, $n \times C$, after the propagation of this step, before the projection (equal to `scores` at $t = 0$).
   */
  readonly propagated: Tensor
  /** The predicted class of every point (argmax of its row; int32). */
  readonly labels: Tensor
  /** The bags' class masses $\sum_{i \in \Bcal_k} F_{ic}$ after the projection, $B \times C$. */
  readonly mass: Tensor
  /** The same before the projection. */
  readonly propagatedMass: Tensor
  /**
   * $\max \lvert \text{mass} - n_k \pi_{k,c} \rvert$ before the projection: how far the propagation pulled the bags off
   * their constraints.
   */
  readonly violation: number
  /** Alternating-projection cycles this step took. */
  readonly projections: number
  /** $\max \lvert \Fmat_t - \Fmat_{t-1} \rvert$ (Infinity at $t = 0$). */
  readonly change: number
}

/** The graph LP-LLP propagates over, each matrix $n \times n$. */
export interface LpLlpGraph {
  /** The affinity $\Wmat$, with a zero diagonal. */
  readonly affinity: Tensor
  /** The random-walk matrix $\Smat = \Dmat^{-1}\Wmat$. */
  readonly walk: Tensor
  /** The propagation matrix $\Pmat = (1 - \alpha)(\Imat - \alpha\Smat)^{-1}$ (or $(\Imat - \alpha\Smat)^{-1}$). */
  readonly propagation: Tensor
}

/**
 * Build LP-LLP's graph from points: $\Wmat$, $\Smat = \Dmat^{-1}\Wmat$ and
 * $\Pmat = (1 - \alpha)(\Imat - \alpha\Smat)^{-1}$ (or $(\Imat - \alpha\Smat)^{-1}$ without `normalise`).
 *
 * @param x The points, $n \times d$.
 * @param options `alpha`, `gamma`, `neighbours` and `normalise` of `LpLlpOptions`; the rest is unused.
 * @returns The affinity, random-walk and propagation matrices.
 *
 * @example Three points on a line: the scaled propagation matrix averages
 * const g = lpllpGraph([[0], [1], [3]])
 * print('S =', g.walk)
 * print('P =', g.propagation)
 * print('row sums of P:', sum(g.propagation, 1))
 */
export function lpllpGraph(x: MatrixLike, options: LpLlpOptions = {}): LpLlpGraph {
  const { alpha = 0.5, gamma = 1, neighbours = 0, normalise = true } = options
  const affinity = pointAffinity(x, { gamma, neighbours })
  const walk = randomWalkMatrix(affinity)
  return { affinity, walk, propagation: spreadingResolvent(walk, alpha, { scaled: normalise }) }
}

/**
 * The argmax of each row (the first column on a tie).
 *
 * @param F The scores, $n \times C$ row-major.
 * @param n The number of rows.
 * @param C The number of columns.
 * @returns The column of each row's largest score.
 */
function argmaxRows(F: Float64Array, n: number, C: number): Int32Array {
  return Int32Array.from({ length: n }, (_, i) => {
    let best = 0
    for (let c = 1; c < C; c++) if (F[i * C + c] > F[i * C + best]) best = c
    return best
  })
}

/**
 * The bags' class masses $\sum_{i \in \Bcal_k} F_{ic}$ (points in no bag are left out).
 *
 * @param F The scores, $n \times C$ row-major.
 * @param info The bags.
 * @returns The masses, $B \times C$ row-major.
 */
function bagMass(F: Float64Array, info: BagProportions): Float64Array {
  const { bags, bagCount: B, classes: C } = info
  const mass = new Float64Array(B * C)
  for (let i = 0; i < bags.length; i++)
    if (bags[i] >= 0) for (let c = 0; c < C; c++) mass[bags[i] * C + c] += F[i * C + c]
  return mass
}

/**
 * The projection of LP-LLP: rows onto the simplex, alternated with the bags' class masses (the entries of points in no
 * bag are free of the mass constraint).
 *
 * @param info The bags, with the class masses they fix.
 * @param n The number of points.
 * @param tolerance The alternating projections' tolerance.
 * @param maxCycles Their most cycles.
 * @returns A function from scores ($n \times C$ row-major) to the projected scores `F` and the `cycles` taken.
 */
function projector(info: BagProportions, n: number, tolerance: number, maxCycles: number) {
  const C = info.classes
  // Entry (i, c) belongs to the group (bag of i, c); entries of points in no bag are free.
  const groups = Int32Array.from({ length: n * C }, (_, idx) => {
    const k = info.bags[Math.floor(idx / C)]
    return k < 0 ? -1 : k * C + (idx % C)
  })
  const toSums = projectGroupSums(groups, info.targets)
  const toRows = projectSimplexRows(C)
  return (F: Float64Array) => {
    // Rows last, so every returned row is a distribution; at convergence the bag masses hold as well.
    const r = alternatingProjections([toSums, toRows], F, { tolerance, maxCycles })
    return { F: Float64Array.from(dense.toF64(r.x, 'lpllp')), cycles: r.cycles }
  }
}

/**
 * LP-LLP as steps (see the file's notes). Step 0 is $\Fmat^{(0)} = \hat{\Ymat}$, each point's bag proportions; each
 * step propagates, then projects, and the state is `converged` when a step moves $\Fmat$ by less than `tolerance`.
 * Throws `ShapeError` unless there is one bag index per point, and as `readBags` does.
 *
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none (such a point starts from the uniform
 *   distribution and is not constrained).
 * @param proportions Each bag's class proportions, $B \times C$.
 * @param options The graph, the propagation and the stopping rules.
 * @returns The algorithm, to run with `run` or `trace` (its input is unused).
 *
 * @example How far each propagation pulls the bags off their proportions, step by step
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const alg = lpllpSteps(x, bags, proportions)
 * for (const t of [0, 1, 2, 5]) {
 *   const st = run(alg, undefined, t)
 *   const hit = y.filter((c, i) => st.labels.data[i] === c).length / y.length
 *   print('step', st.t, ' pulled off by', st.violation, ' change', st.change, ' accuracy', hit)
 * }
 */
export function lpllpSteps(
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: LpLlpOptions = {},
): Algorithm<void, LpLlpState> {
  const info = readBags(bags, proportions, 'lpllpSteps')
  const X = dense.toMatrixF64(x, 'lpllpSteps')
  const n = X.m
  if (info.bags.length !== n)
    throw new ShapeError('lpllpSteps', `lpllpSteps: ${info.bags.length} bag indices for ${n} points`)
  const C = info.classes
  const { tolerance = 1e-4, projectionTolerance = 1e-9, maxProjections = 2000 } = options
  const P = dense.data(lpllpGraph(x, options).propagation)
  const project = projector(info, n, projectionTolerance, maxProjections)
  const state = (t: number, F: Float64Array, G: Float64Array, cycles: number, change: number): LpLlpState => {
    const before = bagMass(G, info)
    let violation = 0
    for (let k = 0; k < before.length; k++) violation = Math.max(violation, Math.abs(before[k] - info.targets[k]))
    return {
      t,
      scores: fromData(F, [n, C]),
      propagated: fromData(G, [n, C]),
      labels: fromData(argmaxRows(F, n, C), [n]),
      mass: fromData(bagMass(F, info), [info.bagCount, C]),
      propagatedMass: fromData(before, [info.bagCount, C]),
      violation,
      projections: cycles,
      change,
      converged: change < tolerance,
    }
  }
  return {
    name: 'lpllp',
    init: () => {
      const F = new Float64Array(n * C)
      for (let i = 0; i < n; i++)
        for (let c = 0; c < C; c++) F[i * C + c] = info.bags[i] < 0 ? 1 / C : info.proportions[info.bags[i] * C + c]
      return state(0, F, F, 0, Infinity)
    },
    step: (s) => {
      const F = dense.data(s.scores)
      const G = dense.matMul(P, F, n, n, C)
      const projected = project(G)
      let change = 0
      for (let k = 0; k < F.length; k++) change = Math.max(change, Math.abs(projected.F[k] - F[k]))
      return state(s.t + 1, projected.F, G, projected.cycles, change)
    },
  }
}

/**
 * LP-LLP run to convergence (at most `maxSteps` propagate-and-project steps, default 300): the final state.
 *
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none.
 * @param proportions Each bag's class proportions, $B \times C$.
 * @param options The options of `lpllpSteps`, and `maxSteps`.
 * @returns The final state: scores, labels and the bags' masses.
 *
 * @example Label every point from four bags' proportions
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const accuracy = (labels) => y.filter((c, i) => labels[i] === c).length / y.length
 * const fit = lpllp(x, bags, proportions)
 * print('steps:', fit.t, ' accuracy:', accuracy(fit.labels.data))
 */
export function lpllp(
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: LpLlpOptions & { maxSteps?: Size } = {},
): LpLlpState {
  const alg = lpllpSteps(x, bags, proportions, options)
  return trace(alg, undefined, options.maxSteps ?? 300, { keep: 'none' }).final
}

/**
 * The paper's choice of $\gamma$: run LP-LLP for each $\gamma$ on a grid and keep the one with the largest smoothness
 * score $\sum_c \bar{\fvec}_c^\top \Smat \bar{\fvec}_c$, where $\bar{\Fmat} = \Fmat^* - 1/C$ is the converged scores
 * centred on the uniform distribution ($\Smat$ of that $\gamma$). A smooth labelling that is confident scores high; the
 * uniform scores of an over-smoothed graph score zero. A tie keeps the earlier $\gamma$.
 *
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none.
 * @param proportions Each bag's class proportions, $B \times C$.
 * @param gammas The grid of $\gamma$ to try (not empty).
 * @param options The options of `lpllp`; their `gamma` is replaced by each of the grid.
 * @returns The chosen `gamma`, the smoothness score of each, and the LP-LLP state of the chosen one.
 *
 * @example Pick the graph's scale
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const accuracy = (labels) => y.filter((c, i) => labels[i] === c).length / y.length
 * const search = lpllpGammaSearch(x, bags, proportions, [0.1, 1, 10], { maxSteps: 20 })
 * print('scores:', search.scores, ' chosen gamma:', search.gamma)
 * print('accuracy with it:', accuracy(search.best.labels.data))
 */
export function lpllpGammaSearch(
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  gammas: readonly number[],
  options: LpLlpOptions & { maxSteps?: Size } = {},
): { gamma: number; scores: number[]; best: LpLlpState } {
  let best: LpLlpState | null = null
  let bestGamma = gammas[0]
  let top = -Infinity
  const scores = gammas.map((gamma) => {
    const s = lpllp(x, bags, proportions, { ...options, gamma })
    const { walk } = lpllpGraph(x, { ...options, gamma })
    const [n, C] = s.scores.shape
    const F = dense.data(s.scores).map((v) => v - 1 / C)
    const SF = dense.matMul(dense.data(walk), F, n, n, C)
    let score = 0
    for (let k = 0; k < F.length; k++) score += F[k] * SF[k]
    if (score > top) {
      top = score
      bestGamma = gamma
      best = s
    }
    return score
  })
  return { gamma: bestGamma, scores, best: best! }
}

// ── Baselines ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Labels for every point from a label-proportion baseline, with its decision values. */
export interface LlpPrediction {
  /** The class of every point. */
  readonly labels: Int32Array
  /** The decision value of every point (two classes: $f > 0$ leans to class 1). */
  readonly decision: Float64Array
}

/**
 * Throws `DomainError` unless the bags have two classes, as the paper's baselines need.
 *
 * @param info The bags.
 * @param where The caller's name, for error messages.
 */
const binaryOnly = (info: BagProportions, where: string) => {
  if (info.classes !== 2) throw new DomainError(where, `${where}: two classes only, as in the paper`)
}

/**
 * Inverse calibration (InvCal; Rüping 2010, ICML): each bag becomes a super-instance at its mean in the kernel's
 * feature space, $\mvec_k = \frac{1}{n_k} \sum_{i \in \Bcal_k} \phi(\xvec_i)$, with target
 * $t_k = -\log(1/\pi_{k,1} - 1)$; an $\varepsilon$-insensitive support vector regression
 * $f(\xvec) = \sum_k \beta_k \langle \mvec_k, \phi(\xvec) \rangle + b$ fits the targets, and a point is labelled 1
 * when $f(\xvec) > 0$ (its calibrated probability $\sigma(f(\xvec))$ exceeds $\tfrac{1}{2}$). Solved in the dual
 * (interior point), a QP in the $2B$ variables $\alpha^+, \alpha^- \in [0, C]$ with
 * $\sum_k (\alpha^+_k - \alpha^-_k) = 0$. Proportions are clipped to $[0.001, 0.999]$ so the targets are finite. Two
 * classes only (`DomainError` otherwise).
 *
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none (still labelled).
 * @param proportions Each bag's class proportions, $B \times 2$.
 * @param options `gamma`, the RBF kernel's $\gamma$ (default 1); `C`, the box bound (default 10); and `epsilon`, the
 *   tube's half-width $\varepsilon$ (default 0.01).
 * @returns The labels and decision values $f(\xvec)$ of every point.
 *
 * @example InvCal on four bags
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const accuracy = (labels) => y.filter((c, i) => labels[i] === c).length / y.length
 * const r = inverseCalibration(x, bags, proportions)
 * print('accuracy:', accuracy(r.labels))
 * print('decision values of the first four points:', r.decision.slice(0, 4), ' their classes:', y.slice(0, 4))
 */
export function inverseCalibration(
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: { gamma?: number; C?: number; epsilon?: number } = {},
): LlpPrediction {
  const { gamma = 1, C = 10, epsilon = 0.01 } = options
  const info = readBags(bags, proportions, 'inverseCalibration')
  binaryOnly(info, 'inverseCalibration')
  const B = info.bagCount
  const K = dense.data(pointAffinity(x, { gamma }))
  const n = info.bags.length
  // The diagonal of the full kernel is exp(0) = 1 (pointAffinity zeroes it).
  const k = (i: number, j: number) => (i === j ? 1 : K[i * n + j])
  const members = Array.from({ length: B }, (_, b) => Array.from(info.bags).flatMap((v, i) => (v === b ? [i] : [])))
  // ⟨m_k, φ(x_j)⟩ for every bag and point, and the bag Gram matrix G_kl = ⟨m_k, m_l⟩.
  const toPoint = new Float64Array(B * n)
  for (let b = 0; b < B; b++)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (const i of members[b]) s += k(i, j)
      toPoint[b * n + j] = members[b].length > 0 ? s / members[b].length : 0
    }
  const G = new Float64Array(B * B)
  for (let a = 0; a < B; a++)
    for (let b = 0; b < B; b++) {
      let s = 0
      for (const i of members[b]) s += toPoint[a * n + i]
      G[a * B + b] = members[b].length > 0 ? s / members[b].length : 0
    }
  const t = Float64Array.from({ length: B }, (_, b) => {
    const p = Math.min(0.999, Math.max(0.001, info.proportions[b * 2 + 1]))
    return -Math.log(1 / p - 1)
  })
  // Variables z = (α⁺, α⁻): ½(α⁺ − α⁻)ᵀG(α⁺ − α⁻) + εΣ(α⁺ + α⁻) − tᵀ(α⁺ − α⁻).
  const m = 2 * B
  const Q = new Float64Array(m * m)
  for (let a = 0; a < B; a++)
    for (let b = 0; b < B; b++) {
      const g = G[a * B + b]
      Q[a * m + b] = g
      Q[(a + B) * m + b + B] = g
      Q[a * m + b + B] = -g
      Q[(a + B) * m + b] = -g
    }
  for (let a = 0; a < m; a++) Q[a * m + a] += 1e-9
  const c = Float64Array.from({ length: m }, (_, a) => (a < B ? epsilon - t[a] : epsilon + t[a - B]))
  const A = new Float64Array(2 * m * m)
  const bound = new Float64Array(2 * m)
  for (let a = 0; a < m; a++) {
    A[a * m + a] = 1
    bound[a] = C
    A[(m + a) * m + a] = -1
  }
  const E = Float64Array.from({ length: m }, (_, a) => (a < B ? 1 : -1))
  const r = quadprog(
    { Q: fromData(Q, [m, m]), c, A: fromData(A, [2 * m, m]), b: bound, E: fromData(E, [1, m]), e: [0] },
    { method: 'interior-point' },
  )
  const z = toFlat(r.x)
  const beta = Float64Array.from({ length: B }, (_, a) => z[a] - z[a + B])
  const Gb = Float64Array.from({ length: B }, (_, a) => {
    let s = 0
    for (let b = 0; b < B; b++) s += G[a * B + b] * beta[b]
    return s
  })
  // b from the bags strictly inside the tube's edge (0 < α < C); otherwise the midpoint of the feasible range.
  const tol = 1e-6 * C
  const bs: number[] = []
  for (let a = 0; a < B; a++) {
    if (z[a] > tol && z[a] < C - tol) bs.push(t[a] - epsilon - Gb[a])
    if (z[a + B] > tol && z[a + B] < C - tol) bs.push(t[a] + epsilon - Gb[a])
  }
  const bias = bs.length > 0 ? bs.reduce((s, v) => s + v, 0) / bs.length : t.reduce((s, v, a) => s + v - Gb[a], 0) / B
  const decision = Float64Array.from({ length: n }, (_, j) => {
    let s = bias
    for (let b = 0; b < B; b++) s += beta[b] * toPoint[b * n + j]
    return s
  })
  return { labels: Int32Array.from(decision, (v) => (v > 0 ? 1 : 0)), decision }
}

/**
 * The soft-margin kernel SVM in the dual with the bias folded into the kernel ($k(\xvec, \xvec') + 1$, a regularised
 * bias as in LIBLINEAR), so the dual is a box-constrained QP: minimise
 * $\tfrac{1}{2}\alphavec^\top(\yvec\yvec^\top \circ (\Kmat + 1))\alphavec - \ones^\top\alphavec$ over
 * $0 \le \alpha_i \le C$, solved by `boxQuadprog` (warm-started; at most 200 steps).
 *
 * @param K The kernel matrix $\Kmat$, $n \times n$ row-major.
 * @param y The labels, 0 or 1 (taken as $-1$ and $+1$).
 * @param C The box bound $C$.
 * @param warm A starting $\alphavec$ (default: the solver's own).
 * @returns `alpha`, the dual solution; `decision`, $f(\xvec_i) = \sum_j \alpha_j y_j (K_{ij} + 1)$; and `wNorm`,
 *   $\lVert \wvec \rVert^2 = \sum_i \alpha_i y_i f(\xvec_i)$ (the bias's square included).
 */
function svmDual(K: Float64Array, y: Int32Array, C: number, warm?: Float64Array) {
  const n = y.length
  const sign = Float64Array.from(y, (v) => (v === 1 ? 1 : -1))
  const Q = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Q[i * n + j] = sign[i] * sign[j] * (K[i * n + j] + 1)
  const s = boxQuadprog(
    {
      Q: fromData(Q, [n, n]),
      c: new Float64Array(n).fill(-1),
      lower: new Float64Array(n),
      upper: new Float64Array(n).fill(C),
    },
    { tolerance: 1e-6, maxSteps: 200, x0: warm },
  )
  const alpha = Float64Array.from(toFlat(s.x))
  const decision = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let f = 0
    for (let j = 0; j < n; j++) f += alpha[j] * sign[j] * (K[i * n + j] + 1)
    decision[i] = f
  }
  let quad = 0
  for (let i = 0; i < n; i++) quad += alpha[i] * sign[i] * decision[i]
  return { alpha, decision, wNorm: quad }
}

/**
 * Alternating $\propto$SVM (alter-$\propto$SVM; Yu et al. 2013, ICML, Algorithm 1) in the limit of a hard proportion
 * constraint: start from random labels with each bag's proportion; repeat: fit a soft-margin SVM with the RBF kernel
 * $\exp(-\gamma \lVert \xvec - \xvec' \rVert^2)$ to the current labels, then relabel each bag by giving class 1 to its
 * $\operatorname{round}(n_k \pi_{k,1})$ points with the largest decision values (the labels that minimise the hinge
 * loss under the proportion), and each point in no bag by the sign of its decision value; stop when no label changes
 * or after `maxRounds` rounds. Of `restarts` random starts, the labelling with the smallest SVM objective
 * $\tfrac{1}{2}\lVert \wvec \rVert^2 + C \sum \text{hinge}$ is kept. The paper's annealing of the label weight is left
 * out, and the SVM's bias is folded into the kernel (logged). Two classes only (`DomainError` otherwise).
 *
 * @param s The stream of the random starts (and of the flips that keep both classes present).
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none.
 * @param proportions Each bag's class proportions, $B \times 2$.
 * @param options `gamma`, the RBF kernel's $\gamma$ (default 1); `C`, the SVM's box bound (default 1); `restarts`, the
 *   random starts (default 5); and `maxRounds`, the most fit-and-relabel rounds per start (default 20).
 * @returns The kept labels, the decision values of the last SVM fitted from that start, and its objective (on the
 *   labels it was fitted to).
 *
 * @example alter-SVM on four bags, with a narrow and a wide kernel
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const accuracy = (labels) => y.filter((c, i) => labels[i] === c).length / y.length
 * for (const gamma of [1, 0.1]) {
 *   const r = alterProportionSvm(stream(22), x, bags, proportions, { gamma, restarts: 2 })
 *   print('gamma =', gamma, ' accuracy:', accuracy(r.labels), ' objective:', r.objective)
 * }
 */
export function alterProportionSvm(
  s: Stream,
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: { gamma?: number; C?: number; restarts?: Size; maxRounds?: Size } = {},
): LlpPrediction & { objective: number } {
  const { gamma = 1, C = 1, restarts = 5, maxRounds = 20 } = options
  const info = readBags(bags, proportions, 'alterProportionSvm')
  binaryOnly(info, 'alterProportionSvm')
  const n = info.bags.length
  const K = Float64Array.from(dense.data(pointAffinity(x, { gamma })))
  for (let i = 0; i < n; i++) K[i * n + i] = 1
  const B = info.bagCount
  const members = Array.from({ length: B }, (_, b) => Array.from(info.bags).flatMap((v, i) => (v === b ? [i] : [])))
  const positives = members.map((m, b) => Math.round(m.length * info.proportions[b * 2 + 1]))
  let best: (LlpPrediction & { objective: number }) | null = null
  for (let r = 0; r < restarts; r++) {
    const y = new Int32Array(n)
    // Points in no bag start at random and are relabelled by the SVM's sign.
    for (let i = 0; i < n; i++) if (info.bags[i] < 0) y[i] = uniform(child(s, 'free', r, i)) < 0.5 ? 1 : 0
    members.forEach((m, b) => {
      const order = toFlat(permutation(child(s, 'start', r, b), m.length))
      order.forEach((o, rank) => (y[m[o]] = rank < positives[b] ? 1 : 0))
    })
    let decision = new Float64Array(n)
    let objective = Infinity
    let warm: Float64Array | undefined
    for (let round = 0; round < maxRounds; round++) {
      // An SVM needs both classes.
      if (y.every((v) => v === y[0])) y[integers(child(s, 'flip', r, round), n)] ^= 1
      const fit = svmDual(K, y, C, warm)
      warm = fit.alpha
      decision = fit.decision
      let hinge = 0
      for (let i = 0; i < n; i++) hinge += Math.max(0, 1 - (y[i] === 1 ? 1 : -1) * decision[i])
      objective = fit.wNorm / 2 + C * hinge
      let changed = 0
      members.forEach((m, b) => {
        const ranked = [...m].sort((i, j) => decision[j] - decision[i])
        ranked.forEach((i, rank) => {
          const v = rank < positives[b] ? 1 : 0
          if (y[i] !== v) changed++
          y[i] = v
        })
      })
      for (let i = 0; i < n; i++)
        if (info.bags[i] < 0) {
          const v = decision[i] > 0 ? 1 : 0
          if (y[i] !== v) changed++
          y[i] = v
        }
      if (changed === 0) break
    }
    if (!best || objective < best.objective) best = { labels: Int32Array.from(y), decision, objective }
  }
  return best!
}

/**
 * Random Fourier features of the RBF kernel $\exp(-\gamma \lVert \xvec - \xvec' \rVert^2)$ (Rahimi and Recht, 2007):
 * $\zvec(\xvec) = \sqrt{2/D} \cos(\Omegamat^\top \xvec + \bvec)$, $\Omegamat \sim \Gauss(\zeros, 2\gamma\Imat)$,
 * $b_k \sim \Unif[0, 2\pi)$.
 *
 * @param s The stream the frequencies and phases are drawn from.
 * @param X The points: `data` row-major, `m` rows and `n` columns.
 * @param D The number of features $D$.
 * @param gamma The kernel's $\gamma$.
 * @returns The features, $m \times D$ row-major.
 */
function fourierFeatures(
  s: Stream,
  X: { data: ArrayLike<number>; m: number; n: number },
  D: Size,
  gamma: number,
): Float64Array {
  const omega = Array.from({ length: D * X.n }, (_, k) => Math.sqrt(2 * gamma) * normal(child(s, 'omega', k)))
  const phase = Array.from({ length: D }, (_, k) => 2 * Math.PI * uniform(child(s, 'phase', k)))
  const out = new Float64Array(X.m * D)
  for (let i = 0; i < X.m; i++)
    for (let k = 0; k < D; k++) {
      let v = phase[k]
      for (let j = 0; j < X.n; j++) v += omega[k * X.n + j] * X.data[i * X.n + j]
      out[i * D + k] = Math.sqrt(2 / D) * Math.cos(v)
    }
  return out
}

/**
 * MeanMap (Quadrianto, Smola, Caetano and Le 2009, JMLR): a conditional exponential family
 * $p(y \mid \xvec, \thetavec) \propto \exp(\thetavec_y^\top \phi(\xvec))$ fitted without labels. Assuming
 * $p(\xvec \mid y, \text{bag}) = p(\xvec \mid y)$, each bag's mean feature is a mixture of the class means,
 * $\mvec_k = \sum_y \pi_{k,y} \muvec_y$, so the class means are the least-squares solution of
 * $\Mmat = \Pimat\Umat$ (needs at least $C$ bags, or `DomainError`); the sufficient statistic
 * $\muvec_{XY} = \sum_y p(y) \evec_y \otimes \muvec_y$ then replaces the labels in the regularised negative
 * log-likelihood $\sum_i \log \sum_y \exp(\thetavec_y^\top \phi(\xvec_i)) - n \muvec_{XY}^\top \thetavec + \lambda
 * \lVert \thetavec \rVert^2$, minimised by L-BFGS from $\thetavec = \zeros$. $\phi$ is the identity with a constant
 * (`linear`) or $D$ random Fourier features of $\exp(-\gamma \lVert \xvec - \xvec' \rVert^2)$ (`rbf`, default), with
 * the constant too. The class prior $p(y)$ is the bags' proportions weighted by their sizes.
 *
 * @param s The stream of the random Fourier features.
 * @param x The points, $n \times d$.
 * @param bags Each point's bag in $0, \dots, B - 1$, or $-1$ for none.
 * @param proportions Each bag's class proportions, $B \times C$.
 * @param options `features`, `linear` or `rbf` (default); `gamma`, the RBF kernel's $\gamma$ (default 1); `dimension`,
 *   the number $D$ of Fourier features (default 100); `l2`, $\lambda$ (default 0.01); and `steps`, the most L-BFGS
 *   steps (default 200).
 * @returns The labels (argmax of the scores), the decision values (for two classes the class-1 score minus the class-0
 *   score, otherwise zeros), and the scores $\thetavec_y^\top \phi(\xvec_i)$, $n \times C$.
 *
 * @example MeanMap with linear features on four bags
 * const s = stream(20)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const { bags, proportions } = bagsByProportion(stream(21), y, [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
 * const accuracy = (labels) => y.filter((c, i) => labels[i] === c).length / y.length
 * const r = meanMap(stream(22), x, bags, proportions, { features: 'linear' })
 * print('accuracy:', accuracy(r.labels))
 */
export function meanMap(
  s: Stream,
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: { features?: 'linear' | 'rbf'; gamma?: number; dimension?: Size; l2?: number; steps?: Size } = {},
): LlpPrediction & { scores: Tensor } {
  const { features = 'rbf', gamma = 1, dimension = 100, l2 = 1e-2, steps = 200 } = options
  const info = readBags(bags, proportions, 'meanMap')
  const X = dense.toMatrixF64(x, 'meanMap')
  const n = X.m
  const C = info.classes
  const B = info.bagCount
  const raw =
    features === 'rbf' ? fourierFeatures(child(s, 'features'), X, dimension, gamma) : Float64Array.from(X.data)
  const d0 = features === 'rbf' ? dimension : X.n
  const D = d0 + 1
  const phi = new Float64Array(n * D)
  for (let i = 0; i < n; i++) {
    phi.set(raw.subarray(i * d0, (i + 1) * d0), i * D)
    phi[i * D + d0] = 1
  }
  if (B < C) throw new DomainError('meanMap', `meanMap: needs at least as many bags (${B}) as classes (${C})`)
  const M = new Float64Array(B * D)
  for (let i = 0; i < n; i++) {
    const b = info.bags[i]
    if (b < 0) continue
    for (let j = 0; j < D; j++) M[b * D + j] += phi[i * D + j] / info.sizes[b]
  }
  const U = dense.data(lstsq(fromData(info.proportions, [B, C]), fromData(M, [B, D])).x as Tensor)
  const prior = new Float64Array(C)
  let bagged = 0
  for (let b = 0; b < B; b++) bagged += info.sizes[b]
  for (let b = 0; b < B; b++)
    for (let c = 0; c < C; c++) prior[c] += (info.sizes[b] / bagged) * info.proportions[b * C + c]
  // n μ_XY as a [D, C] matrix (column y: n p(y) μ_y).
  const stat = new Float64Array(D * C)
  for (let c = 0; c < C; c++) for (let j = 0; j < D; j++) stat[j * C + c] = n * prior[c] * U[c * D + j]
  const Phi = fromData(phi, [n, D])
  const Stat = fromData(stat, [D, C])
  const objective = (theta: Value) => {
    const W = reshape(theta, [D, C])
    const logits = matmul(Phi, W)
    return sub(sum(logsumexp(logits, 1)), sub(sum(mul(Stat, W)), mul(l2, sum(mul(W, W)))))
  }
  const vg = valueAndGrad((theta: Value) => objective(theta))
  const run = trace(
    lbfgs((theta: Tensor) => {
      const r = vg(theta)
      return { value: unwrap(r.value) as number, grad: r.grad as Tensor }
    }),
    { x0: new Float64Array(D * C) },
    steps,
    { keep: 'none' },
  )
  const theta = dense.data(run.final.x as Tensor)
  const scores = dense.matMul(phi, theta, n, D, C)
  const labels = argmaxRows(scores, n, C)
  const decision =
    C === 2 ? Float64Array.from({ length: n }, (_, i) => scores[i * 2 + 1] - scores[i * 2]) : new Float64Array(n)
  return { labels, decision, scores: fromData(scores, [n, C]) }
}

// ── Comparison ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The LLP methods {@link llpComparison} runs: LP-LLP, InvCal, alter-$\propto$SVM, MeanMap and the proportion-loss
 * classifier.
 */
export type LlpMethod = 'lpllp' | 'invcal' | 'alter-svm' | 'meanmap' | 'proportion-loss'

/** Options of {@link llpComparison}. */
export interface LlpComparisonOptions {
  /**
   * One labelled dataset per repeat (features $n \times d$ and labels 0/1), e.g. draws of the Gaussian XOR (the
   * paper's Table 1) or the half-kernel (Table 2) from `aifn-methods/data/synthetic`.
   */
  datasets: readonly { x: Tensor; y: Tensor }[]
  /**
   * Bag sizes to try (each run splits the $n$ points into $\lceil n / \text{size} \rceil$ bags); default
   * $[75, 30, 10]$.
   */
  bagSizes?: readonly number[]
  /**
   * Bag purities $p$: bags alternate proportions $p$ and $1 - p$ of class 1 (0.5: no information); default
   * $[0.6, 0.75, 0.9]$.
   */
  purities?: readonly number[]
  /** The methods to run (default all five). */
  methods?: readonly LlpMethod[]
  /** $\gamma$ of LP-LLP's graph and of the RBF kernels of the baselines (default 4). */
  gamma?: number
  /** Seed of the bags, the baselines' random draws and the proportion-loss classifier's weights (default 0). */
  seed?: number | string
}

/** The progress and results of {@link llpComparison}: mean and sd of the accuracy per method, bag size and purity. */
export interface LlpComparison {
  /** The methods run. */
  readonly methods: readonly LlpMethod[]
  /** The bag sizes tried. */
  readonly bagSizes: readonly number[]
  /** The purities tried. */
  readonly purities: readonly number[]
  /** `mean[m][b][p]`: the mean accuracy over the repeats done so far (NaN before the first). */
  readonly mean: number[][][]
  /** `sd[m][b][p]`: the sample standard deviation of the same (0 after one repeat, NaN before the first). */
  readonly sd: number[][][]
  /** Cells (bag size, purity and dataset) done so far. */
  readonly done: number
  /** Cells in all. */
  readonly total: number
}

/**
 * Accuracy of LP-LLP against the baselines as bag size and purity vary, as a generator: one partial result per cell
 * (dataset, bag size and purity), so a figure fills in while it runs. Every method sees the same bags (made by
 * `bagsByProportion`) and the proportions they realise, and is scored on all $n$ points (transductive, as the paper).
 * alter-$\propto$SVM runs with two restarts, and the proportion-loss classifier is an MLP with 16 hidden units trained
 * for 150 steps.
 *
 * @param options The datasets, the grid of bag sizes and purities, the methods, $\gamma$ and the seed.
 * @returns A generator of partial results, one after each cell; it returns the final one.
 *
 * @example LP-LLP and MeanMap on one small dataset, one bag size and one purity
 * const s = stream(23)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const options = { bagSizes: [10], purities: [0.8], methods: ['lpllp', 'meanmap'], gamma: 1 }
 * let last
 * for (const r of llpComparison({ datasets: [{ x: tensor(x), y: tensor(y) }], ...options })) last = r
 * print(last.methods, ' mean accuracy:', last.mean.map((m) => m[0][0]), ' cells:', last.done, 'of', last.total)
 */
export function* llpComparison(options: LlpComparisonOptions): Generator<LlpComparison, LlpComparison> {
  const {
    datasets,
    bagSizes = [75, 30, 10],
    purities = [0.6, 0.75, 0.9],
    methods = ['lpllp', 'invcal', 'alter-svm', 'meanmap', 'proportion-loss'],
    gamma = 4,
    seed = 0,
  } = options
  const root = makeStream(`llpComparison/${seed}`)
  const acc = methods.map(() => bagSizes.map(() => purities.map(() => [] as number[])))
  const repeats = datasets.length
  const total = bagSizes.length * purities.length * repeats
  let done = 0
  const snapshot = (): LlpComparison => {
    const stat = (f: (v: number[]) => number) => acc.map((m) => m.map((b) => b.map((v) => (v.length ? f(v) : NaN))))
    const mu = (v: number[]) => v.reduce((a, x) => a + x, 0) / v.length
    return {
      methods,
      bagSizes,
      purities,
      mean: stat(mu),
      sd: stat((v) => Math.sqrt(v.reduce((a, x) => a + (x - mu(v)) ** 2, 0) / Math.max(1, v.length - 1))),
      done,
      total,
    }
  }
  for (let r = 0; r < repeats; r++)
    for (let b = 0; b < bagSizes.length; b++)
      for (let p = 0; p < purities.length; p++) {
        const s = child(root, 'cell', r, b, p)
        const x = datasets[r].x
        const y = Int32Array.from(toFlat(datasets[r].y))
        const n = y.length
        const B = Math.max(1, Math.ceil(n / bagSizes[b]))
        const wanted = Array.from({ length: B }, (_, k) => {
          const q = k % 2 === 0 ? purities[p] : 1 - purities[p]
          return [1 - q, q]
        })
        const { bags, proportions } = bagsByProportion(child(s, 'bags'), y, wanted)
        methods.forEach((m, mi) => {
          let labels: ArrayLike<number>
          if (m === 'lpllp') labels = toFlat(lpllp(x, bags, proportions, { gamma }).labels)
          else if (m === 'invcal') labels = inverseCalibration(x, bags, proportions, { gamma }).labels
          else if (m === 'alter-svm')
            labels = alterProportionSvm(child(s, 'svm'), x, bags, proportions, { gamma, restarts: 2 }).labels
          else if (m === 'meanmap') labels = meanMap(child(s, 'meanmap'), x, bags, proportions, { gamma }).labels
          else {
            const model = proportionClassifier(x, bags, proportions, { hidden: [16], steps: 150, seed: `${seed}/${r}` })
            const P = dense.data(model.predict(x))
            labels = Int32Array.from({ length: n }, (_, i) => (P[2 * i + 1] > P[2 * i] ? 1 : 0))
          }
          let hit = 0
          for (let i = 0; i < n; i++) if (labels[i] === y[i]) hit++
          acc[mi][b][p].push(hit / n)
        })
        done++
        yield snapshot()
      }
  return snapshot()
}
