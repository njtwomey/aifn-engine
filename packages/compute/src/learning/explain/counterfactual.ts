/**
 * Counterfactual explanations: the smallest change to an input $\xvec$ that changes a model's decision, and
 * constraints on which changes a person could act on.
 *
 * - `wachterCounterfactual` (Wachter, Mittelstadt and Russell, 2017): minimise
 *   $\lambda (f(\xvec') - y')^2 + \sum_i \lvert x'_i - x_i \rvert / \text{MAD}_i$ by gradient steps (Adam), raising
 *   $\lambda$ until $\lvert f(\xvec') - y' \rvert \le \epsilon$. The L1 distance scaled by each feature's median
 *   absolute deviation favours sparse changes.
 * - `diverseCounterfactuals` (Mothilal, Sharma and Tan, 2020, DiCE): $k$ counterfactuals at once, minimising the mean
 *   hinge loss on the target logit plus $\lambda_1$ times the mean MAD-scaled distance to $\xvec$, minus $\lambda_2$
 *   times the determinant of the kernel $K_{ij} = 1/(1 + \text{dist}(\cvec_i, \cvec_j))$ (a determinantal point process
 *   term that rewards spread).
 * - `faceGraph` and `faceSearch`, together `face` (Poyiadzi, Sokol, Santos-Rodríguez, De Bie and Flach, 2020, FACE):
 *   a counterfactual that is an actual data point, reached from $\xvec$ by a path of short steps through dense regions.
 *   The $f$-distance of a path $\gamma$ is $\int f(p(\gamma(t))) \lvert \gamma'(t) \rvert\, dt$; on a graph over the
 *   data (an $\epsilon$-graph, a kNN graph or an $\epsilon$-graph with KDE weights) each edge weighs
 *   $f(\hat p) \lVert \xvec_i - \xvec_j \rVert$ with $\hat p$ estimated at the edge, $f(p) = -\log p$ by default.
 *   Candidates are data points the classifier gives the target with probability at least $t_p$ and whose density is
 *   at least $t_d$; Dijkstra's algorithm finds the cheapest. Edges that break the actionability conditions are left
 *   out, so every step of the path is feasible.
 * - `growingSpheres` (Laugel et al., 2018): sample uniformly in a ball around $\xvec$, halving its radius until it
 *   holds no enemy (a point classified otherwise), then in growing spherical shells until one does; take the closest
 *   enemy and sparsify it by resetting its smallest changes while the class stays changed.
 * - Actionability (Ustun, Spangher and Liu, 2019): immutable features, features that may only rise or only fall, and
 *   bounds. `isActionable` tests a change; `projectActionable` maps a candidate onto the allowed set.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  abs,
  add,
  dense,
  div,
  expandDims,
  fromData,
  maximum,
  mean,
  mul,
  square,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { det, pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { logGamma } from 'aifn-compute/numerics/special'
import { fromEdges } from 'aifn-compute/graph'
import { dijkstra } from 'aifn-compute/graph/shortest-paths'
import { adamRule } from 'aifn-compute/optim/first-order'
import { median, multivariateKde } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Differentiable } from './gradients'

/**
 * A scalar function's value as a number.
 *
 * @param v A number, or a tensor whose first element is read.
 * @returns The number.
 */
const scalar = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

// ── Actionability ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Which changes are allowed: features that may not change, may only increase or only decrease, and bounds. */
export type Actionability = {
  /** Features that may not change. */
  immutable?: readonly Size[]
  /** Features that may only increase. */
  increasing?: readonly Size[]
  /** Features that may only decrease. */
  decreasing?: readonly Size[]
  /** Lower bound per feature ($d$ values; $-\infty$ for none). */
  lower?: VectorLike
  /** Upper bound per feature ($d$ values; $\infty$ for none). */
  upper?: VectorLike
}

/**
 * Whether moving from one point to another respects the constraints, to within a tolerance.
 *
 * @param from The starting point ($d$ values).
 * @param to The point moved to ($d$ values).
 * @param constraints The constraints (default none).
 * @param tolerance The slack allowed on every comparison.
 * @returns True when every constraint holds.
 *
 * @example Age is fixed, debt may only fall, and nothing may be negative
 * const rules = { immutable: [0], decreasing: [1], lower: [0, 0, 0] }
 * print('pay down debt:', isActionable([30, 5, 2], [30, 3, 4], rules))
 * print('grow older:', isActionable([30, 5, 2], [31, 3, 4], rules))
 * print('borrow more:', isActionable([30, 5, 2], [30, 6, 4], rules))
 */
export function isActionable(
  from: ArrayLike<number>,
  to: ArrayLike<number>,
  constraints: Actionability = {},
  tolerance = 1e-9,
): boolean {
  for (const i of constraints.immutable ?? []) if (Math.abs(to[i] - from[i]) > tolerance) return false
  for (const i of constraints.increasing ?? []) if (to[i] < from[i] - tolerance) return false
  for (const i of constraints.decreasing ?? []) if (to[i] > from[i] + tolerance) return false
  const lo = constraints.lower ? dense.toF64(constraints.lower, 'isActionable') : undefined
  const hi = constraints.upper ? dense.toF64(constraints.upper, 'isActionable') : undefined
  for (let i = 0; i < to.length; i++) {
    if (lo && to[i] < lo[i] - tolerance) return false
    if (hi && to[i] > hi[i] + tolerance) return false
  }
  return true
}

/**
 * The nearest allowed point to a candidate when starting from a given point: bounds clipped, then monotone features
 * clipped at their starting values, then immutable features reset.
 *
 * @param from The starting point ($d$ values).
 * @param z The candidate ($d$ values); not modified.
 * @param constraints The constraints (default none).
 * @returns The projected candidate ($d$ values).
 *
 * @example A candidate that breaks every rule
 * const rules = { immutable: [0], decreasing: [1], lower: [0, 0, 0] }
 * print(projectActionable([30, 5, 2], [25, 7, -1], rules))
 */
export function projectActionable(
  from: ArrayLike<number>,
  z: ArrayLike<number>,
  constraints: Actionability = {},
): Float64Array {
  const out = Float64Array.from(z)
  const lo = constraints.lower ? dense.toF64(constraints.lower, 'projectActionable') : undefined
  const hi = constraints.upper ? dense.toF64(constraints.upper, 'projectActionable') : undefined
  for (let i = 0; i < out.length; i++) {
    if (lo) out[i] = Math.max(out[i], lo[i])
    if (hi) out[i] = Math.min(out[i], hi[i])
  }
  for (const i of constraints.increasing ?? []) out[i] = Math.max(out[i], from[i])
  for (const i of constraints.decreasing ?? []) out[i] = Math.min(out[i], from[i])
  for (const i of constraints.immutable ?? []) out[i] = from[i]
  return out
}

/**
 * The median absolute deviation of each column, $\operatorname{med}_i \lvert x_{ij} - \operatorname{med} x_j \rvert$,
 * with 1 where it is 0 so that it can divide: the per-feature scale of Wachter et al. and DiCE.
 *
 * @param X The data ($n \times d$).
 * @returns The scale of each column ($d$ values).
 *
 * @example One spread column and one constant
 * print(medianAbsoluteDeviation([[1, 10], [2, 10], [4, 10], [10, 10]]))
 */
export function medianAbsoluteDeviation(X: MatrixLike): Float64Array {
  const { data, m, n } = dense.toMatrixF64(X, 'medianAbsoluteDeviation')
  return Float64Array.from({ length: n }, (_, j) => {
    const col = Array.from({ length: m }, (_, i) => data[i * n + j])
    const c = median(col)
    return median(col.map((v) => Math.abs(v - c))) || 1
  })
}

/**
 * A matrix whose rows are the given vectors.
 *
 * @param rows The vectors ($T$ of them, $d$ values each).
 * @param d Their length.
 * @returns The $T \times d$ matrix.
 */
const stackRows = (rows: readonly Float64Array[], d: Size): Tensor => {
  const out = new Float64Array(rows.length * d)
  rows.forEach((r, k) => out.set(r, k * d))
  return fromData(out, [rows.length, d])
}

// ── Wachter et al. ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `wachterCounterfactual`. */
export type WachterOptions = {
  /** The desired model output $y'$ (default 0.5). */
  target?: number
  /** Stop once $\lvert f(\xvec') - y' \rvert \le \text{tolerance}$ (default 0.05). */
  tolerance?: number
  /** The first $\lambda$ (default 0.1). */
  lambda?: number
  /** The factor $\lambda$ is multiplied by after each round that misses (default 2). */
  growth?: number
  /** At most this many rounds, one per value of $\lambda$ (default 12). */
  rounds?: Size
  /** Adam steps per round, from a fresh Adam state (default 100). */
  steps?: Size
  /** Adam's step size (default 0.05). */
  rate?: number
  /** The per-feature scale of the L1 distance (default 1; pass `medianAbsoluteDeviation(data)` as the paper). */
  scale?: VectorLike
  /** Constraints every iterate is projected onto (default none). */
  constraints?: Actionability
  /** Where the search starts (default x). */
  start?: VectorLike
}

/**
 * A Wachter counterfactual of a differentiable scalar model at $\xvec$ (see the file comment): rounds of Adam steps
 * on the penalised objective, each iterate projected onto the constraints, with $\lambda$ raised between rounds
 * until the output is within tolerance of the target. Throws `ShapeError` when `scale` does not match $\xvec$.
 *
 * @param f The model: a differentiable function of one input ($d$ values) to a score or probability.
 * @param x The instance $\xvec$ ($d$ values).
 * @param options The target, the schedule of $\lambda$, the optimiser, the scale and the constraints.
 * @returns `counterfactual`, the last iterate $\xvec'$, and `output`, $f(\xvec')$; `lambda`, the $\lambda$ of the last
 *   round; `valid`, whether the output is within tolerance; `distance`, the scaled L1 distance to $\xvec$; and the
 *   whole search: `path`, every iterate from the start ($T \times d$), with `outputs` and `lambdas` at each.
 *
 * @example The cheapest way to raise a linear score from 3 to 5
 * // f = x0 + 2 x1: one unit of x1 buys twice the score of one unit of x0, so only x1 should move.
 * const f = (x) => add(get(x, 0), mul(2, get(x, 1)))
 * const r = wachterCounterfactual(f, [1, 1], { target: 5, lambda: 1, steps: 50 })
 * print('counterfactual =', r.counterfactual, ' output =', r.output)
 * print('valid =', r.valid, ' lambda =', r.lambda, ' distance =', r.distance)
 */
export function wachterCounterfactual(
  f: Differentiable,
  x: VectorLike,
  options: WachterOptions = {},
): {
  counterfactual: Float64Array
  output: number
  lambda: number
  valid: boolean
  distance: number
  path: Tensor
  outputs: Float64Array
  lambdas: Float64Array
} {
  const xv = Float64Array.from(dense.toF64(x, 'wachterCounterfactual'))
  const d = xv.length
  const {
    target = 0.5,
    tolerance = 0.05,
    growth = 2,
    rounds = 12,
    steps = 100,
    rate = 0.05,
    constraints = {},
  } = options
  const scale = options.scale ? dense.toF64(options.scale, 'wachterCounterfactual') : new Float64Array(d).fill(1)
  if (scale.length !== d) throw new ShapeError('wachterCounterfactual', 'wachterCounterfactual: scale must match x')
  const xt = fromData(Float64Array.from(xv), [d])
  const st = fromData(Float64Array.from(scale), [d])
  let lambda = options.lambda ?? 0.1
  let z: Float64Array = options.start
    ? Float64Array.from(dense.toF64(options.start, 'wachterCounterfactual'))
    : Float64Array.from(xv)
  const output = (p: Float64Array) => scalar(f(fromData(Float64Array.from(p), [d])))
  const path: Float64Array[] = [Float64Array.from(z)]
  const outputs: number[] = [output(z)]
  const lambdas: number[] = [lambda]
  for (let r = 0; r < rounds; r++) {
    const lam = lambda
    const loss = (p: Tensor) => add(mul(lam, square(sub(f(p), target))), sum(div(abs(sub(p, xt)), st)))
    const g = grad(loss)
    const rule = adamRule({ stepSize: rate })
    let state = rule.init(fromData(z, [d]))
    for (let t = 0; t < steps; t++) {
      const p = fromData(Float64Array.from(z), [d])
      const res = rule.update(g(p) as Tensor, state, p)
      state = res.state
      const next = toFlat(add(p, res.updates as Tensor) as Tensor)
      z = projectActionable(xv, next, constraints)
      path.push(Float64Array.from(z))
      outputs.push(output(z))
      lambdas.push(lam)
    }
    if (Math.abs(outputs[outputs.length - 1] - target) <= tolerance) break
    lambda *= growth
  }
  const final = outputs[outputs.length - 1]
  let distance = 0
  for (let i = 0; i < d; i++) distance += Math.abs(z[i] - xv[i]) / scale[i]
  return {
    counterfactual: z,
    output: final,
    lambda: lambdas[lambdas.length - 1],
    valid: Math.abs(final - target) <= tolerance,
    distance,
    path: stackRows(path, d),
    outputs: Float64Array.from(outputs),
    lambdas: Float64Array.from(lambdas),
  }
}

// ── DiCE ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `diverseCounterfactuals`. */
export type DiverseOptions = {
  /** Counterfactuals to find (default 4). */
  count?: Size
  /** The class wanted: 1 (logit > 0, default) or 0. */
  desired?: 0 | 1
  /** $\lambda_1$, the weight on proximity (default 0.5, as DiCE). */
  proximityWeight?: number
  /** $\lambda_2$, the weight on diversity (default 1, as DiCE); 0 drops the term. */
  diversityWeight?: number
  /** Adam steps (default 500). */
  steps?: Size
  /** Adam's step size (default 0.05). */
  rate?: number
  /** Feature scale of the distance (default 1; DiCE uses the MAD of the training data). */
  scale?: VectorLike
  /** Constraints the start and every iterate are projected onto (default none). */
  constraints?: Actionability
  /** Standard deviation of the random start around x, in units of `scale` (default 1). */
  spread?: number
  /** Keep the counterfactuals every this many steps for display (default 10). */
  every?: Size
}

/**
 * DiCE counterfactuals of a logit model at $\xvec$ (see the file comment): $k$ counterfactuals started at random
 * around $\xvec$ and moved together by Adam, each iterate projected onto the constraints. Distances are scaled L1
 * distances divided by $d$, as DiCE. Throws `DomainError` when `count` is not a positive integer.
 *
 * @param f The model: a differentiable function of one input ($d$ values) to the logit of class 1 (batched by `vmap`).
 * @param x The instance $\xvec$ ($d$ values).
 * @param stream The random stream: the starting points are drawn from `child(stream, 'start')`.
 * @param options The number of counterfactuals, the class wanted, the weights, the optimiser and the constraints.
 * @returns `counterfactuals` ($k \times d$) and their `logits`; `valid`, which reach the desired class; `proximity`,
 *   their mean distance to $\xvec$; `diversity`, their mean pairwise distance; `sparsity`, their mean number of changed
 *   features (a change above $0.1$ times the feature's scale); `loss`, the objective before each step; and
 *   `snapshots`, the counterfactuals at the start, every `every` steps and at the end ($T \times k \times d$).
 *
 * @example Three different ways over a linear boundary
 * // The logit x0 + x1 - 3 is negative at (1, 1).
 * const f = (x) => sub(add(get(x, 0), get(x, 1)), 3)
 * const r = diverseCounterfactuals(f, [1, 1], stream(0), { count: 3, steps: 200 })
 * print('counterfactuals =', r.counterfactuals)
 * print('logits =', r.logits, ' valid =', r.valid)
 * print('proximity =', r.proximity, ' diversity =', r.diversity)
 */
export function diverseCounterfactuals(
  f: Differentiable,
  x: VectorLike,
  stream: Stream,
  options: DiverseOptions = {},
): {
  counterfactuals: Tensor
  logits: Float64Array
  valid: boolean[]
  proximity: number
  diversity: number
  sparsity: number
  loss: Float64Array
  snapshots: Tensor
} {
  const xv = Float64Array.from(dense.toF64(x, 'diverseCounterfactuals'))
  const d = xv.length
  const {
    count: k = 4,
    desired = 1,
    proximityWeight = 0.5,
    diversityWeight = 1,
    steps = 500,
    rate = 0.05,
    constraints = {},
    spread = 1,
    every = 10,
  } = options
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('diverseCounterfactuals', 'diverseCounterfactuals: count ≥ 1')
  const scale = options.scale ? dense.toF64(options.scale, 'diverseCounterfactuals') : new Float64Array(d).fill(1)
  const st = fromData(Float64Array.from(scale), [d])
  const xt = fromData(Float64Array.from(xv), [d])
  const z = desired === 1 ? 1 : -1
  const batch = vmap(f)
  // The mean scaled L1 distance per feature, as DiCE (divided by the number of features).
  const loss = (C: Tensor): Value => {
    const yloss = mean(maximum(sub(1, mul(z, batch(C) as Tensor)), 0))
    const proximity = mean(div(abs(sub(C, xt)), st))
    if (k === 1 || diversityWeight === 0) return add(yloss, mul(proximityWeight, proximity))
    const diff = div(abs(sub(expandDims(C, 1), expandDims(C, 0))), st)
    const K = div(1, add(1, mean(diff, -1)))
    return sub(add(yloss, mul(proximityWeight, proximity)), mul(diversityWeight, det(K)))
  }
  const g = grad(loss)
  const noise = toFlat(normal(child(stream, 'start'), 0, spread, { shape: [k * d] }))
  let C = new Float64Array(k * d)
  for (let r = 0; r < k; r++)
    C.set(
      projectActionable(
        xv,
        Float64Array.from({ length: d }, (_, i) => xv[i] + scale[i] * noise[r * d + i]),
        constraints,
      ),
      r * d,
    )
  const rule = adamRule({ stepSize: rate })
  let state = rule.init(fromData(C, [k, d]))
  const losses: number[] = []
  const snaps: Float64Array[] = [Float64Array.from(C)]
  for (let t = 0; t < steps; t++) {
    const P = fromData(Float64Array.from(C), [k, d])
    losses.push(scalar(loss(P)))
    const res = rule.update(g(P) as Tensor, state, P)
    state = res.state
    const next = Float64Array.from(toFlat(add(P, res.updates as Tensor) as Tensor))
    C = new Float64Array(k * d)
    for (let r = 0; r < k; r++) C.set(projectActionable(xv, next.subarray(r * d, (r + 1) * d), constraints), r * d)
    if ((t + 1) % every === 0 || t === steps - 1) snaps.push(Float64Array.from(C))
  }
  const final = fromData(C, [k, d])
  const logits = Float64Array.from(toFlat(batch(final) as Tensor))
  const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    let s = 0
    for (let i = 0; i < d; i++) s += Math.abs(a[i] - b[i]) / scale[i]
    return s / d
  }
  const rows = Array.from({ length: k }, (_, r) => C.subarray(r * d, (r + 1) * d))
  let pairs = 0
  let diversity = 0
  for (let a = 0; a < k; a++)
    for (let b = a + 1; b < k; b++) {
      diversity += dist(rows[a], rows[b])
      pairs++
    }
  let changed = 0
  for (const r of rows) for (let i = 0; i < d; i++) if (Math.abs(r[i] - xv[i]) > 0.1 * scale[i]) changed++
  const snapData = new Float64Array(snaps.length * k * d)
  snaps.forEach((s, t) => snapData.set(s, t * k * d))
  return {
    counterfactuals: final,
    logits,
    valid: Array.from(logits, (l) => (desired === 1 ? l > 0 : l < 0)),
    proximity: rows.reduce((a, r) => a + dist(r, xv), 0) / k,
    diversity: pairs > 0 ? diversity / pairs : 0,
    sparsity: changed / k,
    loss: Float64Array.from(losses),
    snapshots: fromData(snapData, [snaps.length, k, d]),
  }
}

// ── FACE ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** How FACE builds its graph and weighs its edges. */
export type FaceGraphOptions = {
  /**
   * `kde` (default): an $\epsilon$-graph whose edge weight is
   * $f(\hat p((\xvec_i + \xvec_j)/2)) \lVert \xvec_i - \xvec_j \rVert$ with $\hat p$ a Gaussian KDE; `knn`: each point
   * joined to its $k$ nearest (and they to it), weight $f(\hat p_{ij}) \lVert \xvec_i - \xvec_j \rVert$ with the
   * $k$-NN estimate $\hat p_{ij} = (k/n)/(\eta_d \lVert \xvec_i - \xvec_j \rVert^d)$, $\eta_d$ the volume of the unit
   * ball; `epsilon`: an $\epsilon$-graph, whose density estimate $(k/n)/(\eta_d \epsilon^d)$ is the same on every
   * edge, so the weight is the length.
   */
  graph?: 'kde' | 'knn' | 'epsilon'
  /**
   * The $\epsilon$ of the $\epsilon$-graphs: the longest edge (default twice the median, over the data points, of the
   * distance to the 5th nearest node).
   */
  epsilon?: number
  /** Neighbours per point of the kNN graph (default 10). */
  k?: Size
  /** The KDE's bandwidth factor (default Scott's rule, as `multivariateKde`). */
  bandwidth?: number
  /**
   * $f$ in $f(\hat p) \lVert \xvec_i - \xvec_j \rVert$, applied to the density relative to its largest value at a
   * data point ($\tilde p = \hat p / \max \hat p$, at most 1); default $-\log \tilde p$, which is at least 0, so
   * Dijkstra's algorithm applies. A negative cost is raised to 0. Not used by the `epsilon` graph.
   */
  cost?: (density: number) => number
  /** Edges $i \to j$ are kept only when the move from $\xvec_i$ to $\xvec_j$ is actionable. */
  constraints?: Actionability
}

/** A FACE graph over the data (nodes $0, \dots, n - 1$) and the query (node $n$): edges, weights and densities. */
export type FaceGraph = {
  /** The data ($n \times d$). */
  points: Tensor
  /** The query $\xvec$ ($d$ values). */
  query: Float64Array
  /** Directed edges as node pairs `[from, to]`. */
  edges: [number, number][]
  /** The weight of each edge, in the order of `edges`. */
  weights: Float64Array
  /**
   * Relative density $\tilde p \in (0, 1]$ at each data point and at the query (index $n$): the KDE for `kde`, the
   * $k$-NN estimate for `knn`, and the 5-NN estimate for `epsilon`.
   */
  density: Float64Array
  /** The $\epsilon$ used (computed even for the kNN graph, which does not use it). */
  epsilon: number
  /** The kind of graph built. */
  graph: 'kde' | 'knn' | 'epsilon'
}

/**
 * The log volume of the unit ball in $d$ dimensions, $\log \eta_d = \frac{d}{2}\log\pi - \log\Gamma(d/2 + 1)$.
 *
 * @param d The dimension.
 * @returns $\log \eta_d$.
 */
const logUnitBall = (d: Size) => (d / 2) * Math.log(Math.PI) - (logGamma(d / 2 + 1) as number)

/**
 * Build the FACE graph over the rows of $\Xmat$ and the query $\xvec$ (see `FaceGraphOptions`). Edges that break the
 * actionability constraints are left out. Throws `ShapeError` when $\xvec$ does not have $\Xmat$'s width.
 *
 * @param X The data ($n \times d$): nodes $0, \dots, n - 1$.
 * @param x The query $\xvec$ ($d$ values): node $n$.
 * @param options The kind of graph, its $\epsilon$ or $k$, the KDE's bandwidth, the cost and the constraints.
 * @returns The graph.
 *
 * @example An epsilon-graph whose edges are at most 1 long
 * // A dense line of points along x1 = 0 and one isolated point; the classifier says 1 where x0 > 2.
 * const X = [[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [3.5, 0], [4, 0], [2.2, 1.5]]
 * const g = faceGraph(X, [0, 0.8], { graph: 'epsilon', epsilon: 1 })
 * print('edges =', g.edges.length, ' from the query:', g.edges.filter(([i]) => i === 10))
 * print('density =', g.density)
 */
export function faceGraph(X: MatrixLike, x: VectorLike, options: FaceGraphOptions = {}): FaceGraph {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'faceGraph')
  const q = Float64Array.from(dense.toF64(x, 'faceGraph'))
  if (q.length !== d) throw new ShapeError('faceGraph', 'faceGraph: x must have as many features as X')
  const kind = options.graph ?? 'kde'
  const cost = options.cost ?? ((p: number) => -Math.log(Math.max(p, 1e-300)))
  const all = new Float64Array((n + 1) * d)
  all.set(data, 0)
  all.set(q, n * d)
  const N = n + 1
  const D = toFlat(pairwiseDistances(fromData(all, [N, d])))
  // The k-th nearest distance of each node (excluding itself), for the default ε and the k-NN density.
  const kth = (i: number, k: number) => {
    const row = Array.from({ length: N }, (_, j) => (j === i ? Infinity : D[i * N + j])).sort((a, b) => a - b)
    return row[Math.min(k, N - 1) - 1]
  }
  const k = options.k ?? 10
  const epsilon = options.epsilon ?? 2 * median(Array.from({ length: n }, (_, i) => kth(i, 5)))
  const logBall = logUnitBall(d)
  // Log density at every node, then relative to the largest at a data point.
  let logDensity: Float64Array
  let kde: ((at: Float64Array, m: number) => Float64Array) | null = null
  if (kind === 'knn' || kind === 'epsilon') {
    const kk = kind === 'knn' ? k : 5
    logDensity = Float64Array.from({ length: N }, (_, i) => Math.log(kk / n) - logBall - d * Math.log(kth(i, kk)))
  } else {
    const sample = fromData(Float64Array.from(data), [n, d])
    const bw = options.bandwidth
    kde = (at, m) =>
      Float64Array.from(
        toFlat(multivariateKde(sample, fromData(at, [m, d]), bw === undefined ? {} : { bandwidth: bw }).logDensity),
      )
    logDensity = kde(all, N)
  }
  let top = -Infinity
  for (let i = 0; i < n; i++) top = Math.max(top, logDensity[i])
  const density = Float64Array.from(logDensity, (l) => Math.min(1, Math.exp(l - top)))
  const edges: [number, number][] = []
  const lengths: number[] = []
  const linked = (i: number, j: number) => {
    if (kind === 'knn') return D[i * N + j] <= kth(i, k) || D[j * N + i] <= kth(j, k)
    return D[i * N + j] <= epsilon
  }
  const kthCache = kind === 'knn' ? Float64Array.from({ length: N }, (_, i) => kth(i, k)) : null
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      if (i === j) continue
      const ok = kthCache ? D[i * N + j] <= kthCache[i] || D[j * N + i] <= kthCache[j] : linked(i, j)
      if (!ok) continue
      if (
        options.constraints &&
        !isActionable(all.subarray(i * d, (i + 1) * d), all.subarray(j * d, (j + 1) * d), options.constraints)
      )
        continue
      edges.push([i, j])
      lengths.push(D[i * N + j])
    }
  let weights: Float64Array
  if (kind === 'kde' && kde) {
    const mid = new Float64Array(edges.length * d)
    edges.forEach(([i, j], e) => {
      for (let c = 0; c < d; c++) mid[e * d + c] = (all[i * d + c] + all[j * d + c]) / 2
    })
    const lm = edges.length > 0 ? kde(mid, edges.length) : new Float64Array(0)
    weights = Float64Array.from(lengths, (len, e) => Math.max(0, cost(Math.min(1, Math.exp(lm[e] - top)))) * len)
  } else if (kind === 'knn') {
    weights = Float64Array.from(lengths, (len) => {
      const lp = Math.log(k / n) - logBall - d * Math.log(Math.max(len, 1e-300))
      return Math.max(0, cost(Math.min(1, Math.exp(lp - top)))) * len
    })
  } else weights = Float64Array.from(lengths)
  return { points: fromData(Float64Array.from(data), [n, d]), query: q, edges, weights, density, epsilon, graph: kind }
}

/** Options of `faceSearch`. */
export type FaceSearchOptions = {
  /** $t_p$: candidates have target probability at least this (default 0.75). */
  predictionThreshold?: number
  /** $t_d$: candidates have relative density $\tilde p$ at least this (default 0). */
  densityThreshold?: number
}

/** A FACE counterfactual: the data point reached, its path from the query and its cost. */
export type FaceResult = {
  /** Row of $\Xmat$ chosen ($-1$ when no candidate is reachable). */
  index: number
  /** The chosen row ($d$ values), or `null` when no candidate is reachable. */
  counterfactual: Float64Array | null
  /** Node path from the query (node $n$) to the counterfactual; empty when unreachable. */
  path: number[]
  /** The path's cost, the sum of its edge weights (`Infinity` when unreachable). */
  cost: number
  /** Rows meeting both thresholds. */
  candidates: number[]
  /** Shortest-path cost from the query to every node, the query included ($n + 1$; `Infinity` when unreachable). */
  distance: Float64Array
  /**
   * The candidate nearest $\xvec$ in straight-line distance, reachable or not, for comparison ($-1$ when there is
   * none).
   */
  nearest: number
}

/**
 * Search a FACE graph for the cheapest candidate (see the file comment): Dijkstra's algorithm from the query, then the
 * reachable candidate with the smallest path cost. Throws `ShapeError` when there is not one probability per data row.
 *
 * @param g The graph, as `faceGraph` builds it.
 * @param probability The classifier's probability of the target class at each data row ($n$ values).
 * @param options The candidates' thresholds.
 * @returns The counterfactual, its path and cost, the candidates and the costs to every node.
 *
 * @example The path walks along the dense line
 * // A dense line of points along x1 = 0 and one isolated point; the classifier says 1 where x0 > 2.
 * const X = [[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [3.5, 0], [4, 0], [2.2, 1.5]]
 * const g = faceGraph(X, [0, 0.8], { graph: 'epsilon', epsilon: 1 })
 * const r = faceSearch(g, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
 * print('index =', r.index, ' path =', r.path, ' cost =', r.cost)
 * print('candidates =', r.candidates, ' nearest =', r.nearest)
 */
export function faceSearch(g: FaceGraph, probability: ArrayLike<number>, options: FaceSearchOptions = {}): FaceResult {
  const { predictionThreshold = 0.75, densityThreshold = 0 } = options
  const [n, d] = g.points.shape
  if (probability.length !== n) throw new ShapeError('faceSearch', 'faceSearch: one probability per data row')
  const graph = fromEdges(
    n + 1,
    g.edges.map(([i, j], e) => [i, j, g.weights[e]] as const),
    { directed: true },
  )
  const sp = dijkstra(graph, n)
  const distance = Float64Array.from(toFlat(sp.distance))
  const pred = toFlat(sp.predecessor)
  const X = Float64Array.from(toFlat(g.points))
  const candidates: number[] = []
  for (let i = 0; i < n; i++)
    if (probability[i] >= predictionThreshold && g.density[i] >= densityThreshold) candidates.push(i)
  let index = -1
  let best = Infinity
  let nearest = -1
  let near = Infinity
  for (const i of candidates) {
    if (distance[i] < best) {
      best = distance[i]
      index = i
    }
    let s = 0
    for (let c = 0; c < d; c++) s += (X[i * d + c] - g.query[c]) ** 2
    if (s < near) {
      near = s
      nearest = i
    }
  }
  const path: number[] = []
  if (index >= 0) {
    let v = index
    path.push(v)
    while (v !== n && path.length <= n + 1) {
      v = pred[v]
      if (v < 0) break
      path.push(v)
    }
    path.reverse()
  }
  return {
    index,
    counterfactual: index >= 0 ? Float64Array.from(X.subarray(index * d, (index + 1) * d)) : null,
    path,
    cost: best,
    candidates,
    distance,
    nearest,
  }
}

/**
 * FACE in one call: `faceGraph`, then `faceSearch` with the classifier's target probabilities on $\Xmat$.
 *
 * @param X The data ($n \times d$).
 * @param probability The classifier: from the data ($n \times d$ tensor) to the target class's probability per row.
 * @param x The query $\xvec$ ($d$ values).
 * @param options The graph's and the search's options together.
 * @returns The search's result, with the `graph`.
 *
 * @example The nearest candidate is isolated; FACE picks one reachable through the data
 * // A dense line of points along x1 = 0 and one isolated point; the classifier says 1 where x0 > 2.
 * const X = [[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [3.5, 0], [4, 0], [2.2, 1.5]]
 * const probability = (P) => toArray(P).map(([a]) => (a > 2 ? 1 : 0))
 * const r = face(X, probability, [0, 0.8], { epsilon: 1 })
 * print('nearest candidate =', r.nearest, ' chosen =', r.index, r.counterfactual)
 * print('path =', r.path, ' cost =', r.cost)
 */
export function face(
  X: MatrixLike,
  probability: (X: Tensor) => ArrayLike<number> | Tensor,
  x: VectorLike,
  options: FaceGraphOptions & FaceSearchOptions = {},
): FaceResult & { graph: FaceGraph } {
  const graph = faceGraph(X, x, options)
  const p = probability(graph.points)
  const pv = 'shape' in (p as object) && !ArrayBuffer.isView(p) ? toFlat(p as Tensor) : (p as ArrayLike<number>)
  return { ...faceSearch(graph, pv, options), graph }
}

// ── Growing spheres ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `growingSpheres`. */
export type GrowingSpheresOptions = {
  /** Points sampled per ball or shell (default 500). */
  samples?: Size
  /** The first radius $\eta$ (default 1). */
  radius?: number
  /** At most this many halvings and shells (default 200). */
  maxLayers?: Size
  /** The class wanted (default: any class other than $\xvec$'s). */
  target?: number
  /** Constraints every sample is projected onto (default none). */
  constraints?: Actionability
}

/**
 * Growing Spheres counterfactual of a classifier at $\xvec$ (see the file comment). After halving, the shells are
 * $(\eta, 2\eta], (2\eta, 3\eta], \dots$ for the last radius $\eta$ whose ball held no enemy.
 *
 * @param predict The classifier: from a batch of rows ($m \times d$ tensor) to a class per row.
 * @param x The instance $\xvec$ ($d$ values).
 * @param stream The random stream: ball or shell $l$ draws from `child(stream, 'layer', l)`.
 * @param options The samples per layer, the first radius, the most layers, the class wanted and the constraints.
 * @returns `enemy`, the closest enemy found, and `sparse`, it with its smallest changes reset (both `null` when no
 *   layer held an enemy); `layers`, each ball or shell searched, with its inner and outer radius and the enemies found;
 *   `samples`, the last layer's samples; and `evaluations`, the rows classified.
 *
 * @example Only one feature needs to move
 * // Class 1 where x0 > 2.
 * const predict = (X) => toArray(X).map(([a]) => (a > 2 ? 1 : 0))
 * const r = growingSpheres(predict, [1, 1], stream(0), { samples: 100 })
 * print('enemy =', r.enemy, ' sparse =', r.sparse)
 * print('layers =', r.layers.map((l) => [l.inner, l.outer, l.enemies]), ' evaluations =', r.evaluations)
 */
export function growingSpheres(
  predict: (X: Tensor) => ArrayLike<number> | Tensor,
  x: VectorLike,
  stream: Stream,
  options: GrowingSpheresOptions = {},
): {
  enemy: Float64Array | null
  sparse: Float64Array | null
  layers: { inner: number; outer: number; enemies: Size }[]
  samples: Tensor
  evaluations: Size
} {
  const xv = Float64Array.from(dense.toF64(x, 'growingSpheres'))
  const d = xv.length
  const { samples: m = 500, maxLayers = 200, constraints = {} } = options
  let eta = options.radius ?? 1
  let evaluations = 0
  const labels = (rows: Float64Array, count: number): Float64Array => {
    evaluations += count
    const out = predict(fromData(rows, [count, d]))
    return Float64Array.from(
      'shape' in (out as object) && !ArrayBuffer.isView(out) ? toFlat(out as Tensor) : (out as ArrayLike<number>),
    )
  }
  const own = labels(Float64Array.from(xv), 1)[0]
  const isEnemy = (c: number) => (options.target === undefined ? c !== own : c === options.target)
  // Uniform in the shell a0 < ‖z − x‖ ≤ a1: a normal direction and radius (a0^d + u(a1^d − a0^d))^{1/d}.
  let layer = 0
  const shell = (a0: number, a1: number): Float64Array => {
    const s = child(stream, 'layer', layer++)
    const dir = toFlat(normal(child(s, 'direction'), 0, 1, { shape: [m * d] }))
    const rows = new Float64Array(m * d)
    for (let r = 0; r < m; r++) {
      let norm = 0
      for (let i = 0; i < d; i++) norm += dir[r * d + i] ** 2
      norm = Math.sqrt(norm) || 1
      const u = uniform(child(s, 'radius', r)) as number
      const rad = (a0 ** d + u * (a1 ** d - a0 ** d)) ** (1 / d)
      const z = Float64Array.from({ length: d }, (_, i) => xv[i] + (rad * dir[r * d + i]) / norm)
      rows.set(projectActionable(xv, z, constraints), r * d)
    }
    return rows
  }
  const layers: { inner: number; outer: number; enemies: Size }[] = []
  const enemiesIn = (rows: Float64Array) => {
    const y = labels(rows, m)
    return Array.from({ length: m }, (_, r) => r).filter((r) => isEnemy(y[r]))
  }
  let rows = shell(0, eta)
  let found = enemiesIn(rows)
  layers.push({ inner: 0, outer: eta, enemies: found.length })
  while (found.length > 0 && layers.length < maxLayers) {
    eta /= 2
    rows = shell(0, eta)
    found = enemiesIn(rows)
    layers.push({ inner: 0, outer: eta, enemies: found.length })
  }
  let a0 = eta
  let a1 = 2 * eta
  while (found.length === 0 && layers.length < maxLayers) {
    rows = shell(a0, a1)
    found = enemiesIn(rows)
    layers.push({ inner: a0, outer: a1, enemies: found.length })
    a0 = a1
    a1 += eta
  }
  if (found.length === 0) return { enemy: null, sparse: null, layers, samples: fromData(rows, [m, d]), evaluations }
  let best = -1
  let bd = Infinity
  for (const r of found) {
    let s = 0
    for (let i = 0; i < d; i++) s += (rows[r * d + i] - xv[i]) ** 2
    if (s < bd) {
      bd = s
      best = r
    }
  }
  const enemy = Float64Array.from(rows.subarray(best * d, (best + 1) * d))
  // Feature selection: reset the smallest change while the class stays changed.
  let sparse = Float64Array.from(enemy)
  for (;;) {
    let j = -1
    let small = Infinity
    for (let i = 0; i < d; i++) {
      const c = Math.abs(sparse[i] - xv[i])
      if (c > 0 && c < small) {
        small = c
        j = i
      }
    }
    if (j < 0) break
    const trial = Float64Array.from(sparse)
    trial[j] = xv[j]
    if (!isEnemy(labels(trial, 1)[0])) break
    sparse = trial
  }
  return { enemy, sparse, layers, samples: fromData(rows, [m, d]), evaluations }
}
