/**
 * Counterfactual explanations: the smallest change to an input x that changes a model's decision, and constraints on
 * which changes a person could act on.
 *
 * - `wachterCounterfactual` (Wachter, Mittelstadt and Russell, 2017): minimise λ (f(x′) − y′)² + Σᵢ |x′ᵢ − xᵢ| / MADᵢ by
 *   gradient steps (Adam), raising λ until |f(x′) − y′| ≤ ε. The L1 distance scaled by each feature's median absolute
 *   deviation favours sparse changes.
 * - `diverseCounterfactuals` (Mothilal, Sharma and Tan, 2020, DiCE): k counterfactuals at once, minimising the mean
 *   hinge loss on the target logit plus λ₁ × the mean MAD-scaled distance to x, minus λ₂ × the determinant of the
 *   kernel Kᵢⱼ = 1/(1 + dist(cᵢ, cⱼ)) (a determinantal point process term that rewards spread).
 * - `faceGraph` and `faceSearch`, together `face` (Poyiadzi, Sokol, Santos-Rodríguez, De Bie and Flach, 2020, FACE):
 *   a counterfactual that is an actual data point, reached from x by a path of short steps through dense regions.
 *   The f-distance of a path γ is ∫ f(p(γ(t))) |γ′(t)| dt; on a graph over the data (an ε-graph, a kNN graph or an
 *   ε-graph with KDE weights) each edge weighs f(p̂) ‖xᵢ − xⱼ‖ with p̂ estimated at the edge, f(p) = −log p by default.
 *   Candidates are data points the classifier gives the target with probability ≥ t_p and whose density is ≥ t_d;
 *   Dijkstra's algorithm finds the cheapest. Edges that break the actionability conditions are left out, so every step
 *   of the path is feasible.
 * - `growingSpheres` (Laugel et al., 2018): sample uniformly in a ball around x, halving its radius until it holds no
 *   enemy (a point classified otherwise), then in growing spherical shells until one does; take the closest enemy and
 *   sparsify it by resetting its smallest changes while the class stays changed.
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

const scalar = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

// ── Actionability ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Which changes are allowed: features that may not change, may only increase or only decrease, and bounds. */
export type Actionability = {
  immutable?: readonly Size[]
  increasing?: readonly Size[]
  decreasing?: readonly Size[]
  /** Per-feature bounds [d] (±Infinity for none). */
  lower?: VectorLike
  upper?: VectorLike
}

/** True when moving from `from` to `to` respects the constraints (to within `tolerance`). */
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

/** The nearest allowed point to z when starting from `from`: immutable features reset, monotone ones and bounds clipped. */
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

/** The median absolute deviation of each column of X [n, d], med |xᵢⱼ − med xⱼ| (1 where it is 0). */
export function medianAbsoluteDeviation(X: MatrixLike): Float64Array {
  const { data, m, n } = dense.toMatrixF64(X, 'medianAbsoluteDeviation')
  return Float64Array.from({ length: n }, (_, j) => {
    const col = Array.from({ length: m }, (_, i) => data[i * n + j])
    const c = median(col)
    return median(col.map((v) => Math.abs(v - c))) || 1
  })
}

/** Rows [T, d] from a list of vectors. */
const stackRows = (rows: readonly Float64Array[], d: Size): Tensor => {
  const out = new Float64Array(rows.length * d)
  rows.forEach((r, k) => out.set(r, k * d))
  return fromData(out, [rows.length, d])
}

// ── Wachter et al. ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `wachterCounterfactual`. */
export type WachterOptions = {
  /** The desired model output y′ (default 0.5). */
  target?: number
  /** Stop once |f(x′) − y′| ≤ tolerance (default 0.05). */
  tolerance?: number
  /** The first λ (default 0.1), multiplied by `growth` (default 2) after each round that misses. */
  lambda?: number
  growth?: number
  /** At most this many rounds of λ (default 12) and Adam steps per round (default 100) at step size `rate` (0.05). */
  rounds?: Size
  steps?: Size
  rate?: number
  /** The per-feature scale of the L1 distance (default 1; pass `medianAbsoluteDeviation(data)` as the paper). */
  scale?: VectorLike
  constraints?: Actionability
  /** Where the search starts (default x). */
  start?: VectorLike
}

/**
 * A Wachter counterfactual of the scalar model f at x [d]: the point, its output, the λ reached, whether it is within
 * tolerance, its scaled L1 distance, and the path of iterates [T, d] with the output and λ at each.
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
  /** λ₁ on proximity (default 0.5) and λ₂ on diversity (default 1), as DiCE. */
  proximityWeight?: number
  diversityWeight?: number
  /** Adam steps (default 500) and step size (default 0.05). */
  steps?: Size
  rate?: number
  /** Feature scale of the distance (default 1; DiCE uses the MAD of the training data). */
  scale?: VectorLike
  constraints?: Actionability
  /** Standard deviation of the random start around x, in units of `scale` (default 1). */
  spread?: number
  /** Keep the counterfactuals every this many steps for display (default 10). */
  every?: Size
}

/**
 * DiCE counterfactuals of the logit model f at x [d] (see the module comment): the counterfactuals [k, d] and their
 * logits, which of them reach the desired class, the mean scaled distance to x (proximity), the mean pairwise scaled
 * distance (diversity), the mean count of changed features (|Δ| > 0.1 × scale), the loss per step and snapshots
 * [T, k, d].
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
   * `kde` (default): an ε-graph whose edge weight is f(p̂((xᵢ + xⱼ)/2)) ‖xᵢ − xⱼ‖ with p̂ a Gaussian KDE; `knn`: each
   * point joined to its k nearest, weight f(p̂ᵢⱼ) ‖xᵢ − xⱼ‖ with the k-NN estimate p̂ᵢⱼ = (k/n)/(η_d ‖xᵢ − xⱼ‖^d);
   * `epsilon`: an ε-graph, whose density estimate (k/n)/(η_d ε^d) is the same on every edge, so the weight is the length.
   */
  graph?: 'kde' | 'knn' | 'epsilon'
  /** The ε of the ε-graphs: the longest edge (default the median distance to the 5th nearest neighbour × 2). */
  epsilon?: number
  /** Neighbours per point of the kNN graph (default 10). */
  k?: Size
  /** The KDE's bandwidth factor (default Scott's rule, as `multivariateKde`). */
  bandwidth?: number
  /**
   * f in f(p̂) ‖xᵢ − xⱼ‖, applied to the density relative to its largest value at a data point (p̃ = p̂/max p̂, at most
   * 1); default −log p̃, which is ≥ 0, so Dijkstra's algorithm applies.
   */
  cost?: (density: number) => number
  /** Edges i → j are kept only when the move xᵢ → xⱼ is actionable. */
  constraints?: Actionability
}

/** A FACE graph over the data and the query (node n): edges, weights and densities. */
export type FaceGraph = {
  /** The data [n, d] and the query [d]. */
  points: Tensor
  query: Float64Array
  /** Directed edges as node pairs [from, to] and their weights. */
  edges: [number, number][]
  weights: Float64Array
  /** Relative density p̃ ∈ (0, 1] at each data point and at the query (index n). */
  density: Float64Array
  epsilon: number
  graph: 'kde' | 'knn' | 'epsilon'
}

const logUnitBall = (d: Size) => (d / 2) * Math.log(Math.PI) - (logGamma(d / 2 + 1) as number)

/** Build the FACE graph over the rows of X [n, d] and the query x [d] (see `FaceGraphOptions`). */
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
  /** t_p: candidates have target probability ≥ this (default 0.75). */
  predictionThreshold?: number
  /** t_d: candidates have relative density p̃ ≥ this (default 0). */
  densityThreshold?: number
}

/** A FACE counterfactual: the data point reached, its path from the query and its cost. */
export type FaceResult = {
  /** Row of X chosen (−1 when no candidate is reachable). */
  index: number
  counterfactual: Float64Array | null
  /** Node path from the query (n) to the counterfactual; empty when unreachable. */
  path: number[]
  cost: number
  /** Rows meeting both thresholds. */
  candidates: number[]
  /** Shortest-path cost from the query to every node (Infinity when unreachable). */
  distance: Float64Array
  /** The candidate nearest x in straight-line distance, for comparison (−1 when there is none). */
  nearest: number
}

/**
 * Search a FACE graph for the cheapest candidate, given the target-class probability of each row of the graph's data
 * [n] (see the module comment).
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

/** FACE in one call: `faceGraph` then `faceSearch` with the classifier's target probabilities on X. */
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
  /** The first radius η (default 1). */
  radius?: number
  /** At most this many halvings and shells (default 200). */
  maxLayers?: Size
  /** The class wanted (default: any class other than x's). */
  target?: number
  constraints?: Actionability
}

/**
 * Growing Spheres counterfactual of a classifier `predict` (a class per row of a batch [m, d]) at x [d]: the closest
 * enemy found, its sparsified version (`sparse`), the shells searched (inner and outer radius and the enemies found in
 * each) and the last shell's samples [samples, d].
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
