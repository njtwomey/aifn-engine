/**
 * Counterfactual explanations by their laws: Wachter's search reaches the target and, with an L1 distance, changes
 * only the feature with the largest weight per unit scale of a linear model; constraints hold along every path; DiCE's
 * counterfactuals are valid and its diversity term spreads them; FACE's path cost equals the sum of its edge weights,
 * is the least over all candidates (Floyd–Warshall), respects the edge length and actionability, and its weights follow
 * the paper's KDE and k-NN formulas; Growing Spheres finds the nearest enemy of a half-space classifier and sparsifies
 * it to the one feature that matters.
 */
import { describe, expect, it } from 'vitest'
import { fromEdges } from 'aifn-compute/graph'
import { floydWarshall } from 'aifn-compute/graph/shortest-paths'
import { stream } from 'aifn-compute/foundation/random'
import { add, dense, div, exp, fromData, get, mul, neg, sum, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { multivariateKde } from 'aifn-compute/probability/stats'
import {
  diverseCounterfactuals,
  face,
  faceGraph,
  faceSearch,
  growingSpheres,
  isActionable,
  medianAbsoluteDeviation,
  projectActionable,
  wachterCounterfactual,
} from 'aifn-compute/learning/explain'

const W = [3, 0.5]
const B = -1
/** σ(wᵀx + b) of one row [2]. */
const prob = (x: Tensor) => div(1, add(1, exp(neg(add(sum(mul(x, fromData(Float64Array.from(W), [2]))), B)))))
const logit = (x: Tensor) => add(add(mul(W[0], get(x, 0)), mul(W[1], get(x, 1))), B)

describe('actionability', () => {
  const c = { immutable: [0], increasing: [1], upper: [10, 2, 10] }
  it('tests and projects', () => {
    expect(isActionable([1, 1, 1], [1, 2, 0], c)).toBe(true)
    expect(isActionable([1, 1, 1], [1.5, 2, 0], c)).toBe(false)
    expect(isActionable([1, 1, 1], [1, 0.5, 0], c)).toBe(false)
    expect(isActionable([1, 1, 1], [1, 3, 0], c)).toBe(false)
    expect(Array.from(projectActionable([1, 1, 1], [4, 0, 5], c))).toEqual([1, 1, 5])
    expect(Array.from(projectActionable([1, 1, 1], [4, 9, 5], c))).toEqual([1, 2, 5])
  })
  it('scales by the median absolute deviation', () => {
    expect(
      Array.from(
        medianAbsoluteDeviation([
          [1, 5],
          [2, 5],
          [4, 5],
        ]),
      ),
    ).toEqual([1, 1])
  })
})

describe('Wachter counterfactuals', () => {
  it('reach the target, changing only the cheapest feature under L1', () => {
    const r = wachterCounterfactual(prob, [0, 0], { target: 0.7, tolerance: 0.02, steps: 200 })
    expect(r.valid).toBe(true)
    expect(Math.abs(r.output - 0.7)).toBeLessThanOrEqual(0.02)
    expect(Math.abs(r.counterfactual[1])).toBeLessThan(0.05)
    expect(r.counterfactual[0]).toBeGreaterThan(0.3)
    expect(r.path.shape[1]).toBe(2)
  })
  it('keep immutable features along the whole path', () => {
    const r = wachterCounterfactual(prob, [0, 0], { target: 0.7, constraints: { immutable: [0] }, steps: 200 })
    const p = toFlat(r.path)
    for (let t = 0; t < r.path.shape[0]; t++) expect(p[t * 2]).toBe(0)
    expect(r.counterfactual[1]).toBeGreaterThan(1.5)
  })
})

describe('DiCE', () => {
  it('finds valid counterfactuals, spread out by the diversity term', () => {
    const spread = diverseCounterfactuals(logit, [0, 0], stream('dice'), { count: 3, steps: 300 })
    const tight = diverseCounterfactuals(logit, [0, 0], stream('dice'), { count: 3, steps: 300, diversityWeight: 0 })
    expect(spread.valid.every(Boolean)).toBe(true)
    expect(tight.valid.every(Boolean)).toBe(true)
    expect(spread.diversity).toBeGreaterThan(tight.diversity)
    expect(spread.snapshots.shape).toEqual([31, 3, 2])
  })
  it('respects monotone constraints', () => {
    const r = diverseCounterfactuals(logit, [0, 0], stream('dice-c'), {
      count: 2,
      steps: 200,
      constraints: { increasing: [0, 1] },
    })
    const c = toFlat(r.counterfactuals)
    for (const v of c) expect(v).toBeGreaterThanOrEqual(0)
  })
})

/** Two dense blobs joined by a corridor, as rows [n, 2]. */
function corridor(): number[][] {
  const rows: number[][] = []
  for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) rows.push([i * 0.3, j * 0.3])
  for (let i = 0; i < 10; i++) rows.push([1.8 + i * 0.3, 0])
  for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) rows.push([4.8 + i * 0.3, j * 0.3])
  return rows
}

describe('FACE', () => {
  const X = corridor()
  const target = (Z: Tensor) => Float64Array.from({ length: Z.shape[0] }, (_, i) => (toFlat(Z)[i * 2] > 4.5 ? 1 : 0))
  for (const graph of ['kde', 'knn', 'epsilon'] as const) {
    it(`finds the cheapest candidate along short edges (${graph})`, () => {
      const r = face(X, target, [0, 0.9], { graph, epsilon: 0.45, k: 4, predictionThreshold: 0.5 })
      expect(r.index).toBeGreaterThanOrEqual(0)
      expect(r.candidates).toContain(r.index)
      const n = X.length
      expect(r.path[0]).toBe(n)
      expect(r.path.at(-1)).toBe(r.index)
      // The path's cost is the sum of its edge weights, each edge within ε for the ε-graphs.
      const weight = new Map(r.graph.edges.map(([a, b], e) => [`${a}-${b}`, r.graph.weights[e]]))
      let total = 0
      for (let s = 1; s < r.path.length; s++) total += weight.get(`${r.path[s - 1]}-${r.path[s]}`) as number
      expect(total).toBeCloseTo(r.cost, 10)
      // No candidate is cheaper (all-pairs shortest paths).
      const all = floydWarshall(
        fromEdges(
          n + 1,
          r.graph.edges.map(([a, b], e) => [a, b, r.graph.weights[e]] as const),
        ),
      )
      const D = toFlat(all.distance)
      for (const c of r.candidates) expect(D[n * (n + 1) + c]).toBeGreaterThanOrEqual(r.cost - 1e-9)
    })
  }
  it('weights KDE edges by −log of the relative density at the midpoint', () => {
    const g = faceGraph(X, [0, 0.9], { graph: 'kde', epsilon: 0.45 })
    const sample = fromData(Float64Array.from(X.flat()), [X.length, 2])
    const top = Math.max(...Array.from(toFlat(multivariateKde(sample, sample).logDensity)))
    const [a, b] = g.edges[7]
    const pa = X[a] ?? [0, 0.9]
    const pb = X[b] ?? [0, 0.9]
    const mid = fromData(Float64Array.of((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2), [1, 2])
    const lp = toFlat(multivariateKde(sample, mid).logDensity)[0]
    const len = Math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    expect(g.weights[7]).toBeCloseTo(-Math.min(0, lp - top) * len, 10)
  })
  it('weights kNN edges by the k-NN density estimate at the edge length', () => {
    const g = faceGraph(X, [0, 0.9], { graph: 'knn', k: 4 })
    const n = X.length
    const [a, b] = g.edges[3]
    const len = Math.hypot(...[0, 1].map((c) => (X[a] ?? [0, 0.9])[c] - (X[b] ?? [0, 0.9])[c]))
    // log p̂ = log(k/n) − log(π) − 2 log len in two dimensions, relative to the largest node estimate.
    const node = (i: number) => {
      const p = X[i] ?? [0, 0.9]
      const ds = X.map((q) => Math.hypot(q[0] - p[0], q[1] - p[1]))
      if (i === n) ds.push(Infinity)
      else ds.push(Math.hypot(p[0], p[1] - 0.9))
      ds[i] = Infinity
      const kth = ds.sort((u, v) => u - v)[3]
      return Math.log(4 / n) - Math.log(Math.PI) - 2 * Math.log(kth)
    }
    const top = Math.max(...Array.from({ length: n }, (_, i) => node(i)))
    const lp = Math.log(4 / n) - Math.log(Math.PI) - 2 * Math.log(len)
    expect(g.weights[3]).toBeCloseTo(-Math.min(0, lp - top) * len, 9)
  })
  it('keeps every step actionable', () => {
    const g = faceGraph(X, [0, 0.9], { graph: 'epsilon', epsilon: 0.45, constraints: { increasing: [0] } })
    const pts = [...X, [0, 0.9]]
    for (const [a, b] of g.edges) expect(pts[b][0]).toBeGreaterThanOrEqual(pts[a][0] - 1e-12)
    const r = faceSearch(g, target(fromData(Float64Array.from(X.flat()), [X.length, 2])), { predictionThreshold: 0.5 })
    for (let s = 1; s < r.path.length; s++) expect(pts[r.path[s]][0]).toBeGreaterThanOrEqual(pts[r.path[s - 1]][0])
  })
  it('honours the density threshold', () => {
    const g = faceGraph(X, [0, 0.9], { graph: 'kde', epsilon: 0.45 })
    const p = target(fromData(Float64Array.from(X.flat()), [X.length, 2]))
    const r = faceSearch(g, p, { predictionThreshold: 0.5, densityThreshold: 0.6 })
    for (const c of r.candidates) expect(g.density[c]).toBeGreaterThanOrEqual(0.6)
  })
})

describe('Growing Spheres', () => {
  const predict = (Z: Tensor) => {
    const z = dense.data(Z)
    return Float64Array.from({ length: Z.shape[0] }, (_, i) => (z[i * 3] > 1 ? 1 : 0))
  }
  it('finds an enemy just past the boundary and keeps only the feature that matters', () => {
    const r = growingSpheres(predict, [0, 0, 0], stream('gs'), { radius: 0.5, samples: 400 })
    expect(r.enemy).not.toBeNull()
    const e = r.enemy as Float64Array
    expect(e[0]).toBeGreaterThan(1)
    expect(Math.hypot(...e)).toBeLessThan(1 + 0.5 + 1e-9)
    const s = r.sparse as Float64Array
    expect(s[1]).toBe(0)
    expect(s[2]).toBe(0)
    expect(s[0]).toBe(e[0])
    expect(r.layers.at(-1)?.enemies).toBeGreaterThan(0)
  })
})
