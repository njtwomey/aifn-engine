/** Link prediction (aifn-methods/neural/graph), against the site's self-check (homophily-and-links). */
import { describe, expect, it } from 'vitest'
import { linkPredictionRun, splitEdges } from 'aifn-methods/neural/graph'
import { contextualBlockModel } from 'aifn-methods/data/synthetic'
import { auroc } from 'aifn-compute/learning/metrics'
import { completeGraph } from 'aifn-compute/graph/structures'
import { stream } from 'aifn-compute/foundation/random'
import { eye } from 'aifn-compute/foundation/tensor'

const key = (u: number, v: number) => (u < v ? `${u},${v}` : `${v},${u}`)
const d = contextualBlockModel(stream('link/graph'), { sizes: [20, 20, 20], pIn: 0.3, pOut: 0.02 })
const split = splitEdges(stream('link/split'), d.graph, 0.15)

describe('splitEdges', () => {
  it('counts add up, held-out edges are not trained on, negatives are non-edges, as many as positives', () => {
    const all = new Set(d.graph.edges.map((e) => key(e.from, e.to)))
    const kept = new Set(split.train.edges.map((e) => key(e.from, e.to)))
    expect(split.train.edges.length + split.testPositive.length).toBe(d.graph.edges.length)
    expect(split.testPositive.some(([u, v]) => kept.has(key(u, v)))).toBe(false)
    expect(split.testNegative.some(([u, v]) => all.has(key(u, v)))).toBe(false)
    expect(split.testNegative.length).toBe(split.testPositive.length)
    expect(new Set(split.testNegative.map(([u, v]) => key(u, v))).size).toBe(split.testNegative.length)
  })
  it('throws when the share leaves no test edges, or the graph has too few non-edges', () => {
    expect(() => splitEdges(stream(0), d.graph, 0)).toThrow()
    expect(() => splitEdges(stream(0), completeGraph(5), 0.5)).toThrow()
  })
})

describe('linkPredictionRun', () => {
  it('AUC by counting: positives (2.1, 0.9, 0.4) against negatives (1.2, 0.3, -0.5) is 7/9', () => {
    expect(auroc([1, 1, 1, 0, 0, 0], [2.1, 0.9, 0.4, 1.2, 0.3, -0.5])).toBeCloseTo(7 / 9, 12)
  })
  it('the GAE lifts the held-out AUC above 0.68 within 200 steps on the 60-node block model', () => {
    let last = 0
    let count = 0
    for (const c of linkPredictionRun({ split, x: eye(60), steps: 200, stepSize: 0.02, every: 50, seed: 1 })) {
      last = c.auc
      count++
      expect(c.scores.length).toBe(split.testPositive.length + split.testNegative.length)
    }
    expect(count).toBe(5)
    expect(last).toBeGreaterThan(0.68)
  })
})
