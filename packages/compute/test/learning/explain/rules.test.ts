/**
 * Rule explanations by their laws: KL bounds sit at the requested divergence; KL-LUCB identifies the best Bernoulli
 * arms; an anchor of a threshold model is the one predicate that decides it, with precision 1 and the bin's coverage,
 * and its estimated precision agrees with a large independent sample; a decision list learns a conjunction; a tree's
 * rules partition the space and give the tree's predictions; fidelity is agreement or R².
 */
import { describe, expect, it } from 'vitest'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  anchor,
  applyRuleList,
  bernoulliKl,
  fidelity,
  klBounds,
  klLucb,
  quantileEdges,
  ruleCoverage,
  ruleList,
  satisfies,
  treeRules,
  type ShapTree,
} from 'aifn-compute/learning/explain'

const gaussianRows = (seed: string, n: number, d: number) =>
  fromData(Float64Array.from(toFlat(normal(stream(seed), 0, 1, { shape: [n * d] }))), [n, d])

describe('KL confidence bounds', () => {
  it('sit at the requested divergence', () => {
    for (const [p, level] of [
      [0.3, 0.05],
      [0.9, 0.01],
      [0.5, 0.2],
    ]) {
      const b = klBounds(p, level)
      expect(b.lower).toBeLessThan(p)
      expect(b.upper).toBeGreaterThan(p)
      expect(bernoulliKl(p, b.upper)).toBeCloseTo(level, 6)
      expect(bernoulliKl(p, b.lower)).toBeCloseTo(level, 6)
    }
  })
})

describe('KL-LUCB', () => {
  it('identifies the two best arms', () => {
    const means = [0.2, 0.85, 0.5, 0.9, 0.6]
    const s = stream('lucb')
    let k = 0
    const draw = (a: number, m: number) => {
      const r = child(s, 'draw', k++)
      let hits = 0
      for (let i = 0; i < m; i++) if ((uniform(r) as number) < means[a]) hits++
      return hits
    }
    const stats = means.map(() => ({ draws: 0, successes: 0 }))
    const res = klLucb(draw, stats, { top: 2, epsilon: 0.05, delta: 0.05 })
    expect([...res.chosen].sort()).toEqual([1, 3])
  })
})

describe('anchors', () => {
  const background = gaussianRows('anchor-bg', 2000, 3)
  const predict = (X: Tensor) => {
    const x = dense.data(X)
    return Float64Array.from({ length: X.shape[0] }, (_, i) => (x[i * 3] > 0 ? 1 : 0))
  }
  it('of a threshold model is the deciding predicate', () => {
    const r = anchor(predict, [1.5, 0, 0], background, stream('anchor'), { threshold: 0.95 })
    expect(r.valid).toBe(true)
    expect(r.rule).toHaveLength(1)
    expect(r.rule[0].feature).toBe(0)
    expect(r.rule[0].upper).toBe(Infinity)
    expect(r.precision).toBe(1)
    expect(r.coverage).toBeCloseTo(ruleCoverage(background, r.rule), 12)
    expect(r.coverage).toBeGreaterThan(0.2)
    expect(r.coverage).toBeLessThan(0.3)
  })
  it('estimates precision as a large independent sample does', () => {
    // A noisier model: the label also depends on x₁; the anchor needs both or accepts some error.
    const noisy = (X: Tensor) => {
      const x = dense.data(X)
      return Float64Array.from({ length: X.shape[0] }, (_, i) => (x[i * 3] + 0.6 * x[i * 3 + 1] > 0 ? 1 : 0))
    }
    const r = anchor(noisy, [0.8, 0.3, 0], background, stream('anchor-2'), { threshold: 0.9 })
    // Brute force: the anchor's region under the same perturbation, by rejection from the background.
    const bg = dense.data(background)
    let n = 0
    let hit = 0
    const label = noisy(fromData(Float64Array.of(0.8, 0.3, 0), [1, 3]))[0]
    for (let i = 0; i < 2000; i++) {
      const row = bg.subarray(i * 3, i * 3 + 3)
      if (!satisfies(row, r.rule)) continue
      n++
      if (noisy(fromData(Float64Array.from(row), [1, 3]))[0] === label) hit++
    }
    expect(n).toBeGreaterThan(50)
    expect(Math.abs(hit / n - r.precision)).toBeLessThan(0.08)
  })
})

describe('decision lists', () => {
  it('learn a conjunction', () => {
    const X = gaussianRows('list', 400, 3)
    const x = dense.data(X)
    const y = Float64Array.from({ length: 400 }, (_, i) => (x[i * 3] > 0 && x[i * 3 + 1] > 0 ? 1 : 0))
    const list = ruleList(X, y, { bins: 4 })
    const { labels, fired } = applyRuleList(list, X)
    expect(fidelity(y, labels)).toBeGreaterThan(0.95)
    expect(list.rules.length).toBeGreaterThan(0)
    for (let i = 0; i < 400; i++)
      if (fired[i] >= 0) expect(satisfies(x.subarray(i * 3, i * 3 + 3), list.rules[fired[i]].rule)).toBe(true)
  })
  it('use quantile edges per feature', () => {
    const e = quantileEdges([[1], [2], [3], [4]], 2)
    expect(Array.from(e[0])).toEqual([2.5])
  })
})

describe('tree rules and fidelity', () => {
  const tree: ShapTree = {
    root: 0,
    nodes: [
      { feature: 0, threshold: 0, weight: 10, value: [0], children: [1, 2] },
      { feature: 1, threshold: 1, weight: 6, value: [0], children: [3, 4] },
      { feature: -1, threshold: 0, weight: 4, value: [5], children: [] },
      { feature: -1, threshold: 0, weight: 4, value: [1], children: [] },
      { feature: -1, threshold: 0, weight: 2, value: [2], children: [] },
    ].map((n, id) => ({ ...n, id })),
  }
  it('partition the space and give the tree’s predictions', () => {
    const rules = treeRules(tree)
    expect(rules).toHaveLength(3)
    for (const p of [
      [-1, 0, 1],
      [-1, 2, 2],
      [3, 0, 5],
    ]) {
      const hits = rules.filter((r) => satisfies(p.slice(0, 2), r.rule))
      expect(hits).toHaveLength(1)
      expect(hits[0].value[0]).toBe(p[2])
    }
  })
  it('measures agreement and R²', () => {
    expect(fidelity([1, 0, 1, 1], [1, 1, 1, 1])).toBe(0.75)
    expect(fidelity([1, 2, 3], [1, 2, 3], 'r2')).toBe(1)
    expect(fidelity([1, 2, 3], [2, 2, 2], 'r2')).toBeCloseTo(0, 12)
  })
})
