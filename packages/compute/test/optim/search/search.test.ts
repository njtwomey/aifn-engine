import { describe, expect, it } from 'vitest'
import { trace } from 'aifn-compute/foundation/trace'
import {
  filterRedundant,
  offerResult,
  refinementSearch,
  refinementSearchSteps,
  searchHistory,
  type SearchSpace,
  type SearchState,
  type SearchVisit,
} from 'aifn-compute/optim/search'

// A toy space: nodes are increasing index sets of items with signed weights; quality is the weight sum minus a cost
// per item, and the optimistic estimate adds every positive weight still available after the last index.
const WEIGHTS = [3, -1, 2.5, -4, 1.5, 2, -0.5, 1, -2, 0.75]
const COST = 0.4
const quality = (s: readonly number[]) => s.reduce((a, i) => a + WEIGHTS[i] - COST, 0)
const space: SearchSpace<readonly number[]> = {
  root: [],
  refine: (s) => {
    const last = s.length ? s[s.length - 1] : -1
    return WEIGHTS.map((_, i) => i)
      .filter((i) => i > last)
      .map((i) => [...s, i])
  },
  quality,
  bound: (s) => {
    const last = s.length ? s[s.length - 1] : -1
    let extra = 0
    for (let i = last + 1; i < WEIGHTS.length; i++) extra += Math.max(0, WEIGHTS[i] - COST)
    return quality(s) + extra
  },
}

/** Every node of depth 1..d by brute force. */
function enumerate(d: number): (readonly number[])[] {
  const out: (readonly number[])[] = []
  const walk = (s: readonly number[]) => {
    if (s.length) out.push(s)
    if (s.length < d) for (const c of space.refine(s)) walk(c)
  }
  walk([])
  return out
}
const topQualities = (nodes: (readonly number[])[], k: number) =>
  nodes
    .map(quality)
    .sort((a, b) => b - a)
    .slice(0, k)
const qualities = (s: SearchState<readonly number[]>) => s.results.map((r) => r.quality)

describe('refinementSearch', () => {
  for (const depth of [2, 3, 4])
    for (const k of [1, 5, 12]) {
      const truth = topQualities(enumerate(depth), k)
      it(`exhaustive depth-first equals brute force (depth ${depth}, k ${k})`, () => {
        const s = refinementSearch(space, { strategy: 'depth-first', maxDepth: depth, k, prune: false })
        expect(qualities(s)).toEqual(truth)
        expect(s.pruned).toBe(0)
      })
      for (const strategy of ['depth-first', 'best-first', 'breadth-first'] as const)
        it(`${strategy} branch and bound finds the same top k (depth ${depth}, k ${k})`, () => {
          const s = refinementSearch(space, { strategy, maxDepth: depth, k })
          expect(qualities(s).map((q) => +q.toFixed(12))).toEqual(truth.map((q) => +q.toFixed(12)))
        })
    }

  it('prunes, and evaluates fewer nodes than exhaustive search', () => {
    const full = refinementSearch(space, { strategy: 'depth-first', maxDepth: 4, k: 3, prune: false })
    const bb = refinementSearch(space, { strategy: 'best-first', maxDepth: 4, k: 3 })
    expect(bb.pruned).toBeGreaterThan(0)
    expect(bb.evaluated).toBeLessThan(full.evaluated)
  })

  it('never prunes a node with a refinement better than the k-th result at the time', () => {
    for (const strategy of ['depth-first', 'best-first', 'breadth-first', 'beam'] as const) {
      const t = trace(refinementSearchSteps(space, { strategy, maxDepth: 4, k: 4, beamWidth: 3 }), undefined, 10_000)
      for (const s of t.steps) {
        const pruned = [...s.generated.filter((v) => v.fate === 'pruned'), ...s.discarded]
        for (const v of pruned) {
          const below = enumerate(4).filter(
            (n) => n.length > v.node.length && v.node.every((x, i) => n[i] === x) && n.length <= 4,
          )
          for (const n of below) expect(quality(n)).toBeLessThanOrEqual(s.threshold + 1e-12)
          expect(v.bound).toBeLessThanOrEqual(s.threshold)
        }
      }
    }
  })

  it('beam search wide enough is exhaustive; width 1 is greedy', () => {
    const wide = refinementSearch(space, { strategy: 'beam', beamWidth: 1000, maxDepth: 3, k: 5, prune: false })
    expect(qualities(wide)).toEqual(topQualities(enumerate(3), 5))
    const greedy = trace(refinementSearchSteps(space, { strategy: 'beam', beamWidth: 1, maxDepth: 4 }), undefined, 10)
    for (const s of greedy.steps.slice(1)) expect(s.frontier.length).toBeLessThanOrEqual(1)
    // Each level keeps the best child of the previous beam.
    const s1 = greedy.steps[1]
    const best = Math.max(...s1.generated.map((v) => v.quality))
    expect(s1.frontier[0].quality).toBe(best)
  })

  it('records a consistent history: one generation per node, expansions after generation', () => {
    const t = trace(refinementSearchSteps(space, { strategy: 'best-first', maxDepth: 3, k: 3 }), undefined, 10_000)
    const h = searchHistory(t.steps)
    expect(h.visits.length).toBe(t.final.evaluated)
    h.visits.forEach((v, id) => {
      expect(v.id).toBe(id)
      if (h.expandedAt[id] >= 0) expect(h.expandedAt[id]).toBeGreaterThan(h.generatedAt[id])
    })
    expect(t.final.terminated).toBe(true)
    expect(t.final.frontier.length).toBe(0)
  })

  it('stops at maxNodes', () => {
    const s = refinementSearch(space, { strategy: 'breadth-first', maxDepth: 5, maxNodes: 20, prune: false })
    expect(s.evaluated).toBe(20)
    expect(s.terminated).toBe(true)
  })

  it('reaches each node once when refinement is not canonical, by key', () => {
    const loose: SearchSpace<readonly number[]> = {
      ...space,
      refine: (s) =>
        WEIGHTS.map((_, i) => i)
          .filter((i) => !s.includes(i))
          .map((i) => [...s, i].sort((a, b) => a - b)),
      bound: undefined,
    }
    const t = trace(
      refinementSearchSteps(loose, { strategy: 'breadth-first', maxDepth: 2, prune: false }),
      undefined,
      10_000,
    )
    const keys = t.steps.flatMap((s) => s.generated.filter((v) => v.fate !== 'duplicate').map((v) => v.key))
    expect(new Set(keys).size).toBe(keys.length)
    expect(t.final.evaluated).toBe(1 + 10 + 45)
  })
})

describe('result sets', () => {
  const visit = (id: number, q: number, node: readonly number[]): SearchVisit<readonly number[]> => ({
    id,
    parent: -1,
    depth: node.length,
    node,
    key: node.join(','),
    quality: q,
    bound: Infinity,
    fate: 'leaf',
  })
  const sameFirst = (a: readonly number[], b: readonly number[]) => a[0] === b[0]

  it('offerResult keeps the top k, best first, and applies the redundancy filter', () => {
    let r: readonly SearchVisit<readonly number[]>[] = []
    r = offerResult(r, visit(0, 1, [0]), 2)
    r = offerResult(r, visit(1, 3, [1]), 2)
    r = offerResult(r, visit(2, 2, [2]), 2)
    expect(r.map((v) => v.id)).toEqual([1, 2])
    r = offerResult(r, visit(3, 2.5, [1, 4]), 2, { redundant: sameFirst })
    expect(r.map((v) => v.id)).toEqual([1, 2])
    r = offerResult(r, visit(4, 5, [2, 3]), 2, { redundant: sameFirst })
    expect(r.map((v) => v.id)).toEqual([4, 1])
    expect(offerResult([], visit(5, -1, [0]), 2, { minQuality: 0 })).toEqual([])
  })

  it('filterRedundant greedily keeps the best of each redundant group', () => {
    const vs = [visit(0, 1, [0, 1]), visit(1, 4, [0]), visit(2, 3, [2]), visit(3, 2, [2, 5])]
    expect(filterRedundant(vs, sameFirst).map((v) => v.id)).toEqual([1, 2])
    expect(filterRedundant(vs, sameFirst, 1).map((v) => v.id)).toEqual([1])
  })

  it('search with a redundancy filter returns pairwise non-redundant results', () => {
    const s = refinementSearch(space, {
      strategy: 'depth-first',
      maxDepth: 3,
      k: 5,
      redundant: sameFirst,
      prune: false,
    })
    const firsts = s.results.map((v) => v.node[0])
    expect(new Set(firsts).size).toBe(firsts.length)
  })
})
