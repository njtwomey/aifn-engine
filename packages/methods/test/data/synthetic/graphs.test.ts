/**
 * Graph datasets (aifn-methods/data/synthetic/graphs): the contextual block model reproduces the site candidate's
 * draws, and its probabilities give the homophily asked for.
 */
import { describe, expect, it } from 'vitest'
import { datasetRegistry, generate } from 'aifn-methods/data'
import { blockModelProbabilities, contextualBlockModel, graphFamilies } from 'aifn-methods/data/synthetic'
import { edgeHomophily } from 'aifn-compute/graph/measures'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('contextualBlockModel', () => {
  it("reproduces the site's graph (stream 3, sizes [40, 40], pIn 0.12, pOut 0.02)", () => {
    const d = contextualBlockModel(stream(3), { sizes: [40, 40], pIn: 0.12, pOut: 0.02 })
    expect(d.graph.edges.length).toBe(227)
    expect(d.homophily).toBeCloseTo(0.8678414096916299, 12)
    expect(d.homophily).toBe(edgeHomophily(d.graph, d.y))
    expect(d.y.dtype).toBe('int32')
    expect(d.graph.directed).toBe(false)
  })
  it("draws the features as the site did: scalar normals from child 'features', row by row", () => {
    const d = contextualBlockModel(stream('link/graph'), { sizes: [20, 20, 20], pIn: 0.3, pOut: 0.02, features: 4 })
    expect(d.graph.edges.length).toBe(224)
    const sf = child(stream('link/graph'), 'features')
    const y = toFlat(d.y)
    const want = Array.from({ length: 60 * 4 }, (_, k) => (k % 4 === y[Math.floor(k / 4)] ? 1 : 0) + normal(sf))
    expect(Array.from(toFlat(d.x))).toEqual(want)
  })
  it('mean homophily over five seeds matches the catalogue probes (0.854 and 0.494 expected)', () => {
    const mean = (pIn: number, pOut: number) =>
      [0, 1, 2, 3, 4].reduce((a, s) => a + contextualBlockModel(stream(s), { pIn, pOut }).homophily / 5, 0)
    expect(Math.abs(mean(0.12, 0.02) - 0.854)).toBeLessThan(0.03)
    expect(Math.abs(mean(0.06, 0.06) - 0.494)).toBeLessThan(0.04)
  })
  it('throws on an empty block, a bad probability or too few features', () => {
    expect(() => contextualBlockModel(stream(0), { sizes: [3, 0] })).toThrow()
    expect(() => contextualBlockModel(stream(0), { pIn: 2 })).toThrow()
    expect(() => contextualBlockModel(stream(0), { sizes: [3, 3, 3], features: 2 })).toThrow()
  })
  it('is registered and runs from its knobs', () => {
    const entry = datasetRegistry.contextualBlockModel
    const d = generate(entry, stream(1), defaults(entry.info.knobs))
    expect((d as { graph: { nodes: number } }).graph.nodes).toBe(80)
  })
})

describe('blockModelProbabilities', () => {
  it('blocks of 100, degree 6, homophily 0.2: pIn = 1.2/99, pOut = 4.8/100, measured near 0.2', () => {
    const p = blockModelProbabilities({ classes: 2, blockSize: 100, degree: 6, homophily: 0.2 })
    expect(p.pIn).toBeCloseTo(1.2 / 99, 12)
    expect(p.pOut).toBeCloseTo(4.8 / 100, 12)
    let mean = 0
    for (let s = 0; s < 5; s++) mean += contextualBlockModel(stream(s), { sizes: [100, 100], ...p }).homophily / 5
    expect(Math.abs(mean - 0.2)).toBeLessThan(0.04)
  })
  it('throws when the degree is too large for the blocks', () => {
    expect(() => blockModelProbabilities({ classes: 2, blockSize: 3, degree: 10, homophily: 0.5 })).toThrow()
  })
})

describe('graphFamilies', () => {
  it('90 graphs, 30 per family, sizes in [6, 13]; cycles have n edges, stars and trees n - 1', () => {
    const fam = graphFamilies(stream(1))
    expect([0, 1, 2].map((f) => fam.family.filter((x) => x === f).length)).toEqual([30, 30, 30])
    expect(fam.size.every((n) => n >= 6 && n <= 13)).toBe(true)
    fam.graphs.forEach((g, i) => {
      expect(g.nodes).toBe(fam.size[i])
      expect(g.edges.length).toBe(fam.family[i] === 0 ? g.nodes : g.nodes - 1)
      expect(g.directed).toBe(false)
    })
  })
  it('throws for a size range that a cycle cannot have', () => {
    expect(() => graphFamilies(stream(1), { minNodes: 2 })).toThrow()
  })
})
