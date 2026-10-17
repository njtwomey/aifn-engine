/** Homophily measures (aifn-compute/graph/measures), against the site's self-check (homophily-and-links). */
import { describe, expect, it } from 'vitest'
import { adjustedHomophily, edgeHomophily, nodeHomophily } from 'aifn-compute/graph/measures'
import { blockMembership, cycleGraph, stochasticBlockModel } from 'aifn-compute/graph/structures'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { fromData } from 'aifn-compute/foundation/tensor'

describe('homophily', () => {
  it('the 6-cycle labelled A A B B A B: edge and node 1/3, adjusted -1/3', () => {
    const g = cycleGraph(6)
    const y = [0, 0, 1, 1, 0, 1]
    expect(edgeHomophily(g, y)).toBeCloseTo(1 / 3, 12)
    expect(nodeHomophily(g, y)).toBeCloseTo(1 / 3, 12)
    expect(adjustedHomophily(g, y)).toBeCloseTo(-1 / 3, 12)
    // Labels as an int32 tensor, as a Dataset holds them.
    expect(edgeHomophily(g, fromData(Int32Array.from(y)))).toBeCloseTo(1 / 3, 12)
  })
  it('random wiring with a 90/10 class split: edge homophily near 0.82, adjusted near 0', () => {
    const g = stochasticBlockModel(stream(5), [180, 20], { pIn: 0.05, pOut: 0.05 })
    const y = blockMembership([180, 20])
    expect(Math.abs(edgeHomophily(g, y) - 0.82)).toBeLessThan(0.03)
    expect(Math.abs(adjustedHomophily(g, y))).toBeLessThan(0.06)
  })
  it('self-loops are ignored; no edges gives NaN', () => {
    const g = fromEdges(
      3,
      [
        [0, 0],
        [0, 1],
        [1, 2],
      ],
      { directed: false },
    )
    expect(edgeHomophily(g, [0, 0, 1])).toBeCloseTo(0.5, 12)
    expect(edgeHomophily(fromEdges(2, [], { directed: false }), [0, 1])).toBeNaN()
    expect(nodeHomophily(fromEdges(2, [], { directed: false }), [0, 1])).toBeNaN()
  })
  it('throws for labels of the wrong length or that are not class indices', () => {
    expect(() => edgeHomophily(cycleGraph(3), [0, 1])).toThrow()
    expect(() => edgeHomophily(cycleGraph(3), [0, 1, -1])).toThrow()
  })
})
