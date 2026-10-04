/** Flows (aifn-compute/graph/flows). */
import { describe, expect, it } from 'vitest'
import { edmondsKarpSteps, maxFlow } from 'aifn-compute/graph/flows'
import { fromEdges } from 'aifn-compute/graph'
import { checkProtocol } from '../../protocol'
import { randomGraph } from '../helpers'

describe('maximum flow', () => {
  it('maximum flow equals the minimum cut capacity (CLRS Figure 26.1)', () => {
    // s v1 v2 v3 v4 t
    const g = fromEdges(6, [
      [0, 1, 16],
      [0, 2, 13],
      [2, 1, 4],
      [1, 3, 12],
      [3, 2, 9],
      [2, 4, 14],
      [4, 3, 7],
      [3, 5, 20],
      [4, 5, 4],
    ])
    const r = maxFlow(g, 0, 5)
    expect(r.value).toBe(23)
    expect(r.cutCapacity).toBe(23)
    expect(r.sourceSide.data[0]).toBe(1)
    expect(r.sourceSide.data[5]).toBe(0)
    for (let seed = 0; seed < 10; seed++) {
      const h = randomGraph(300 + seed, 8, 0.4, seed % 2 === 0)
      const f = maxFlow(h, 0, 7)
      expect(f.value).toBeCloseTo(f.cutCapacity, 12)
    }
  })
  it('Edmonds–Karp follows the Algorithm protocol', () => {
    const g = fromEdges(4, [
      [0, 1, 3],
      [0, 2, 2],
      [1, 2, 1],
      [1, 3, 2],
      [2, 3, 3],
    ])
    checkProtocol(edmondsKarpSteps(g, { source: 0, sink: 3 }), undefined, { steps: 6 })
  })
})
