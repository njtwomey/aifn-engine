/** Minimum spanning trees (aifn-compute/graph/spanning-trees). */
import { describe, expect, it } from 'vitest'
import { kruskalSteps, minimumSpanningTree, primSteps } from 'aifn-compute/graph/spanning-trees'
import { connectedComponents } from 'aifn-compute/graph/traversal'
import { checkProtocol } from '../../protocol'
import { randomGraph } from '../helpers'

describe('minimum spanning trees', () => {
  it('Kruskal and Prim give minimum spanning trees of equal weight', () => {
    for (let seed = 0; seed < 10; seed++) {
      const g = randomGraph(200 + seed, 12, 0.4, false)
      const k = minimumSpanningTree(g)
      const p = minimumSpanningTree(g, { method: 'prim' })
      expect(p.weight).toBe(k.weight)
      expect(p.trees).toBe(k.trees)
      expect(k.trees).toBe(connectedComponents(g).count)
    }
  })
  it('Kruskal and Prim follow the Algorithm protocol', () => {
    const g = randomGraph(7, 8, 0.5, false)
    checkProtocol(kruskalSteps(g), undefined, { steps: 12 })
    checkProtocol(primSteps(g), undefined, { steps: 12 })
  })
})
