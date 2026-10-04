/** Structured graphs (aifn-compute/graph/structured): templates, unrolling, shapes, Markov blankets and diagrams. */
import { describe, expect, it } from 'vitest'
import {
  chainTemplate,
  latticeTemplate,
  markovBlanket,
  nodesWithRole,
  shape,
  structured,
  toDiagram,
  treeTemplate,
  unroll,
} from 'aifn-compute/graph/structured'

describe('templates and unrolling', () => {
  it('a hidden Markov model: a chain of latents with an observation each', () => {
    const hmm = chainTemplate('T', { observed: 'x' })
    expect(hmm.kind).toBe('graph')
    const g = unroll(hmm, { T: 4 })
    expect(g.attributes.map((n) => n.name)).toEqual(['z[0]', 'z[1]', 'z[2]', 'z[3]', 'x[0]', 'x[1]', 'x[2]', 'x[3]'])
    expect(g.edges.length).toBe(3 + 4)
    expect(nodesWithRole(g, 'observed').length).toBe(4)
    expect(shape(hmm, { T: 4 })).toBe('chain')
  })
  it('lattices and trees have their shapes', () => {
    expect(shape(latticeTemplate(3, 3, { directed: false }))).toBe('lattice')
    expect(shape(treeTemplate(2, 3))).toBe('tree')
  })
  it('a plate of observations sharing a parameter: observed nodes are conditioned on, latent ones form a star', () => {
    const m = structured('coin', (b) => {
      b.size('N')
      const theta = b.latent('theta')
      const data = b.plate('data', 'N')
      const x = data.observed('x')
      b.edge(theta, x)
    })
    // With x observed, θ is the only free variable: a chain of one.
    expect(shape(m, { N: 3 })).toBe('chain')
    const latent = structured('mixture', (b) => {
      b.size('N')
      const theta = b.latent('theta')
      b.edge(theta, b.plate('data', 'N').latent('z'))
    })
    expect(shape(latent, { N: 3 })).toBe('tree')
    const g = unroll(m, { N: 3 })
    expect(g.nodes).toBe(4)
    expect(markovBlanket(m, 'theta', { N: 3 }).children.sort()).toEqual(['x[0]', 'x[1]', 'x[2]'])
    // The diagram is plain data: nodes, edges and the plate.
    const d = toDiagram(m)
    expect(JSON.parse(JSON.stringify(d))).toEqual(d)
    expect(d.groups.length).toBe(1)
  })
})
