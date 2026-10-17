/** Weight-free propagation (aifn-methods/neural/graph), against the site's depth and computation-graph checks. */
import { describe, expect, it } from 'vitest'
import { meanPropagation, propagationStack } from 'aifn-methods/neural/graph'
import { adjacencySpectrum, dirichletEnergy, sampledComputationGraph } from 'aifn-compute/nn/graph'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** Two triangles joined by the edge 2-3 (the oversmoothing note). */
const tt = fromEdges(
  6,
  [
    [0, 1],
    [0, 2],
    [1, 2],
    [3, 4],
    [3, 5],
    [4, 5],
    [2, 3],
  ],
  { directed: false },
)
const x0 = [[3], [1], [2], [0], [1], [-1]]

describe('propagationStack', () => {
  it('plain: energies 5.0479, 0.6104, 0.4459, 0.2441 at layers 0, 1, 2 and 4, within the spectral bound', () => {
    const plain = propagationStack(tt, x0, { depth: 16 })
    ;[5.0479, 0.6104, 0.4459, 0.2441].forEach((e, i) =>
      expect(dirichletEnergy(tt, plain[[0, 1, 2, 4][i]])).toBeCloseTo(e, 4),
    )
    const lambda = adjacencySpectrum(tt).contraction
    for (let k = 0; k < 16; k++)
      expect(dirichletEnergy(tt, plain[k + 1])).toBeLessThanOrEqual(lambda ** 2 * dirichletEnergy(tt, plain[k]) + 1e-12)
  })
  it('the plain limit is proportional to sqrt(d tilde): (0.946, 0.946, 1.093, 1.093, 0.946, 0.946)', () => {
    const deep = propagationStack(tt, x0, { depth: 200 })[200].map((r) => r[0])
    ;[0.946, 0.946, 1.093, 1.093, 0.946, 0.946].forEach((v, i) => expect(deep[i]).toBeCloseTo(v, 3))
  })
  it('initial residual at alpha 0.1: the fixed point (1.528, 1.328, 1.429, 0.735, 0.577, 0.377)', () => {
    const ir = propagationStack(tt, x0, { depth: 200, variant: 'initial-residual', alpha: 0.1 })[200].map((r) => r[0])
    ;[1.528, 1.328, 1.429, 0.735, 0.577, 0.377].forEach((v, i) => expect(ir[i]).toBeCloseTo(v, 3))
  })
  it('PairNorm keeps zero column means and unit mean squared norm; DropEdge gives depth + 1 layers', () => {
    const pn = propagationStack(tt, x0, { depth: 5, variant: 'pairnorm' })[5]
    expect(pn.reduce((s, r) => s + r[0], 0)).toBeCloseTo(0, 12)
    expect(pn.reduce((s, r) => s + r[0] ** 2, 0) / 6).toBeCloseTo(1, 12)
    const de = propagationStack(tt, x0, { depth: 3, variant: 'dropedge', dropProbability: 0.5, stream: stream(1) })
    expect(de.length).toBe(4)
    expect(() => propagationStack(tt, x0, { depth: 3, variant: 'dropedge' })).toThrow()
  })
})

describe('meanPropagation', () => {
  /** The triangle 0-1-2 with a tail 2-3. */
  const g = fromEdges(
    4,
    [
      [0, 1],
      [0, 2],
      [1, 2],
      [2, 3],
    ],
    { directed: false },
  )
  const h = tensor([[1], [2], [3], [4]])
  it('one round on h = (1, 2, 3, 4): (1.75, 2, 8/3, 3.5)', () => {
    const y = Array.from(toFlat(meanPropagation(g, h, 1) as Tensor))
    ;[1.75, 2, 8 / 3, 3.5].forEach((v, i) => expect(y[i]).toBeCloseTo(v, 12))
  })
  it('two rounds on samples with a large fan-out equal the exact ones', () => {
    const big = sampledComputationGraph(g, [0, 1, 2, 3], [5, 5], stream(1))
    const exact = Array.from(toFlat(meanPropagation(g, h, 2) as Tensor))
    const sampled = Array.from(toFlat(meanPropagation(g, h, 2, big.samples) as Tensor))
    sampled.forEach((v, i) => expect(v).toBeCloseTo(exact[i], 12))
    ;[2.0417, 2.1042, 2.5417, 3.0833].forEach((v, i) => expect(exact[i]).toBeCloseTo(v, 4))
  })
})
