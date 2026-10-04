import { describe, expect, it } from 'vitest'
import { isingInference, isingLattice, isingModel, isingShape } from 'aifn-methods/inference/lattice-models'
import { beliefPropagation } from 'aifn-compute/inference/message-passing'
import { enumerate } from 'aifn-compute/inference/exact'
import { factorGraphGibbs, gibbsMarginals } from 'aifn-compute/inference/stochastic'
import { gridGraph } from 'aifn-compute/graph/structures'
import { stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'

describe('Ising models', () => {
  const g = isingLattice(3, 3, 0.3, 0.1)
  const exact = enumerate(g)
  it('the lattice and the grid graph give the same model', () => {
    const other = enumerate(isingModel(gridGraph(3, 3), 0.3, 0.1))
    expect(other.logZ).toBeCloseTo(exact.logZ, 12)
  })
  it('a lattice reads its shape from the lattice template; a plain grid graph reads as general', () => {
    expect(isingShape(g)).toBe('lattice')
    expect(isingShape(isingLattice(4, 3, 0.3, 0, { periodic: true }))).toBe('lattice')
    expect(isingShape(isingModel(gridGraph(3, 3), 0.3, 0.1))).toBe('general')
    expect(isingShape(isingLattice(1, 5, 0.3, 0))).toBe('chain')
    expect(isingInference(g).shape).toBe('lattice')
    expect(isingInference(isingLattice(1, 5, 0.3, 0)).shape).toBe('chain')
  })
  it('loopy BP converges near the exact marginals for weak coupling', () => {
    const r = beliefPropagation(g)
    expect(r.converged).toBe(true)
    r.marginals.forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.02))
    expect(Math.abs(r.logZ - exact.logZ)).toBeLessThan(0.05)
  })
  it('factor-graph Gibbs estimates the marginals', () => {
    const s = run(factorGraphGibbs(g), undefined, 5000, { stream: stream(1) })
    gibbsMarginals(s).forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.04))
  })
})
