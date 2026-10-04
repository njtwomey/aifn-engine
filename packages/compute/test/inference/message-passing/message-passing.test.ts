import { describe, expect, it } from 'vitest'
import {
  beliefPropagation,
  beliefPropagationSteps,
  decodeBeliefs,
  gaussianBeliefPropagation,
  gaussianBeliefPropagationSteps,
} from 'aifn-compute/inference/message-passing'
import { enumerate, variableElimination } from 'aifn-compute/inference/exact'
import { isTree } from 'aifn-compute/inference/model'
import { inverse } from 'aifn-compute/numerics/linalg'
import { fromRows, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'
import { isingGrid, randomTree } from '../graphs'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

describe('trees: sum-product and max-product are exact', () => {
  const g = randomTree(11)
  const exact = enumerate(g)
  it('recognises a tree', () => expect(isTree(g)).toBe(true))
  it('tree schedule: marginals and Bethe log Z equal enumeration after one sweep', () => {
    const r = beliefPropagation(g)
    expect(r.sweeps).toBe(1)
    r.marginals.forEach((m, v) => close(m.data, exact.marginals[v].data, 1e-12))
    expect(r.logZ).toBeCloseTo(exact.logZ, 10)
  })
  it('flooding converges to the same marginals', () => {
    const r = beliefPropagation(g, { schedule: 'flooding' })
    expect(r.converged).toBe(true)
    r.marginals.forEach((m, v) => close(m.data, exact.marginals[v].data, 1e-9))
  })
  it('max-product decodes the MAP assignment', () => {
    const r = beliefPropagation(g, { mode: 'max' })
    expect(toFlat(decodeBeliefs(r.state))).toEqual(toFlat(exact.map))
  })
  it('evidence: BP agrees with variable elimination', () => {
    const bp = beliefPropagation(g, { evidence: { 3: 1 } })
    const ve = variableElimination(g, [0], { evidence: { 3: 1 } })
    close(bp.marginals[0].data, ve.marginal.data, 1e-12)
    expect(bp.logZ).toBeCloseTo(ve.logZ, 10)
  })
  it('message granularity exposes one update per step', () => {
    const t = trace(beliefPropagationSteps(g), undefined, 100)
    expect(t.steps[1].updated).toHaveLength(1)
    expect(t.meta.stopped).toBe('done')
    expect(t.meta.steps).toBe(t.steps[0].schedule.length)
  })
})

describe('loopy BP on a 3 × 3 Ising grid', () => {
  const g = isingGrid(3, 3, 0.3, 0.1)
  const exact = enumerate(g)
  it('converges near the exact marginals for weak coupling', () => {
    const r = beliefPropagation(g)
    expect(isTree(g)).toBe(false)
    expect(r.converged).toBe(true)
    r.marginals.forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.02))
    expect(Math.abs(r.logZ - exact.logZ)).toBeLessThan(0.05)
  })
  it('damping and a sequential schedule reach the same fixed point', () => {
    const a = beliefPropagation(g)
    const b = beliefPropagation(g, { schedule: 'sequential', damping: 0.5 })
    a.marginals.forEach((m, v) => close(m.data, b.marginals[v].data, 1e-6))
  })
})

describe('Gaussian belief propagation', () => {
  const solve = (J: number[][], h: number[]) => {
    const S = toRows(inverse(fromRows(J)))
    return { mean: S.map((r) => r.reduce((s, v, j) => s + v * h[j], 0)), variance: S.map((r, i) => r[i]) }
  }
  it('is exact on a tree', () => {
    const J = [
      [3, -1, 0, 0.5],
      [-1, 4, 1, 0],
      [0, 1, 2, 0],
      [0.5, 0, 0, 2],
    ]
    const h = [1, -2, 0.5, 1]
    const r = gaussianBeliefPropagation(J, h)
    const ex = solve(J, h)
    expect(r.converged).toBe(true)
    close(toFlat(r.means), ex.mean, 1e-9)
    close(toFlat(r.variances), ex.variance, 1e-9)
  })
  it('gets the means right on a loop (diagonally dominant), variances only approximately', () => {
    const J = [
      [4, 1, 0, 1],
      [1, 4, 1, 0],
      [0, 1, 4, 1],
      [1, 0, 1, 4],
    ]
    const h = [1, 2, 3, 4]
    const r = gaussianBeliefPropagation(J, h, { schedule: 'sequential' })
    const ex = solve(J, h)
    close(toFlat(r.means), ex.mean, 1e-8)
    close(toFlat(r.variances), ex.variance, 0.02)
  })
})

describe('protocol', () => {
  it('discrete and Gaussian BP satisfy the Algorithm protocol', () => {
    const g = isingGrid(2, 2, 0.4, 0.1)
    checkProtocol(beliefPropagationSteps(g), undefined, { steps: 12, record: { c: (s) => s.change } })
    checkProtocol(beliefPropagationSteps(randomTree(3), { mode: 'max' }), undefined, { steps: 12 })
    checkProtocol(beliefPropagationSteps(g, { schedule: 'sequential', damping: 0.3 }), undefined, { steps: 12 })
    const J = fromRows([
      [3, 1, 0],
      [1, 3, 1],
      [0, 1, 3],
    ])
    checkProtocol(gaussianBeliefPropagationSteps(J, [1, 0, 1]), undefined, { steps: 12 })
  })
})
