/** Depth measures (aifn-compute/nn/graph), against the site's self-check (depth-long-range). */
import { describe, expect, it } from 'vitest'
import {
  adjacencySpectrum,
  dirichletEnergy,
  dropEdge,
  gcnCoefficients,
  gcnMatrix,
  pairNorm,
  propagationSensitivity,
} from 'aifn-compute/nn/graph'
import { withVirtualNode } from 'aifn-compute/graph/rewiring'
import { barbellGraph, chainGraph, cycleGraph } from 'aifn-compute/graph/structures'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { grad } from 'aifn-compute/foundation/autodiff'
import { matmul, sum, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

const rows = (t: Value) => toRows(t as Tensor) as number[][]

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

describe('gcnMatrix and adjacencySpectrum', () => {
  it('path 0-1-2: A hat (0, 1) = 1/sqrt(6), eigenvalues (1, 1/2, -1/6)', () => {
    expect(rows(gcnMatrix(chainGraph(3)))[0][1]).toBeCloseTo(1 / Math.sqrt(6), 14)
    const values = Array.from(toFlat(adjacencySpectrum(chainGraph(3)).values))
    ;[1, 0.5, -1 / 6].forEach((v, k) => expect(values[k]).toBeCloseTo(v, 12))
  })
  it('the dense matrix holds the coefficients of gcnCoefficients', () => {
    const { edges, coefficients } = gcnCoefficients(tt, { normalisation: 'random-walk' })
    const A = rows(gcnMatrix(tt, { normalisation: 'random-walk' }))
    toFlat(coefficients).forEach((c, k) => expect(A[edges.destination[k]][edges.source[k]]).toBeCloseTo(c, 14))
  })
  it('two triangles: mu_2 = contraction = 0.8604', () => {
    const spec = adjacencySpectrum(tt)
    expect(toFlat(spec.values)[1]).toBeCloseTo(0.8604, 4)
    expect(spec.contraction).toBeCloseTo(0.8604, 4)
  })
})

describe('dirichletEnergy', () => {
  const A = gcnMatrix(tt)
  const layers: Value[] = [tensor([[3], [1], [2], [0], [1], [-1]])]
  for (let k = 0; k < 16; k++) layers.push(matmul(A, layers[k]))
  it('x = (3, 1, 2, 0, 1, -1), plain propagation: energies 5.0479, 0.6104, 0.4459, 0.2441 at layers 0, 1, 2, 4', () => {
    ;[5.0479, 0.6104, 0.4459, 0.2441].forEach((e, i) =>
      expect(dirichletEnergy(tt, layers[[0, 1, 2, 4][i]] as Tensor)).toBeCloseTo(e, 4),
    )
  })
  it('the energy bound E(A H) <= lambda^2 E(H) holds for 16 layers', () => {
    const lambda = adjacencySpectrum(tt).contraction
    for (let k = 0; k < 16; k++)
      expect(dirichletEnergy(tt, layers[k + 1] as Tensor)).toBeLessThanOrEqual(
        lambda ** 2 * dirichletEnergy(tt, layers[k] as Tensor) + 1e-12,
      )
  })
  it('zero for rows proportional to sqrt(d tilde); throws for the wrong number of rows', () => {
    const d = [3, 3, 4, 4, 3, 3].map((x) => [Math.sqrt(x)])
    expect(dirichletEnergy(tt, d)).toBeCloseTo(0, 14)
    expect(() => dirichletEnergy(tt, [[1]])).toThrow()
  })
})

describe('pairNorm', () => {
  it('zero column means and mean squared row norm s^2 = 4', () => {
    const y = rows(
      pairNorm(
        tensor([
          [1, 2],
          [3, 4],
          [5, 9],
        ]),
        2,
      ),
    )
    ;[0, 1].forEach((k) => expect(y.reduce((s, r) => s + r[k], 0)).toBeCloseTo(0, 12))
    expect(y.reduce((s, r) => s + r[0] ** 2 + r[1] ** 2, 0) / 3).toBeCloseTo(4, 12)
  })
  it('equal rows give zeros, and it is differentiable', () => {
    expect(
      Array.from(
        toFlat(
          pairNorm(
            tensor([
              [1, 1],
              [1, 1],
            ]),
          ) as Tensor,
        ),
      ),
    ).toEqual([0, 0, 0, 0])
    const g = grad((x: Value) => sum(pairNorm(x, 1.5)))(
      tensor([
        [1, 2],
        [0, 5],
        [4, 1],
      ]),
    ) as Tensor
    expect(toFlat(g).every(Number.isFinite)).toBe(true)
  })
})

describe('dropEdge', () => {
  it('keeps each edge when its uniform is at least p, drawing from the stream itself', () => {
    expect(dropEdge(stream(0), cycleGraph(6), 0).edges.length).toBe(6)
    expect(dropEdge(stream(0), cycleGraph(6), 1).edges.length).toBe(0)
    const a = dropEdge(stream(1), cycleGraph(40), 0.5).edges.length
    expect(a).toBeGreaterThan(10)
    expect(a).toBeLessThan(30)
    expect(() => dropEdge(stream(0), cycleGraph(3), 2)).toThrow()
  })
})

describe('propagationSensitivity', () => {
  const bb = barbellGraph(5, 3)
  it('barbell(5, 3): (A^5)_{12,0} = 0 at distance 6, (A^6)_{12,0} = 1/4860', () => {
    const s = rows(propagationSensitivity(bb, 12, 6))
    expect(s[5][0] === 0).toBe(true)
    expect(s[6][0] * 4860).toBeCloseTo(1, 10)
    expect(s[0][12]).toBe(1)
  })
  it('a virtual node brings node 0 within two hops of node 12: (A^2)_{12,0} = 1/84', () => {
    expect(rows(propagationSensitivity(withVirtualNode(bb), 12, 2))[2][0] * 84).toBeCloseTo(1, 10)
  })
})
