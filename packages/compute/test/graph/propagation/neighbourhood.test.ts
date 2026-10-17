/**
 * `neighbourhoodGraph` on the site's self-check: the rules on the points 0, 1, 3 of a line, the $k$-NN heat graph
 * against `pointAffinity`, the $b$-matching degree bound, and the Laplacian of a weighted path.
 */
import { describe, expect, it } from 'vitest'
import { neighbourhoodGraph, pointAffinity, type NeighbourhoodRule } from 'aifn-compute/graph/propagation'
import { laplacian } from 'aifn-compute/graph/matrices'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

describe('neighbourhoodGraph', () => {
  const x = [[0], [1], [3]]
  const pairs = (r: NeighbourhoodRule) =>
    neighbourhoodGraph(x, r, { kind: 'binary' })
      .edges.map((e) => `${e.from}-${e.to}`)
      .join()

  it('joins the points 0, 1, 3 by each rule', () => {
    expect(pairs({ kind: 'knn', k: 1 })).toBe('0-1,1-2')
    expect(pairs({ kind: 'mutual-knn', k: 1 })).toBe('0-1')
    expect(pairs({ kind: 'epsilon', epsilon: 1.5 })).toBe('0-1')
    expect(pairs({ kind: 'epsilon', epsilon: 2 })).toBe('0-1,1-2')
    expect(pairs({ kind: 'b-matching', b: 1 })).toBe('0-1')
    expect(pairs({ kind: 'full' })).toBe('0-1,0-2,1-2')
    expect(pairs({ kind: 'knn', k: 5 })).toBe('0-1,0-2,1-2')
  })

  it('reports distances, binary or heat weights, counts and a symmetric affinity', () => {
    const g = neighbourhoodGraph(x, { kind: 'full' }, { kind: 'heat', sigma: 1 })
    expect(g.edges.map((e) => e.distance)).toEqual([1, 3, 2])
    expect(g.edges[0].weight).toBeCloseTo(Math.exp(-0.5), 15)
    expect(g.counts).toEqual([2, 2, 2])
    const W = toRows(g.affinity) as number[][]
    expect(W[0][2]).toBeCloseTo(Math.exp(-4.5), 15)
    expect(W[2][0]).toBe(W[0][2])
    expect(W[1][1]).toBe(0)
    expect(g.graph.edges.length).toBe(3)
  })

  it('gives pointAffinity’s k-NN heat graph to 1e-12, and b-matching keeps degrees at most b', () => {
    const pts = toRows(normals(stream(7), [60, 2])) as number[][]
    const sigma = 0.3
    const mine = toFlat(neighbourhoodGraph(pts, { kind: 'knn', k: 5 }, { kind: 'heat', sigma }).affinity)
    const engine = toFlat(pointAffinity(pts, { gamma: 1 / (2 * sigma * sigma), neighbours: 5 }))
    expect(Math.max(...mine.map((v, i) => Math.abs(v - engine[i])))).toBeLessThan(1e-12)
    const b = neighbourhoodGraph(pts, { kind: 'b-matching', b: 4 }, { kind: 'binary' })
    expect(Math.max(...b.counts)).toBeLessThanOrEqual(4)
  })

  it('the Laplacian of the weighted path is [[1, -1, 0], [-1, 3, -2], [0, -2, 2]]', () => {
    expect(
      toRows(
        laplacian([
          [0, 1, 0],
          [1, 0, 2],
          [0, 2, 0],
        ]),
      ),
    ).toEqual([
      [1, -1, 0],
      [-1, 3, -2],
      [0, -2, 2],
    ])
  })

  it('refuses bad parameters', () => {
    expect(() => neighbourhoodGraph(x, { kind: 'knn', k: 0 })).toThrow(/k must/)
    expect(() => neighbourhoodGraph(x, { kind: 'epsilon', epsilon: -1 })).toThrow(/epsilon/)
    expect(() => neighbourhoodGraph(x, { kind: 'full' }, { kind: 'heat', sigma: 0 })).toThrow(/sigma/)
  })
})
