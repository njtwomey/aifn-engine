/** aifn-methods/learning/semi-supervised: the density-sensitive ρ-path distance, against the note's hand values. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { moons } from 'aifn-methods/data/synthetic'
import {
  rhoDistance,
  rhoEdgeCost,
  rhoPathDistances,
  rhoPathExtend,
  rhoPathNodes,
  rhoQueryNeighbours,
} from 'aifn-methods/learning/semi-supervised'

// Ten points on a line, labelled at 0 (class 0) and 4.2 (class 1); the distances are measured to 2.4 (node 6).
const X = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.4, 3.8, 4.2].map((v) => [v, 0])

describe('rhoPathDistances', () => {
  it('gives the hand values to 2.4: 2.4/1.8 at ρ = 0, 1.3740/1.3089 at ρ = 1, 1.0613/1.1432 at ρ = 2', () => {
    const want: Record<number, [number, number]> = {
      0: [2.4, 1.8],
      1: [Math.log(1 + 6 * Math.expm1(0.4)), Math.log(1 + Math.expm1(1) + 2 * Math.expm1(0.4))],
      2: [Math.log(1 + 6 * Math.expm1(0.8)) / 2, Math.log(1 + Math.expm1(2) + 2 * Math.expm1(0.8)) / 2],
    }
    for (const rho of [0, 1, 2]) {
      const r = rhoPathDistances(X, [0, 9], { rho })
      expect(Math.abs(r.distances[0][6] - want[rho][0])).toBeLessThan(1e-9)
      expect(Math.abs(r.distances[1][6] - want[rho][1])).toBeLessThan(1e-9)
    }
    expect(rhoPathDistances(X, [0, 9], { rho: 1 }).distances[0][6]).toBeCloseTo(1.374, 3)
    expect(rhoPathDistances(X, [0, 9], { rho: 2 }).distances[1][6]).toBeCloseTo(1.1432, 4)
  })

  it('tends to the minimax edge as ρ grows (0.43 and 1.00 at ρ = 60; limits 0.4 and 1.0)', () => {
    const big = rhoPathDistances(X, [0, 9], { rho: 60 })
    expect(Math.abs(big.distances[0][6] - 0.4)).toBeLessThan(0.05)
    expect(Math.abs(big.distances[1][6] - 1.0)).toBeLessThan(0.05)
  })

  it('the cheapest path from 0 to 2.4 walks the run, and the crossover ρ of 2.4 is near 1.356', () => {
    expect(rhoPathNodes(rhoPathDistances(X, [0, 9], { rho: 1 }), 0, 6)).toEqual([0, 1, 2, 3, 4, 5, 6])
    let lo = 1
    let hi = 2
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2
      const r = rhoPathDistances(X, [0, 9], { rho: m })
      if (r.distances[0][6] > r.distances[1][6]) lo = m
      else hi = m
    }
    expect(Math.abs(lo - 1.356)).toBeLessThan(0.001)
  })

  it('reuses a graph, keeps unreachable points at Infinity, and refuses an overflowing path cost', () => {
    const first = rhoPathDistances(X, [0], { rho: 1, neighbours: 2 })
    const again = rhoPathDistances(X, [0], { rho: 3, neighbours: 2, graph: first.graph })
    expect(again.graph).toBe(first.graph)
    // Two far clusters joined by no edge of the 1-nearest-neighbour graph.
    const apart = rhoPathDistances(
      [[0], [0.1], [10], [10.1]].map((r) => [r[0], 0]),
      [0],
      { rho: 1, neighbours: 1 },
    )
    expect(apart.distances[0][2]).toBe(Infinity)
    expect(rhoPathNodes(apart, 0, 2)).toEqual([])
    expect(() =>
      rhoPathDistances(
        [
          [0, 0],
          [10, 0],
        ],
        [0],
        { rho: 100 },
      ),
    ).toThrow(/overflows/)
    expect(() => rhoPathDistances(X, [10], { rho: 1 })).toThrow(/not a point/)
  })

  it('on the figure’s moons (seed 3) nearest-label accuracy rises with ρ: 0.601 at ρ = 0 to 1 at ρ = 16', () => {
    const d = moons(stream(3), { n: 200, noise: 0.1, spacing: 'random' })
    const rows = (toArray(d.x) as number[][]).map(([a, b]) => [a - 0.5, b - 0.25])
    const y = Array.from(toFlat(d.y))
    const lab = [14, 100]
    expect(lab.map((i) => y[i])).toEqual([0, 1])
    const graph = rhoPathDistances(rows, lab, { rho: 0 }).graph
    const accuracy = (rho: number) => {
      const r = rhoPathDistances(rows, lab, { rho, graph })
      const right = rows.filter(
        (_, i) => !lab.includes(i) && (r.distances[0][i] <= r.distances[1][i] ? 0 : 1) === y[i],
      ).length
      return right / (rows.length - 2)
    }
    expect(accuracy(0)).toBeCloseTo(0.601010101010101, 12)
    expect(accuracy(16)).toBe(1)
  })
})

describe('rhoPathExtend', () => {
  it('reaches new points through their neighbours with finite distances', () => {
    const r = rhoPathDistances(X, [0, 9], { rho: 2 })
    const ext = rhoPathExtend(X, [[2.9, 0]], r)
    expect(ext.distances.every((d) => Number.isFinite(d[0]))).toBe(true)
    expect(ext.distances[0][0]).toBeCloseTo(1.154856186711262, 12)
    expect(ext.distances[1][0]).toBeCloseTo(0.8213748013474539, 12)
    expect(ext.entry.map((e) => e[0])).toEqual([6, 7])
    // A query on a point joins it by a zero-cost edge.
    const on = rhoPathExtend(X, [[2.4, 0]], r)
    expect(on.distances[0][0]).toBeCloseTo(r.distances[0][6], 12)
  })

  it('rhoQueryNeighbours lists the nearest points first, and the edge cost maps back to its length', () => {
    const near = rhoQueryNeighbours(
      [
        [0, 0],
        [1, 0],
        [3, 0],
      ],
      [[0.8, 0]],
      2,
    )
    expect(near.indices).toEqual([[1, 0]])
    near.distances[0].forEach((v, k) => expect(v).toBeCloseTo([0.2, 0.8][k], 12))
    expect(rhoDistance(rhoEdgeCost(0.4, 3), 3)).toBeCloseTo(0.4, 14)
    expect(rhoEdgeCost(0.4, 0)).toBe(0.4)
  })
})
