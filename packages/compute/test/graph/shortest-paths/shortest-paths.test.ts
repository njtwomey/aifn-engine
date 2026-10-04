/** Shortest paths (aifn-compute/graph/shortest-paths). */
import { describe, expect, it } from 'vitest'
import {
  aStar,
  aStarSteps,
  bellmanFord,
  bellmanFordSteps,
  dijkstra,
  dijkstraSteps,
  floydWarshall,
  floydWarshallSteps,
} from 'aifn-compute/graph/shortest-paths'
import { fromEdges } from 'aifn-compute/graph'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { checkProtocol } from '../../protocol'
import { randomGraph } from '../helpers'

describe('shortest paths', () => {
  it('Dijkstra agrees with Bellman–Ford and Floyd–Warshall on random graphs', () => {
    for (let seed = 0; seed < 20; seed++) {
      const g = randomGraph(100 + seed, 10, 0.25, seed % 2 === 0)
      const d = toFlat(dijkstra(g, 0).distance)
      expect(toFlat(bellmanFord(g, 0).distance)).toEqual(d)
      expect(toRows(floydWarshall(g).distance)[0]).toEqual(d)
      // A* with a zero heuristic is Dijkstra.
      for (let t = 1; t < 10; t++) expect(aStar(g, 0, t, () => 0).distance).toBe(d[t])
    }
  })

  it('A* with a Manhattan heuristic finds the same length on a grid and expands fewer nodes', () => {
    const W = 8
    const edges: [number, number][] = []
    for (let r = 0; r < W; r++)
      for (let c = 0; c < W; c++) {
        if (c + 1 < W && !(c === 3 && r < W - 1)) edges.push([r * W + c, r * W + c + 1])
        if (r + 1 < W) edges.push([r * W + c, (r + 1) * W + c])
      }
    const g = fromEdges(W * W, edges, { directed: false })
    const target = W * W - 1
    const manhattan = (v: number) => Math.abs(Math.floor(v / W) - (W - 1)) + Math.abs((v % W) - (W - 1))
    const a = aStar(g, 0, target, manhattan)
    const z = aStar(g, 0, target, () => 0)
    expect(a.distance).toBe(z.distance)
    expect(a.path.shape[0]).toBe(a.distance + 1)
    expect(a.expanded).toBeLessThan(z.expanded)
  })

  it('Bellman–Ford returns a negative cycle as a witness', () => {
    const g = fromEdges(5, [
      [0, 1, 1],
      [1, 2, 2],
      [2, 3, -4],
      [3, 1, 1],
      [3, 4, 1],
    ])
    const r = bellmanFord(g, 0)
    const cycle = toFlat(r.negativeCycle!)
    expect(new Set(cycle)).toEqual(new Set([1, 2, 3]))
    let total = 0
    cycle.forEach((v, i) => {
      const e = g.edges.find((e) => e.from === v && e.to === cycle[(i + 1) % cycle.length])
      expect(e).toBeDefined()
      total += e!.weight!
    })
    expect(total).toBeLessThan(0)
  })
})

describe('the Algorithm protocol', () => {
  it('Dijkstra, A*, Bellman–Ford and Floyd–Warshall', () => {
    const g = randomGraph(5, 8, 0.35)
    checkProtocol(dijkstraSteps(g, { source: 0 }), undefined, { steps: 10 })
    checkProtocol(aStarSteps(g, { source: 0, target: 7 }), undefined, { steps: 10 })
    checkProtocol(bellmanFordSteps(g, { source: 0 }), undefined, { steps: 10 })
    checkProtocol(floydWarshallSteps(g), undefined, { steps: 10 })
  })
})
