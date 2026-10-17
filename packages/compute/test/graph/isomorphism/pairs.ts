/**
 * The site's expressiveness fixtures (graph-neural-networks/expressiveness-of-graph-neural-networks/_shared/
 * graph-pairs.ts), without drawing positions: pairs of graphs that 1-WL separates or not.
 */
import { fromEdges, type Graph } from 'aifn-compute/graph'

const undirected = (n: number, edges: [number, number][]): Graph => fromEdges(n, edges, { directed: false })
const cycleEdges = (n: number, offset = 0): [number, number][] =>
  Array.from({ length: n }, (_, i) => [offset + i, offset + ((i + 1) % n)])

export const hexagon = undirected(6, cycleEdges(6))
export const twoTriangles = undirected(6, [...cycleEdges(3), ...cycleEdges(3, 3)])
export const path4 = undirected(4, [
  [0, 1],
  [1, 2],
  [2, 3],
])
export const star3 = undirected(4, [
  [0, 1],
  [0, 2],
  [0, 3],
])
/** Two hexagons sharing the edge 0-1. */
export const decalin = undirected(10, [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 0],
  [1, 6],
  [6, 7],
  [7, 8],
  [8, 9],
  [9, 0],
])
/** Two pentagons joined by the edge 0-5. */
export const bicyclopentyl = undirected(10, [...cycleEdges(5), ...cycleEdges(5, 5), [0, 5]])

const grid4 = (x: number, y: number) => 4 * (((y % 4) + 4) % 4) + (((x % 4) + 4) % 4)

/** The 4 x 4 rook's graph, strongly regular (16, 6, 2, 2). */
export const rook = (() => {
  const edges: [number, number][] = []
  for (let a = 0; a < 16; a++)
    for (let b = a + 1; b < 16; b++) if (a % 4 === b % 4 || Math.floor(a / 4) === Math.floor(b / 4)) edges.push([a, b])
  return undirected(16, edges)
})()

/** The Shrikhande graph, also strongly regular (16, 6, 2, 2). */
export const shrikhande = (() => {
  const seen = new Set<string>()
  const edges: [number, number][] = []
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++)
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const [a, b] = [grid4(x, y), grid4(x + dx, y + dy)]
        const k = `${Math.min(a, b)}-${Math.max(a, b)}`
        if (!seen.has(k)) {
          seen.add(k)
          edges.push([Math.min(a, b), Math.max(a, b)])
        }
      }
  return undirected(16, edges)
})()
