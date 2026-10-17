/** Node encodings (aifn-compute/graph/matrices), against the site's self-check (graph-encodings). */
import { describe, expect, it } from 'vitest'
import { flipSigns, laplacianEncoding, randomWalkEncoding } from 'aifn-compute/graph/matrices'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

const undirected = (n: number, edges: [number, number][]): Graph => fromEdges(n, edges, { directed: false })
const cycleEdges = (n: number, offset = 0): [number, number][] =>
  Array.from({ length: n }, (_, i) => [offset + i, offset + ((i + 1) % n)])
const hexagon = undirected(6, cycleEdges(6))
const twoTriangles = undirected(6, [...cycleEdges(3), ...cycleEdges(3, 3)])
const path4 = undirected(4, [
  [0, 1],
  [1, 2],
  [2, 3],
])
const decalin = undirected(10, [...cycleEdges(6), [1, 6], [6, 7], [7, 8], [8, 9], [9, 0]])
const rook = (() => {
  const edges: [number, number][] = []
  for (let a = 0; a < 16; a++)
    for (let b = a + 1; b < 16; b++) if (a % 4 === b % 4 || Math.floor(a / 4) === Math.floor(b / 4)) edges.push([a, b])
  return undirected(16, edges)
})()
const shrikhande = (() => {
  const at = (x: number, y: number) => 4 * (((y % 4) + 4) % 4) + (((x % 4) + 4) % 4)
  const seen = new Set<string>()
  const edges: [number, number][] = []
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++)
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const [a, b] = [Math.min(at(x, y), at(x + dx, y + dy)), Math.max(at(x, y), at(x + dx, y + dy))]
        if (!seen.has(`${a}-${b}`)) {
          seen.add(`${a}-${b}`)
          edges.push([a, b])
        }
      }
  return undirected(16, edges)
})()

const rows = (t: Parameters<typeof toRows>[0]) => toRows(t) as number[][]

describe('randomWalkEncoding', () => {
  it('two triangles, steps 1-3: (0, 0.5, 0.25)', () => {
    rows(randomWalkEncoding(twoTriangles, 3))[0].forEach((v, k) => expect(v).toBeCloseTo([0, 0.5, 0.25][k], 12))
  })
  it('the 6-cycle, steps 1-4: (0, 0.5, 0, 0.375)', () => {
    rows(randomWalkEncoding(hexagon, 4))[0].forEach((v, k) => expect(v).toBeCloseTo([0, 0.5, 0, 0.375][k], 12))
  })
  it('decalin is bipartite: step 5 is 0 at every node', () => {
    expect(Math.max(...rows(randomWalkEncoding(decalin, 5)).map((r) => r[4]))).toBe(0)
  })
  it("rook's graph and Shrikhande: identical encodings at every node and step", () => {
    const key = (g: Graph) => rows(randomWalkEncoding(g, 8)).map((r) => r.map((v) => v.toFixed(10)).join())
    expect(new Set([...key(rook), ...key(shrikhande)]).size).toBe(1)
  })
  it('an isolated node walks nowhere', () => {
    expect(rows(randomWalkEncoding(undirected(3, [[0, 1]]), 2))[2]).toEqual([0, 0])
  })
})

describe('laplacianEncoding', () => {
  it('the 4-path: eigenvalues (0.5, 1.5, 2), first eigenvector |entries| (0.5774, 0.4082, 0.4082, 0.5774)', () => {
    const { encoding, values } = laplacianEncoding(path4, 3)
    toFlat(values).forEach((v, k) => expect(v).toBeCloseTo([0.5, 1.5, 2][k], 10))
    const first = rows(encoding).map((r) => Math.abs(r[0]))
    first.forEach((v, k) => expect(v).toBeCloseTo([0.5774, 0.4082, 0.4082, 0.5774][k], 4))
  })
  it('the 6-cycle: double eigenvalues 0.5 and 1.5', () => {
    const values = toFlat(laplacianEncoding(hexagon, 5).values)
    values.forEach((v, k) => expect(v).toBeCloseTo([0.5, 0.5, 1.5, 1.5, 2][k], 10))
  })
  it('pads with zero columns past the last eigenvalue', () => {
    const { encoding, values } = laplacianEncoding(path4, 4)
    expect(values.shape).toEqual([3])
    expect(rows(encoding).map((r) => r[3])).toEqual([0, 0, 0, 0])
  })
})

describe('flipSigns', () => {
  it('keeps absolute values and flips whole columns', () => {
    const { encoding } = laplacianEncoding(path4, 3)
    const f = flipSigns(stream(4), encoding)
    const a = rows(encoding)
    rows(f.encoding).forEach((r, i) => r.forEach((v, j) => expect(v).toBe(a[i][j] * f.signs[j])))
    expect(f.signs.every((s) => s === 1 || s === -1)).toBe(true)
  })
})
