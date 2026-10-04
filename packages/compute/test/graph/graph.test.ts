/** The graph representation (graph.ts), the heap and union–find (heap.ts). */
import { describe, expect, it } from 'vitest'
import {
  adjacencyMatrix,
  createHeap,
  fromAdjacency,
  fromEdges,
  fromMatrix,
  heapPop,
  heapPush,
  inDegree,
  neighbours,
  outDegree,
  path,
  reverse,
  subgraph,
  unionFind,
  unionFindRoot,
  unite,
} from 'aifn-compute/graph'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

describe('representation', () => {
  it('builds the same graph from edges, adjacency lists and a matrix', () => {
    const a = fromEdges(3, [
      [0, 1, 2],
      [0, 2, 5],
      [1, 2, 1],
    ])
    const b = fromAdjacency([[1, 2], [2], []], { weights: [[2, 5], [1], []] })
    const c = fromMatrix([
      [0, 2, 5],
      [0, 0, 1],
      [0, 0, 0],
    ])
    expect(toRows(adjacencyMatrix(a))).toEqual(toRows(adjacencyMatrix(b)))
    expect(toRows(adjacencyMatrix(a))).toEqual(toRows(adjacencyMatrix(c)))
    expect(toFlat(neighbours(a, 0))).toEqual([1, 2])
    expect(toFlat(outDegree(a))).toEqual([2, 1, 0])
    expect(toFlat(inDegree(a))).toEqual([0, 1, 2])
    expect(toFlat(inDegree(reverse(a)))).toEqual([2, 1, 0])
    const sub = subgraph(a, [2, 1])
    expect(sub.edges).toEqual([{ from: 1, to: 0, weight: 1 }])
    expect(sub.labels).toEqual(['2', '1'])
    // Undirected adjacency lists may list a pair from both ends; the matrix is symmetric.
    const u = fromAdjacency([[1], [0, 2], [1]], { directed: false })
    expect(u.edges.length).toBe(2)
    expect(toRows(adjacencyMatrix(u))).toEqual([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ])
    expect(toFlat(path([-1, 0, 1, 1], 3))).toEqual([0, 1, 3])
    expect(() => fromEdges(2, [[0, 2]])).toThrow()
  })

  it('heap pops in priority order (ties first-in) and union–find merges sets', () => {
    const h = createHeap<string>()
    ;[5, 1, 4, 1, 3].forEach((p, k) => heapPush(h, `${p}${'abcde'[k]}`, p))
    const out: string[] = []
    for (let e = heapPop(h); e; e = heapPop(h)) out.push(e.value)
    expect(out).toEqual(['1b', '1d', '3e', '4c', '5a'])
    const uf = unionFind(5)
    expect(unite(uf, 0, 1)).toBe(true)
    expect(unite(uf, 3, 4)).toBe(true)
    expect(unite(uf, 1, 0)).toBe(false)
    expect(uf.count).toBe(3)
    expect(unionFindRoot(uf, 0)).toBe(unionFindRoot(uf, 1))
    expect(unionFindRoot(uf, 2)).not.toBe(unionFindRoot(uf, 3))
  })
})
