/** Traversal, order and components (aifn-compute/graph/traversal). */
import { describe, expect, it } from 'vitest'
import {
  bipartite,
  breadthFirstSearch,
  breadthFirstSteps,
  condensation,
  connectedComponents,
  depthFirstSearch,
  depthFirstSteps,
  findCycle,
  isDag,
  iterativeDeepening,
  iterativeDeepeningSteps,
  kahnSteps,
  kosarajuSteps,
  stronglyConnectedComponents,
  tarjanSteps,
  topologicalSort,
  unweightedShortestPaths,
} from 'aifn-compute/graph/traversal'
import { fromEdges, path } from 'aifn-compute/graph'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { checkProtocol } from '../../protocol'
import { byName, lettered, names, randomGraph } from '../helpers'

describe('traversal', () => {
  // CLRS Figure 22.3: breadth-first search from s.
  const clrs223 = lettered('rstuvwxy', ['rs', 'rv', 'sw', 'wt', 'wx', 'tx', 'tu', 'xu', 'xy', 'uy'], false)
  // CLRS Figure 22.4: depth-first search, neighbours in alphabetical order.
  const clrs224 = lettered('uvwxyz', ['uv', 'ux', 'vy', 'wy', 'wz', 'xv', 'yx', 'zz'])

  it('breadth-first search gives CLRS 22.3 layers', () => {
    const at = byName(names('rstuvwxy'))
    const r = breadthFirstSearch(clrs223, at('s'))
    const letters = (t: ArrayLike<number>) => Array.from(t, (v) => 'rstuvwxy'[v]).join('')
    expect(r.layers.map((l) => letters(l.data))).toEqual(['s', 'rw', 'vtx', 'uy'])
    expect(toFlat(r.depth)).toEqual([1, 0, 2, 3, 2, 1, 2, 3])
    expect(letters(path(r.parent, at('y')).data)).toEqual('swxy')
    expect(toFlat(unweightedShortestPaths(clrs223, at('s')).distance)).toEqual([1, 0, 2, 3, 2, 1, 2, 3])
  })

  it('depth-first search gives CLRS 22.4 times and edge classes', () => {
    const r = depthFirstSearch(clrs224)
    expect(toFlat(r.discovery)).toEqual([1, 2, 9, 4, 3, 10])
    expect(toFlat(r.finish)).toEqual([8, 7, 12, 5, 6, 11])
    // Edges: uv ux vy wy wz xv yx zz.
    expect(r.edgeClass).toEqual(['tree', 'forward', 'tree', 'cross', 'tree', 'back', 'tree', 'back'])
    expect(toFlat(r.preorder)).toEqual([0, 1, 4, 3, 2, 5])
    expect(toFlat(r.postorder)).toEqual([3, 4, 1, 0, 5, 2])
    // An undirected graph has only tree and back edges.
    const u = depthFirstSearch(clrs223)
    expect(new Set(u.edgeClass)).toEqual(new Set(['tree', 'back']))
    expect(u.edgeClass.filter((c) => c === 'tree').length).toBe(7)
  })

  it('iterative deepening finds a shortest path and the breadth-first depths', () => {
    const at = byName(names('rstuvwxy'))
    const r = iterativeDeepening(clrs223, at('s'), at('u'))
    expect(r.found).toBe(true)
    expect(r.path.shape[0]).toBe(4)
    expect(r.limit).toBe(3)
    const all = iterativeDeepening(clrs223, at('s'))
    expect(toFlat(all.depth)).toEqual(toFlat(breadthFirstSearch(clrs223, at('s')).depth))
  })

  it('follows the trace protocol', () => {
    const record = { t: (s: { t: number }) => s.t }
    checkProtocol(breadthFirstSteps(clrs224), undefined, { steps: 12, record })
    checkProtocol(depthFirstSteps(clrs224), undefined, { steps: 12, record })
    checkProtocol(iterativeDeepeningSteps(clrs223, { source: 1, target: 3 }), undefined, { steps: 12, record })
    checkProtocol(breadthFirstSteps(clrs224), undefined, { steps: 100 })
  })
})

describe('order and cycles', () => {
  it('topological orders are valid, and cycles are found', () => {
    for (let seed = 0; seed < 10; seed++) {
      // A random DAG: edges only from lower to higher index, then relabelled by a permutation.
      const g0 = randomGraph(seed, 9, 0.3)
      const perm = [4, 7, 0, 2, 8, 1, 6, 3, 5]
      const dag = fromEdges(
        9,
        g0.edges.filter((e) => e.from < e.to).map((e) => [perm[e.from], perm[e.to]] as const),
      )
      expect(isDag(dag)).toBe(true)
      for (const method of ['kahn', 'depth-first'] as const) {
        const { order, cycle } = topologicalSort(dag, { method })
        expect(cycle).toBeNull()
        const rank = new Map(Array.from(order.data, (v, i) => [v, i]))
        expect(rank.size).toBe(9)
        for (const e of dag.edges) expect(rank.get(e.from)!).toBeLessThan(rank.get(e.to)!)
      }
    }
    const cyclic = fromEdges(5, [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 1],
      [3, 4],
    ])
    expect(isDag(cyclic)).toBe(false)
    expect(toFlat(findCycle(cyclic)!)).toEqual([1, 2, 3])
    const sorted = topologicalSort(cyclic)
    expect(toFlat(sorted.order)).toEqual([0])
    expect(sorted.cycle).not.toBeNull()
    expect(toFlat(findCycle(fromEdges(2, [[1, 1]]))!)).toEqual([1])
  })
})

describe('components', () => {
  // CLRS Figure 22.9.
  const clrs229 = lettered('abcdefgh', [
    'ab',
    'bc',
    'be',
    'bf',
    'cd',
    'cg',
    'dc',
    'dh',
    'ea',
    'ef',
    'fg',
    'gf',
    'gh',
    'hh',
  ])
  const partitionOf = (members: { data: ArrayLike<number> }[]) =>
    members
      .map((m) =>
        Array.from(m.data, (v) => 'abcdefgh'[v])
          .sort()
          .join(''),
      )
      .sort()

  it('Tarjan and Kosaraju find the CLRS 22.9 components', () => {
    const tarjan = stronglyConnectedComponents(clrs229)
    const kosaraju = stronglyConnectedComponents(clrs229, { method: 'kosaraju' })
    expect(partitionOf(tarjan.members)).toEqual(['abe', 'cd', 'fg', 'h'])
    expect(partitionOf(kosaraju.members)).toEqual(['abe', 'cd', 'fg', 'h'])
    // Tarjan numbers components in reverse topological order, Kosaraju in topological order.
    expect(partitionOf([tarjan.members[0]])).toEqual(['h'])
    expect(partitionOf([kosaraju.members[0]])).toEqual(['abe'])
    const c = condensation(clrs229)
    expect(isDag(c.graph)).toBe(true)
    expect(c.graph.edges.length).toBe(5)
    expect(c.graph.labels).toContain('a,b,e')
  })

  it('connected components ignore direction', () => {
    const g = fromEdges(6, [
      [0, 1],
      [2, 1],
      [3, 4],
    ])
    const r = connectedComponents(g)
    expect(r.count).toBe(3)
    expect(toFlat(r.labels)).toEqual([0, 0, 0, 1, 1, 2])
  })

  it('bipartite graphs get a two-colouring, others an odd cycle', () => {
    const square = fromEdges(
      4,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
      ],
      { directed: false },
    )
    const r = bipartite(square)
    expect(r.bipartite).toBe(true)
    if (r.bipartite) for (const e of square.edges) expect(r.colour.data[e.from]).not.toBe(r.colour.data[e.to])
    const pentagon = fromEdges(
      6,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0],
        [4, 5],
      ],
      { directed: false },
    )
    const p = bipartite(pentagon)
    expect(p.bipartite).toBe(false)
    if (!p.bipartite) {
      const cycle = toFlat(p.oddCycle)
      expect(cycle.length % 2).toBe(1)
      const adjacent = (a: number, b: number) =>
        pentagon.edges.some((e) => (e.from === a && e.to === b) || (e.from === b && e.to === a))
      cycle.forEach((v, i) => expect(adjacent(v, cycle[(i + 1) % cycle.length])).toBe(true))
    }
  })
})

describe('the Algorithm protocol for order and components', () => {
  it('Kahn, Tarjan and Kosaraju', () => {
    const g = randomGraph(3, 8, 0.3)
    checkProtocol(tarjanSteps(g), undefined, { steps: 20 })
    checkProtocol(kosarajuSteps(g), undefined, { steps: 20 })
    const dag = fromEdges(6, [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [3, 4],
      [5, 4],
    ])
    checkProtocol(kahnSteps(dag), undefined, { steps: 10 })
  })
})
