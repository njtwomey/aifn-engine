/**
 * Tests of the `Tree` type in aifn-compute/graph and the graph producers that return one (breadth- and depth-first forests,
 * spanning trees). References are hand-checked. Huffman trees are tested in applications (information/coding) and
 * the branch-and-bound search tree in optim/programming.
 */

import { describe, expect, it } from 'vitest'
import {
  ancestors,
  binaryTree,
  depth,
  depths,
  foldTree,
  fromEdges,
  height,
  inOrder,
  lca,
  leaves,
  levelOrder,
  mapTree,
  pathToRoot,
  postOrder,
  preOrder,
  rightChild,
  leftChild,
  spanningForestOf,
  spanningTreeOf,
  subtreeSize,
  treeFromChildren,
  treeFromNested,
  treeFromParents,
} from 'aifn-compute/graph'
import { breadthFirstSearch, depthFirstSearch } from 'aifn-compute/graph/traversal'
import { minimumSpanningTree } from 'aifn-compute/graph/spanning-trees'

//        0
//      / | \
//     1  2  3
//    / \     \
//   4   5     6
//             |
//             7
const PARENTS = [-1, 0, 0, 0, 1, 1, 3, 6]

describe('graph: Tree constructors', () => {
  it('parents, children and nested agree', () => {
    const a = treeFromParents(PARENTS, { labels: PARENTS.map((_, i) => `n${i}`) })
    const b = treeFromChildren([[1, 2, 3], [4, 5], [], [6], [], [], [7], []])
    expect(a.root).toBe(0)
    expect(a.nodes.map((n) => n.children)).toEqual(b.nodes.map((n) => n.children))
    expect(a.nodes[4].label).toBe('n4')
    expect(a.edges[0]).toBeNull()
    const c = treeFromNested<{ v?: number }>({
      label: 'r',
      children: [
        { label: 'a', v: 1 },
        { label: 'b', edge: { label: 'e' } },
      ],
    })
    expect(preOrder(c).map((i) => c.nodes[i].label)).toEqual(['r', 'a', 'b'])
    expect(c.edges[2]).toEqual({ label: 'e' })
    expect(c.nodes[1].v).toBe(1)
    // A tree survives JSON.
    expect(JSON.parse(JSON.stringify(a))).toEqual(a)
  })

  it('rejects forests, cycles and unknown ids', () => {
    expect(() => treeFromParents([-1, -1])).toThrow(/one root/)
    expect(() => treeFromChildren([[1], [0]])).toThrow()
    expect(() => treeFromChildren([[3]])).toThrow(/unknown child/)
  })

  it('binary trees keep the side of a lone child', () => {
    const t = binaryTree({ label: '2', left: { label: '1' }, right: { label: '4', left: { label: '3' } } })
    expect(t.arity).toBe(2)
    expect(inOrder(t).map((i) => t.nodes[i].label)).toEqual(['1', '2', '3', '4'])
    const lone = binaryTree({ label: 'a', right: { label: 'b' } })
    expect(leftChild(lone, 0)).toBeNull()
    expect(rightChild(lone, 0)).toBe(1)
    expect(inOrder(lone).map((i) => lone.nodes[i].label)).toEqual(['a', 'b'])
  })
})

describe('graph: Tree queries and traversals', () => {
  const t = treeFromParents(PARENTS)
  it('queries', () => {
    expect(depth(t, 7)).toBe(3)
    expect(depths(t)).toEqual([0, 1, 1, 1, 2, 2, 2, 3])
    expect(height(t)).toBe(3)
    expect(height(t, 1)).toBe(1)
    expect(leaves(t)).toEqual([4, 5, 2, 7])
    expect(ancestors(t, 7)).toEqual([6, 3, 0])
    expect(pathToRoot(t, 5)).toEqual([5, 1, 0])
    expect(lca(t, 4, 5)).toBe(1)
    expect(lca(t, 4, 7)).toBe(0)
    expect(lca(t, 6, 7)).toBe(6)
    expect(subtreeSize(t)).toBe(8)
    expect(subtreeSize(t, 3)).toBe(3)
  })
  it('traversals, map and fold', () => {
    expect(preOrder(t)).toEqual([0, 1, 4, 5, 2, 3, 6, 7])
    expect(postOrder(t)).toEqual([4, 5, 1, 2, 7, 6, 3, 0])
    expect(levelOrder(t)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    const m = mapTree(t, (n) => ({ size: subtreeSize(t, n.id) }))
    expect(m.nodes.map((n) => n.size)).toEqual([8, 3, 1, 3, 1, 1, 2, 1])
    expect(m.nodes[3].children).toEqual([6])
    expect(foldTree(t, (n, xs: number[]) => n.id + xs.reduce((a, b) => a + b, 0))).toBe(28)
  })
})

describe('graph: trees from searches', () => {
  const g = fromEdges(6, [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
  ])
  it('breadth- and depth-first forests', () => {
    const bfs = breadthFirstSearch(g, 0)
    expect(bfs.trees).toHaveLength(1)
    const b = bfs.trees[0]
    expect(b.nodes.map((n) => n.vertex)).toEqual([0, 1, 3, 4, 2])
    expect(b.nodes[0].children.map((c) => b.nodes[c].vertex)).toEqual([1, 2])
    expect(b.edges[1]!.edge).toBe(0)
    const dfs = depthFirstSearch(g)
    // Vertex 5 is isolated: a second tree.
    expect(dfs.trees.map((t) => t.nodes.length)).toEqual([5, 1])
    expect(height(dfs.trees[0])).toBe(3)
  })
  it('a spanning tree from an edge set', () => {
    const w = fromEdges(
      4,
      [
        [0, 1, 1],
        [1, 2, 5],
        [0, 2, 2],
        [2, 3, 1],
      ],
      { directed: false, labels: ['a', 'b', 'c', 'd'] },
    )
    const mst = minimumSpanningTree(w)
    const t = spanningTreeOf(w, { edges: mst.edges.data }, 0)
    expect(t.nodes.map((n) => n.label)).toEqual(['a', 'b', 'c', 'd'])
    expect(t.edges.reduce((s, e) => s + (e?.weight ?? 0), 0)).toBe(mst.weight)
    expect(spanningForestOf(w, { edges: mst.edges.data })).toHaveLength(1)
  })
})
