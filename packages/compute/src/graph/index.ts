/**
 * `aifn-compute/graph`: graphs and trees as plain data, and the algorithms on them, as networkx.
 *
 * The shared layer sits at the family's root: `Graph` (a node count and an edge list, built by `fromEdges`,
 * `fromAdjacency` or `fromMatrix`, read through `adjacency`, whose neighbour order fixes every traversal's), `Tree`
 * (rooted, ordered trees with their constructors, queries and traversals, and `spanningTreeOf` for a search's tree),
 * a binary min-heap and union-find. The modules:
 *
 * - `traversal`: breadth- and depth-first search, topological order, cycles and connected components.
 * - `shortest-paths`: Dijkstra, A*, Bellman-Ford and Floyd-Warshall.
 * - `spanning-trees`: minimum spanning trees by Kruskal or Prim, with their steps.
 * - `flows`: maximum flow with a minimum cut (Edmonds-Karp), and minimum-cost flow.
 * - `structured`: intentional graphs for graphical models, with node roles, plates and templates, `unroll` and
 *   `shape`.
 * - `structures`: standard, random and point-cloud graphs.
 * - `matrices`: the adjacency, degree, Laplacian and incidence matrices.
 * - `propagation`: differentiable message passing and label propagation.
 */

export {
  adjacency,
  fromAdjacency,
  fromEdges,
  fromMatrix,
  inDegree,
  neighbours,
  outDegree,
  path,
  reverse,
  subgraph,
  type Arc,
  type Edge,
  type EdgeInput,
  type Graph,
  type GraphOptions,
} from './graph'
export {
  ancestors,
  binaryTree,
  depth,
  depths,
  foldTree,
  height,
  inOrder,
  isLeaf,
  lca,
  leaves,
  leftChild,
  levelOrder,
  mapTree,
  pathFromRoot,
  pathToRoot,
  postOrder,
  preOrder,
  rightChild,
  spanningForestOf,
  spanningTreeOf,
  subtreeSize,
  treeFromChildren,
  treeFromNested,
  treeFromParents,
  type NestedBinaryTree,
  type NestedTree,
  type SpanningInput,
  type SpanningTreeEdge,
  type SpanningTreeNode,
  type Tree,
  type TreeEdge,
  type TreeNode,
  type TreeOptions,
} from './tree'
export {
  createHeap,
  heapCopy,
  heapPeek,
  heapPop,
  heapPush,
  heapSorted,
  unionFind,
  unionFindCopy,
  unionFindRoot,
  unite,
  type Heap,
  type HeapEntry,
  type UnionFind,
} from './heap'
export { breadthFirstSearch, depthFirstSearch, topologicalSort, connectedComponents } from './traversal'
export { dijkstra, shortestPath } from './shortest-paths'
export { minimumSpanningTree } from './spanning-trees'
export { maxFlow } from './flows'
export { shape, structured, unroll, type StructuredGraph } from './structured'
export { gridGraph, kNearestNeighbourGraph } from './structures'
export { adjacencyMatrix, laplacian } from './matrices'
export { propagate } from './propagation'
