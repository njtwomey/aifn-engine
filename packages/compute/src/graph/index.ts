/**
 * `aifn-compute/graph`: graphs and trees as plain data (the shared layer: `Graph`, `Tree`, heaps and union–find) and the algorithms on them,
 * as networkx. Children: traversal, shortest-paths, spanning-trees, flows; structured (intentional graphs: roles,
 * plates and templates, `unroll`, `shape`), structures (standard, random and point-cloud graphs), matrices (adjacency,
 * Laplacian, incidence) and propagation (differentiable message passing).
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
