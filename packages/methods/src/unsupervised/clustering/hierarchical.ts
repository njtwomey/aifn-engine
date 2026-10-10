/**
 * Agglomerative hierarchical clustering (Everitt et al., 2011, "Cluster Analysis", ch. 4): start from singletons and
 * repeatedly merge the two closest clusters under single, complete, average (UPGMA) or Ward linkage. The result is a
 * linkage matrix in SciPy's format and a tree of merges with heights; `cutTree` cuts it into flat clusters and
 * `dendrogram` lays it out.
 *
 * Cluster distances are computed from the current partition at each step: single, complete and average linkage from
 * the point distances, Ward's from sizes and centroids,
 * $d(A, B) = \sqrt{2\lvert A \rvert \lvert B \rvert / (\lvert A \rvert + \lvert B \rvert)}
 * \, \lVert \cvec_A - \cvec_B \rVert$ (the height SciPy reports; Ward, 1963). Each step costs $O(n^2)$, so the
 * whole hierarchy $O(n^3)$.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type { Dataset, Estimator, FitOptions, Trained } from 'aifn-compute/learning/estimators'
import type { Tree, TreeNode } from 'aifn-compute/graph'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { canonical, ints, matrix } from './util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, space } from 'aifn-compute/foundation/space'

/**
 * How the distance between two clusters is measured: `'single'` the closest pair of their points, `'complete'` the
 * farthest pair, `'average'` the mean over all pairs (UPGMA), `'ward'` Ward's distance between their centroids.
 */
export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One state of agglomeration. */
export interface AgglomerationState extends Status {
  /**
   * The cluster id of each point, $n$ values (int32): point $i$ starts in cluster $i$; merge $r$ creates cluster
   * $n + r$ (SciPy's ids).
   */
  labels: Tensor
  /** The ids of the clusters that remain. */
  active: number[]
  /** Merges so far, $r \times 4$: the two ids (smaller first), the height and the new cluster's size. */
  merges: Tensor
  /** The merge made in this step ([a, b, height]), or null at the start. */
  last: [number, number, number] | null
  /** Merges done. */
  t: number
}

/**
 * Agglomeration as a traceable algorithm on the rows of `x`: each step merges the closest pair of active clusters
 * (ties to the smallest pair of ids). It is done after $n - 1$ merges; a step after that returns the state unchanged.
 *
 * @param x The data, $n \times d$, one point per row.
 * @param params The `linkage` (default `'ward'`).
 * @returns The algorithm, whose `init` takes nothing and whose state is an `AgglomerationState`.
 *
 * @example Two single-linkage merges on a line
 * const x = tensor([[0], [1], [3], [7]])
 * const state = run(agglomerativeSteps(x, { linkage: 'single' }), undefined, 2)
 * print('cluster ids', state.labels)
 * print('active', state.active)
 * print('merges', state.merges)
 */
export function agglomerativeSteps(x: Tensor, params: { linkage?: Linkage } = {}): Algorithm<void, AgglomerationState> {
  const { n, d, v } = matrix(x, 'agglomerativeSteps')
  const linkage = params.linkage ?? 'ward'
  const D = linkage === 'ward' ? null : dense.data(pairwiseDistances(x) as Tensor)
  return {
    name: `agglomerative-${linkage}`,
    init: () => ({
      labels: fromData(
        Int32Array.from({ length: n }, (_, i) => i),
        [n],
      ),
      active: Array.from({ length: n }, (_, i) => i),
      merges: fromData(new Float64Array(0), [0, 4]),
      last: null,
      t: 0,
    }),
    step: (state) => {
      const labels = state.labels.data as Int32Array
      const ids = state.active
      const c = ids.length
      if (c < 2) return state
      const index = new Map(ids.map((id, j) => [id, j]))
      const size = new Float64Array(c)
      for (const l of labels) size[index.get(l)!]++
      const dist = new Float64Array(c * c)
      if (linkage === 'ward') {
        const centre = new Float64Array(c * d)
        for (let i = 0; i < n; i++) {
          const j = index.get(labels[i])!
          for (let t = 0; t < d; t++) centre[j * d + t] += v[i * d + t]
        }
        for (let j = 0; j < c; j++) for (let t = 0; t < d; t++) centre[j * d + t] /= size[j]
        for (let a = 0; a < c; a++) {
          for (let b = a + 1; b < c; b++) {
            let s = 0
            for (let t = 0; t < d; t++) s += (centre[a * d + t] - centre[b * d + t]) ** 2
            dist[a * c + b] = Math.sqrt(((2 * size[a] * size[b]) / (size[a] + size[b])) * s)
          }
        }
      } else {
        if (linkage === 'single') dist.fill(Infinity)
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            let a = index.get(labels[i])!
            let b = index.get(labels[j])!
            if (a === b) continue
            if (a > b) [a, b] = [b, a]
            const dij = D![i * n + j]
            const k = a * c + b
            if (linkage === 'single') dist[k] = Math.min(dist[k], dij)
            else if (linkage === 'complete') dist[k] = Math.max(dist[k], dij)
            else dist[k] += dij
          }
        }
        if (linkage === 'average')
          for (let a = 0; a < c; a++) for (let b = a + 1; b < c; b++) dist[a * c + b] /= size[a] * size[b]
      }
      let best: [number, number] = [0, 1]
      let bestD = Infinity
      for (let a = 0; a < c; a++) {
        for (let b = a + 1; b < c; b++) {
          const t = dist[a * c + b]
          // Ties go to the pair of smallest ids (ids are ascending in `active`).
          if (t < bestD) {
            bestD = t
            best = [a, b]
          }
        }
      }
      const [ia, ib] = [ids[best[0]], ids[best[1]]]
      const id = n + state.t
      const next = Int32Array.from(labels, (l) => (l === ia || l === ib ? id : l))
      const merges = new Float64Array((state.t + 1) * 4)
      merges.set(state.merges.data as Float64Array)
      merges.set([Math.min(ia, ib), Math.max(ia, ib), bestD, size[best[0]] + size[best[1]]], state.t * 4)
      return {
        labels: fromData(next, [n]),
        active: [...ids.filter((j) => j !== ia && j !== ib), id],
        merges: fromData(merges, [state.t + 1, 4]),
        last: [Math.min(ia, ib), Math.max(ia, ib), bestD],
        t: state.t + 1,
      }
    },
    done: (state) => state.active.length <= 1,
  }
}

/** What a merge-tree node records: the number of points below it (the height is the tree node's `height`). */
export interface MergeData {
  /** The number of points below the node (1 for a point). */
  size: number
}

/**
 * A merge tree: an `aifn-compute/graph` binary `Tree` with SciPy's ids (nodes $0, \dots, n - 1$ are the points, node
 * $n + r$ is merge $r$; the root is last). Each node's `height` is its merge distance (0 for points), its children are
 * the merged pair (smaller id first), and each edge's `weight` is the branch length (parent height minus child
 * height).
 */
export type MergeTree = Tree<MergeData>

/**
 * The merge tree of a linkage matrix. Points are labelled by their ids; merge nodes are unlabelled.
 *
 * @param merges The linkage matrix, $(n - 1) \times 4$ in SciPy's format, as `linkage` returns it (float64).
 * @returns The `MergeTree` of $2n - 1$ nodes, rooted at the last merge.
 *
 * @example The tree of three points
 * const tree = mergeTree(linkage(tensor([[0], [1], [5]]), 'single'))
 * print('heights', tree.nodes.map((node) => node.height))
 * print('children of the root', tree.nodes[tree.root].children)
 * print('branch lengths', tree.edges.map((e) => (e ? e.weight : null)))
 */
export function mergeTree(merges: Tensor): MergeTree {
  const r = merges.shape[0]
  const n = r + 1
  const Z = merges.data as Float64Array
  const nodes: TreeNode<MergeData>[] = Array.from({ length: n }, (_, i) => ({
    id: i,
    parent: null,
    children: [],
    height: 0,
    size: 1,
    label: String(i),
  }))
  for (let k = 0; k < r; k++) {
    const a = Z[k * 4]
    const b = Z[k * 4 + 1]
    nodes.push({ id: n + k, parent: null, children: [a, b], height: Z[k * 4 + 2], size: Z[k * 4 + 3], label: '' })
    nodes[a].parent = n + k
    nodes[a].slot = 0
    nodes[b].parent = n + k
    nodes[b].slot = 1
  }
  const edges = nodes.map((node) =>
    node.parent === null ? null : { weight: nodes[node.parent].height! - node.height! },
  )
  return { kind: 'tree', nodes, root: nodes.length - 1, edges, arity: 2 }
}

/**
 * Flat clusters from a linkage matrix: cut into `clusters` groups, or at `height` (merges above it are undone). The
 * merges are taken to be in order of height, as the linkages here produce them.
 *
 * @param merges The linkage matrix, $(n - 1) \times 4$ in SciPy's format (float64).
 * @param cut Where to cut: `clusters`, the number of groups (clamped to $1, \dots, n$), or `height`, keeping the
 *   merges at or below that height.
 * @returns The label of each point, $n$ values (int32), numbered $0, 1, \dots$ in order of first appearance.
 *
 * @example Two clusters, or a cut at height 1.5
 * const merges = linkage(tensor([[0], [1], [3], [7]]), 'single')
 * print('2 clusters', cutTree(merges, { clusters: 2 }))
 * print('height 1.5', cutTree(merges, { height: 1.5 }))
 */
export function cutTree(merges: Tensor, cut: { clusters: number } | { height: number }): Tensor {
  const r = merges.shape[0]
  const n = r + 1
  const Z = merges.data as Float64Array
  const keep =
    'clusters' in cut
      ? n - Math.max(1, Math.min(cut.clusters, n))
      : Array.from({ length: r }, (_, k) => Z[k * 4 + 2]).filter((h) => h <= cut.height).length
  const label = Int32Array.from({ length: n }, (_, i) => i)
  // Union the first `keep` merges (they are in order of height).
  const parent = Int32Array.from({ length: n + r }, (_, i) => i)
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  for (let k = 0; k < keep; k++) {
    parent[find(Z[k * 4])] = n + k
    parent[find(Z[k * 4 + 1])] = n + k
  }
  for (let i = 0; i < n; i++) label[i] = find(i)
  return fromData(canonical(label), [n])
}

/** A dendrogram layout: leaf order and, for every merge, the bracket-shaped link as $x$ and $y$ coordinates. */
export interface Dendrogram {
  /** Point ids from left to right (SciPy's `leaves`), $n$ values (int32). */
  order: Tensor
  /**
   * The $x$ position of every node, $2n - 1$ values indexed by id (leaves at $0, 1, \dots$ in `order`; a merge at the
   * mean of its children's).
   */
  x: Tensor
  /**
   * Per merge, the four corners of its link, $r \times 4 \times 2$: $(x_a, h_a)$, $(x_a, h)$, $(x_b, h)$, $(x_b, h_b)$,
   * for children $a$ and $b$ and merge height $h$.
   */
  links: Tensor
  /** The merge tree laid out. */
  tree: MergeTree
}

/**
 * Lays out a dendrogram: the left child of each merge is its first id, as SciPy's `dendrogram` (whose leaves sit at
 * $5, 15, 25, \dots$ rather than $0, 1, 2, \dots$).
 *
 * @param merges The linkage matrix, $(n - 1) \times 4$ in SciPy's format (float64).
 * @returns The layout: leaf `order`, node positions `x`, the `links` to draw and the `tree`.
 *
 * @example The layout of four points
 * const layout = dendrogram(linkage(tensor([[0], [1], [3], [7]]), 'single'))
 * print('leaves', layout.order)
 * print('x', layout.x)
 * print('last link', toArray(layout.links)[2])
 */
export function dendrogram(merges: Tensor): Dendrogram {
  const tree = mergeTree(merges)
  const n = merges.shape[0] + 1
  const order: number[] = []
  const x = new Float64Array(tree.nodes.length)
  const place = (id: number): number => {
    const node = tree.nodes[id]
    if (!node.children.length) {
      x[id] = order.length
      order.push(id)
      return x[id]
    }
    x[id] = (place(node.children[0]) + place(node.children[1])) / 2
    return x[id]
  }
  if (n > 1) place(tree.root)
  else order.push(0)
  const r = n - 1
  const links = new Float64Array(r * 8)
  for (let k = 0; k < r; k++) {
    const node = tree.nodes[n + k]
    const [a, b] = node.children
    links.set([x[a], tree.nodes[a].height!, x[a], node.height!, x[b], node.height!, x[b], tree.nodes[b].height!], k * 8)
  }
  return { order: ints(order), x: fromData(x, [x.length]), links: fromData(links, [r, 4, 2]), tree }
}

/** A fitted agglomerative clustering. */
export interface AgglomerativeModel extends Trained<AgglomerationState> {
  readonly kind: 'model'
  /** Agglomeration partitions the training rows only: it cannot place new inputs. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'agglomerative'
  /** The linkage it was fitted with. */
  readonly linkage: Linkage
  /** The linkage matrix, $(n - 1) \times 4$, in SciPy's format. */
  readonly merges: Tensor
  /** The merge tree of `merges`. */
  readonly tree: MergeTree
  /** Flat labels of the training rows from the cut given to the estimator (or a single cluster). */
  readonly labels: Tensor
  /** Flat labels of the training rows from another cut (`cutTree` on `merges`). */
  cut(at: { clusters: number } | { height: number }): Tensor
}

/**
 * Agglomerative clustering: the full merge sequence, kept as a trace (one merge per step), with flat labels from
 * `clusters` or `height` if given. The fit is deterministic and transductive: `cut` relabels the training rows, but
 * new rows cannot be placed.
 *
 * @param params The hyperparameters.
 * @param params.linkage How cluster distances are measured (default `'ward'`).
 * @param params.clusters The number of flat clusters for `labels`; takes precedence over `height`.
 * @param params.height The height at which to cut for `labels`; with neither, `labels` is one cluster.
 * @returns The estimator; `fit({ x })` takes the data, $n \times d$.
 *
 * @example Merge heights of four points, and two clusters
 * const x = tensor([[0, 0], [0, 1], [4, 0], [4, 3]])
 * const model = agglomerative({ linkage: 'average', clusters: 2 }).fit({ x })
 * print('merges (a, b, height, size)', model.merges)
 * print('labels', model.labels)
 */
export function agglomerative(
  params: { linkage?: Linkage; clusters?: number; height?: number } = {},
): Estimator<Dataset<Tensor>, AgglomerativeModel> {
  const { linkage = 'ward' } = params
  return {
    name: 'agglomerative',
    params,
    fit({ x }, options: FitOptions = {}) {
      const { n } = matrix(x, 'agglomerative')
      const training: Trace<AgglomerationState> = trace(agglomerativeSteps(x, { linkage }), undefined, n, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: { clusters: (s) => s.active.length },
      })
      const merges = linkageFromTrace(training)
      const cut = (at: { clusters: number } | { height: number }) => cutTree(merges, at)
      return {
        kind: 'model',
        transductive: true,
        name: 'agglomerative',
        linkage,
        merges,
        tree: mergeTree(merges),
        labels:
          params.clusters !== undefined
            ? cut({ clusters: params.clusters })
            : params.height !== undefined
              ? cut({ height: params.height })
              : cut({ clusters: 1 }),
        cut,
        training,
      }
    },
  }
}

/**
 * The linkage matrix of a finished agglomeration.
 *
 * @param t The trace of `agglomerativeSteps` run to the end.
 * @returns The merges of its final state, $(n - 1) \times 4$.
 */
function linkageFromTrace(t: Trace<AgglomerationState>): Tensor {
  return t.final.merges
}

/**
 * The linkage matrix of the rows of `x` (SciPy's `linkage(x, method)` format): row $r$ holds the ids merged (smaller
 * first), the merge height and the new cluster's size.
 *
 * @param x The data, $n \times d$, one point per row.
 * @param method The linkage.
 * @returns The merges, $(n - 1) \times 4$ (float64).
 *
 * @example Single and Ward linkage of four points
 * const x = tensor([[0, 0], [0, 1], [4, 0], [4, 3]])
 * print('single', linkage(x, 'single'))
 * print('ward', linkage(x))
 */
export function linkage(x: Tensor, method: Linkage = 'ward'): Tensor {
  return run(agglomerativeSteps(x, { linkage: method }), undefined, x.shape[0]).merges
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'agglomerative',
    module: 'unsupervised/clustering',
    name: 'Agglomerative clustering',
    summary: 'Repeated merging of the closest clusters under a linkage, cut into flat clusters.',
    task: 'clustering',
    capabilities: [],
    transductive: true,
    hyper: space({ linkage: oneOf(['ward', 'single', 'complete', 'average']), clusters: int(1, 20, { default: 2 }) }),
    notes: ['hierarchical-clustering'],
    cite: ['ward1963'],
  },
  agglomerative,
)
