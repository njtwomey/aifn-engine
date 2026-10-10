/**
 * TreeSHAP (Lundberg, Erion and Lee, 2018, Algorithm 2; Lundberg et al., 2020): exact Shapley values of a decision
 * tree's output under the path-dependent value function $v(S) = \expect[f(\xvec) \mid \xvec_S]$, where a split on a
 * feature outside $S$ sends $\xvec$ down both branches weighted by their training cover. It follows every root-to-leaf
 * path once, keeping for the features on the path the proportions of the paths through it that coalitions with ("one")
 * and without ("zero") the feature take, and the polynomial weights of each subset size: $O(L D^2)$ for $L$ leaves and
 * depth $D$, against the $O(L\, 2^d)$ of enumerating coalitions.
 *
 * Trees are `aifn-compute/graph` binary trees whose nodes carry the split (`feature`, $-1$ at a leaf; `threshold`, the
 * left child taking $x_{\text{feature}} \le \text{threshold}$), the training `weight` reaching the node (its cover) and
 * the node's `value` vector (the explained output is `value[output]`), as the decision trees of `aifn-methods` are.
 * The covers of a split's two children must add up to its own.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense } from 'aifn-compute/foundation/tensor'

/** The node data TreeSHAP reads. */
export type ShapNode = {
  /** The feature split on, or $-1$ at a leaf. */
  feature: number
  /** The split point: rows with $x_{\text{feature}} \le \text{threshold}$ go left. Unused at a leaf. */
  threshold: number
  /** The cover: the training weight (or count) of the rows reaching the node. */
  weight: number
  /** The node's outputs; the one explained is `value[output]`. Only leaves' values are read. */
  value: readonly number[]
}

/**
 * A binary tree TreeSHAP can explain: `nodes`, each with `children` (the indices in `nodes` of the left and right
 * child at a split, `[]` at a leaf; in `aifn-compute/graph` trees `nodes[i].id === i`), and the index of the `root`.
 */
export type ShapTree = {
  readonly nodes: readonly (ShapNode & { readonly children: readonly number[] })[]
  readonly root: number
}

/**
 * One feature on the current root-to-leaf path: its index `d` ($-1$ for the root's placeholder), the fractions `z` and
 * `o` of the paths through it taken when the feature is unknown ("zero") and known ("one"), and the permutation weight
 * `w` of the subsets of that size.
 */
type PathEntry = { d: number; z: number; o: number; w: number }

/**
 * Add a feature to the path and update the subset-size weights (EXTEND of Algorithm 2). Returns a new path; `m` is not
 * modified.
 *
 * @param m The path so far.
 * @param pz The zero fraction of the new entry: the share of the cover following this branch.
 * @param po The one fraction of the new entry: 1 when $\xvec$ takes this branch, 0 when it does not.
 * @param pi The feature index of the new entry.
 * @returns The extended path, one entry longer.
 */
function extend(m: PathEntry[], pz: number, po: number, pi: number): PathEntry[] {
  const l = m.length
  const out = m.map((e) => ({ ...e }))
  out.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 })
  for (let i = l - 1; i >= 0; i--) {
    out[i + 1].w += (po * out[i].w * (i + 1)) / (l + 1)
    out[i].w = (pz * out[i].w * (l - i)) / (l + 1)
  }
  return out
}

/**
 * The path with entry $i$ removed, undoing its `extend` (UNWIND of Algorithm 2). Returns a new path; `m` is not
 * modified.
 *
 * @param m The path.
 * @param i The 0-based position in `m` of the entry to remove.
 * @returns The path one entry shorter, with the weights it would have had without that entry.
 */
function unwind(m: PathEntry[], i: number): PathEntry[] {
  const l = m.length - 1
  let n = m[l].w
  const out = m.slice(0, l).map((e) => ({ ...e }))
  const { o, z } = m[i]
  for (let j = l - 1; j >= 0; j--) {
    if (o !== 0) {
      const t = out[j].w
      out[j].w = (n * (l + 1)) / ((j + 1) * o)
      n = t - (out[j].w * z * (l - j)) / (l + 1)
    } else out[j].w = (out[j].w * (l + 1)) / (z * (l - j))
  }
  for (let j = i; j < l; j++) {
    out[j].d = m[j + 1].d
    out[j].z = m[j + 1].z
    out[j].o = m[j + 1].o
  }
  return out
}

/**
 * The sum of the path weights after unwinding entry $i$: the Shapley weight of the leaf for that entry's feature.
 *
 * @param m The path ending at a leaf.
 * @param i The 0-based position in `m` of the entry.
 * @returns The sum of the weights of the unwound path.
 */
function unwoundSum(m: PathEntry[], i: number): number {
  return unwind(m, i).reduce((a, e) => a + e.w, 0)
}

/**
 * TreeSHAP values of one tree at $\xvec$: exact Shapley values of the path-dependent value function (see the file
 * comment), in $O(L D^2)$.
 *
 * @param tree The tree to explain.
 * @param x The instance $\xvec$ ($d$ values; the tree's feature indices refer to it).
 * @param output Which entry of the leaves' `value` vectors is explained.
 * @returns The values ($d$, zero for features the tree does not split on), the base value (the cover-weighted mean
 *   leaf output, $\expect[f]$) and the tree's output at $\xvec$; $\text{base} + \sum_i \phi_i = \text{output}$.
 *
 * @example A two-feature AND tree splits the credit evenly
 * // Leaves: x0 <= 0.5 gives 0; otherwise x1 <= 0.5 gives 0 and x1 > 0.5 gives 1. Covers 2, 1 and 1.
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const tree = {
 *   root: 0,
 *   nodes: [
 *     { feature: 0, threshold: 0.5, weight: 4, value: [0.25], children: [1, 2] },
 *     leaf(2, 0),
 *     { feature: 1, threshold: 0.5, weight: 2, value: [0.5], children: [3, 4] },
 *     leaf(1, 0),
 *     leaf(1, 1),
 *   ],
 * }
 * print(treeShap(tree, [1, 1]))
 * print('only x0 on:', treeShap(tree, [1, 0]))
 */
export function treeShap(
  tree: ShapTree,
  x: VectorLike,
  output: Size = 0,
): { values: Float64Array; base: number; output: number } {
  const xv = dense.toF64(x, 'treeShap')
  return {
    values: conditionedShap(tree, xv, output, 0, -1),
    base: expectedValue(tree, output),
    output: predictTree(tree, xv, output),
  }
}

/**
 * TreeSHAP with one feature held in or out of every coalition (as the shap package's `tree_shap_recursive` with a
 * condition). The values of the other features are then Shapley values of the game on $d - 1$ players with that
 * feature fixed in or out.
 *
 * @param tree The tree to explain.
 * @param xv The instance $\xvec$ ($d$ values).
 * @param output Which entry of the leaves' `value` vectors is explained.
 * @param condition 1 keeps `feature` always known (follow $\xvec$'s branch, never on the path), $-1$ always unknown
 *   (both branches by cover), 0 conditions on nothing (plain TreeSHAP).
 * @param feature The feature held in or out; ignored when `condition` is 0 (pass $-1$).
 * @returns The values, $d$ of them.
 */
function conditionedShap(
  tree: ShapTree,
  xv: ArrayLike<number>,
  output: Size,
  condition: -1 | 0 | 1,
  feature: number,
): Float64Array {
  const phi = new Float64Array(xv.length)
  const nodes = tree.nodes
  const recurse = (j: number, path: PathEntry[], pz: number, po: number, pi: number, fraction: number) => {
    if (fraction === 0) return
    let m = condition === 0 || pi !== feature ? extend(path, pz, po, pi) : path.map((e) => ({ ...e }))
    const node = nodes[j]
    if (node.children.length === 0) {
      const leaf = node.value[output] * fraction
      for (let i = 1; i < m.length; i++) phi[m[i].d] += unwoundSum(m, i) * (m[i].o - m[i].z) * leaf
      return
    }
    const [left, right] = node.children
    const hot = xv[node.feature] <= node.threshold ? left : right
    const cold = hot === left ? right : left
    let iz = 1
    let io = 1
    const k = m.findIndex((e, idx) => idx > 0 && e.d === node.feature)
    if (k > 0) {
      iz = m[k].z
      io = m[k].o
      m = unwind(m, k)
    }
    const w = node.weight
    const hz = nodes[hot].weight / w
    const cz = nodes[cold].weight / w
    let hotFraction = fraction
    let coldFraction = fraction
    if (condition > 0 && node.feature === feature) coldFraction = 0
    else if (condition < 0 && node.feature === feature) {
      hotFraction *= hz
      coldFraction *= cz
    }
    recurse(hot, m, iz * hz, io, node.feature, hotFraction)
    recurse(cold, m, iz * cz, 0, node.feature, coldFraction)
  }
  recurse(tree.root, [], 1, 1, -1, 1)
  return phi
}

/**
 * Exact SHAP interaction values of a tree at $\xvec$ (Lundberg, Erion and Lee, 2018, §3; Lundberg et al., 2020): for
 * $i \ne j$, $\Phi_{ij} = (\phi_i^{j\,\text{known}} - \phi_i^{j\,\text{unknown}})/2$, each from a conditioned TreeSHAP
 * pass, and $\Phi_{ii} = \phi_i - \sum_{j \ne i} \Phi_{ij}$, so each row sums to $\phi_i$. $O(d L D^2)$ against
 * enumeration's $O(L\, 2^d)$; as the shap package's `shap_interaction_values`.
 *
 * @param tree The tree to explain.
 * @param x The instance $\xvec$ ($d$ values).
 * @param output Which entry of the leaves' `value` vectors is explained.
 * @returns `values`, the $d \times d$ matrix $\Phi$ row-major ($\Phi_{ij}$ at `i * d + j`; zero rows and columns for
 *   features the tree does not split on), `shapley`, the TreeSHAP values, and `base` and `output` as `treeShap`'s.
 *
 * @example The AND tree's credit, split into main effects and interaction
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const tree = {
 *   root: 0,
 *   nodes: [
 *     { feature: 0, threshold: 0.5, weight: 4, value: [0.25], children: [1, 2] },
 *     leaf(2, 0),
 *     { feature: 1, threshold: 0.5, weight: 2, value: [0.5], children: [3, 4] },
 *     leaf(1, 0),
 *     leaf(1, 1),
 *   ],
 * }
 * const r = treeShapInteractions(tree, [1, 1])
 * print('Phi =', [Array.from(r.values.slice(0, 2)), Array.from(r.values.slice(2, 4))])
 * print('shapley =', r.shapley)
 */
export function treeShapInteractions(
  tree: ShapTree,
  x: VectorLike,
  output: Size = 0,
): { values: Float64Array; shapley: Float64Array; base: number; output: number } {
  const xv = dense.toF64(x, 'treeShapInteractions')
  const d = xv.length
  const shapley = conditionedShap(tree, xv, output, 0, -1)
  const values = new Float64Array(d * d)
  const used = new Set(tree.nodes.filter((n) => n.children.length > 0).map((n) => n.feature))
  for (let j = 0; j < d; j++) {
    if (!used.has(j)) continue
    const on = conditionedShap(tree, xv, output, 1, j)
    const off = conditionedShap(tree, xv, output, -1, j)
    for (let i = 0; i < d; i++) if (i !== j) values[i * d + j] = (on[i] - off[i]) / 2
  }
  for (let i = 0; i < d; i++) {
    let rest = 0
    for (let j = 0; j < d; j++) if (j !== i) rest += values[i * d + j]
    values[i * d + i] = shapley[i] - rest
  }
  return { values, shapley, base: expectedValue(tree, output), output: predictTree(tree, xv, output) }
}

/**
 * SHAP interaction values of an additive ensemble $f = \text{offset} + \sum_t s_t f_t$: the scaled sum of each tree's
 * `treeShapInteractions` (as `ensembleTreeShap`).
 *
 * @param trees The ensemble's trees $f_t$.
 * @param x The instance $\xvec$ ($d$ values).
 * @param options The ensemble's weights and the output explained.
 * @param options.scales The weight $s_t$ of each tree (default $1/T$ each, a forest's average).
 * @param options.offset The constant added to the ensemble's output, such as a boosted model's initial prediction
 *   (default 0); it enters `base` and `output`, not the values.
 * @param options.output Which entry of the leaves' `value` vectors is explained (default 0).
 * @returns `values`, the $d \times d$ interaction matrix row-major, and the ensemble's `base` and `output`.
 *
 * @example Two stumps never interact
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const stump = (feature, low, high) => ({
 *   root: 0,
 *   nodes: [
 *     { feature, threshold: 0.5, weight: 2, value: [(low + high) / 2], children: [1, 2] },
 *     leaf(1, low),
 *     leaf(1, high),
 *   ],
 * })
 * const r = ensembleTreeShapInteractions([stump(0, 0, 2), stump(1, 0, 4)], [1, 1], { scales: [1, 1] })
 * print('Phi =', [Array.from(r.values.slice(0, 2)), Array.from(r.values.slice(2, 4))])
 * print('base =', r.base, ' output =', r.output)
 */
export function ensembleTreeShapInteractions(
  trees: readonly ShapTree[],
  x: VectorLike,
  options: { scales?: readonly number[]; offset?: number; output?: Size } = {},
): { values: Float64Array; base: number; output: number } {
  const { output = 0, offset = 0 } = options
  const scales = options.scales ?? trees.map(() => 1 / trees.length)
  const xv = dense.toF64(x, 'ensembleTreeShapInteractions')
  const values = new Float64Array(xv.length * xv.length)
  let base = offset
  let out = offset
  trees.forEach((t, k) => {
    const r = treeShapInteractions(t, xv, output)
    for (let i = 0; i < values.length; i++) values[i] += scales[k] * r.values[i]
    base += scales[k] * r.base
    out += scales[k] * r.output
  })
  return { values, base, output: out }
}

/**
 * The cover-weighted mean of the leaf outputs: $\expect[f]$ under the training distribution the covers record, and the
 * base value of `treeShap`.
 *
 * @param tree The tree.
 * @param output Which entry of the leaves' `value` vectors is averaged.
 * @returns $\sum_{\text{leaves}} (\text{cover}/\text{root cover}) \cdot \text{value}[\text{output}]$.
 *
 * @example A quarter of the cover reaches the leaf worth 1
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const tree = {
 *   root: 0,
 *   nodes: [
 *     { feature: 0, threshold: 0.5, weight: 4, value: [0.25], children: [1, 2] },
 *     leaf(2, 0),
 *     { feature: 1, threshold: 0.5, weight: 2, value: [0.5], children: [3, 4] },
 *     leaf(1, 0),
 *     leaf(1, 1),
 *   ],
 * }
 * print(expectedValue(tree))
 */
export function expectedValue(tree: ShapTree, output: Size = 0): number {
  const visit = (j: number): number => {
    const node = tree.nodes[j]
    if (node.children.length === 0) return node.value[output]
    return node.children.reduce((a, c) => a + (tree.nodes[c].weight / node.weight) * visit(c), 0)
  }
  return visit(tree.root)
}

/**
 * The tree's output at one row: follow the splits from the root to a leaf.
 *
 * @param tree The tree.
 * @param x The row ($d$ values, indexed by the tree's features).
 * @param output Which entry of the leaf's `value` vector is returned.
 * @returns The leaf's `value[output]`.
 */
function predictTree(tree: ShapTree, x: ArrayLike<number>, output: Size): number {
  let j = tree.root
  for (;;) {
    const node = tree.nodes[j]
    if (node.children.length === 0) return node.value[output]
    j = x[node.feature] <= node.threshold ? node.children[0] : node.children[1]
  }
}

/**
 * The path-dependent value function of a tree at $\xvec$: $v(S) = \expect[f(\xvec) \mid \xvec_S]$, with a split on a
 * feature outside the coalition averaged over both branches by cover. Exponential when enumerated over coalitions;
 * TreeSHAP computes its Shapley values directly.
 *
 * @param tree The tree.
 * @param x The instance $\xvec$ ($d$ values).
 * @param output Which entry of the leaves' `value` vectors is valued.
 * @returns $v$: a function of a membership mask (`mask[i]` true when feature $i$ is known).
 *
 * @example Enumerating the game gives TreeSHAP's values
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const tree = {
 *   root: 0,
 *   nodes: [
 *     { feature: 0, threshold: 0.5, weight: 4, value: [0.25], children: [1, 2] },
 *     leaf(2, 0),
 *     { feature: 1, threshold: 0.5, weight: 2, value: [0.5], children: [3, 4] },
 *     leaf(1, 0),
 *     leaf(1, 1),
 *   ],
 * }
 * const v = pathDependentValue(tree, [1, 1])
 * print('v(none), v({0}), v(all) =', v([false, false]), v([true, false]), v([true, true]))
 * print('exactShapley:', exactShapley(v, 2).values)
 * print('treeShap:', treeShap(tree, [1, 1]).values)
 */
export function pathDependentValue(
  tree: ShapTree,
  x: VectorLike,
  output: Size = 0,
): (mask: readonly boolean[]) => number {
  const xv = dense.toF64(x, 'pathDependentValue')
  return (mask) => {
    const visit = (j: number): number => {
      const node = tree.nodes[j]
      if (node.children.length === 0) return node.value[output]
      const [left, right] = node.children
      if (mask[node.feature]) return visit(xv[node.feature] <= node.threshold ? left : right)
      return (tree.nodes[left].weight * visit(left) + tree.nodes[right].weight * visit(right)) / node.weight
    }
    return visit(tree.root)
  }
}

/**
 * TreeSHAP for an additive ensemble $f = \text{offset} + \sum_t s_t f_t$: the values are the scaled sum of each tree's
 * (Shapley values are linear in the game).
 *
 * @param trees The ensemble's trees $f_t$.
 * @param x The instance $\xvec$ ($d$ values).
 * @param options The ensemble's weights and the output explained.
 * @param options.scales The weight $s_t$ of each tree: default $1/T$ each (a forest's average); a boosted ensemble
 *   passes its learning rate for every tree.
 * @param options.offset The constant added to the output (default 0), such as a boosted ensemble's initial
 *   prediction; it enters `base` and `output`, not the values.
 * @param options.output Which entry of the leaves' `value` vectors is explained (default 0).
 * @returns The values ($d$), the ensemble's `base` and its `output` at $\xvec$;
 *   $\text{base} + \sum_i \phi_i = \text{output}$.
 *
 * @example A boosted pair of stumps
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const stump = (feature, low, high) => ({
 *   root: 0,
 *   nodes: [
 *     { feature, threshold: 0.5, weight: 2, value: [(low + high) / 2], children: [1, 2] },
 *     leaf(1, low),
 *     leaf(1, high),
 *   ],
 * })
 * const trees = [stump(0, -1, 1), stump(1, -2, 2)]
 * const r = ensembleTreeShap(trees, [1, 0], { scales: [0.5, 0.5], offset: 10 })
 * print(r)
 * print('treeEnsembleOutput:', treeEnsembleOutput(trees, [[1, 0]], { scales: [0.5, 0.5], offset: 10 }))
 */
export function ensembleTreeShap(
  trees: readonly ShapTree[],
  x: VectorLike,
  options: { scales?: readonly number[]; offset?: number; output?: Size } = {},
): { values: Float64Array; base: number; output: number } {
  const { output = 0, offset = 0 } = options
  const scales = options.scales ?? trees.map(() => 1 / trees.length)
  const xv = dense.toF64(x, 'ensembleTreeShap')
  const values = new Float64Array(xv.length)
  let base = offset
  let out = offset
  trees.forEach((t, k) => {
    const r = treeShap(t, xv, output)
    for (let i = 0; i < values.length; i++) values[i] += scales[k] * r.values[i]
    base += scales[k] * r.base
    out += scales[k] * r.output
  })
  return { values, base, output: out }
}

/**
 * The output of an additive tree ensemble, $\text{offset} + \sum_t s_t f_t(\xvec)$, on each row of $\Xmat$: the
 * function `ensembleTreeShap` explains.
 *
 * @param trees The ensemble's trees $f_t$.
 * @param X The rows to predict ($m \times d$).
 * @param options The ensemble's weights and the output predicted.
 * @param options.scales The weight $s_t$ of each tree (default $1/T$ each, a forest's average).
 * @param options.offset The constant added to every output (default 0).
 * @param options.output Which entry of the leaves' `value` vectors is predicted (default 0).
 * @returns The ensemble's output for each row ($m$ values).
 *
 * @example A forest of two stumps averages them
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const stump = (feature, low, high) => ({
 *   root: 0,
 *   nodes: [
 *     { feature, threshold: 0.5, weight: 2, value: [(low + high) / 2], children: [1, 2] },
 *     leaf(1, low),
 *     leaf(1, high),
 *   ],
 * })
 * print(treeEnsembleOutput([stump(0, 0, 2), stump(1, 0, 4)], [[0, 0], [1, 0], [0, 1], [1, 1]]))
 */
export function treeEnsembleOutput(
  trees: readonly ShapTree[],
  X: MatrixLike,
  options: { scales?: readonly number[]; offset?: number; output?: Size } = {},
): Float64Array {
  const { output = 0, offset = 0 } = options
  const scales = options.scales ?? trees.map(() => 1 / trees.length)
  const { data, m, n: d } = dense.toMatrixF64(X, 'treeEnsembleOutput')
  const out = new Float64Array(m).fill(offset)
  for (let i = 0; i < m; i++) {
    const row = data.subarray(i * d, (i + 1) * d)
    trees.forEach((t, k) => (out[i] += scales[k] * predictTree(t, row, output)))
  }
  return out
}
