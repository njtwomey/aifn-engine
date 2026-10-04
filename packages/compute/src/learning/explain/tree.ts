/**
 * TreeSHAP (Lundberg, Erion and Lee, 2018, Algorithm 2; Lundberg et al., 2020): exact Shapley values of a decision
 * tree's output under the path-dependent value function v(S) = E[f(x) | x_S], where a split on a feature outside S
 * sends x down both branches weighted by their training cover. It follows every root-to-leaf path once, keeping for
 * the features on the path the proportions of the paths through it that coalitions with ("one") and without ("zero")
 * the feature take, and the polynomial weights of each subset size: O(L D²) for L leaves and depth D, against the
 * O(L 2^d) of enumerating coalitions.
 *
 * Trees are `aifn-compute/graph` binary trees whose nodes carry the split (`feature`, −1 at a leaf; `threshold`, the left child
 * taking x[feature] ≤ threshold), the training `weight` reaching the node (its cover) and the node's `value` vector
 * (the explained output is `value[output]`), as the decision trees of `aifn-methods` are.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense } from 'aifn-compute/foundation/tensor'

/** The node data TreeSHAP reads. */
export type ShapNode = {
  feature: number
  threshold: number
  weight: number
  value: readonly number[]
}

/** A binary tree TreeSHAP can explain: `nodes[i].id === i`, `children` = [left, right] at a split, [] at a leaf. */
export type ShapTree = {
  readonly nodes: readonly (ShapNode & { readonly children: readonly number[] })[]
  readonly root: number
}

type PathEntry = { d: number; z: number; o: number; w: number }

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

/** The path with entry i removed, undoing its `extend` (0-based i). */
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

/** The sum of the path weights after unwinding entry i (the Shapley weight of the leaf for that feature). */
function unwoundSum(m: PathEntry[], i: number): number {
  return unwind(m, i).reduce((a, e) => a + e.w, 0)
}

/**
 * TreeSHAP values of one tree at x [d] for output `output` (default 0). Returns the values [d], the base value (the
 * cover-weighted mean leaf output, E[f]) and the tree's output at x; base + Σ values = output.
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
 * condition): `condition` 1 keeps `feature` always known (follow x's branch, never on the path), −1 always unknown
 * (both branches by cover), 0 none. The values of the other features are then Shapley values of the game on d − 1
 * players with that feature fixed in or out.
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
 * Exact SHAP interaction values of a tree at x [d] (Lundberg, Erion and Lee, 2018, §3; Lundberg et al., 2020): for
 * i ≠ j, Φᵢⱼ = (φᵢ with j always known − φᵢ with j always unknown)/2, each from a conditioned TreeSHAP pass, and
 * Φᵢᵢ = φᵢ − Σ_{j≠i} Φᵢⱼ. O(d L D²) against enumeration's O(L 2^d). Returns the matrix [d, d] row-major and the
 * Shapley values; as the shap package's `shap_interaction_values`.
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

/** SHAP interaction values of an additive ensemble: the scaled sum of each tree's (as `ensembleTreeShap`). */
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

/** The cover-weighted mean of the leaf outputs, E[f] under the training distribution the covers record. */
export function expectedValue(tree: ShapTree, output: Size = 0): number {
  const visit = (j: number): number => {
    const node = tree.nodes[j]
    if (node.children.length === 0) return node.value[output]
    return node.children.reduce((a, c) => a + (tree.nodes[c].weight / node.weight) * visit(c), 0)
  }
  return visit(tree.root)
}

function predictTree(tree: ShapTree, x: ArrayLike<number>, output: Size): number {
  let j = tree.root
  for (;;) {
    const node = tree.nodes[j]
    if (node.children.length === 0) return node.value[output]
    j = x[node.feature] <= node.threshold ? node.children[0] : node.children[1]
  }
}

/**
 * The path-dependent value function of a tree at x: E[f(x) | x_S] with features outside the coalition averaged over
 * both branches by cover. Exponential when enumerated over coalitions; TreeSHAP computes its Shapley values directly.
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
 * TreeSHAP for an additive ensemble f = offset + Σ_t scale_t f_t: the values are the scaled sum of each tree's (Shapley
 * values are linear in the game). `scales` defaults to 1/T (a forest's average); a boosted ensemble passes its
 * learning rate for every tree and its initial prediction as `offset`.
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
 * The output of an additive tree ensemble, offset + Σ_t scale_t f_t(x), on each row of X [m, d] (scales default 1/T,
 * a forest's average): the function `ensembleTreeShap` explains.
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
