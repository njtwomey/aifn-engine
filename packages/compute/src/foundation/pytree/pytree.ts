/**
 * Pytrees (design K §4.5): nested arrays and plain objects whose leaves are numbers, tensors or traced values. The
 * autodiff transforms treeFlatten their arguments into leaves, differentiate with respect to the leaves and rebuild results
 * of the same structure; layers return their parameters as trees; optimisers update them leaf by leaf; L-BFGS and
 * Nelder–Mead read them as one flat vector (`ravel`). This is the one implementation (JAX's `jax.tree_util` at teaching
 * scale).
 *
 * - A **leaf** is a number, a branded tensor or a traced value. Arrays and plain objects (prototype `Object` or null)
 *   are **nodes**. Anything else (strings, booleans, null, undefined, functions, class instances) is **static**:
 *   carried through unchanged, never differentiated.
 * - Leaves are visited depth first, arrays in index order and object keys in insertion order. Every function here uses
 *   that order, so `treeFlatten(t).leaves`, `treeLeaves(t)` and `ravel(t).vector` line up.
 * - A `TreeDef` is plain data (structured-cloneable), so a structure can cross to a worker and be rebuilt there.
 */

import {
  avalOf,
  fromData,
  isTensor,
  isTraced,
  reshape,
  toFlat,
  unwrap,
  zerosOf,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { Raw, Scalar, Shape, Size, Value } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A nested structure of numbers, tensors and traced values (arrays and plain objects). */
export type Tree = Value | readonly Tree[] | { readonly [key: string]: Tree }

/**
 * A tree of parameters: tensors and numbers in nested objects and arrays. `undefined` marks an absent optional
 * parameter (a layer without bias); it is static, not a leaf.
 */
export type Params = Tensor | Scalar | undefined | readonly Params[] | { readonly [key: string]: Params }

/** A leaf value: a number or a tensor (or, inside a transform, a traced value). */
export type LeafValue = Raw

/** A leaf with its path, e.g. `layers[0].weight`. `V` is the leaf type the caller expects (raw by default). */
export type Leaf<V extends Value = LeafValue> = { path: string; value: V }

/**
 * The structure of a tree without its leaves: plain data. `leaf` marks a leaf position; `array` and `object` nodes
 * list their children (objects with their keys, in order); `static` holds a non-leaf, non-node value as it was.
 */
export type TreeDef =
  | { readonly kind: 'leaf' }
  | { readonly kind: 'array'; readonly children: readonly TreeDef[] }
  | { readonly kind: 'object'; readonly keys: readonly string[]; readonly children: readonly TreeDef[] }
  | { readonly kind: 'static'; readonly value: unknown }

/** A flattened tree: its leaves in order, a readable path per leaf, and the structure to rebuild it. */
export type Flat<V extends Value = Value> = {
  readonly leaves: V[]
  /** A readable path per leaf: `x`, `w[1]`, `layer.bias` (prefixed by the root name given to `treeFlatten`). */
  readonly paths: string[]
  readonly treedef: TreeDef
}

/** True for a pytree leaf: a number, a tensor or a traced value. */
function isLeaf(x: unknown): x is Value {
  return typeof x === 'number' || isTensor(x) || isTraced(x)
}

function isPlainObject(x: unknown): x is Record<string, unknown> {
  if (typeof x !== 'object' || x === null) return false
  const proto = Object.getPrototypeOf(x) as unknown
  return proto === Object.prototype || proto === null
}

const childPath = (path: string, key: string | number): string =>
  typeof key === 'number' ? `${path}[${key}]` : path ? `${path}.${key}` : key

/**
 * Flatten a tree into its leaves (depth first), their paths and its structure. `root` names the whole tree in the
 * paths (e.g. `x` gives `x[0]`, `x.w`).
 *
 * @example treeFlatten({ w: tensor([1, 2]), b: 0 }) // leaves [w, 0], paths ['w', 'b']
 */
export function treeFlatten<V extends Value = Value>(tree: unknown, root = ''): Flat<V> {
  const leaves: V[] = []
  const paths: string[] = []
  const walk = (node: unknown, path: string): TreeDef => {
    if (isLeaf(node)) {
      leaves.push(node as V)
      paths.push(path)
      return { kind: 'leaf' }
    }
    if (Array.isArray(node)) return { kind: 'array', children: node.map((c, i) => walk(c, childPath(path, i))) }
    if (isPlainObject(node)) {
      const keys = Object.keys(node)
      return { kind: 'object', keys, children: keys.map((k) => walk(node[k], childPath(path, k))) }
    }
    return { kind: 'static', value: node }
  }
  const treedef = walk(tree, root)
  return { leaves, paths, treedef }
}

/** The number of leaves a structure holds. */
export function leafCount(treedef: TreeDef): Size {
  if (treedef.kind === 'leaf') return 1
  if (treedef.kind === 'static') return 0
  return treedef.children.reduce((n, c) => n + leafCount(c), 0)
}

/**
 * Rebuild a tree of structure `treedef` from leaves in flattening order (the inverse of `treeFlatten`). The leaves may be
 * of any kind (the transforms rebuild gradients, tangents and traced inputs this way).
 */
export function treeUnflatten<T = unknown>(treedef: TreeDef, leaves: readonly unknown[]): T {
  const expected = leafCount(treedef)
  if (leaves.length !== expected)
    throw new ShapeError('treeUnflatten', `treeUnflatten: the structure has ${expected} leaves, got ${leaves.length}`)
  let k = 0
  const build = (def: TreeDef): unknown => {
    switch (def.kind) {
      case 'leaf':
        return leaves[k++]
      case 'static':
        return def.value
      case 'array':
        return def.children.map(build)
      case 'object': {
        const out: Record<string, unknown> = {}
        def.children.forEach((c, i) => (out[def.keys[i]] = build(c)))
        return out
      }
    }
  }
  return build(treedef) as T
}

/** The leaves of a tree with their paths, depth first. `V` is the leaf type the caller expects. */
export function treeLeaves<V extends Value = LeafValue>(tree: unknown, root = ''): Leaf<V>[] {
  const { leaves, paths } = treeFlatten<V>(tree, root)
  return leaves.map((value, k) => ({ path: paths[k], value }))
}

/**
 * Map the leaves of several trees of the same structure together: `treeZip([a, b], ([x, y], path) => …)` calls f on
 * each tuple of corresponding leaves and rebuilds the structure of the first tree. Static values come from the first
 * tree. The trees must have the same structure (the other trees' leaves are read at the first tree's leaf paths).
 */
export function treeZip<T, V extends Value = LeafValue>(
  trees: readonly [T, ...unknown[]],
  f: (leaves: V[], path: string) => Value,
): T {
  const [first, ...rest] = trees
  const main = treeFlatten<V>(first)
  const others = rest.map((t) => treeFlatten<V>(t))
  others.forEach((o, i) => {
    if (o.leaves.length !== main.leaves.length)
      throw new ShapeError(
        'treeZip',
        `treeZip: tree ${i + 2} has ${o.leaves.length} leaves, the first has ${main.leaves.length}`,
      )
    // Equal counts are not enough: the leaves must sit at the same paths (a.x with b.x, not with b.y).
    const k = o.paths.findIndex((p, j) => p !== main.paths[j])
    if (k >= 0)
      throw new ShapeError(
        'treeZip',
        `treeZip: tree ${i + 2} has a leaf at '${o.paths[k]}' where the first has '${main.paths[k]}'`,
      )
  })
  const mapped = main.leaves.map((leaf, k) => f([leaf, ...others.map((o) => o.leaves[k])], main.paths[k]))
  return treeUnflatten<T>(main.treedef, mapped)
}

/** Map the leaves of one tree, keeping its structure. */
export function treeMap<T, V extends Value = LeafValue>(tree: T, f: (leaf: V, path: string) => Value): T {
  return treeZip<T, V>([tree], ([leaf], path) => f(leaf, path))
}

/**
 * A tree of zeros with the structure of `tree`: 0 for a number leaf, `zeros(shape)` for a tensor (or traced) leaf
 * (complex128 for a complex leaf, float64 otherwise).
 */
export function zerosLike<T>(tree: T): T {
  return treeMap<T, Value>(tree, (leaf) => zerosOf(avalOf(leaf)))
}

/** The number of scalar entries in a tree's leaves (a number counts 1, a tensor its size). */
export function countParams(tree: unknown): Size {
  return treeFlatten(tree).leaves.reduce<number>((n, leaf) => n + avalOf(leaf).shape.reduce((a, b) => a * b, 1), 0)
}

/** A tree read as one flat vector, and the map back (JAX's `ravel_pytree`). */
export type Raveled<T> = {
  /** Every leaf's entries, row-major, concatenated in leaf order. */
  readonly vector: Float64Array
  /** Rebuild a tree of the same structure (numbers stay numbers, tensors keep their shapes) from a vector. */
  unravel(vector: ArrayLike<Scalar> | Tensor): T
}

/**
 * Ravel a tree of raw leaves into one Float64Array (for L-BFGS, Nelder–Mead and other vector optimisers), with its
 * inverse. Traced leaves are read through their values; the vector is not differentiable. A complex128 leaf takes two
 * entries per element, (re, im) interleaved (the ℝ² view), and is rebuilt complex.
 */
export function ravel<T>(tree: T): Raveled<T> {
  const { leaves, treedef } = treeFlatten(tree)
  const shapes: (Shape | null)[] = leaves.map((leaf) => {
    const v = unwrap(leaf)
    return typeof v === 'number' ? null : v.shape
  })
  const complex = leaves.map((leaf) => avalOf(leaf).dtype === 'complex128')
  const sizes = shapes.map((s, k) => (s === null ? 1 : s.reduce((a, b) => a * b, 1) * (complex[k] ? 2 : 1)))
  const total = sizes.reduce((a, b) => a + b, 0)
  const vector = new Float64Array(total)
  let at = 0
  leaves.forEach((leaf, k) => {
    const v = unwrap(leaf)
    if (typeof v === 'number') vector[at] = v
    else vector.set(toFlat(v), at)
    at += sizes[k]
  })
  const unravel = (input: ArrayLike<Scalar> | Tensor): T => {
    const values = isTensor(input) ? toFlat(input) : input
    if (values.length !== total)
      throw new ShapeError('ravel', `ravel: the tree has ${total} entries, the vector ${values.length}`)
    let offset = 0
    const rebuilt = shapes.map((shape, k) => {
      const n = sizes[k]
      const part = Float64Array.from({ length: n }, (_, i) => values[offset + i])
      offset += n
      if (shape === null) return part[0]
      return complex[k] ? fromData(part, shape, 'complex128') : reshape(fromData(part), shape)
    })
    return treeUnflatten<T>(treedef, rebuilt)
  }
  return { vector, unravel }
}
