/**
 * `aifn-compute/foundation/pytree`: nested arrays and plain objects of numbers, tensors and traced values (pytrees),
 * defined once (design K §4.5).
 *
 * - Structure: `treeFlatten` splits a tree into its leaves, their paths and a plain-data `TreeDef`; `treeUnflatten`
 *   rebuilds it from new leaves; `leafCount` counts a structure's leaves.
 * - Leaf-wise maps: `treeMap` over one tree, `treeZip` over several of the same structure, `treeLeaves` to read the
 *   leaves by path, `zerosLike` for a tree of zeros.
 * - Flat views: `countParams` counts the scalar entries, and `ravel` reads a tree as one vector, with its inverse, for
 *   optimisers that work on one flat vector.
 *
 * Every function visits leaves in one order (depth first, arrays by index, object keys in insertion order), and
 * values that are neither leaves nor nodes are static: carried through unchanged. The autodiff transforms, the layers
 * of `aifn-compute/nn`, the optimisers of `aifn-compute/optim` and the kernels of `aifn-compute/learning/kernels` all
 * walk trees with these.
 */

export {
  countParams,
  treeFlatten,
  leafCount,
  ravel,
  treeLeaves,
  treeMap,
  treeZip,
  treeUnflatten,
  zerosLike,
  type Flat,
  type Leaf,
  type LeafValue,
  type Params,
  type Raveled,
  type Tree,
  type TreeDef,
} from './pytree'
