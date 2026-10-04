/**
 * `aifn-compute/foundation/pytree`: nested arrays and plain objects of numbers, tensors and traced values (pytrees), defined
 * once (design K §4.5). `treeFlatten`/`treeUnflatten` with a plain-data `TreeDef`; `treeMap`, `treeZip`, `treeLeaves`,
 * `zerosLike`, `countParams`; `ravel` for optimisers that work on one flat vector. The autodiff transforms, the layers
 * of `aifn-compute/nn`, the optimisers of `aifn-compute/optim` and the kernels of `aifn-compute/learning/kernels` all walk trees with these.
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
