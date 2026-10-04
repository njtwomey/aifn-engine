/**
 * `aifn-methods/inference/lattice-models`: the Ising model on any graph or on a lattice declared with
 * `aifn-compute/graph/structured`, with inference chosen by the graph's shape.
 */
export { isingInference, isingLattice, isingModel, isingShape, type IsingModel } from './ising'
export { latticeModelFunctions } from './registry'
