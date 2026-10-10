/**
 * `aifn-methods/inference/lattice-models`: the Ising model on any graph or on a lattice declared with
 * `aifn-compute/graph/structured`, with inference chosen by the graph's shape.
 *
 * - Building a model: `isingModel` puts spins on the nodes and couplings on the edges of any graph; `isingLattice` on
 *   a $\text{rows} \times \text{cols}$ grid or torus, keeping the lattice declaration as its `structure`.
 * - Inference: `isingShape` reads the shape (chain, tree, lattice or general), and `isingInference` picks exact
 *   forward-backward on a chain, tree belief propagation on a tree, and loopy belief propagation otherwise.
 * - Spins $x_i \in \{-1, +1\}$ are the values 0 and 1 of binary variables, and
 *   $p(\xvec) \propto \exp\paren{\sum_{(i, j)} J_{ij} x_i x_j + \sum_i h_i x_i}$.
 * - `latticeModelFunctions` registers the functions with the notes they serve.
 */
export { isingInference, isingLattice, isingModel, isingShape, type IsingModel } from './ising'
export { latticeModelFunctions } from './registry'
