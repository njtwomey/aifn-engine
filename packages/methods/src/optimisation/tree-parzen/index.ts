/**
 * `aifn-methods/optimisation/tree-parzen`: the tree-structured Parzen estimator (Bergstra et al., 2011), with the
 * defaults of Watanabe's (2023) tutorial, as a step-through algorithm over a `Space`.
 *
 * - The search: `treeParzenSteps` minimises an objective over real (linear or log scale), integer, choice and boolean
 *   dimensions; each step evaluates one point and proposes the next as the candidate from $\ell$ with the largest
 *   $\ell / g$ (`TreeParzenOptions`).
 * - What a figure draws: each state holds the observations and the split $y^\star$, the good and bad densities
 *   $\ell(\lambda)$ and $g(\lambda)$ and their ratio on a grid per dimension (and jointly for two continuous
 *   dimensions), the candidates with their scores, and the chosen point (`TreeParzenState`, `TreeParzenMarginal`).
 *
 * Densities are on the unit-cube coordinates of the space, so a log-scale dimension is modelled in $\log \lambda$.
 * Randomness comes only from the runner's streams. `treeParzenAlgorithms` registers the algorithm.
 */

export { treeParzenSteps, type TreeParzenMarginal, type TreeParzenOptions, type TreeParzenState } from './tpe'
export { treeParzenAlgorithms } from './registry'
