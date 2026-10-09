/**
 * `aifn-compute/graph/matrices`: the matrices of a graph as dense float64 Tensors.
 *
 * - Adjacency and degree: `adjacencyMatrix` ($\Amat$, $V \times V$, parallel edges summed), `degrees` (its row sums,
 *   out-degrees when directed) and `degreeMatrix` ($\Dmat = \diag(\dvec)$).
 * - `laplacian`: $\Lmat = \Dmat - \Amat$, or its symmetric ($\Imat - \Dmat^{-1/2}\Amat\Dmat^{-1/2}$) or random-walk
 *   ($\Imat - \Dmat^{-1}\Amat$) normalisation, for spectral clustering and graph convolutions.
 * - `incidenceMatrix`: $\Bmat$, $V \times E$, oriented or not; $\Bmat\Bmat^\top = \Lmat$ for the oriented, unweighted
 *   matrix of an undirected graph.
 * - `matricesFunctions`: the registry entries of these functions.
 *
 * Edge weights are used unless `weighted: false` (`incidenceMatrix` is unweighted unless `weighted: true`). The
 * matrices are plain data, built outside autodiff.
 */

export { adjacencyMatrix, degreeMatrix, degrees, incidenceMatrix, laplacian, type WeightOptions } from './matrices'
export { matricesFunctions } from './registry'
