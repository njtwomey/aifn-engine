/**
 * `aifn-compute/graph/matrices`: the matrices of a graph as dense float64 Tensors.
 *
 * - Adjacency and degree: `adjacencyMatrix` ($\Amat$, $V \times V$, parallel edges summed), `degrees` (its row sums,
 *   out-degrees when directed) and `degreeMatrix` ($\Dmat = \diag(\dvec)$).
 * - `laplacian`: $\Lmat = \Dmat - \Amat$, or its symmetric ($\Imat - \Dmat^{-1/2}\Amat\Dmat^{-1/2}$) or random-walk
 *   ($\Imat - \Dmat^{-1}\Amat$) normalisation, of a graph or of an affinity matrix $\Wmat$, for spectral clustering,
 *   graph-based semi-supervised learning and graph convolutions; `rescaledLaplacian`
 *   ($\tilde\Lmat = 2\Lmat/\lambda_{\max} - \Imat$, spectrum in $[-1, 1]$) for Chebyshev filters.
 * - `incidenceMatrix`: $\Bmat$, $V \times E$, oriented or not; $\Bmat\Bmat^\top = \Lmat$ for the oriented, unweighted
 *   matrix of an undirected graph.
 * - `hypergraphIncidence`: $\Hmat$, $n \times m$, one column per hyperedge.
 * - Node encodings for graph networks: `randomWalkEncoding` (return probabilities $(\Pmat^k)_{vv}$) and
 *   `laplacianEncoding` (eigenvectors of the normalised Laplacian), with `flipSigns` for sign augmentation.
 * - `matricesFunctions`: the registry entries of these functions.
 *
 * Edge weights are used unless `weighted: false` (`incidenceMatrix` is unweighted unless `weighted: true`). The
 * matrices are plain data, built outside autodiff.
 */

export {
  adjacencyMatrix,
  degreeMatrix,
  degrees,
  hypergraphIncidence,
  incidenceMatrix,
  laplacian,
  rescaledLaplacian,
  type RescaleOptions,
  type WeightOptions,
} from './matrices'
export { flipSigns, laplacianEncoding, randomWalkEncoding, type LaplacianEncoding } from './encodings'
export { matricesFunctions } from './registry'
