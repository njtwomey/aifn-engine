/**
 * `aifn-compute/graph/matrices`: the matrices of a graph as Tensors: adjacency, degree, the Laplacian (unnormalised,
 * symmetric and random-walk normalised) and incidence.
 */

export { adjacencyMatrix, degreeMatrix, degrees, incidenceMatrix, laplacian, type WeightOptions } from './matrices'
export { matricesFunctions } from './registry'
