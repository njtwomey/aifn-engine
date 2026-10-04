/**
 * `aifn-compute/text/representations`: classical word representations. Term–document and term–term matrices with raw,
 * binary, log, TF-IDF and PPMI weighting; latent semantic analysis (term and document coordinates from a truncated
 * SVD); cosine similarities, nearest neighbours and analogies by vector offset; random indexing.
 */

export {
  termDocumentMatrix,
  termTermMatrix,
  weightMatrix,
  type MatrixWeighting,
  type TermDocumentOptions,
  type TermMatrix,
  type TermTermOptions,
  type WeightOptions,
} from './matrices'
export {
  analogy,
  cosineMap,
  cosineSimilarities,
  lsa,
  nearestByCosine,
  type AnalogyAnswer,
  type Lsa,
  type Neighbour,
} from './lsa'
export {
  indexVector,
  randomIndexing,
  type IndexVectorOptions,
  type RandomIndexing,
  type RandomIndexingOptions,
} from './random-indexing'
export { representationsFunctions } from './registry'
