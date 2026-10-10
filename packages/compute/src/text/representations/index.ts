/**
 * `aifn-compute/text/representations`: classical word representations, from term matrices to latent semantic analysis
 * and random indexing.
 *
 * - Term matrices: `termDocumentMatrix` (terms $\times$ documents, the transpose of the bag of words) and
 *   `termTermMatrix` (words $\times$ contexts in a window), each with its counts and names, and `weightMatrix` to
 *   reweight counts as presence, $\log(1 + \text{count})$, TF-IDF per column or PPMI.
 * - Latent semantic analysis: `lsa` gives term and column coordinates from a truncated SVD.
 * - Similarity: `cosineSimilarities` of all pairs of rows, `nearestByCosine` for the nearest rows to a row or vector,
 *   `analogy` by vector offset (3CosAdd), and `cosineMap`, a two-dimensional picture of the rows' cosine geometry.
 * - Random indexing: `indexVector` (a token's fixed sparse ternary vector) and `randomIndexing` (the sum of the index
 *   vectors of each word's contexts).
 *
 * Rows are terms and matrices are dense float64 tensors. The counts come from `aifn-compute/text/features` and
 * `aifn-compute/text/cooccurrence`; a zero vector has cosine 0 with everything.
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
