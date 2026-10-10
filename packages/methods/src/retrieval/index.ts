/**
 * `aifn-methods/retrieval`: recommendation and retrieval: ranking and retrieval losses, recommender models, and a
 * benchmark of approximate nearest-neighbour search.
 *
 * Child modules:
 *
 * - `aifn-methods/retrieval/losses`: what to train a ranker or retriever with. Pointwise, pairwise and listwise
 *   ranking losses (RankNet, LambdaRank, BPR, WARP, ListNet, ListMLE, ApproxNDCG, ...), sampled softmax, negative
 *   sampling, NCE and in-batch softmax for large catalogues, and the triplet and contrastive losses on embeddings,
 *   collected in `retrievalLossRegistry`.
 * - `aifn-methods/retrieval/recommenders`: what recommends. Popularity and neighbourhood baselines, matrix
 *   factorisation (ALS, SGD, implicit ALS as in the `implicit` library), gradient-trained models from logistic MF to
 *   two-tower and SASRec, Matchbox, held-out ranking evaluation, a streamed training run and a feedback-loop
 *   simulator; registered in `recommenderAlgorithms` and `recommenderFunctions`.
 * - `aifn-methods/retrieval/ann`: how fast candidates can be retrieved. A recall-against-speed benchmark of the
 *   k-d tree, LSH, IVF, PQ and HNSW indexes of `aifn-compute/numerics/neighbours`, in the manner of ann-benchmarks
 *   (the index families of FAISS).
 *
 * This level re-exports the registries `retrievalLossRegistry`, `recommenderAlgorithms` and `recommenderFunctions`.
 */

export { retrievalLossRegistry } from './losses'
export { recommenderAlgorithms, recommenderFunctions } from './recommenders'
