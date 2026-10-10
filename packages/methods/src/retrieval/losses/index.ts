/**
 * `aifn-methods/retrieval/losses`: ranking and retrieval losses (pointwise, pairwise and listwise; sampled softmax,
 * negative sampling, NCE, in-batch softmax, triplet, contrastive), collected in `retrievalLossRegistry`.
 *
 * - Pointwise, each item on its own: `pointwiseBce` (relevant or not, scores as logits) and `pointwiseSquaredError`
 *   (regression on the grades).
 * - Pairwise, each pair with a higher grade first: `rankNet` (logistic), `pairwiseHinge` (RankSVM), `lambdaRank`
 *   (RankNet weighted by `lambdaWeights`, the NDCG change of each swap), and, for one positive against sampled
 *   negatives, `bpr` and `warp` (hinge weighted by the positive's estimated rank through `warpWeights` and
 *   `warpRankWeight`).
 * - Listwise, the whole list at once: `listwiseSoftmax` (normalised grades as target), `listNet` (top-one
 *   probabilities), `listMle` (Plackett–Luce likelihood of the ideal order) and `approxNdcg` (a smooth NDCG).
 * - Large output spaces: `sampledSoftmax` (with the logQ correction), `negativeSampling` and
 *   `noiseContrastiveEstimation` (which recovers normalised log-probabilities), and `inBatchSoftmax` for two-tower
 *   retrieval.
 * - Metric learning on embeddings: `triplet` and `contrastive`.
 *
 * Every loss is a `defineLoss` loss of `aifn-compute/learning/losses`: differentiable in its scores or embeddings,
 * with grades, weights and sampling probabilities held constant, and reduced over lists or examples by `reduction`
 * (mean by default). Ranking losses take one list of $n$ items (`[n]`) or $B$ of them (`[B, n]`).
 */

export {
  approxNdcg,
  bpr,
  lambdaRank,
  lambdaWeights,
  listMle,
  listNet,
  listwiseSoftmax,
  pairwiseHinge,
  pointwiseBce,
  pointwiseSquaredError,
  rankNet,
  warp,
  warpRankWeight,
  warpWeights,
  type ApproxNdcgOptions,
  type Gain,
  type LambdaOptions,
  type PairwiseHingeOptions,
  type RankNetOptions,
  type WarpOptions,
} from './ranking'
export {
  contrastive,
  inBatchSoftmax,
  negativeSampling,
  noiseContrastiveEstimation,
  sampledSoftmax,
  triplet,
  type MarginOptions,
  type NceOptions,
  type SampledSoftmaxOptions,
} from './retrieval'
export { retrievalLossRegistry } from './registry'
