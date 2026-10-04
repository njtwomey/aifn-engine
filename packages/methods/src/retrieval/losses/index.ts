/**
 * `aifn-methods/retrieval/losses`: ranking and retrieval losses (pointwise, pairwise and listwise; sampled softmax,
 * negative sampling, NCE, in-batch softmax, triplet, contrastive), collected in `retrievalLossRegistry`.
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
