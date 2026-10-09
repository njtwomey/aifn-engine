/**
 * `aifn-compute/probability/information`: information measures of distributions, from probability vectors, joint
 * tables, distribution objects and samples, as scipy.stats.entropy and scikit-learn's `mutual_info_score`.
 *
 * - Entropies: `entropy` (per row of a batch), `jointEntropy`, `conditionalEntropy` and `crossEntropy` of probability
 *   vectors and tables; `differentialEntropy` of a distribution object.
 * - Divergences and distances between two distributions: `klDivergence`, `jensenShannonDivergence` and
 *   `jensenShannonDistance` (finite and symmetric), `totalVariation`, `hellingerDistance`, and `fDivergence` for any
 *   generator (`fGenerators` holds the standard ones, `FGenerator` describes one).
 * - Mutual information: `mutualInformation` and `pointwiseMutualInformation` (with its normalised form) of a joint
 *   table; `gaussianMutualInformation` of blocks of a Gaussian from its covariance; `ksgMutualInformation`, the
 *   Kraskov–Stögbauer–Grassberger $k$-nearest-neighbour estimate from samples.
 * - Helpers: `asValue` (arrays to tensors) and `flatProbabilities` (normalised plain numbers, for coding algorithms);
 *   `informationFunctions` lists the functions with their metadata.
 *
 * Inputs need not be normalised: counts are fine. Results are in nats unless `base` is given (2 for bits). The
 * measures of vectors and tables are compositions of primitives, so they are differentiable; `fDivergence`,
 * `flatProbabilities` and `ksgMutualInformation` work on plain numbers.
 */

export {
  asValue,
  conditionalEntropy,
  crossEntropy,
  differentialEntropy,
  entropy,
  fDivergence,
  flatProbabilities,
  fGenerators,
  hellingerDistance,
  jensenShannonDistance,
  jensenShannonDivergence,
  jointEntropy,
  klDivergence,
  mutualInformation,
  pointwiseMutualInformation,
  totalVariation,
  type BaseOption,
  type FGenerator,
  type Probabilities,
} from './measures'
export { gaussianMutualInformation, ksgMutualInformation } from './continuous'
export { informationFunctions } from './registry'
