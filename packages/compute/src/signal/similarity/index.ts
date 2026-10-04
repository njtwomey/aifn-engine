/**
 * `aifn-compute/signal/similarity`: similarity search in time series, as stumpy, the UCR suite and SAX.
 *
 * - Distance profiles: `distanceProfile` (MASS), `slidingDotProduct`, `slidingMeanStd`, `zDistance`, `zNormalise`.
 * - Matrix profiles: `matrixProfile` (STOMP), `scrimpSteps` (SCRIMP++, anytime), `motifs`, `discords`.
 * - Warping: `dtw` (Sakoe–Chiba band, accumulated cost and path), `keoghEnvelope`, `lbKeogh`, `lbKim`.
 * - Symbolic: `paa`, `sax`, `saxBreakpoints`, `saxMinDist`.
 */

export {
  distanceProfile,
  slidingDotProduct,
  slidingMeanStd,
  zDistance,
  zNormalise,
  type DistanceProfileOptions,
} from './profile'
export {
  discords,
  matrixProfile,
  motifs,
  scrimpProfile,
  scrimpSteps,
  type Discord,
  type MatrixProfile,
  type MatrixProfileOptions,
  type Motif,
  type PickOptions,
  type ScrimpOptions,
  type ScrimpState,
} from './matrix-profile'
export { dtw, dtwProgram, keoghEnvelope, lbKeogh, lbKim, type DtwCost, type DtwOptions, type DtwResult } from './dtw'
export { paa, sax, saxBreakpoints, saxMinDist, type SaxOptions, type SaxWord } from './sax'
export { similarityAlgorithms, similarityFunctions } from './registry'
