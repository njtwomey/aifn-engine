/**
 * `aifn-compute/signal/similarity`: similarity search in time series, as stumpy, the UCR suite and SAX.
 *
 * - Distance profiles, the distance from a query to every window of a series: `distanceProfile` (MASS, by one FFT),
 *   and its parts `slidingDotProduct`, `slidingMeanStd`, `zDistance` and `zNormalise`.
 * - Matrix profiles, every window's nearest neighbour: `matrixProfile` (STOMP, exact) or `scrimpSteps` (SCRIMP++, an
 *   anytime algorithm run step by step, read with `scrimpProfile`); then `motifs` (the closest pairs) and `discords`
 *   (the most unusual windows).
 * - Warping: `dtw` (Sakoe–Chiba band, accumulated cost and path) and `dtwProgram` (the same as a stepped dynamic
 *   program), with the lower bounds `lbKim` and `lbKeogh` (on the envelope of `keoghEnvelope`) that prune a search.
 * - Symbolic: `paa` (frame means), `sax` (words over an alphabet cut at `saxBreakpoints`), `saxMinDist` (a lower
 *   bound on the distance between the series from their words).
 *
 * Series are single-channel signals or their samples. Distances between windows are z-normalised unless an option
 * says otherwise, so they compare shapes, not levels or scales; a constant window follows stumpy's convention. Window
 * lengths and counts that do not fit the series throw `DomainError`.
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
