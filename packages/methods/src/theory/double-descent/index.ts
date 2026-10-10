/**
 * `aifn-methods/theory/double-descent`: random-features regression across the interpolation threshold, with
 * minimum-norm or ridge least squares (`doubleDescent`, `randomFeatures`, `featureCounts`).
 *
 * - The study: `doubleDescent` fits $p$ random features to $n$ noisy points for each $p$ and reports the test and
 *   training errors and the weight norm (`DoubleDescentOptions`, `DoubleDescentResult`); the test error peaks at
 *   $p = n$ unless a ridge penalty is set. `featureCounts` gives its default counts, dense near $n$.
 * - Features: `randomFeatureMap` draws a `FeatureMap` (ReLU or Fourier, `RandomFeatureKind`), and `randomFeatures`
 *   applies it to points.
 *
 * The fits are the least squares of the shared layer of `aifn-methods/theory`. `doubleDescentFunctions` registers the
 * functions.
 */

export {
  doubleDescent,
  featureCounts,
  randomFeatureMap,
  randomFeatures,
  type DoubleDescentOptions,
  type DoubleDescentResult,
  type FeatureMap,
  type RandomFeatureKind,
} from './double-descent'
export { doubleDescentFunctions } from './registry'
