/**
 * `aifn-methods/theory/double-descent`: random-features regression across the interpolation threshold, with
 * minimum-norm or ridge least squares (`doubleDescent`, `randomFeatures`, `featureCounts`).
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
