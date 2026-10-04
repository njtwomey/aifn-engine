/**
 * `aifn-methods/theory`: simulators that demonstrate results of learning theory and probability. Shared layer:
 * one-dimensional regression problems and least-squares fits (`regression.ts`). Modules: `bias-variance` (the
 * decomposition by resampling), `double-descent` (random-features regression across the interpolation threshold),
 * `capacity` (shattering and Rademacher complexity) and `concentration` (tail bounds against simulation, the law of
 * large numbers and the central limit theorem).
 */

export {
  applyWeights,
  drawTrainingSet,
  leastSquaresWeights,
  targetFunction,
  unitGrid,
  type TargetName,
} from './regression'
