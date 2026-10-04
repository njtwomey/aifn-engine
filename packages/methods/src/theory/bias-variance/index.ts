/**
 * `aifn-methods/theory/bias-variance`: the bias–variance decomposition by resampling training sets: one model
 * (`biasVariance`), or a sweep over complexity (`biasVarianceSweep`); polynomial least squares and k-NN regression.
 */

export {
  biasVariance,
  biasVarianceSweep,
  fitAndPredict,
  type BiasVarianceOptions,
  type BiasVarianceResult,
  type ResampledModel,
} from './bias-variance'
export { biasVarianceFunctions } from './registry'
