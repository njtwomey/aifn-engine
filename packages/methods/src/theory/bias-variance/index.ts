/**
 * `aifn-methods/theory/bias-variance`: the bias–variance decomposition by resampling training sets: one model
 * (`biasVariance`), or a sweep over complexity (`biasVarianceSweep`); polynomial least squares and $k$-NN regression.
 *
 * - Studies: `biasVariance` gives $\text{bias}^2$, variance and noise at each $x$ and averaged
 *   (`BiasVarianceOptions`, `BiasVarianceResult`); `biasVarianceSweep` gives the averages and the training error for
 *   each polynomial degree or $k$.
 * - Models: `fitAndPredict` fits a `ResampledModel` (a polynomial, optionally ridge, or $k$-nearest neighbours) to one
 *   training set.
 *
 * Studies draw their training sets from child streams, so the same stream gives every model the same sets.
 * `biasVarianceFunctions` registers the functions.
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
