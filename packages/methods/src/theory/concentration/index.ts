/**
 * `aifn-methods/theory/concentration`: concentration bounds (Chebyshev, Hoeffding, Bernstein, Chernoff, McDiarmid)
 * against Monte Carlo tails, and the law of large numbers and central limit theorem by simulation.
 */

export {
  bernoulliKl,
  concentrationStudy,
  lawMoments,
  mcdiarmidStudy,
  runningMeans,
  sampleMeans,
  standardisedSums,
  tailBounds,
  type ConcentrationOptions,
  type ConcentrationStudy,
  type McdiarmidOptions,
  type SummandLaw,
} from './concentration'
export { concentrationFunctions } from './registry'
