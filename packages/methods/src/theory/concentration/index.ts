/**
 * `aifn-methods/theory/concentration`: concentration bounds (Chebyshev, Hoeffding, Bernstein, Chernoff, McDiarmid)
 * against Monte Carlo tails, and the law of large numbers and central limit theorem by simulation.
 *
 * - Bounds: `tailBounds` gives the four bounds on $P(\lvert \bar{X}_n - \mu \rvert \ge t)$ for variables in
 *   $[0, 1]$, the Chernoff one through `bernoulliKl`.
 * - Studies: `concentrationStudy` sets them against simulated tails of a bounded law (`ConcentrationOptions`,
 *   `ConcentrationStudy`), and `mcdiarmidStudy` does the same for McDiarmid's bound on the fraction of empty bins
 *   (`McdiarmidOptions`).
 * - Limit theorems: `runningMeans` for the law of large numbers, `standardisedSums` for the central limit theorem, and
 *   `sampleMeans` beneath both.
 * - Laws: `SummandLaw` names them, and `lawMoments` gives their mean and variance; the Pareto law has infinite
 *   variance, where the central limit theorem fails.
 *
 * Simulations draw each trial from a child stream, so the stream given is not advanced. `concentrationFunctions`
 * registers the functions.
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
