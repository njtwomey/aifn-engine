/**
 * `aifn-compute/inference/expectation-propagation`: expectation propagation, assumed density filtering and the
 * message algebra they share.
 *
 * - Gaussian messages in natural parameters (precision $\tau$, shift $\nu$; numbers or batches of tensors):
 *   `naturalGaussian`, `gaussianMoments`, `multiplyGaussians`, `divideGaussians` (the cavity), `powerGaussian`,
 *   `dampGaussian`, `UNIFORM_GAUSSIAN`, and to and from a Normal, `gaussianToNormal` and `normalToGaussian`.
 *   Multivariate ($\Lambdamat$, $\etavec$): `naturalMvGaussian`, `mvGaussianMoments`, `multiplyMvGaussians`,
 *   `divideMvGaussians`.
 * - Any exponential family as an `ExpFamilyMessage`: `messageOf`, `multiplyMessages`, `divideMessages`,
 *   `dampMessages`, `powerMessage`, and back to a distribution with `messageToDistribution`.
 * - Tilted moments, a Gaussian cavity times one factor: in closed form `stepTilted`, `probitTilted` and
 *   `intervalTilted` (elementwise over tensors, through `lift`), and for any factor `tiltedByQuadrature`.
 * - A scalar parameter: `expectationPropagation` (EP and power EP, stepped site by site, with damping),
 *   `assumedDensityFiltering` (one pass) and `epLogEvidence`. The factors enter as a `TiltedFn`.
 * - A vector parameter whose factors each see one projection $\avec_i^\top \thetavec$:
 *   `multivariateExpectationPropagation` (rank-one sites; GP classification, probit regression, paired comparisons),
 *   which reports the log evidence as it goes.
 * - A linear-Gaussian model of the model language with interval and Gaussian evidence (TrueSkill):
 *   `compileGaussianModel` and `modelExpectationPropagation`.
 *
 * Every EP run skips and counts an update whose cavity is improper rather than hiding it, and converges once a sweep
 * moves no site by more than its tolerance. The algebra is differentiable; the tilted moments and the EP loops are
 * not.
 */

export {
  dampGaussian,
  dampMessages,
  divideGaussians,
  divideMessages,
  divideMvGaussians,
  gaussianMoments,
  gaussianToNormal,
  messageOf,
  messageToDistribution,
  multiplyGaussians,
  multiplyMessages,
  multiplyMvGaussians,
  mvGaussianMoments,
  naturalGaussian,
  naturalMvGaussian,
  normalToGaussian,
  powerGaussian,
  powerMessage,
  UNIFORM_GAUSSIAN,
  type ExpFamilyMessage,
  type GaussianMoments,
  type NaturalGaussian,
  type NaturalMvGaussian,
} from './gaussian'
export {
  intervalTilted,
  lift,
  probitTilted,
  stepTilted,
  tiltedByQuadrature,
  type Out,
  type ProbitOptions,
  type QuadratureTiltOptions,
  type Tilted,
} from './tilted'
export {
  assumedDensityFiltering,
  epLogEvidence,
  expectationPropagation,
  type AdfOptions,
  type AdfState,
  type EpOptions,
  type EpState,
  type TiltedFn,
} from './ep'
export {
  compileGaussianModel,
  modelExpectationPropagation,
  type CompiledGaussianModel,
  type ModelEpOptions,
  type ModelEpState,
} from './model'
export { multivariateExpectationPropagation, type MvEpOptions, type MvEpState } from './multivariate'
export { expectationPropagationAlgorithms, expectationPropagationFunctions } from './registry'
