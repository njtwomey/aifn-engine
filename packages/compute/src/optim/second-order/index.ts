/**
 * `aifn-compute/optim/second-order`: second-order and quasi-Newton methods: Newton (damped, trust region), BFGS, L-BFGS, OWL-QN
 * (L-BFGS for f + C‖x‖₁),
 * Gauss–Newton and Levenberg–Marquardt for least squares.
 */

export {
  newton,
  trustRegion,
  type DoglegKind,
  type NewtonOptions,
  type NewtonState,
  type TrustRegionOptions,
  type TrustRegionState,
} from './newton'
export {
  bfgs,
  lbfgs,
  owlqn,
  pseudoGradient,
  type BfgsState,
  type CurvaturePair,
  type LbfgsOptions,
  type LbfgsState,
  type OwlqnOptions,
  type OwlqnState,
  type QuasiNewtonOptions,
} from './quasiNewton'
export {
  gaussNewton,
  leastSquares,
  levenbergMarquardt,
  type GaussNewtonOptions,
  type GaussNewtonState,
  type LeastSquaresResult,
  type LeastSquaresState,
  type LevenbergMarquardtOptions,
  type LevenbergMarquardtState,
  type ResidualFunction,
} from './leastSquares'
export { secondOrderAlgorithms } from './registry'
