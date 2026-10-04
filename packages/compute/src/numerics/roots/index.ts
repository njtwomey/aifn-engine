/**
 * `aifn-compute/numerics/roots`: roots of nonlinear equations, every iterative method a traceable `Algorithm`: bisection,
 * regula falsi, secant, Newton and Brent for scalar equations $f(x) = 0$; Newton and Broyden for systems $\mathbf{f}(\mathbf{x}) = \mathbf{0}$; fixed-point iteration;
 * continuation (Newton homotopy); the drivers `findRoot` and `solveSystem`; and `minimizeScalar`, Brent's and
 * golden-section minimisation of a function of one variable (the bracketing methods of root finding turned to minima).
 */

export {
  bisection,
  brent,
  newtonRoot,
  regulaFalsi,
  secant,
  type BracketOptions,
  type BracketState,
  type BrentState,
  type NewtonRootState,
  type RegulaFalsiState,
  type RootState,
  type RootTolerance,
  type ScalarFunction,
  type ScalarWithDerivative,
  type SecantState,
} from './scalar'
export {
  broyden,
  continuation,
  fixedPoint,
  newtonHomotopy,
  newtonSystem,
  type BroydenState,
  type ContinuationOptions,
  type ContinuationState,
  type FixedPointState,
  type Homotopy,
  type NewtonSystemState,
  type SystemFunction,
  type SystemState,
  type SystemTolerance,
  type SystemWithJacobian,
} from './systems'
export { findRoot, solveSystem, type RootResult, type SystemResult } from './convenience'
export { minimizeScalar, type MinimizeScalarOptions, type MinimizeScalarResult } from './minimize'
export { rootsAlgorithms, rootsFunctions } from './registry'
