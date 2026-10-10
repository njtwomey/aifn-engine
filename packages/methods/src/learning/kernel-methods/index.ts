/**
 * `aifn-methods/learning/kernel-methods`: support vector machines, for two classes with any kernel or linear, and
 * the Crammer–Singer multiclass SVM.
 *
 * - The kernel SVM: `supportVectorMachine` (as scikit-learn's `SVC`, with optional Platt scaling), and its solver
 *   `smoSteps`, sequential minimal optimisation on the dual in LIBSVM's form; `dualDecision` turns any dual variables
 *   into the decision function $f(\xvec) = \sum_t \alpha_t y_t k(\xvec_t, \xvec) + b$.
 * - The linear SVM: `linearSvm` (as `LinearSVC(loss='hinge')`), by `dualCoordinateSteps` (exact at convergence) or
 *   `pegasosSteps` (stochastic subgradient steps, never declared converged).
 * - Many classes in one problem: `crammerSinger` and its solver `crammerSingerSteps`, LIBLINEAR's sequential dual
 *   method.
 * - The registry: `kernelMethodsAlgorithms` and `kernelMethodsFunctions`.
 *
 * The estimators take labels 0/1 (or $0, \dots, K - 1$ for Crammer–Singer); the step algorithms take the $\pm 1$ of
 * the textbook dual. Every solver is a traceable algorithm (`run`, `trace`), and each estimator keeps its run in
 * `training`. The linear solvers learn the bias as the weight of a constant feature, regularised with the rest, as
 * LIBLINEAR does. Platt scaling of the scores into probabilities is in `aifn-compute/learning/calibration`.
 */

export {
  dualCoordinateSteps,
  dualDecision,
  linearSvm,
  pegasosSteps,
  smoSteps,
  supportVectorMachine,
  type LinearSvmModel,
  type LinearSvmProblem,
  type LinearSvmState,
  type SmoProblem,
  type SmoState,
  type SupportVectorMachineModel,
} from './svm'
export {
  type CrammerSingerProblem,
  type CrammerSingerState,
  crammerSingerSteps,
  type CrammerSingerModel,
  crammerSinger,
} from './crammerSinger'
export { kernelMethodsAlgorithms, kernelMethodsFunctions } from './registry'
