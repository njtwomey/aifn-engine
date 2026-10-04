/**
 * `aifn-methods/learning/kernel-methods`: kernel methods: support vector machines (SMO, Pegasos, dual coordinate
 * descent, linear SVM) and the Crammer–Singer multiclass SVM. Platt scaling of their scores into probabilities is in
 * `aifn-compute/learning/calibration`.
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
