/**
 * `aifn-compute/signal/decompositions`: empirical mode decomposition (`emd`, `eemd`, `ceemdan`) and variational mode
 * decomposition (`vmd`) as `Decomposition`s, with sifting and VMD's ADMM sweeps as traceable steps.
 */

export {
  ceemdan,
  eemd,
  emd,
  extrema,
  RILLING_RULE,
  siftImf,
  siftSteps,
  type CeemdanOptions,
  type EmdOptions,
  type SiftOptions,
  type SiftState,
  type StopRule,
} from './emd'
export { vmd, vmdModes, vmdSteps, type VmdOptions, type VmdState } from './vmd'
export { decompositionsAlgorithms, decompositionsFunctions } from './registry'
