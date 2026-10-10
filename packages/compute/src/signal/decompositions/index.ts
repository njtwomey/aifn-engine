/**
 * `aifn-compute/signal/decompositions`: adaptive decompositions of a signal into modes, as PyEMD and vmdpy.
 *
 * - Empirical mode decomposition: `emd` sifts intrinsic mode functions out of the signal, fastest first; `eemd`
 *   averages the IMFs of noise-added copies (the sum then carries the averaged noise); `ceemdan` extracts one mode at a
 *   time from noise-added residues, so its modes sum to the signal exactly. The ensemble methods draw from a `Stream`.
 * - Sifting one IMF: `siftSteps` as a traceable algorithm, `siftImf` run to the end, with the stopping rules of
 *   `StopRule` (`RILLING_RULE` among them) and `extrema`, the extrema and zero crossings that sifting reads.
 * - Variational mode decomposition: `vmd` fits $K$ narrow-band modes and their centre frequencies together by ADMM;
 *   `vmdSteps` is its sweeps as a traceable algorithm, and `vmdModes` the time-domain modes of one of its states.
 * - `decompositionsAlgorithms` and `decompositionsFunctions`: the module's algorithms and functions with the notes
 *   and citations that define them.
 *
 * The decompositions return a `Decomposition` over the signal's time axis, whose components and `residual` sum to
 * `original` (exactly for every method but `eemd`). Signals are single-channel `Signal`s or bare samples; nothing here
 * is differentiable.
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
