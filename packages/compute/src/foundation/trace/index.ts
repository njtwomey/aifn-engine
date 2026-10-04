/**
 * `aifn-compute/foundation/trace`: the runners that drive every iterative algorithm in aifn.
 *
 * An `Algorithm` (defined in `aifn-compute/foundation/contracts`) is a pure description: `init(start, stream)`, `step(state,
 * ctx)` and an optional `done`, with a `Status` on every state. The runners supply the randomness, `child(root,
 * 'init')` to `init` and `child(root, 'step', t)` to step t, and stop on the status flags. `run` returns the final
 * state; `trace` returns recorded series over the kept steps with the stored states (`keep`), timing (`timing`) and
 * metadata; `seek` returns the state at one step (from stored states when given a trace); `extend` continues a trace;
 * `live` is a generator for play loops; `timeSliced` yields partial traces for interfaces, each in O(1); `decimate`
 * thins a trace for drawing; `profile` times named phases in a step. `unrolled` differentiates through an
 * algorithm's steps (`atConvergence`, implicitly at convergence, is in `aifn-compute/numerics/implicit`). See `README.md` (Traces).
 */

export type {
  Algorithm,
  Checkpoints,
  KeepStates,
  Recorded,
  Recorder,
  Status,
  StepContext,
  StopReason,
  TimingMode,
  Trace,
  TraceOptions,
  TraceTiming,
} from 'aifn-compute/foundation/contracts'
export { decimate, extend, live, now, profile, run, seek, timeSliced, trace, type RunOptions } from './runners'
export { seriesComponents } from './series'
export { unrolled, type UnrolledOptions } from './differentiate'
