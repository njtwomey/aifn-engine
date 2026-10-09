/**
 * Iterative computation: the `Algorithm` protocol, the `Status` every state carries, and the `Trace` a run records
 * (design K §7, S §2.3).
 *
 * Algorithms are made by factories that close over the problem. `init(start, stream)` makes step 0 and
 * `step(state, ctx)` makes the next state, drawing only from `ctx.stream`, which the runner derives as
 * `child(root, 'step', t)`. So states are plain data and never hold a stream, and step $t$'s randomness is fixed by the
 * root key and $t$ alone: a run resumed from any stored state reproduces the rest exactly.
 */

import type { Kinded } from './kinds'
import type { Index, Scalar, Size, Tensor, TensorWire } from './numbers'
import type { Key, Stream } from './random'

// ── Algorithm ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What every state reports to the runner. The runner stops on `diverged`, then on `converged`, `terminated` or the
 * algorithm's `done`. `stalled` is informative (no progress, not an error).
 */
export interface Status {
  /** Steps taken; 0 in the initial state. */
  t: Size
  /** The method's convergence test has passed: the runners stop with `done`. */
  converged?: boolean
  /** The iterate or a value is no longer finite or has blown up: the runners stop with `diverged`. */
  diverged?: boolean
  /** No progress is being made; informative only, the runners do not stop on it. */
  stalled?: boolean
  /** The process has reached a natural end (an episode, an exhausted search): the runners stop with `done`. */
  terminated?: boolean
}

/**
 * What the runner passes to `step`: the step number of the state being stepped and its stream,
 * `child(root, 'step', t)`.
 */
export interface StepContext {
  /** The step number of the state being stepped (0 for the step from the initial state). */
  readonly t: Size
  /** The only randomness `step` may draw from, made on first use. */
  readonly stream: Stream
}

/** JSON-like data: what a serialisable reference or a wire form may hold. */
export type PlainData = null | boolean | number | string | readonly PlainData[] | { readonly [key: string]: PlainData }

/** A serialisable reference to an algorithm: its registry id and the plain arguments of its factory. */
export interface AlgorithmRef {
  /** `module/key`, e.g. `cluster/kmeans`. */
  readonly id: string
  /** The arguments of the factory, as plain data. */
  readonly args: PlainData
}

/**
 * An iterative algorithm in factory form: the problem is closed over by the factory, `start` is an optional starting
 * point (`undefined` when the algorithm has none), and every state is plain data carrying a `Status`. `init` draws only
 * from its stream and `step` only from `ctx.stream`; both are pure.
 */
export interface Algorithm<Start, State extends Status> {
  /** A readable name, kept in a trace's metadata and used in error messages. */
  readonly name: string
  /** Set by factories whose arguments are plain data, so a worker can rebuild the algorithm. */
  readonly ref?: AlgorithmRef
  /** The initial state (step 0, `t: 0`). The runner passes `child(root, 'init')`. */
  init(start: Start, stream: Stream): State
  /** The next state; pure. */
  step(state: State, ctx: StepContext): State
  /** True once the algorithm has finished for a reason `Status` does not express. */
  done?(state: State): boolean
}

// ── Trace ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a recorder may return: a number, a flat array, a nested array (rectangular), or a `Tensor` (any strides or
 * dtype; recorded as float64).
 */
export type Recorded = Scalar | ArrayLike<number> | readonly Recorded[] | Tensor

/** Maps a state to a recorded quantity; runs on kept steps only. */
export type Recorder<State> = (state: State, step: Index) => Recorded

/**
 * Why a run stopped: `done` (the state is `converged` or `terminated`, or the algorithm's `done` holds), `limit` (the
 * step limit), or `diverged` (the state is flagged `diverged`, or a recording is not finite).
 */
export type StopReason = 'done' | 'limit' | 'diverged'

/**
 * Which states a trace stores: `all` (every kept step, and the checkpoints), `checkpoints` (only the checkpoints), or
 * `none`. Series are always recorded and the final state is always stored; `seek` recomputes from the nearest stored
 * state (or from `init`).
 */
export type KeepStates = 'all' | 'checkpoints' | 'none'

/**
 * How a trace is timed: `step` (each step, two clock reads per step), `total` (the whole run only, for microsecond
 * steps where the clock reads would dominate), or `false` (not at all).
 */
export type TimingMode = 'step' | 'total' | false

/** Timing of a traced run. All times are in milliseconds from `performance.now()` (or `Date.now()` without it). */
export interface TraceTiming {
  /**
   * With `timing: 'step'`, the time spent in `step` for each computed step (entry $t$: from state $t$ to state
   * $t + 1$); else empty.
   */
  stepMs: Float64Array
  /**
   * With `timing: 'step'`, wall-clock time since the run began at each kept step (aligned with `index`); else empty.
   */
  elapsedMs: Float64Array
  /** Total time spent stepping (the sum of `stepMs` with `timing: 'step'`); 0 with `timing: false`. */
  totalMs: number
  /**
   * Steps per second of step time: `steps / (totalMs / 1000)`; 0 before the first step, `Infinity` when too fast to
   * measure, NaN untimed.
   */
  perSecond: number
  /**
   * Time per named phase: `init` and `record` (the recorders) with `timing: 'step'`, and every `profile(name, fn)`
   * called inside a step unless `timing: false`.
   */
  phases: Record<string, number>
}

/** Stored states for `seek`, keyed by step number (ascending). */
export interface Checkpoints<State> {
  /** The step number of each stored state, ascending. */
  readonly index: readonly Index[]
  /** The stored states, aligned with `index`. */
  readonly states: readonly State[]
}

/** What a trace records about its run. */
export interface TraceMeta {
  /** The algorithm's `name`. */
  algorithm: string
  /** Why the run stopped (`limit` for a partial trace of a run still going). */
  stopped: StopReason
  /** The number of steps computed (the step number of the final state). */
  steps: Size
  /** The spacing of the kept steps, as the trace was run with. */
  every: Size
  /** The spacing of the stored checkpoints, or null for step 0 only. */
  checkpointEvery: Size | null
  /** Which states the trace stores. */
  keep: KeepStates
  /** How the run was timed. */
  timing: TimingMode
  /** The start passed to `init`. */
  start: unknown
  /** The root key: step $t$ drew from `child(root, 'step', t)`, so `seek` and `extend` reproduce the run from it. */
  key: Key
  /** The names of the recorders, so `extend` (also across a worker boundary) can check it has the same set. */
  recorders: readonly string[]
}

/** How to run a trace. */
export interface TraceOptions<State> {
  /** Keep every `every`-th step (step numbers divisible by `every`), plus the final step. Default 1. */
  every?: Size
  /** Named recorders; each becomes a series stacked over kept steps. */
  record?: Record<string, Recorder<State>>
  /** Also store the state at every multiple of this many steps, for fast `seek`. Default: step 0 only. */
  checkpointEvery?: Size
  /** Which states to store. Default `all`. */
  keep?: KeepStates
  /** How to time the run. Default `step`. */
  timing?: TimingMode
  /** The root stream (only its key is used). Default `stream(0)`. */
  stream?: Stream
  /**
   * Stop with `stopped: 'diverged'` when a recording is infinite, or NaN after that series has been finite. A NaN
   * before a series has had any finite value means "not defined yet" (e.g. a step size at step 0). Default true.
   */
  stopOnNonFinite?: boolean
}

/**
 * A traced run (`kind: 'trace'`): recorded series over the kept steps, the stored states, timing and metadata. Series,
 * `index` and timing are views of the runner's buffers, so taking a trace of a run in progress costs $O(1)$.
 */
export interface Trace<State> extends Kinded<'trace'> {
  /** The step number of each kept step (steps divisible by `every`, and the final step). */
  index: Int32Array
  /** Each recorded quantity stacked over kept steps: a contiguous float64 tensor `[index.length, ...valueShape]`. */
  series: Record<string, Tensor>
  /** With `keep: 'all'`, the state at each kept step (aligned with `index`); otherwise empty. */
  steps: readonly State[]
  /** The final state (step `meta.steps`). */
  final: State
  /** States stored every `checkpointEvery` steps and at step 0 (empty with `keep: 'none'`). */
  checkpoints: Checkpoints<State>
  /** How long the run took, in total, per step and per phase. */
  timing: TraceTiming
  /** How the trace was run and why it stopped. */
  meta: TraceMeta
}

/** A trace on the wire: series and timing without states, which are not always serialisable. */
export interface TraceWire {
  /** The brand of a trace. */
  readonly kind: 'trace'
  /** The step number of each kept step. */
  readonly index: readonly Index[]
  /** Each recorded series, by recorder name. */
  readonly series: Readonly<Record<string, TensorWire>>
  /** The total step time in milliseconds and the steps per second, as in `TraceTiming`. */
  readonly timing: { readonly totalMs: number; readonly perSecond: number }
  /** How the trace was run and why it stopped. */
  readonly meta: TraceMeta
}
