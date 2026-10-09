/**
 * The runner protocol, internal to `aifn-compute/foundation/trace`: how every runner (`run`, `seek`, `live`, `trace`,
 * `extend`, `timeSliced`) and every differentiating driver (`unrolled`, `atConvergence`) derives an algorithm's
 * randomness from one root key and decides when to stop. One definition, so a state reached by any of them is the
 * state reached by `run` with the same root stream.
 *
 * - `rootKey(stream)`: the key every stream derives from (default `stream(0)`'s).
 * - `initStream(key)`: `child(root, 'init')`, the stream of `init`.
 * - `stepContext(key, t)`: the context of step $t$, whose stream `child(root, 'step', t)` is made on first use.
 * - `stopReason(alg, state)`: 'diverged' before 'done' (converged, terminated or `alg.done`), else null.
 *
 * Not exported from the package: a new runner lives in this folder and imports it from here.
 */

import type { Algorithm, Key, Size, Status, StepContext, StopReason, Stream } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'

/**
 * The key every stream of a run derives from: the given stream's, or `stream(0)`'s.
 *
 * @param s The run's root stream, as passed in a runner's `options.stream`; only its key is used. Left out, the root
 *   is `stream(0)`.
 * @returns The root key.
 */
export const rootKey = (s: Stream | undefined): Key => (s ?? stream(0)).key

/**
 * The stream `init` draws from: `child(root, 'init')`.
 *
 * @param key The run's root key, from `rootKey`.
 * @returns A new stream for `init`.
 */
export const initStream = (key: Key): Stream => child(key, 'init')

/**
 * The context of step $t$: its stream is `child(root, 'step', t)`, made on first use, so a step that draws nothing
 * costs nothing.
 *
 * @param key The run's root key, from `rootKey`.
 * @param t The step number of the state being stepped (0 for the step from the initial state).
 * @returns The `StepContext` passed to `step`, with `t` and a lazily made `stream`.
 */
export function stepContext(key: Key, t: Size): StepContext {
  let s: Stream | undefined
  return {
    t,
    get stream() {
      return (s ??= child(key, 'step', t))
    },
  }
}

/**
 * Why stepping must stop at this state, if it must: divergence first, then convergence, termination or `done`.
 *
 * @param alg The algorithm, whose optional `done` is consulted after the `Status` flags.
 * @param state The state just reached.
 * @returns `'diverged'` when `state.diverged` is true; `'done'` when `converged` or `terminated` is true or
 *   `alg.done(state)` holds; otherwise null (keep stepping).
 */
export function stopReason<S extends Status>(alg: Algorithm<never, S>, state: S): StopReason | null {
  if (state.diverged === true) return 'diverged'
  if (state.converged === true || state.terminated === true || alg.done?.(state)) return 'done'
  return null
}
