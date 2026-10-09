/**
 * Differentiating through an `Algorithm` (design K §4.3, §7.3). A trace records raw values for display; derivatives go
 * through `unrolled` (and `atConvergence`) instead, which call `init` and `step` directly under the active transform.
 *
 * - `unrolled` runs the steps on traced values, so the result is differentiated through every step (backpropagation
 *   through the iterations). With `checkpointEvery: k` each run of $k$ steps is a `checkpoint`: the reverse pass
 *   stores one state per segment and recomputes the segment during the backward sweep ($k \approx \sqrt{n}$ stores
 *   $O(\sqrt{n})$ states).
 * - `atConvergence` (in `aifn-compute/numerics/implicit`, since it needs a linear solve) runs the algorithm to
 *   convergence on raw values and differentiates the converged point by the implicit function theorem.
 *
 * Only algorithms written with primitives can be unrolled. One computing on raw arrays (`dense` loops) drops the trace
 * of its inputs; `unrolled` detects that and raises `NotDifferentiableError` rather than returning a zero gradient.
 */

import type { Algorithm, Size, Status } from 'aifn-compute/foundation/contracts'
import { checkpoint } from 'aifn-compute/foundation/autodiff'
import { DomainError, NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { treeFlatten } from 'aifn-compute/foundation/pytree'
import { isTraced } from 'aifn-compute/foundation/tensor'
import { initStream, rootKey, stepContext, stopReason } from './protocol'
import type { RunOptions } from './runners'

/**
 * Whether any leaf of a value is traced by an active transform.
 *
 * @param tree A state or parameters: any pytree (nested arrays and plain objects) whose leaves are checked.
 * @returns True when at least one leaf is a traced value.
 */
const traced = (tree: unknown): boolean => treeFlatten(tree).leaves.some(isTraced)

/** Options of `unrolled`. */
export type UnrolledOptions = RunOptions & {
  /**
   * Checkpoint every $k$ steps: store one state per $k$ steps and recompute each segment in the backward pass. A
   * positive integer; anything else throws `DomainError`. Default: no checkpointing.
   */
  checkpointEvery?: Size
}

/**
 * The state after at most `n` steps of `alg` from `start` (stopping early on a `Status` flag, as `run` does),
 * computed on traced values so that the active transform differentiates through every step. The draws are those of
 * `run` with the same root stream.
 *
 * `alg` is an algorithm, or a factory `(params) => Algorithm` given `options.params`. With `checkpointEvery`, traced
 * values the algorithm depends on (hyperparameters) must reach it as `params`, not through a closure, because each
 * segment is recomputed from its arguments. A step that turns a traced state (or traced `params`) into an untraced one
 * throws `NotDifferentiableError`: the algorithm computes on raw arrays and cannot be unrolled.
 *
 * @param alg The algorithm, or a factory that makes it from `options.params` (called once, and again for each
 *   checkpointed segment).
 * @param start The starting point passed to `init`; traced to differentiate with respect to it.
 * @param n The largest number of steps to take.
 * @param options The root stream (`stream`), the checkpoint interval (`checkpointEvery`) and, with a factory, the
 *   `params` it is called with.
 * @returns The state after the last step, holding traced values when anything it depends on was traced.
 *
 * @example The gradient of an iterate with respect to the learning rate
 * // Gradient descent on f(x) = x²/2 scales x by (1 − lr) each step, so after 10 steps x = (1 − lr)¹⁰.
 * const descent = (lr) => ({
 *   name: 'descent',
 *   init: (x) => ({ t: 0, x }),
 *   step: (s) => ({ t: s.t + 1, x: mul(s.x, sub(1, lr)) }),
 * })
 * print('d x10 / d lr =', grad((lr) => unrolled(descent, 1, 10, { params: lr }).x)(0.1))
 * print('-10 (1 - lr)^9 =', -10 * 0.9 ** 9)
 *
 * @example Checkpointing gives the same gradient with fewer stored states
 * const descent = (lr) => ({
 *   name: 'descent',
 *   init: (x) => ({ t: 0, x }),
 *   step: (s) => ({ t: s.t + 1, x: mul(s.x, sub(1, lr)) }),
 * })
 * const x10With = (k) => (lr) => unrolled(descent, 1, 10, { params: lr, checkpointEvery: k }).x
 * print('every 3 steps:', grad(x10With(3))(0.1))
 * print('every step:', grad(x10With(1))(0.1))
 */
export function unrolled<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  n: Size,
  options?: UnrolledOptions,
): S
export function unrolled<P, Start, S extends Status>(
  alg: (params: P) => Algorithm<Start, S>,
  start: Start,
  n: Size,
  options: UnrolledOptions & { params: P },
): S
export function unrolled<P, Start, S extends Status>(
  alg: Algorithm<Start, S> | ((params: P) => Algorithm<Start, S>),
  start: Start,
  n: Size,
  options: UnrolledOptions & { params?: P } = {},
): S {
  const make = typeof alg === 'function' ? alg : () => alg
  const params = options.params as P
  const first = make(params)
  const key = rootKey(options.stream)
  const paramsTraced = traced(params)
  let state = first.init(start, initStream(key))

  // A step that turns a traced state (or traced parameters) into an untraced one has computed on raw values.
  const guard = (before: boolean, next: S): void => {
    if ((before || paramsTraced) && !traced(next))
      throw new NotDifferentiableError(
        first.name,
        `unrolled: ${first.name} dropped the trace of its inputs (it computes on raw arrays); only algorithms written with primitives can be unrolled — use atConvergence for converged solutions`,
      )
  }

  const k = options.checkpointEvery
  if (k === undefined) {
    for (let t = 0; t < n && !stopReason(first, state); t++) {
      const next = first.step(state, stepContext(key, t))
      guard(traced(state), next)
      state = next
    }
    return state
  }
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('unrolled', `unrolled: checkpointEvery must be a positive integer`)
  const segment = checkpoint((s: S, p: P, t0: number, m: number): S => {
    const a = make(p)
    for (let j = 0; j < m && !stopReason(a, s); j++) s = a.step(s, stepContext(key, t0 + j))
    return s
  })
  for (let t = 0; t < n && !stopReason(first, state); t += k) {
    const next = segment(state, params, t, Math.min(k, n - t))
    guard(traced(state), next)
    state = next
  }
  return state
}
