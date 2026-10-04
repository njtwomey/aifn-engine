/**
 * Differentiating through an `Algorithm` (design K §4.3, §7.3). A trace records raw values for display; derivatives go
 * through `unrolled` (and `atConvergence`) instead, which call `init` and `step` directly under the active transform.
 *
 * - `unrolled` runs the steps on traced values, so the result is differentiated through every step (backpropagation
 *   through the iterations). With `checkpointEvery: k` each run of k steps is a `checkpoint`: the reverse pass stores
 *   one state per segment and recomputes the segment during the backward sweep (k ≈ √n stores O(√n) states).
 * - `atConvergence` (in `aifn-compute/numerics/implicit`, since it needs a linear solve) runs the algorithm to convergence on
 *   raw values and differentiates the converged point by the implicit function theorem.
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

const traced = (tree: unknown): boolean => treeFlatten(tree).leaves.some(isTraced)

/** Options of `unrolled`. */
export type UnrolledOptions = RunOptions & {
  /** Checkpoint every k steps: store one state per k steps and recompute each segment in the backward pass. */
  checkpointEvery?: Size
}

/**
 * The state after at most `n` steps of `alg` from `start` (stopping early on a `Status` flag, as `run` does),
 * computed on traced values so that the active transform differentiates through every step.
 *
 * `alg` is an algorithm, or a factory `(params) => Algorithm` given `options.params`. With `checkpointEvery`, traced
 * values the algorithm depends on (hyperparameters) must reach it as `params`, not through a closure, because each
 * segment is recomputed from its arguments.
 *
 * @example
 * // The gradient of the loss after 50 steps of gradient descent, with respect to the learning rate.
 * grad((lr: number) => loss(unrolled(gd, x0, 50, { params: lr, checkpointEvery: 7 }).x))(0.1)
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
