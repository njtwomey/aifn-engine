/**
 * The Algorithm protocol checks every factory's tests run (design K §7): the same key gives the same trace, a different
 * key a different one when the algorithm draws; `seek(i)` equals `run(i)`, from scratch and from a trace's
 * checkpoints; `extend` equals a longer trace; and a state cloned with `structuredClone` (a worker boundary, then
 * `revive` for its tensors) steps the same as the original, so states are plain data. The applications' twin is
 * `packages/methods/test/protocol.ts` (`expectProtocol`).
 *
 * ```ts
 * checkProtocol(kmeans(data, { k: 3 }), undefined, { steps: 20, record: { loss: (s) => s.loss } })
 * ```
 */

import { expect } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { revive, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { extend, run, seek, trace, type Algorithm, type Recorder } from 'aifn-compute/foundation/trace'
import type { Status } from 'aifn-compute/foundation/contracts'

export type ProtocolOptions<S> = {
  /** Steps to run (the algorithm may stop earlier). Default 12. */
  readonly steps?: number
  /** Recorders compared between traces. Default: none (states are compared). */
  readonly record?: Record<string, Recorder<S>>
  /** The algorithm draws from its streams, so a different key must give a different trace. Default false. */
  readonly random?: boolean
  /** Compare states structurally (default), or with this projection when states hold closures or huge buffers. */
  readonly view?: (state: S) => unknown
}

/** Plain, comparable data of a state: tensors become their shape and values; functions are dropped. */
export function plainOf(x: unknown): unknown {
  if (typeof x === 'function') return undefined
  if (x === null || typeof x !== 'object') return x
  if ('shape' in x && 'strides' in x && 'data' in x) {
    const t = x as Tensor
    return { shape: [...t.shape], dtype: t.dtype, values: toFlat(t) }
  }
  if (ArrayBuffer.isView(x)) return Array.from(x as unknown as ArrayLike<number>)
  if (Array.isArray(x)) return x.map(plainOf)
  if (x instanceof Map) return [...x.entries()].map(([k, v]) => [plainOf(k), plainOf(v)])
  if (x instanceof Set) return [...x].map(plainOf)
  return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, plainOf(v)]))
}

const seriesOf = (t: { series: Record<string, Tensor>; index: Int32Array }) => ({
  index: Array.from(t.index),
  series: Object.fromEntries(Object.entries(t.series).map(([k, v]) => [k, toFlat(v)])),
})

/** Run the four protocol checks on `alg` from `start`. */
export function checkProtocol<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  options: ProtocolOptions<S> = {},
): void {
  const n = options.steps ?? 12
  const record = options.record ?? {}
  const view = options.view ?? ((s: S) => plainOf(s))
  const key = stream('protocol')

  // Same key, same trace (series and states); another key, another trace when the algorithm draws.
  const a = trace(alg, start, n, { record, stream: key, checkpointEvery: 3 })
  const b = trace(alg, start, n, { record, stream: stream('protocol') })
  expect(seriesOf(b)).toEqual(seriesOf(a))
  expect(a.steps.map(view)).toEqual(b.steps.map(view))
  expect(a.final.t).toBe(a.meta.steps)
  if (options.random) {
    const c = trace(alg, start, n, { record, stream: stream('protocol/other') })
    expect(c.steps.map(view)).not.toEqual(a.steps.map(view))
  }

  // seek(i) = run(i): from scratch and from the trace's checkpoints.
  const steps = a.meta.steps
  for (const i of new Set([0, 1, Math.floor(steps / 2), steps])) {
    const direct = view(run(alg, start, i, { stream: key }))
    expect(view(seek(alg, start, i, { stream: key }))).toEqual(direct)
    expect(view(seek(alg, start, i, { checkpoints: a }))).toEqual(direct)
  }

  // extend = a longer trace.
  if (a.meta.stopped === 'limit' && n >= 2) {
    const m = Math.floor(n / 2)
    const short = trace(alg, start, n - m, { record, stream: key, checkpointEvery: 3 })
    const extended = extend(short, alg, m)
    expect(seriesOf(extended)).toEqual(seriesOf(a))
    expect(view(extended.final)).toEqual(view(a.final))
  }

  // A state cloned across a worker boundary (then re-branded by `revive`, as a worker does) steps the same.
  const t = Math.min(steps, Math.max(0, Math.floor(steps / 2)))
  const s = seek(alg, start, t, { stream: key })
  if (!alg.done?.(s) && !s.converged && !s.diverged && !s.terminated) {
    // The runner's context for stepping state t: its stream is child(root, 'step', t).
    const ctx = () => ({ t, stream: child(key, 'step', t) })
    expect(view(alg.step(revive(structuredClone(s)), ctx()))).toEqual(view(alg.step(s, ctx())))
  }
}
