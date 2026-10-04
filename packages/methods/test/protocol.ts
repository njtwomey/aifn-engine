/**
 * The Algorithm protocol, checked for one algorithm: the same root key gives the same trace; `seek(i)` equals
 * `run(i)` (from `init` and from a trace's stored states); `extend` of a shorter trace equals a longer trace; and a
 * state that went through `structuredClone` (then `revive`) steps to the same next state.
 */
import type { Algorithm, Recorder, Status } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { isTensor, revive, toFlat } from 'aifn-compute/foundation/tensor'
import { extend, run, seek, trace } from 'aifn-compute/foundation/trace'
import { expect } from 'vitest'

/** A comparable copy of a state: tensors become `{ shape, dtype, data }`, typed arrays plain arrays, functions go. */
export function canon(x: unknown): unknown {
  if (isTensor(x)) return { shape: [...x.shape], dtype: x.dtype, data: Array.from(toFlat(x)) }
  if (ArrayBuffer.isView(x)) return Array.from(x as unknown as ArrayLike<number>)
  if (Array.isArray(x)) return x.map(canon)
  if (x instanceof Map) return [...x].map(([k, v]) => [canon(k), canon(v)])
  if (x instanceof Set) return [...x].map(canon)
  if (typeof x === 'function') return undefined
  if (typeof x === 'object' && x !== null) {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(x)) if (typeof v !== 'function') out[k] = canon(v)
    return out
  }
  return x
}

export interface ProtocolOptions<S> {
  /** Steps of the long trace (default 8). */
  n?: number
  /** Recorders for the traces (default: `t` only). */
  record?: Record<string, Recorder<S>>
  /** Root seed (default 7). */
  seed?: number | string
  /** Skip the structuredClone check (a state that legitimately holds non-plain data). */
  noClone?: boolean
}

const seriesOf = (tr: ReturnType<typeof trace>) =>
  Object.fromEntries(Object.entries(tr.series).map(([k, v]) => [k, Array.from(toFlat(v))]))

/** Runs every protocol check on `alg` from `start`. */
export function expectProtocol<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  options: ProtocolOptions<S> = {},
): void {
  const n = options.n ?? 8
  const record = options.record ?? { t: (s: S) => s.t }
  const root = () => stream(options.seed ?? 7)

  // Same key, same trace; and the final state is plain data equal across runs.
  const a = trace(alg, start, n, { stream: root(), record, keep: 'all' })
  const b = trace(alg, start, n, { stream: root(), record, keep: 'all' })
  expect(seriesOf(b), `${alg.name}: same key, same trace`).toEqual(seriesOf(a))
  expect(canon(b.final)).toEqual(canon(a.final))
  const steps = a.meta.steps

  // seek(i) = run(i), from init and from the trace's stored states.
  for (const i of new Set([0, 1, Math.floor(steps / 2), steps])) {
    const r = canon(run(alg, start, i, { stream: root() }))
    expect(canon(seek(alg, start, i, { stream: root() })), `${alg.name}: seek(${i}) = run(${i})`).toEqual(r)
    expect(canon(seek(alg, start, i, { checkpoints: a })), `${alg.name}: seek(${i}) from a trace`).toEqual(r)
  }
  expect(canon(run(alg, start, n, { stream: root() }))).toEqual(canon(a.final))

  // extend(trace(k), n − k) = trace(n).
  const k = Math.max(1, Math.floor(steps / 2))
  if (k < n) {
    const short = trace(alg, start, k, { stream: root(), record, keep: 'all' })
    const long = extend(short, alg as Algorithm<never, S>, n - k)
    expect(seriesOf(long), `${alg.name}: extend = longer trace`).toEqual(seriesOf(a))
    expect(canon(long.final)).toEqual(canon(a.final))
  }

  // structuredClone(state) steps the same.
  if (!options.noClone && steps > 0) {
    const t = Math.min(k, steps - 1)
    const s = run(alg, start, t, { stream: root() })
    const cloned = revive(structuredClone(s))
    const ctx = (tt: number) => ({ t: tt, stream: child(root().key, 'step', tt) })
    expect(canon(cloned), `${alg.name}: clone is equal`).toEqual(canon(s))
    expect(canon(alg.step(cloned, ctx(t))), `${alg.name}: clone steps the same`).toEqual(canon(alg.step(s, ctx(t))))
  }
}
