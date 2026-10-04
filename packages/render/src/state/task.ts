/**
 * Worker tasks: a computation named by aifn addresses and plain inputs, so it can cross to a Web Worker
 * (`useComputed(…, { mode: 'worker' })`, DESIGN.md §8a). A task is data, not a closure:
 *
 *   call('foundation/trace/trace',
 *     call('inference/stochastic/hmc', call('data/targets/funnel'), { stepSize: 0.3, steps: 25 }),
 *     { x0: [-4.5, 0.2] }, 40, { stream: call('foundation/random/stream', 'showcase-hmc') })
 *
 * An address is `<module>/<export>`: an entry's catalog address (`inference/stochastic/hmc`, whose key equals its
 * export name) or any other export of an aifn module (`foundation/trace/trace`); applications' modules resolve with or
 * without the `applied/` prefix. Arguments are plain values (numbers, strings, arrays, plain objects, typed arrays),
 * tensors, or nested calls, which the worker evaluates first. Results come back by structured clone, tensors re-branded
 * by aifn's `revive`; function-valued fields are dropped on the way.
 */
import { revive } from 'aifn-compute/foundation/tensor'

const CALL = '$call'

/** A call of an aifn export by address with plain arguments; `R` is the result type, stated by the caller. */
export type Task<R = unknown> = {
  readonly $call: string
  readonly args: readonly unknown[]
  /** Phantom: the result type. */
  readonly __result?: R
}

/** The task "call the aifn export at `address` with `args`" (see the module comment). */
export function call<R = unknown>(address: string, ...args: unknown[]): Task<R> {
  return { [CALL]: address, args } as Task<R>
}

export const isTask = (x: unknown): x is Task =>
  typeof x === 'object' && x !== null && typeof (x as { $call?: unknown }).$call === 'string'

const cloneable = (x: object) =>
  ArrayBuffer.isView(x) ||
  x instanceof ArrayBuffer ||
  x instanceof Date ||
  x instanceof RegExp ||
  x instanceof Map ||
  x instanceof Set

/**
 * `x` made safe for `postMessage`: functions (methods and closures on results such as a trace's `extend`) are dropped,
 * arrays and objects walked; shared sub-objects stay shared and cycles are kept.
 */
export function toMessage(x: unknown, seen = new Map<object, unknown>()): unknown {
  if (typeof x === 'function' || typeof x === 'symbol') return undefined
  if (typeof x !== 'object' || x === null || cloneable(x)) return x
  const known = seen.get(x)
  if (known !== undefined) return known
  if (Array.isArray(x)) {
    const out: unknown[] = []
    seen.set(x, out)
    for (const v of x) out.push(toMessage(v, seen))
    return out
  }
  const out: Record<string, unknown> = {}
  seen.set(x, out)
  for (const [k, v] of Object.entries(x)) {
    if (typeof v === 'function') continue
    out[k] = toMessage(v, seen)
  }
  return out
}

/** A value received from a worker (or sent to one), with its tensors re-branded. */
export const fromMessage = <T>(x: unknown): T => revive(x) as T

/** A request to the compute worker and its answer. */
export type WorkerRequest = { id: number; task: unknown }
export type WorkerResponse = { id: number; ms: number } & (
  { ok: true; value: unknown; partial?: boolean } | { ok: false; error: string }
)
