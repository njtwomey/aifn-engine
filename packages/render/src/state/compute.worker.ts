/**
 * The compute worker of `useComputed(…, { mode: 'worker' })`: it evaluates one task at a time (`state/task.ts`),
 * resolving each address to an aifn export by importing that module on first use, and answers with the result made
 * cloneable. A result that is a generator (headless training, `aifn-methods/gym` `training`) streams: every yielded
 * value is posted as a partial answer, then the last one as the result. aifn is DOM-free, so its modules run here
 * unchanged. A stale job is cancelled by the page, which terminates this worker and starts a fresh one.
 */
import { fromMessage, isTask, toMessage, type WorkerRequest, type WorkerResponse } from './task'

// Every aifn module (a directory with an index.ts; `_` folders are private), imported lazily by path.
const MODULES = {
  ...import.meta.glob(['../../../compute/src/**/index.ts', '!**/_*/**']),
  ...import.meta.glob(['../../../methods/src/**/index.ts', '!**/_*/**']),
} as Record<string, () => Promise<Record<string, unknown>>>

const CORE = '../../../compute/src/'
const APPLIED = '../../../methods/src/'

/** The aifn export at `address` (`<module>/<export>`), from the compute or the applications. */
async function resolve(address: string): Promise<unknown> {
  const cut = address.lastIndexOf('/')
  if (cut <= 0) throw new Error(`worker: '${address}' is not an address (<module>/<export>)`)
  const module = address.slice(0, cut)
  const key = address.slice(cut + 1)
  const local = module.replace(/^applied\//, '')
  const load = MODULES[`${CORE}${module}/index.ts`] ?? MODULES[`${APPLIED}${local}/index.ts`]
  if (!load) throw new Error(`worker: no aifn module '${module}' (from '${address}')`)
  const ns = await load()
  if (key in ns) return ns[key]
  // An entry kept only in a table of entries (a registry) is found by its key.
  for (const v of Object.values(ns))
    if (v !== null && typeof v === 'object' && key in v) {
      const entry = (v as Record<string, unknown>)[key] as { info?: { key?: unknown } } | undefined
      if (entry?.info?.key === key) return entry
    }
  throw new Error(`worker: module '${module}' has no export '${key}'`)
}

/** Evaluate a task tree: calls (innermost first), arrays and plain objects; leaves re-branded. */
async function evaluate(x: unknown): Promise<unknown> {
  if (isTask(x)) {
    const f = await resolve(x.$call)
    const args = await Promise.all(x.args.map(evaluate))
    if (typeof f === 'function') return (f as (...a: unknown[]) => unknown)(...args)
    if (args.length > 0) throw new Error(`worker: '${x.$call}' is not a function`)
    return f
  }
  if (Array.isArray(x)) return Promise.all(x.map(evaluate))
  if (typeof x === 'object' && x !== null && Object.getPrototypeOf(x) === Object.prototype) {
    if ('shape' in x && 'data' in x) return fromMessage(x)
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(x)) out[k] = await evaluate(v)
    return out
  }
  return x
}

/** A generator's iterator (not an array or other built-in iterable). */
const isIterator = (x: unknown): x is Iterator<unknown> =>
  typeof x === 'object' &&
  x !== null &&
  !Array.isArray(x) &&
  !ArrayBuffer.isView(x) &&
  typeof (x as { next?: unknown }).next === 'function' &&
  typeof (x as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function'

// The worker's global scope; the lab's TypeScript program has the DOM library, not the worker one.
const scope = self as unknown as {
  postMessage(r: WorkerResponse): void
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null
}
const post = (r: WorkerResponse) => scope.postMessage(r)

scope.onmessage = async (e) => {
  const { id, task } = e.data
  const t0 = performance.now()
  try {
    let result = await evaluate(task)
    // A generator streams: each value it yields goes back as a partial answer, and the last is the result.
    if (isIterator(result)) {
      let last: unknown = undefined
      for (let step = result.next(); !step.done; step = result.next()) {
        last = step.value
        post({ id, ok: true, partial: true, value: toMessage(last), ms: performance.now() - t0 })
      }
      result = last
    }
    const value = toMessage(result)
    post({ id, ok: true, value, ms: performance.now() - t0 })
  } catch (err) {
    post({ id, ok: false, error: err instanceof Error ? err.message : String(err), ms: performance.now() - t0 })
  }
}
