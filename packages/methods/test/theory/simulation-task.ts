/**
 * Worker task evaluation for the theory tests, as `packages/render/src/state/compute.worker.ts` does it: an address
 * `<module>/<export>` resolves to an export of `aifn-compute/<module>`, else of `aifn-methods/<module>`, and a task
 * tree is evaluated innermost first.
 */

type Task = { $call: string; args: unknown[] }

const isTask = (x: unknown): x is Task =>
  typeof x === 'object' && x !== null && typeof (x as { $call?: unknown }).$call === 'string'

/** The export at `address`, from compute first, else from the applications. */
async function resolve(address: string): Promise<unknown> {
  const cut = address.lastIndexOf('/')
  const module = address.slice(0, cut)
  const key = address.slice(cut + 1)
  let ns: Record<string, unknown>
  try {
    ns = (await import(`aifn-compute/${module}`)) as Record<string, unknown>
  } catch {
    ns = (await import(`aifn-methods/${module.replace(/^applied\//, '')}`)) as Record<string, unknown>
  }
  if (!(key in ns)) throw new Error(`no export '${key}' in '${module}'`)
  return ns[key]
}

/** Evaluate a task tree: calls (innermost first), arrays and plain objects. */
export async function evaluateTask(x: unknown): Promise<unknown> {
  if (isTask(x)) {
    const f = await resolve(x.$call)
    const args = await Promise.all(x.args.map(evaluateTask))
    return typeof f === 'function' ? (f as (...a: unknown[]) => unknown)(...args) : f
  }
  if (Array.isArray(x)) return Promise.all(x.map(evaluateTask))
  if (typeof x === 'object' && x !== null && Object.getPrototypeOf(x) === Object.prototype) {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(x)) out[k] = await evaluateTask(v)
    return out
  }
  return x
}
