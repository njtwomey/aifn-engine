/**
 * Replicates: run a stochastic function on child streams $0, 1, \dots, n - 1$ of a stream. Replicate $k$ always uses
 * `child(s, k)`, so its result depends only on the stream's key and $k$: raising $n$ keeps the first replicates (prefix
 * reuse), and running two methods over the same replicates gives common random numbers.
 */

import type { Size, Stream } from 'aifn-compute/foundation/contracts'
import { child } from './stream'

/** A cache of replicate results keyed by the child stream's key path. */
export type ReplicateCache<T> = Map<string, T>

// Default caches, one per function object. A new function (e.g. a closure over new parameters) gets a fresh cache.
const caches = new WeakMap<object, ReplicateCache<unknown>>()

/**
 * Run `fn(child(s, k), k)` for $k = 0, \dots, n - 1$ and return the results in order. Results are cached by the
 * child's key path: by default in a cache attached to `fn` itself, so calling again with the same function and a
 * larger $n$ computes only the new replicates. `fn` must therefore be a pure function of its stream and index; when its
 * inputs change, pass a new function (as a React `useCallback` with dependencies does) or an explicit `cache`.
 *
 * @param n The number of replicates.
 * @param s The stream whose children the replicates draw from. Only its key is used: it is not advanced, and its
 *   position does not matter.
 * @param fn One replicate: given its own stream `child(s, k)` and its index $k$, it returns the replicate's result.
 * @param options How results are cached.
 * @param options.cache A map from child key paths to results to read and fill (shared between functions if you
 *   choose), `false` to compute every replicate afresh, or omitted for the cache attached to `fn`.
 * @returns The $n$ results, replicate $k$ at index $k$.
 *
 * @example Each replicate draws from its own child stream
 * const s = stream(1)
 * const draw = (r) => uniform(r)
 * print('3 replicates =', replicate(3, s, draw))
 * print('5 replicates =', replicate(5, s, draw))
 *
 * @example Two estimators on common random numbers
 * const s = stream(2)
 * const mean = replicate(4, s, (r) => sum(normal(r, 0, 1, { shape: [10] })) / 10)
 * const first = replicate(4, s, (r) => normal(r, 0, 1, { shape: [10] }).data[0])
 * print('mean of 10 =', mean)
 * print('first of 10 =', first)
 */
export function replicate<T>(
  n: Size,
  s: Stream,
  fn: (s: Stream, k: Size) => T,
  options: { cache?: ReplicateCache<T> | false } = {},
): T[] {
  let cache: ReplicateCache<T> | undefined
  if (options.cache === false) cache = undefined
  else if (options.cache) cache = options.cache
  else {
    cache = caches.get(fn) as ReplicateCache<T> | undefined
    if (!cache) {
      cache = new Map()
      caches.set(fn, cache as ReplicateCache<unknown>)
    }
  }
  const out: T[] = []
  for (let k = 0; k < n; k++) {
    const r = child(s, k)
    if (cache?.has(r.key.path)) {
      out.push(cache.get(r.key.path) as T)
      continue
    }
    const value = fn(r, k)
    cache?.set(r.key.path, value)
    out.push(value)
  }
  return out
}
