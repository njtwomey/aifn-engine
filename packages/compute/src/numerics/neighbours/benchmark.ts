/**
 * Measuring approximate search the way ann-benchmarks does (Aumüller, Bernhardsson and Faithfull 2020,
 * "ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms", Information Systems 87): recall@k
 * against the exact answer, queries per second, and the distance evaluations per query, a cost that does not depend on
 * the machine.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { now } from 'aifn-compute/foundation/trace'
import type { Neighbours } from './search'

/**
 * Recall@k: the mean over queries of |A ∩ E| / k, where A holds the first k indices found and E the exact k nearest.
 * Both are m × (≥ k) index matrices (or `Neighbours`); −1 entries (no answer) never match.
 */
export function searchRecall(found: Tensor | Neighbours, exact: Tensor | Neighbours, k?: Size): number {
  const a = 'kind' in found && found.kind === 'neighbours' ? found.indices : (found as Tensor)
  const e = 'kind' in exact && exact.kind === 'neighbours' ? exact.indices : (exact as Tensor)
  const [m, ka] = a.shape
  const [me, ke] = e.shape
  if (m !== me) throw new ShapeError('searchRecall', `searchRecall: ${m} queries found, ${me} exact`)
  const kk = k ?? Math.min(ka, ke)
  if (kk > ka || kk > ke) throw new ShapeError('searchRecall', `searchRecall: k = ${kk} exceeds the answers' width`)
  const A = dense.data(a)
  const E = dense.data(e)
  let hits = 0
  for (let i = 0; i < m; i++) {
    const want = new Set<number>()
    for (let j = 0; j < kk; j++) want.add(E[i * ke + j])
    for (let j = 0; j < kk; j++) if (A[i * ka + j] >= 0 && want.has(A[i * ka + j])) hits++
  }
  return m ? hits / (m * kk) : NaN
}

/** One method's measurements over a query set. */
export interface SearchBenchmark {
  /** Recall@k against the exact answer. */
  readonly recall: number
  /** Wall-clock seconds for all queries (median of `repeats`). */
  readonly seconds: number
  readonly queriesPerSecond: number
  /** Mean distance evaluations per query. */
  readonly distancesPerQuery: number
}

/**
 * Time `search` over m queries (`repeats` times, default 3, the median kept) and score its answer against `exact`.
 * Timing is the machine's; recall and distances per query are reproducible.
 */
export function benchmarkSearch(
  search: () => Neighbours,
  exact: Neighbours,
  options: { k?: Size; repeats?: Size } = {},
): SearchBenchmark {
  const times: number[] = []
  let result: Neighbours | null = null
  for (let r = 0; r < Math.max(1, options.repeats ?? 3); r++) {
    const start = now()
    result = search()
    times.push((now() - start) / 1000)
  }
  times.sort((a, b) => a - b)
  const seconds = times[Math.floor(times.length / 2)]
  const m = result!.indices.shape[0]
  return {
    recall: searchRecall(result!, exact, options.k),
    seconds,
    queriesPerSecond: seconds > 0 ? m / seconds : Infinity,
    distancesPerQuery: m ? result!.distanceEvaluations / m : 0,
  }
}
