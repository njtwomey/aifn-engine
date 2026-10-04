/** The functions of `aifn-methods/retrieval/ann`: the nearest-neighbour index benchmark. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as benchmark from './benchmark'

const fn = definer<FunctionInfo>('function', 'retrieval/ann')
fn(
  {
    key: 'annBenchmark',
    name: 'Nearest-neighbour index benchmark',
    summary: 'Recall@k against queries per second for k-d trees, LSH, IVF, PQ and HNSW at several settings each.',
    role: 'simulation',
    random: true,
    notes: ['approximate-nearest-neighbour-benchmarking', 'approximate-nearest-neighbour-search'],
    cite: ['aumuller2020'],
  },
  benchmark.annBenchmark,
)

/** The functions of the module, keyed by name. */
export const annFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', benchmark) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
