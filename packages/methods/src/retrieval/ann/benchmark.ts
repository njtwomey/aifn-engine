/**
 * A recall-against-speed benchmark of nearest-neighbour indexes in the manner of ann-benchmarks (Aumüller,
 * Bernhardsson and Faithfull 2020): the last `queries` rows of the data are the queries, the rest the indexed points;
 * every method is built once and queried at each setting of its speed knob (tree leaf size, LSH tables, IVF probes, PQ
 * codewords, HNSW beam width), and each setting reports recall@k against brute force, queries per second, distance
 * evaluations per query and build time. A generator, so a worker can stream methods as they finish.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { now } from 'aifn-compute/foundation/trace'
import {
  benchmarkSearch,
  bruteForceNeighbours,
  hnswIndex,
  hnswSearch,
  ivfIndex,
  ivfSearch,
  kdTree,
  lshIndex,
  lshSearch,
  pqEncode,
  pqSearch,
  productQuantiser,
  pStableFamily,
  treeSearch,
  type Neighbours,
} from 'aifn-compute/numerics/neighbours'

/** The indexes the benchmark runs. */
export const ANN_METHODS = ['kd-tree', 'lsh', 'ivf', 'pq', 'hnsw'] as const
export type AnnMethod = (typeof ANN_METHODS)[number]

/** Options of `annBenchmark`. */
export interface AnnBenchmarkOptions {
  /** The rows: the last `queries` are queries, the rest are indexed. */
  data: { x: Tensor }
  queries?: Size
  k?: Size
  methods?: readonly AnnMethod[]
  seed?: string | number
}

/** One setting of one method. */
export interface AnnPoint {
  /** The knob's value, e.g. `probes = 4`. */
  readonly setting: string
  readonly recall: number
  readonly queriesPerSecond: number
  readonly distancesPerQuery: number
}

/** One method's curve. */
export interface AnnCurve {
  readonly method: AnnMethod
  readonly buildSeconds: number
  readonly points: readonly AnnPoint[]
}

/** A snapshot: the curves of the methods finished so far, and brute force's speed. */
export interface AnnBenchmarkSnapshot {
  readonly done: boolean
  readonly n: Size
  readonly d: Size
  readonly queries: Size
  readonly bruteForce: { queriesPerSecond: number; distancesPerQuery: number }
  readonly curves: readonly AnnCurve[]
}

/** Run every method's settings (module notes), yielding after each method. */
export function* annBenchmark(options: AnnBenchmarkOptions): Generator<AnnBenchmarkSnapshot> {
  const { k = 10, methods = ANN_METHODS } = options
  const all = options.data.x
  const [rows, d] = all.shape
  const m = Math.min(options.queries ?? 100, Math.floor(rows / 2))
  const n = rows - m
  const v = dense.data(all)
  const x = fromData(v.slice(0, n * d), [n, d])
  const q = fromData(v.slice(n * d), [m, d])
  const root = stream(options.seed ?? 'ann')
  const truth = bruteForceNeighbours(x, q, k)
  const brute = benchmarkSearch(() => bruteForceNeighbours(x, q, k), truth, { k, repeats: 1 })
  const curves: AnnCurve[] = []
  const snapshot = (done: boolean): AnnBenchmarkSnapshot => ({
    done,
    n,
    d,
    queries: m,
    bruteForce: { queriesPerSecond: brute.queriesPerSecond, distancesPerQuery: brute.distancesPerQuery },
    curves: [...curves],
  })
  const point = (setting: string, search: () => Neighbours): AnnPoint => {
    const b = benchmarkSearch(search, truth, { k, repeats: 1 })
    return { setting, recall: b.recall, queriesPerSecond: b.queriesPerSecond, distancesPerQuery: b.distancesPerQuery }
  }
  const timed = <T>(build: () => T): [T, number] => {
    const start = now()
    const out = build()
    return [out, (now() - start) / 1000]
  }
  yield snapshot(false)
  for (const method of methods) {
    const s = child(root, method)
    if (method === 'kd-tree') {
      const points = [4, 16, 64].map((leaf) => {
        const tree = kdTree(x, { leafSize: leaf })
        return point(`leaf ${leaf}`, () => treeSearch(tree, q, k))
      })
      curves.push({ method, buildSeconds: timed(() => kdTree(x))[1], points })
    } else if (method === 'lsh') {
      // Bucket width about the typical nearest-neighbour distance scale of the data.
      const width = Math.max(1e-6, dense.data(truth.distances)[k - 1]) * 4
      const t0 = now()
      const indexes = [1, 2, 4, 8, 16].map((tables) => ({
        tables,
        index: lshIndex(x, pStableFamily(d, { tables, hashesPerTable: 4, width, stream: s })),
      }))
      const buildSeconds = (now() - t0) / 1000 / indexes.length
      curves.push({
        method,
        buildSeconds,
        points: indexes.map(({ tables, index }) => point(`${tables} tables`, () => lshSearch(index, q, k))),
      })
    } else if (method === 'ivf') {
      const lists = Math.max(4, Math.round(Math.sqrt(n)))
      const [index, buildSeconds] = timed(() => ivfIndex(x, { lists, stream: s }))
      const probes = [1, 2, 4, 8, 16].filter((p) => p <= lists)
      curves.push({
        method,
        buildSeconds,
        points: probes.map((p) => point(`${p} probes`, () => ivfSearch(index, q, k, { probes: p }))),
      })
    } else if (method === 'pq') {
      const M = [4, 2, 1].find((c) => d % c === 0 && d / c >= 1) ?? 1
      const t0 = now()
      const points = [4, 16, 64].map((K) => {
        const pq = productQuantiser(x, { subspaces: M, codewords: K, stream: child(s, 'K', K), iterations: 10 })
        const codes = pqEncode(pq, x)
        return point(`${M} × ${K} codewords`, () => pqSearch(pq, codes, q, k))
      })
      curves.push({ method, buildSeconds: (now() - t0) / 1000 / 3, points })
    } else {
      const [index, buildSeconds] = timed(() => hnswIndex(x, { M: 8, efConstruction: 64, stream: s }))
      curves.push({
        method,
        buildSeconds,
        points: [k, 2 * k, 4 * k, 8 * k].map((ef) => point(`ef ${ef}`, () => hnswSearch(index, q, k, { ef }))),
      })
    }
    yield snapshot(false)
  }
  yield snapshot(true)
}
