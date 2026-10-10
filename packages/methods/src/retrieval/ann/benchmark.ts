/**
 * A recall-against-speed benchmark of nearest-neighbour indexes in the manner of ann-benchmarks (Aumüller,
 * Bernhardsson and Faithfull, 2020).
 *
 * The last `queries` rows of the data are the queries and the rest the indexed points. Every index of
 * `aifn-compute/numerics/neighbours` is queried at several settings of its speed knob: the k-d tree's leaf size, the
 * number of LSH tables, IVF's probed lists, PQ's codewords per subspace and HNSW's beam width (the indexes FAISS calls
 * `IndexIVFFlat`, `IndexPQ` and `IndexHNSWFlat`). Each setting reports recall@$k$ against brute force, queries per
 * second and distance evaluations per query, and each method its build time. Recall and distance counts are
 * reproducible from the seed; the timings are the machine's. A generator, so a worker can stream methods as they
 * finish.
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
/** The name of one index of the benchmark, an entry of `ANN_METHODS`. */
export type AnnMethod = (typeof ANN_METHODS)[number]

/** Options of `annBenchmark`. */
export interface AnnBenchmarkOptions {
  /** The rows, an $N \times d$ matrix in `x`: the last `queries` are queries, the rest are indexed. */
  data: { x: Tensor }
  /** The number of queries $m$ (default 100, and at most half the rows). */
  queries?: Size
  /** The neighbours $k$ found per query, against which recall is measured (default 10). */
  k?: Size
  /** The indexes to run, in order (default all of `ANN_METHODS`). */
  methods?: readonly AnnMethod[]
  /** The root seed of the indexes' random construction (default `'ann'`). */
  seed?: string | number
}

/** One setting of one method. */
export interface AnnPoint {
  /** The knob's value as a label, e.g. `4 probes` or `ef 20`. */
  readonly setting: string
  /** Recall@$k$: the share of the true $k$ nearest neighbours found, over all queries. */
  readonly recall: number
  /** Throughput of the search, timed once on this machine. */
  readonly queriesPerSecond: number
  /** Mean distance evaluations per query (brute force needs $n$). */
  readonly distancesPerQuery: number
}

/** One method's curve. */
export interface AnnCurve {
  /** The index. */
  readonly method: AnnMethod
  /** Seconds to build the index (for LSH and PQ, the mean over the indexes built for its settings). */
  readonly buildSeconds: number
  /** One point per setting, in the order `annBenchmark` lists them. */
  readonly points: readonly AnnPoint[]
}

/** A snapshot: the curves of the methods finished so far, and brute force's speed. */
export interface AnnBenchmarkSnapshot {
  /** True on the last snapshot, after every method. */
  readonly done: boolean
  /** The number of indexed points $n$. */
  readonly n: Size
  /** The dimension $d$ of the points. */
  readonly d: Size
  /** The number of queries $m$. */
  readonly queries: Size
  /** The exact search's throughput and distance evaluations per query, the baseline. */
  readonly bruteForce: { queriesPerSecond: number; distancesPerQuery: number }
  /** The curves of the methods finished so far, in the order run. */
  readonly curves: readonly AnnCurve[]
}

/**
 * Run every method at each of its settings, yielding a snapshot before the first method, after each method and once
 * more at the end (`done`). The settings are leaf sizes 4, 16 and 64 for the k-d tree (always exact); 1 to 16 tables of
 * four hashes for LSH, with a bucket width of four times the first query's $k$-th neighbour distance; 1 to 16 probes of
 * $\sqrt{n}$ lists for IVF; 4, 16 and 64 codewords on 4, 2 or 1 subspaces for PQ; and a beam width $k$ to $8k$ for HNSW
 * ($M = 8$, construction beam 64).
 *
 * @param options The data, the number of queries and neighbours $k$, the methods to run and the seed.
 * @returns A generator of snapshots, each holding the curves finished so far.
 *
 * @example Recall of the true nearest point of 200 points rises with each index's knob
 * const x = normal(stream(0), 0, 1, { shape: [220, 2] })
 * const snapshots = [...annBenchmark({ data: { x }, queries: 20, k: 1, methods: ['kd-tree', 'ivf', 'hnsw'] })]
 * const last = snapshots[snapshots.length - 1]
 * print('indexed', last.n, 'points; brute force evaluates', last.bruteForce.distancesPerQuery, 'distances per query')
 * for (const c of last.curves)
 *   print(c.method, c.points.map((p) => `${p.setting}: recall ${p.recall}, ${p.distancesPerQuery} evals`).join('; '))
 */
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
