/**
 * Nearest-neighbour search: brute force and the k-d and ball trees against scikit-learn, the vantage-point tree against brute force (exact indices and distances
 * in three metrics), the trees' pruning, LSH collision laws and recall, IVF/PQ/OPQ/HNSW/NN-descent recall against
 * brute force, codebooks and the recall metric.
 */
import { describe, expect, it } from 'vitest'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  assignNearest,
  ballTree,
  benchmarkSearch,
  bruteForceNeighbours,
  bruteForceQuery,
  hnswIndex,
  hnswQuery,
  hnswSearch,
  hyperplaneFamily,
  ivfIndex,
  ivfQuery,
  ivfSearch,
  kdTree,
  kmeansPlusPlus,
  lloydUpdate,
  lshCollisionProbability,
  lshIndex,
  lshQuery,
  lshSearch,
  lshSignatures,
  nearestNeighbourDescent,
  nodeLowerBound,
  optimisedProductQuantiser,
  pqDecode,
  pqDistanceTable,
  pqEncode,
  pqQuery,
  pqSearch,
  productQuantiser,
  pStableFamily,
  searchRecall,
  trainCodebook,
  treeQuery,
  treeSearch,
  vpTree,
  type TreeMetric,
} from 'aifn-compute/numerics/neighbours'
import { fixture } from '../../fixtures'

type Answer = { indices: number[][]; distances: number[][] }
const F = fixture('numerics/neighbours') as {
  data: number[][]
  queries: number[][]
  k: number
  self: Answer
} & Record<TreeMetric, { kdTree: Answer; ballTree: Answer; brute: Answer }>

const rows = (t: Tensor) => toRows(t) as number[][]

/** Gaussian clusters in d dimensions: a cheap stand-in for embedding data. */
function clusters(seed: string, n: number, d: number, centres = 8): Tensor {
  const s = stream(seed)
  const c = toFlat(normal(child(s, 'centres'), 0, 3, { shape: [centres, d] }))
  const z = toFlat(normal(child(s, 'noise'), 0, 1, { shape: [n, d] }))
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    const j = Math.floor(uniform(child(s, 'pick', i)) * centres)
    for (let t = 0; t < d; t++) out[i * d + t] = c[j * d + t] + z[i * d + t]
  }
  return fromData(out, [n, d])
}

describe('exact search against scikit-learn', () => {
  for (const metric of ['euclidean', 'manhattan', 'chebyshev'] as const) {
    it(`brute force, k-d tree and ball tree agree with scikit-learn (${metric})`, () => {
      const want = F[metric].brute
      const brute = bruteForceNeighbours(F.data, F.queries, F.k, { metric })
      expect(rows(brute.indices)).toEqual(want.indices)
      rows(brute.distances).forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(want.distances[i][j], 12)))
      for (const [name, build] of [
        ['kdTree', kdTree],
        ['ballTree', ballTree],
      ] as const) {
        const tree = build(F.data, { leafSize: 10, metric })
        const got = treeSearch(tree, F.queries, F.k)
        expect(rows(got.indices)).toEqual(F[metric][name].indices)
        rows(got.distances).forEach((r, i) =>
          r.forEach((v, j) => expect(v).toBeCloseTo(F[metric][name].distances[i][j], 12)),
        )
        // The tree computes fewer distances than the scan.
        expect(got.distanceEvaluations).toBeLessThan(brute.distanceEvaluations)
      }
    })
  }
  for (const metric of ['euclidean', 'sqeuclidean', 'manhattan', 'chebyshev'] as const) {
    it(`the vantage-point tree returns brute force's answer with fewer distances (${metric})`, () => {
      const brute = bruteForceNeighbours(F.data, F.queries, F.k, { metric })
      const got = treeSearch(vpTree(F.data, { leafSize: 10, metric }), F.queries, F.k)
      expect(rows(got.indices)).toEqual(rows(brute.indices))
      rows(got.distances).forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(rows(brute.distances)[i][j], 12)))
      expect(got.distanceEvaluations).toBeLessThan(brute.distanceEvaluations)
    })
  }
  it('excludeSelf gives the k-NN graph of the data', () => {
    const got = bruteForceNeighbours(F.data, F.data, F.k, { excludeSelf: true })
    expect(rows(got.indices)).toEqual(F.self.indices)
  })

  it('a zero row (NaN cosine distance) ranks last and never displaces a real neighbour', () => {
    // Row 0 is zero, so its cosine distance to the query is NaN; offered first, it used to stay at the head.
    const data = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]
    const got = bruteForceNeighbours(data, [[1, 0.1]], 2, { metric: 'cosine' })
    expect(rows(got.indices)).toEqual([[1, 3]])
    const all = bruteForceNeighbours(data, [[1, 0.1]], 4, { metric: 'cosine' })
    expect(rows(all.indices)[0][3]).toBe(0)
    expect(Number.isNaN(toFlat(all.distances)[3])).toBe(true)
  })
})

describe('trees', () => {
  it('partition the points, keep leaves small and bound every point of a node from below', () => {
    for (const build of [kdTree, ballTree, vpTree]) {
      const tree = build(F.data, { leafSize: 5 })
      expect([...tree.order].sort((a, b) => a - b)).toEqual(Array.from({ length: tree.n }, (_, i) => i))
      for (const [id, nd] of tree.nodes.entries()) {
        if (nd.left < 0) expect(nd.end - nd.start).toBeLessThanOrEqual(5)
        else {
          expect(tree.nodes[nd.left].start).toBe(nd.start)
          expect(tree.nodes[nd.right].end).toBe(nd.end)
        }
        const q = F.queries[id % F.queries.length]
        const bound = nodeLowerBound(tree, id, q)
        for (let i = nd.start; i < nd.end; i++) {
          const p = F.data[tree.order[i]]
          const dist = Math.hypot(...p.map((v, c) => v - q[c]))
          expect(bound).toBeLessThanOrEqual(dist + 1e-12)
        }
      }
    }
  })
  it('records a visit order that scans nearer leaves first and prunes the rest', () => {
    const tree = kdTree(F.data, { leafSize: 8 })
    const r = treeQuery(tree, F.queries[0], 3)
    expect(r.visits[0].node).toBe(0)
    expect(r.visits.some((v) => v.action === 'prune')).toBe(true)
    const scanned = r.visits.filter((v) => v.action === 'scan')
    expect(r.distanceEvaluations).toBe(
      scanned.reduce((s, v) => s + tree.nodes[v.node].end - tree.nodes[v.node].start, 0),
    )
    // A pruned node's bound exceeded the k-th best distance held then.
    for (const v of r.visits) if (v.action === 'prune') expect(v.bound).toBeGreaterThan(v.worst)
    expect(r.indices).toEqual(bruteForceQuery(F.data, F.queries[0], 3).indices)
  })
})

describe('locality-sensitive hashing', () => {
  it('random hyperplanes collide with probability 1 − θ/π', () => {
    const fam = hyperplaneFamily(2, { tables: 1, hashesPerTable: 4000, stream: stream('hp') })
    const theta = 1
    const sig = toRows(
      lshSignatures(fam, [
        [1, 0],
        [Math.cos(theta), Math.sin(theta)],
      ]),
    ) as number[][]
    const agree = sig[0].filter((b, j) => b === sig[1][j]).length / 4000
    expect(agree).toBeCloseTo(lshCollisionProbability(fam, theta), 1)
    expect(Math.abs(agree - (1 - theta / Math.PI))).toBeLessThan(0.03)
  })
  it('p-stable hashes collide with the probability of Datar et al.', () => {
    const w = 2
    const fam = pStableFamily(3, { tables: 1, hashesPerTable: 6000, width: w, stream: stream('ps') })
    for (const c of [0.5, 1.5, 4]) {
      const sig = toRows(
        lshSignatures(fam, [
          [0, 0, 0],
          [c, 0, 0],
        ]),
      ) as number[][]
      const agree = sig[0].filter((b, j) => b === sig[1][j]).length / 6000
      expect(Math.abs(agree - lshCollisionProbability(fam, c))).toBeLessThan(0.03)
    }
    expect(lshCollisionProbability(fam, 0)).toBe(1)
  })
  it('answers exactly over its candidates, and more tables raise recall', () => {
    const all = toRows(clusters('lsh', 640, 8)) as number[][]
    const x = all.slice(0, 600)
    const q = all.slice(600)
    const exact = bruteForceNeighbours(x, q, 5)
    const recall = (tables: number) => {
      const idx = lshIndex(x, pStableFamily(8, { tables, hashesPerTable: 4, width: 6, stream: stream('t') }))
      return searchRecall(lshSearch(idx, q, 5), exact)
    }
    const r1 = recall(1)
    const r8 = recall(10)
    expect(r8).toBeGreaterThan(r1)
    expect(r8).toBeGreaterThan(0.8)
    const idx = lshIndex(x, hyperplaneFamily(8, { tables: 4, hashesPerTable: 6, stream: stream('h') }))
    const one = lshQuery(idx, q[0], 3)
    expect(one.buckets.length).toBe(4)
    expect(one.distanceEvaluations).toBe(one.candidates.length)
    for (const i of one.indices) expect(one.candidates).toContain(i)
  })
})

describe('codebooks', () => {
  it('k-means++ is reproducible and Lloyd never raises the quantisation error', () => {
    const x = clusters('cb', 300, 2, 4)
    const a = kmeansPlusPlus(stream(4), x, 4)
    expect(toFlat(kmeansPlusPlus(stream(4), x, 4).indices)).toEqual(toFlat(a.indices))
    let c = a.centroids
    let prev = assignNearest(x, c)
    for (let t = 0; t < 10; t++) {
      c = lloydUpdate(x, prev.labels, c).centroids
      const next = assignNearest(x, c)
      expect(next.inertia).toBeLessThanOrEqual(prev.inertia + 1e-9)
      prev = next
    }
    const book = trainCodebook(x, 4, { stream: stream(1) })
    expect(book.inertia).toBeLessThanOrEqual(assignNearest(x, a.centroids).inertia)
    expect(toFlat(book.sizes).reduce((s, v) => s + v, 0)).toBe(300)
    // A warm start must have one codeword per cluster: k > n trains n codewords, so k initial rows are refused.
    expect(() =>
      trainCodebook(
        [
          [0, 0],
          [1, 1],
        ],
        3,
        {
          stream: stream(1),
          initial: [
            [0, 0],
            [1, 1],
            [2, 2],
          ],
        },
      ),
    ).toThrow(/expected 2 × 2/)
  })
})

describe('approximate indexes against brute force', () => {
  const all = toRows(clusters('ann', 1550, 8)) as number[][]
  const x = all.slice(0, 1500)
  const q = all.slice(1500)
  const k = 10
  const exact = bruteForceNeighbours(x, q, k)

  it('IVF: one probe misses some, all probes are exact, and probing more never lowers recall', () => {
    const ivf = ivfIndex(x, { lists: 16, stream: stream(2) })
    expect(ivf.lists.flat().length).toBe(1500)
    const r1 = searchRecall(ivfSearch(ivf, q, k, { probes: 1 }), exact)
    const r4 = searchRecall(ivfSearch(ivf, q, k, { probes: 4 }), exact)
    const all = ivfSearch(ivf, q, k, { probes: 16 })
    expect(r4).toBeGreaterThanOrEqual(r1)
    expect(r4).toBeGreaterThan(0.9)
    expect(rows(all.indices)).toEqual(rows(exact.indices))
    const one = ivfQuery(ivf, q[0], k, { probes: 2 })
    expect(one.probed.length).toBe(2)
  })
  it('PQ: ADC equals the distance to the decoded vector; OPQ lowers the distortion', () => {
    const pq = productQuantiser(x, { subspaces: 4, codewords: 16, stream: stream(3) })
    const codes = pqEncode(pq, x)
    expect(codes.shape).toEqual([1500, 4])
    const decoded = toRows(pqDecode(pq, codes)) as number[][]
    const q0 = q[0]
    const r = pqQuery(pq, codes, q0, 5)
    for (const [j, i] of r.indices.entries())
      expect(r.distances[j]).toBeCloseTo(Math.hypot(...decoded[i].map((v, c) => v - q0[c])), 9)
    expect(pqDistanceTable(pq, q0).shape).toEqual([4, 16])
    expect(searchRecall(pqSearch(pq, codes, q, k), exact)).toBeGreaterThan(0.3)
    // A correlated cloud: a rotation spreads the variance across sub-spaces, so OPQ beats PQ.
    const z = toRows(clusters('opq', 600, 4, 3)) as number[][]
    const skew = z.map(([a, b, c, d]) => [a + b + c + d, a - b, 0.1 * c, 0.1 * d])
    const opq = optimisedProductQuantiser(skew, { subspaces: 2, codewords: 8, rounds: 8, stream: stream(5) })
    for (let r = 1; r < opq.distortions.length; r++)
      expect(opq.distortions[r]).toBeLessThanOrEqual(opq.distortions[r - 1] + 1e-9)
    expect(opq.distortions.at(-1)!).toBeLessThan(0.9 * opq.distortions[0])
    const R = toRows(opq.rotation!) as number[][]
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) expect(R.reduce((s, row) => s + row[i] * row[j], 0)).toBeCloseTo(i === j ? 1 : 0, 9)
    const opqCodes = pqEncode(opq, skew)
    const back = toRows(pqDecode(opq, opqCodes)) as number[][]
    const mse = back.reduce((s, r, i) => s + r.reduce((t, v, c) => t + (v - skew[i][c]) ** 2, 0), 0) / 600
    expect(mse).toBeCloseTo(opq.distortion, 6)
  })
  it('HNSW: high recall at a fraction of the distances, with a greedy descent through the layers', () => {
    const index = hnswIndex(x, { M: 8, efConstruction: 64, stream: stream(6) })
    expect(index.topLayer).toBeGreaterThan(0)
    for (let l = 0; l <= index.topLayer; l++)
      for (let i = 0; i < index.n; i++) {
        if (index.levels[i] < l) expect(index.links[l][i].length).toBe(0)
        expect(index.links[l][i].length).toBeLessThanOrEqual(l === 0 ? 16 : 8)
      }
    const found = hnswSearch(index, q, k, { ef: 64 })
    expect(searchRecall(found, exact)).toBeGreaterThan(0.95)
    expect(found.distanceEvaluations / 50).toBeLessThan(1500 / 2)
    const r = hnswQuery(index, q[0], k, { ef: 32 })
    expect(r.layers.map((l) => l.layer)).toEqual(
      Array.from({ length: index.topLayer + 1 }, (_, i) => index.topLayer - i),
    )
    expect(r.layers[0].expanded[0]).toBe(index.entry)
    // Each layer starts where the one above ended.
    for (let i = 1; i < r.layers.length; i++) expect(r.layers[i].expanded[0]).toBe(r.layers[i - 1].nearest)
  })
  it('NN-descent: the k-NN graph with high recall, sorted lists, no self', () => {
    const found = nearestNeighbourDescent(x, k, { stream: stream(1) })
    const graph = bruteForceNeighbours(x, x, k, { excludeSelf: true })
    expect(searchRecall(found, graph)).toBeGreaterThan(0.9)
    const I = rows(found.indices)
    const D = rows(found.distances)
    I.forEach((r, i) => {
      expect(r.includes(i)).toBe(false)
      expect(new Set(r).size).toBe(k)
      for (let j = 1; j < k; j++) expect(D[i][j]).toBeGreaterThanOrEqual(D[i][j - 1])
    })
    expect(found.updates.length).toBe(found.rounds)
  })
})

describe('measurement', () => {
  it('recall@k counts the overlap with the exact answer, and the benchmark reports it', () => {
    const exact = fromData(Int32Array.from([0, 1, 2, 3, 4, 5]), [2, 3])
    const found = fromData(Int32Array.from([2, 0, 9, -1, 4, 5]), [2, 3])
    expect(searchRecall(found, exact)).toBeCloseTo(4 / 6, 12)
    const x = clusters('bench', 200, 3)
    const ex = bruteForceNeighbours(x, x, 3)
    const b = benchmarkSearch(() => treeSearch(kdTree(x), x, 3), ex, { repeats: 1 })
    expect(b.recall).toBe(1)
    expect(b.queriesPerSecond).toBeGreaterThan(0)
    expect(b.distancesPerQuery).toBeGreaterThan(0)
  })
})
