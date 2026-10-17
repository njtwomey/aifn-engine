/** aifn-methods/unsupervised/embedding/graph: `netmfMatrix` against its closed form and against sampled walks. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import { ppmi, truncatedSvd, wordVectors } from 'aifn-compute/text/cooccurrence'
import { netmfMatrix, node2vecWalks, walkCooccurrence } from 'aifn-methods/unsupervised/embedding/graph'

const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t))

/** $\sum_{r=1}^{T} \Pmat^r$ for a row-major $n \times n$ matrix, by plain loops. */
function powerSum(P: number[], n: number, T: number): number[] {
  const sum = new Array<number>(n * n).fill(0)
  let power = P.slice()
  for (let r = 1; r <= T; r++) {
    for (let k = 0; k < n * n; k++) sum[k] += power[k]
    const next = new Array<number>(n * n).fill(0)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) next[i * n + j] += power[i * n + k] * P[k * n + j]
    power = next
  }
  return sum
}

// A triangle with a tail: connected, not bipartite and not regular.
const kite = fromEdges(
  4,
  [
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 3],
  ],
  { directed: false },
)

describe('netmfMatrix', () => {
  it('on a regular graph is n / (bT) times the sum of the powers of P (vol = nd, D = dI)', () => {
    const n = 5
    const cycle = fromEdges(
      n,
      Array.from({ length: n }, (_, i) => [i, (i + 1) % n] as [number, number]),
      { directed: false },
    )
    const P = new Array<number>(n * n).fill(0)
    for (let i = 0; i < n; i++) P[i * n + ((i + 1) % n)] = P[i * n + ((i + n - 1) % n)] = 0.5
    for (const [T, b] of [
      [1, 1],
      [3, 2],
      [5, 1],
    ]) {
      const sum = powerSum(P, n, T)
      const M = flat(netmfMatrix(cycle, { window: T, negatives: b, raw: true }))
      M.forEach((v, k) => expect(v).toBeCloseTo((n / (b * T)) * sum[k], 12))
    }
  })

  it('is symmetric on an undirected graph, and the default is log max(M, 1)', () => {
    const raw = flat(netmfMatrix(kite, { window: 3, raw: true }))
    const log = flat(netmfMatrix(kite, { window: 3 }))
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(raw[i * 4 + j]).toBeCloseTo(raw[j * 4 + i], 12)
    log.forEach((v, k) => expect(v).toBeCloseTo(Math.log(Math.max(raw[k], 1)), 14))
  })

  it('is the limit of the shifted PPMI of long walks (Qiu et al. 2018, Theorem 2.3)', () => {
    const T = 2
    const walks = node2vecWalks(stream(11), kite, { walksPerNode: 1, length: 25000 })
    const sampled = flat(ppmi(walkCooccurrence(walks, 4, T), { shift: 1 }))
    const exact = flat(netmfMatrix(kite, { window: T }))
    sampled.forEach((v, k) => expect(Math.abs(v - exact[k])).toBeLessThan(0.1))
  })

  it('ppmi with shift b is the walk PMI max(log(#(w, c) |D| / (b #(w) #(c))), 0) of symmetric counts', () => {
    const counts = [
      [0, 6, 3, 1],
      [6, 2, 4, 0],
      [3, 4, 0, 5],
      [1, 0, 5, 8],
    ]
    const row = counts.map((r) => r.reduce((a, v) => a + v, 0))
    const total = row.reduce((a, v) => a + v, 0)
    for (const b of [1, 2, 5]) {
      const got = flat(ppmi(counts, { shift: b }))
      counts.forEach((r, i) =>
        r.forEach((v, j) => {
          const want = v > 0 ? Math.log(Math.max((v * total) / (b * row[i] * row[j]), 1)) : 0
          expect(got[i * 4 + j]).toBeCloseTo(want, 12)
        }),
      )
    }
  })

  it('the NetMF embedding U_d S_d^(1/2) is wordVectors', () => {
    const M = netmfMatrix(kite, { window: 2 })
    const { U, S } = truncatedSvd(M, 2)
    const u = flat(U)
    const s = flat(S)
    const v = flat(wordVectors(M, 2))
    v.forEach((x, k) => expect(x).toBeCloseTo(u[k] * Math.sqrt(s[k % 2]), 12))
  })

  it('refuses a node of degree zero and a bad window', () => {
    const isolated = fromEdges(3, [[0, 1]], { directed: false })
    expect(() => netmfMatrix(isolated)).toThrow(/at least one edge/)
    expect(() => netmfMatrix(kite, { window: 0 })).toThrow(/window/)
    expect(() => netmfMatrix(kite, { negatives: 0 })).toThrow(/negatives/)
  })
})
