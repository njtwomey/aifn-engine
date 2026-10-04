/** The ANN benchmark: every method reports a recall curve that rises with its knob, against exact brute force. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { blobs } from 'aifn-methods/data/synthetic'
import { annBenchmark, type AnnBenchmarkSnapshot } from 'aifn-methods/retrieval/ann'
import type { Tensor } from 'aifn-compute/foundation/tensor'

describe('annBenchmark', () => {
  it('streams each method, with recall in [0, 1] that never falls as the knob loosens (IVF, HNSW, LSH)', () => {
    const data = blobs(stream('ann'), { n: 700, centers: 6, dim: 8 })
    let final: AnnBenchmarkSnapshot | undefined
    let yields = 0
    for (const s of annBenchmark({ data: { x: data.x as Tensor }, queries: 50, k: 5 })) {
      final = s
      yields++
    }
    expect(yields).toBe(7)
    expect(final!.done).toBe(true)
    expect(final!.curves.map((c) => c.method)).toEqual(['kd-tree', 'lsh', 'ivf', 'pq', 'hnsw'])
    for (const c of final!.curves) {
      for (const p of c.points) expect(p.recall >= 0 && p.recall <= 1).toBe(true)
      if (c.method === 'kd-tree') for (const p of c.points) expect(p.recall).toBe(1)
      if (c.method === 'ivf' || c.method === 'hnsw' || c.method === 'lsh')
        for (let i = 1; i < c.points.length; i++)
          expect(c.points[i].recall).toBeGreaterThanOrEqual(c.points[i - 1].recall)
    }
    expect(final!.curves.find((c) => c.method === 'hnsw')!.points.at(-1)!.recall).toBeGreaterThan(0.95)
  })
})
