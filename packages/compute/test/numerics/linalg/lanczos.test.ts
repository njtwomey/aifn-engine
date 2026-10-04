import { describe, expect, it } from 'vitest'
import { eigh, eigsh, type EigshWhich } from 'aifn-compute/numerics/linalg'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Case = {
  n: number
  rows: number[]
  cols: number[]
  vals: number[]
} & Record<EigshWhich, { values: number[]; vectors: number[][] }>
const F = fixture<{ eigsh: Record<string, Case> }>('numerics/linalg').eigsh

/** A matrix-free product from the fixture's coordinate list. */
function operator(c: Case) {
  let products = 0
  const matvec = (v: Vector) => {
    products++
    const x = toFlat(v)
    const out = new Float64Array(c.n)
    for (let e = 0; e < c.rows.length; e++) out[c.rows[e]] += c.vals[e] * x[c.cols[e]]
    return out
  }
  return { matvec, count: () => products }
}

describe('eigsh (thick-restart Lanczos)', () => {
  for (const key of Object.keys(F))
    it.each(['largest', 'smallest', 'magnitude'] as const)(
      `matches scipy eigsh (ARPACK) on ${key}, which = %s`,
      (which) => {
        const c = F[key]
        const op = operator(c)
        const r = eigsh(op.matvec, c.n, { k: 5, which })
        expect(r.converged).toBe(true)
        toFlat(r.values).forEach((v, i) => expect(Math.abs(v - c[which].values[i])).toBeLessThan(1e-8))
        // Eigenvectors up to sign; clustered eigenvalues (the Laplacian's) are compared through their residuals.
        const V = toFlat(r.vectors)
        for (let j = 0; j < 5; j++) {
          const x = Array.from({ length: c.n }, (_, i) => V[i * 5 + j])
          const Ax = Array.from(op.matvec(fromData(Float64Array.from(x), [c.n])))
          const res = Math.hypot(...Ax.map((v, i) => v - r.values.data[j] * x[i]))
          expect(res).toBeLessThan(1e-7)
          const ref = c[which].vectors.map((row) => row[j])
          const dot = x.reduce((s, v, i) => s + v * ref[i], 0)
          expect(Math.abs(Math.abs(dot) - 1)).toBeLessThan(1e-6)
        }
        expect(r.products).toBe(op.count() - 5)
      },
    )

  it('agrees with dense eigh on a small matrix, and takes a dense matrix as well as a function', () => {
    const n = 40
    const a = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        const v = Math.sin(i * 7 + j * 3) + (i === j ? i / 4 : 0)
        a[i * n + j] = v
        a[j * n + i] = v
      }
    const A = Array.from({ length: n }, (_, i) => Array.from(a.subarray(i * n, (i + 1) * n)))
    const dense = toFlat(eigh(fromData(a, [n, n])).values)
    const top = eigsh(A, n, { k: 3 })
    toFlat(top.values).forEach((v, i) => expect(v).toBeCloseTo(dense[i], 9))
    const bottom = eigsh(A, n, { k: 3, which: 'smallest', basis: 10 })
    toFlat(bottom.values).forEach((v, i) => expect(v).toBeCloseTo(dense[n - 1 - i], 9))
    expect(bottom.restarts).toBeGreaterThan(0)
  })

  it('handles an invariant starting subspace and repeated eigenvalues', () => {
    // diag(5, 5, 5, 1, …): three copies of the top eigenvalue, found by restarting with fresh directions.
    const n = 30
    const d = Array.from({ length: n }, (_, i) => (i < 3 ? 5 : 1 + i / 100))
    const r = eigsh((v) => toFlat(v).map((x, i) => d[i] * x), n, { k: 3, start: stream('rep') })
    toFlat(r.values).forEach((v) => expect(v).toBeCloseTo(5, 10))
    const e1 = new Float64Array(n)
    e1[0] = 1
    const s = eigsh((v) => toFlat(v).map((x, i) => d[i] * x), n, { k: 2, start: e1 })
    expect(toFlat(s.values)[0]).toBeCloseTo(5, 10)
  })

  it('rejects k outside [1, n)', () => {
    expect(() => eigsh([[1]], 1, { k: 1 })).toThrow()
    expect(() =>
      eigsh(
        [
          [1, 0],
          [0, 2],
        ],
        2,
        { k: 2 },
      ),
    ).toThrow(/k must/)
  })
})
