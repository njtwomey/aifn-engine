/**
 * aifn-compute/nn/sequence `deltaRule`: the delta-rule note's worked example, exact overwriting with orthonormal keys,
 * the gated form against the explicit matrix formula, and gradients in the keys under `grad`.
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { normals, stream } from 'aifn-compute/foundation/random'
import { eye, matmul, mul, slice, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { deltaRule } from 'aifn-compute/nn/sequence'

const maxError = (got: number[], want: number[]) => Math.max(...got.map((g, i) => Math.abs(g - want[i])))
const r = Math.SQRT1_2

describe('deltaRule', () => {
  it('reproduces the worked example: S_4 = (2.3284, 3.3284, 3) and exact reads', () => {
    const k = tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [r, r, 0],
    ])
    const { states, outputs } = deltaRule(k, k, tensor([[1], [2], [3], [4]]), 1)
    const S4 = toFlat(slice(states, 3) as Tensor)
    const e = 4 - 3 * r
    expect(maxError(S4, [1 + e * r, 2 + e * r, 3])).toBeLessThan(1e-12)
    expect(S4.map((x) => +x.toFixed(4))).toEqual([2.3284, 3.3284, 3])
    const keys = tensor([
      [1, 0, 0, r],
      [0, 1, 0, r],
      [0, 0, 1, 0],
    ])
    const reads = toFlat(matmul(slice(states, 3), keys) as Tensor)
    expect(maxError(reads, [1 + e * r, 2 + e * r, 3, 4])).toBeLessThan(1e-12)
    expect(toFlat(outputs as Tensor).at(-1)).toBeCloseTo(4, 12)
  })

  it('overwrites a rewritten key exactly with orthonormal keys and beta 1', () => {
    const k = tensor([
      [1, 0],
      [0, 1],
      [1, 0],
    ])
    const v = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    const { states } = deltaRule(k, k, v, 1)
    expect(maxError(toFlat(slice(states, 2) as Tensor), [5, 3, 6, 4])).toBeLessThan(1e-12)
  })

  it('matches S_t = alpha S_{t-1}(I - beta k kT) + beta v kT with per-step decay and strength', () => {
    const T = 10
    const d = 3
    const s = stream(4)
    const k = normals(s, [T, d])
    const v = normals(s, [T, d])
    const q = normals(s, [T, d])
    const alpha = [0.9, 0.8, 1, 0.95, 0.7, 1, 0.99, 0.5, 0.9, 0.85]
    const beta = [0.5, 0.2, 0.9, 0.3, 0.6, 0.4, 0.1, 0.8, 0.7, 0.5]
    const { outputs } = deltaRule(q, k, v, tensor(beta), { decay: tensor(alpha) })
    const [K, V, Q, I] = [toFlat(k), toFlat(v), toFlat(q), toFlat(eye(d))]
    let S = new Array<number>(d * d).fill(0)
    const want: number[] = []
    for (let t = 0; t < T; t++) {
      const kt = K.slice(t * d, t * d + d)
      const vt = V.slice(t * d, t * d + d)
      const qt = Q.slice(t * d, t * d + d)
      const M = I.map((x, idx) => x - beta[t] * kt[Math.floor(idx / d)] * kt[idx % d])
      const next = new Array<number>(d * d).fill(0)
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++) {
          let acc = 0
          for (let c = 0; c < d; c++) acc += S[a * d + c] * M[c * d + b]
          next[a * d + b] = alpha[t] * acc + beta[t] * vt[a] * kt[b]
        }
      S = next
      for (let a = 0; a < d; a++) want.push(S[a * d] * qt[0] + S[a * d + 1] * qt[1] + S[a * d + 2] * qt[2])
    }
    expect(maxError(toFlat(outputs as Tensor), want)).toBeLessThan(1e-12)
  })

  it('differentiates in the keys (traced shapes are read through shapeOfValue)', () => {
    const s = stream('delta-grad')
    const k = normals(s, [4, 2])
    const v = normals(s, [4, 2])
    const w = normals(s, [4, 2])
    const loss = (kk: Value) => sum(mul(deltaRule(kk, kk, v, 0.5).outputs, w))
    expect(gradCheck(loss, k, { rtol: 1e-5, atol: 1e-7 }).ok).toBe(true)
  })
})
