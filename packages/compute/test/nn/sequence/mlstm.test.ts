/**
 * aifn-compute/nn/sequence `mlstmRecurrent` and `mlstmParallel`: the xLSTM note's three-step worked example, both forms
 * against an unstabilised float64 loop, and finite, equal outputs when the gates are far beyond float64's range.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { add, mul, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { mlstmParallel, mlstmRecurrent } from 'aifn-compute/nn/sequence'

const flat = (v: Value) => toFlat(v as Tensor)
const maxError = (got: number[], want: number[]) => Math.max(...got.map((g, i) => Math.abs(g - want[i])))

function naive(q: number[], k: number[], v: number[], iPre: number[], fPre: number[], T: number, d: number) {
  const C = new Array<number>(d * d).fill(0)
  const n = new Array<number>(d).fill(0)
  const out: number[] = []
  for (let t = 0; t < T; t++) {
    const f = 1 / (1 + Math.exp(-fPre[t]))
    const i = Math.exp(iPre[t])
    for (let a = 0; a < d; a++) {
      n[a] = f * n[a] + i * k[t * d + a]
      for (let b = 0; b < d; b++) C[a * d + b] = f * C[a * d + b] + i * v[t * d + a] * k[t * d + b]
    }
    let nq = 0
    for (let b = 0; b < d; b++) nq += n[b] * q[t * d + b]
    const den = Math.max(Math.abs(nq), 1)
    for (let a = 0; a < d; a++) {
      let s = 0
      for (let b = 0; b < d; b++) s += C[a * d + b] * q[t * d + b]
      out.push(s / den)
    }
  }
  return out
}

describe('mLSTM', () => {
  it('reproduces the worked example: h_3 = (12.5, 4)/4.5 and m = (0, 0, ln 4) in both forms', () => {
    const e = tensor([
      [1, 0],
      [0, 1],
      [1, 0],
    ])
    const v = tensor([
      [1, 0],
      [0, 2],
      [3, 1],
    ])
    const iPre = tensor([0, 0, Math.log(4)])
    const fPre = tensor([0, 0, Math.log(0.5)])
    const want = [1, 0, 0, 2, 12.5 / 4.5, 4 / 4.5]
    const rec = mlstmRecurrent(e, e, v, iPre, fPre, { forget: 'exp' })
    const par = mlstmParallel(e, e, v, iPre, fPre, { forget: 'exp' })
    expect(maxError(flat(rec.outputs), want)).toBeLessThan(1e-12)
    expect(maxError(flat(par.outputs), want)).toBeLessThan(1e-12)
    expect(maxError(flat(rec.stabiliser), [0, 0, Math.log(4)])).toBeLessThan(1e-12)
    expect(maxError(flat(par.stabiliser), [0, 0, Math.log(4)])).toBeLessThan(1e-12)
  })

  it('both forms equal an unstabilised loop with sigmoid forget gates (T = 20)', () => {
    const T = 20
    const d = 3
    const s = stream(5)
    const q = normals(s, [T, d])
    const k = normals(s, [T, d])
    const v = normals(s, [T, d])
    const iPre = normals(s, [T], 0, 2)
    const fPre = add(normals(s, [T]), 2)
    const want = naive(toFlat(q), toFlat(k), toFlat(v), toFlat(iPre), flat(fPre), T, d)
    expect(maxError(flat(mlstmRecurrent(q, k, v, iPre, fPre).outputs), want)).toBeLessThan(1e-10)
    expect(maxError(flat(mlstmParallel(q, k, v, iPre, fPre).outputs), want)).toBeLessThan(1e-10)
  })

  it('stays finite with exponential gates near e^20, and the forms agree', () => {
    const T = 64
    const d = 4
    const s = stream(9)
    const q = normals(s, [T, d])
    const k = normals(s, [T, d])
    const v = normals(s, [T, d])
    const iPre = add(normals(s, [T]), 30)
    const fPre = add(mul(normals(s, [T]), 0.1), 20)
    const rec = mlstmRecurrent(q, k, v, iPre, fPre, { forget: 'exp' })
    const par = mlstmParallel(q, k, v, iPre, fPre, { forget: 'exp' })
    expect(flat(rec.stabiliser)[T - 1]).toBeGreaterThan(709)
    const r = flat(rec.outputs)
    expect(r.every(Number.isFinite)).toBe(true)
    expect(maxError(r, flat(par.outputs))).toBeLessThan(1e-9)
  })
})
