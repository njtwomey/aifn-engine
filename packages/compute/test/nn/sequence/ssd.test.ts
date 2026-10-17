/**
 * aifn-compute/nn/sequence SSD: the Mamba-2 note's six-token mask and outputs, the recurrence and the chunked algorithm
 * against the quadratic form for every chunk length, `selectiveScan` with a scalar transition against the quadratic
 * form, a zero decay (exact zeros, no NaN), the domain check, and gradients in the decays.
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { mul, sum, tensor, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { selectiveScan, semiseparableMask, ssdChunked, ssdQuadratic, ssdRecurrent } from 'aifn-compute/nn/sequence'

const rows = (v: Value) => toRows(v as Tensor)
const maxDiff = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((r, i) => r.map((v, j) => Math.abs(v - b[i][j]))))

describe('SSD', () => {
  const a = [0.9, 0.5, 1, 0.8, 0.2, 0.9]
  const ones = a.map(() => [1])
  const x = [[1], [0], [2], [0], [0], [1]]

  it('builds the worked example mask and outputs y = (1, 0.5, 2.5, 2, 0.4, 1.36)', () => {
    const want = [
      [1, 0, 0, 0, 0, 0],
      [0.5, 1, 0, 0, 0, 0],
      [0.5, 1, 1, 0, 0, 0],
      [0.4, 0.8, 0.8, 1, 0, 0],
      [0.08, 0.16, 0.16, 0.2, 1, 0],
      [0.072, 0.144, 0.144, 0.18, 0.9, 1],
    ]
    expect(maxDiff(rows(semiseparableMask(a)), want)).toBeLessThan(1e-12)
    const y = rows(ssdQuadratic(x, a, ones, ones))
    expect(maxDiff(y, [[1], [0.5], [2.5], [2], [0.4], [1.36]])).toBeLessThan(1e-12)
    const chunked = ssdChunked(x, a, ones, ones, 3)
    expect(maxDiff(rows(chunked.y), y)).toBeLessThan(1e-12)
    expect(chunked.cost).toEqual({ intraChunk: 24, stateProducts: 12, sequentialSteps: 2 })
  })

  it('agrees three ways and with selectiveScan on random data (T = 24, N = 4, P = 2)', () => {
    const r = stream(4)
    const [T, N, P] = [24, 4, 2]
    const av = Array.from({ length: T }, () => 0.5 + 0.5 * uniform(r))
    const xv = Array.from({ length: T }, () => Array.from({ length: P }, () => normal(r)))
    const Bv = Array.from({ length: T }, () => Array.from({ length: N }, () => normal(r)))
    const Cv = Array.from({ length: T }, () => Array.from({ length: N }, () => normal(r)))
    const q = rows(ssdQuadratic(xv, av, Bv, Cv))
    expect(maxDiff(rows(ssdRecurrent(xv, av, Bv, Cv).y), q)).toBeLessThan(1e-10)
    let worst = 0
    for (let Q = 1; Q <= T; Q++) worst = Math.max(worst, maxDiff(rows(ssdChunked(xv, av, Bv, Cv, Q).y), q))
    expect(worst).toBeLessThan(1e-10)
    expect(maxDiff(rows(ssdChunked(xv, av, Bv, Cv, 5).final), rows(ssdRecurrent(xv, av, Bv, Cv).final))).toBeLessThan(
      1e-10,
    )
    // Δ_t = −log a_t for every channel, A = −1 and B_t/Δ_t (the Euler B̄ = ΔB undoes the division).
    const delta = av.map((v) => -Math.log(v))
    const ys = selectiveScan(
      tensor(xv),
      tensor(delta.map((d) => Array<number>(P).fill(d))),
      tensor(Array.from({ length: P }, () => Array<number>(N).fill(-1))),
      tensor(Bv.map((row, t) => row.map((b) => b / delta[t]))),
      tensor(Cv),
    ).outputs
    expect(maxDiff(rows(ys), q)).toBeLessThan(1e-9)
  })

  it('cuts the mask to exact zeros at a zero decay, and rejects a negative one', () => {
    const L = rows(semiseparableMask([0.9, 0, 0.5]))
    expect(L).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0.5, 1],
    ])
    const one = [[1], [1], [1]]
    expect(rows(ssdQuadratic(one, [0.9, 0, 0.5], one, one)).flat()).toEqual([1, 1, 1.5])
    expect(() => semiseparableMask([0.5, -0.1])).toThrow(/non-negative/)
  })

  it('differentiates the quadratic form in the decays', () => {
    const w = tensor([[1], [-1], [2], [0.5], [1], [-2]])
    const f = (av: Value) => sum(mul(ssdQuadratic(x, av, ones, ones), w))
    expect(gradCheck(f, tensor(a), { rtol: 1e-5, atol: 1e-7 }).ok).toBe(true)
  })
})
