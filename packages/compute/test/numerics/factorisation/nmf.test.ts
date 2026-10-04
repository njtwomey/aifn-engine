/**
 * NMF by multiplicative updates against scikit-learn's `NMF(solver='mu')` from the same initial factors
 * (`fixtures/numerics/factorisation.json`), and the laws: the objective never increases and the factors stay
 * non-negative.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { nmf, nmfSteps, type NmfLoss } from 'aifn-compute/numerics/factorisation'
import { fixture } from '../../fixtures'

type Run = { loss: NmfLoss; steps: number; W: number[][]; H: number[][]; objective: number }
const F = fixture<{ x: number[][]; W0: number[][]; H0: number[][]; runs: Run[]; hals: Omit<Run, 'loss'>[] }>(
  'numerics/factorisation',
)

const close = (got: ArrayLike<number>, want: number[], rtol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w)).toBeLessThanOrEqual(rtol * Math.abs(w) + 1e-12))

describe('nmfSteps', () => {
  const X = fromRows(F.x)
  for (const r of F.runs)
    it(`matches scikit-learn after ${r.steps} sweep(s), ${r.loss}`, () => {
      const s = run(
        nmfSteps(X, { rank: 3, loss: r.loss, init: { W: F.W0, H: F.H0 }, tolerance: 0 }),
        undefined,
        r.steps,
      )
      expect(s.t).toBe(r.steps)
      close(toFlat(s.W), r.W.flat(), 1e-8)
      close(toFlat(s.H), r.H.flat(), 1e-8)
      expect(s.objective).toBeCloseTo(r.objective, 8)
    })

  for (const loss of ['frobenius', 'kullback-leibler'] as const)
    it(`never increases the ${loss} objective and keeps the factors non-negative`, () => {
      const t = trace(nmfSteps(X, { rank: 2, loss, tolerance: 0 }), undefined, 60, { stream: stream(4) })
      const obj = t.steps.map((s) => s.objective)
      for (let i = 1; i < obj.length; i++) expect(obj[i]).toBeLessThanOrEqual(obj[i - 1] * (1 + 1e-12))
      const last = t.final
      for (const v of [...toFlat(last.W), ...toFlat(last.H)]) expect(v).toBeGreaterThanOrEqual(0)
    })

  it('recovers an exactly low-rank non-negative matrix', () => {
    const W = [
      [1, 0],
      [0, 2],
      [1, 1],
      [3, 0],
    ]
    const H = [
      [1, 2, 0, 1],
      [0, 1, 3, 1],
    ]
    const Xr = W.map((w) => H[0].map((_, j) => w[0] * H[0][j] + w[1] * H[1][j]))
    const r = nmf(Xr, { rank: 2, maxSteps: 3000, tolerance: 1e-14, stream: stream(1) })
    expect(r.objective).toBeLessThan(1e-4)
  })

  for (const r of F.hals)
    it(`HALS matches scikit-learn's coordinate descent after ${r.steps} sweep(s)`, () => {
      const s = run(
        nmfSteps(X, { rank: 3, solver: 'hals', init: { W: F.W0, H: F.H0 }, tolerance: 0 }),
        undefined,
        r.steps,
      )
      close(toFlat(s.W), r.W.flat(), 1e-8)
      close(toFlat(s.H), r.H.flat(), 1e-8)
      expect(s.objective).toBeCloseTo(r.objective, 8)
    })

  it('HALS never increases the objective and reaches a lower one than MU in the same sweeps', () => {
    const t = trace(nmfSteps(X, { rank: 2, solver: 'hals', tolerance: 0 }), undefined, 30, { stream: stream(4) })
    const obj = t.steps.map((s) => s.objective)
    for (let i = 1; i < obj.length; i++) expect(obj[i]).toBeLessThanOrEqual(obj[i - 1] * (1 + 1e-12))
    const mu = run(nmfSteps(X, { rank: 2, tolerance: 0 }), undefined, 30, { stream: stream(4) })
    expect(t.final.objective).toBeLessThanOrEqual(mu.objective)
    expect(() => nmfSteps(X, { rank: 2, solver: 'hals', loss: 'kullback-leibler' })).toThrow()
  })

  it('rejects negative entries', () => {
    expect(() => nmfSteps([[1, -1]], { rank: 1 })).toThrow()
  })
})
