import { describe, expect, it } from 'vitest'
import { lms, nlms, rls, statisticalAlgorithms } from 'aifn-compute/signal/statistical'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'
import { fixture } from '../../fixtures'

type Run = { weights: Record<string, number[]>; errors: number[] }
type Fixture = {
  x: number[]
  d: number[]
  order: number
  plant: number[]
  leastSquares: number[]
  lms: Run & { stepSize: number }
  nlms: Run & { stepSize: number; epsilon: number }
  rls: Run & { forgetting: number; delta: number; P: number[][] }
}
const F = fixture<Fixture>('signal/statistical')

const close = (got: ArrayLike<number>, want: number[], tol: number) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `[${i}] ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol))
}

const filters = {
  lms: () => lms(F.x, F.d, { order: F.order, stepSize: F.lms.stepSize }),
  nlms: () => nlms(F.x, F.d, { order: F.order, stepSize: F.nlms.stepSize, epsilon: F.nlms.epsilon }),
  rls: () => rls(F.x, F.d, { order: F.order, forgetting: F.rls.forgetting, delta: F.rls.delta }),
}

describe('adaptive filters match a direct numpy run (system identification)', () => {
  for (const [name, make] of Object.entries(filters))
    it(`${name}: weights at every checkpoint and the a-priori errors`, () => {
      const ref = F[name as keyof typeof filters]
      const t = trace(make(), undefined, 10_000, { record: { e: (s) => s.e } })
      expect(t.meta.stopped).toBe('done')
      expect(t.final.t).toBe(F.x.length)
      close(toFlat(t.series.e).slice(1), ref.errors, 1e-10)
      for (const [step, w] of Object.entries(ref.weights))
        close(toFlat(run(make(), undefined, Number(step)).w), w, 1e-10)
    })
  it('every filter identifies the plant; RLS reaches the least-squares taps', () => {
    for (const make of Object.values(filters)) close(toFlat(run(make(), undefined, 10_000).w), F.plant, 0.01)
    const r = run(filters.rls(), undefined, 10_000)
    close(toFlat(r.w), F.leastSquares, 2e-3)
    toRows(r.P).forEach((row, i) => close(row, F.rls.P[i], 1e-8 * Math.max(...F.rls.P[i].map(Math.abs))))
  })
  it('RLS converges in far fewer samples than LMS on coloured input', () => {
    const error = (s: { w: { data: ArrayLike<number> } }) =>
      Math.hypot(...Array.from(toFlat(s.w as never), (v, i) => v - F.plant[i]))
    expect(error(run(filters.rls(), undefined, 50))).toBeLessThan(0.05)
    expect(error(run(filters.lms(), undefined, 50))).toBeGreaterThan(0.2)
  })
  it('a start w0 is used', () => {
    const s = run(lms(F.x, F.d, { order: F.order, stepSize: 0 }), { w0: F.plant }, 3)
    close(toFlat(s.w), F.plant, 0)
  })
})

describe('the Algorithm protocol and the registry', () => {
  it('protocol', () => {
    for (const make of Object.values(filters))
      checkProtocol(make(), undefined, { steps: 12, record: { e: (s) => s.e } })
  })
  it('registered with their notes', () => {
    expect(Object.keys(statisticalAlgorithms).sort()).toEqual(['lms', 'nlms', 'rls'])
    expect(statisticalAlgorithms.rls.info.notes).toContain('recursive-least-squares-filter')
  })
})
