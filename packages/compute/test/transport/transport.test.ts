import { describe, expect, it } from 'vitest'
import {
  barycenter1d,
  costMatrix,
  exactTransport,
  gromovWasserstein,
  gromovWassersteinSteps,
  monotonePlan,
  sinkhorn,
  sinkhornSteps,
  slicedWasserstein,
  uniformWeights,
  wasserstein1d,
} from 'aifn-compute/transport'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../fixtures'
import { checkProtocol } from '../protocol'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('transport')

const close = (a: number[], b: number[], tol: number) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol))
}

describe('costs and exact transport', () => {
  it('costMatrix is the squared distance by default', () => {
    close(toFlat(costMatrix(F.assignment.x, F.assignment.y)), (F.assignment.cost as number[][]).flat(), 1e-12)
    expect(toFlat(costMatrix([0, 1], [3], { p: 1 }))).toEqual([3, 2])
  })

  it('uniform equal masses use the Hungarian algorithm and match scipy', () => {
    const r = exactTransport(uniformWeights(6), uniformWeights(6), F.assignment.cost)
    expect(r.method).toBe('hungarian')
    expect(toFlat(r.assignment!)).toEqual(F.assignment.cols)
    expect(r.cost).toBeCloseTo(F.assignment.value, 10)
  })

  it('unequal masses solve the linear program, with marginals and duals', () => {
    const e = F.exact
    const r = exactTransport(e.a, e.b, e.cost)
    expect(r.method).toBe('simplex')
    expect(r.cost).toBeCloseTo(e.value, 9)
    const P = toFlat(r.plan)
    for (let i = 0; i < 5; i++) expect(P.slice(i * 4, i * 4 + 4).reduce((s, v) => s + v, 0)).toBeCloseTo(e.a[i], 10)
    const f = toFlat(r.f!)
    const g = toFlat(r.g!)
    const dual = f.reduce((s, v, i) => s + v * e.a[i], 0) + g.reduce((s, v, j) => s + v * e.b[j], 0)
    expect(dual).toBeCloseTo(e.value, 8)
  })
})

describe('Sinkhorn', () => {
  const S = F.sinkhorn
  const alg = () => sinkhornSteps(S.a, S.b, S.cost, { epsilon: S.eps, tolerance: 0 })

  it('matches a direct log-domain implementation after 50 iterations', () => {
    const s = run(alg(), {}, 50)
    close(toFlat(s.f), S.f, 1e-10)
    close(toFlat(s.g), S.g, 1e-10)
    close(toFlat(s.plan), (S.plan as number[][]).flat(), 1e-12)
    expect(s.transportCost).toBeCloseTo(S.transport, 10)
  })

  it('converges to the exact cost as ε shrinks, with the dual increasing', () => {
    const e = F.exact
    const small = sinkhorn(e.a, e.b, e.cost, { epsilon: 0.005, maxSteps: 5000 })
    expect(small.converged).toBe(true)
    expect(small.transportCost).toBeCloseTo(e.value, 2)
    const t = trace(sinkhornSteps(e.a, e.b, e.cost, { epsilon: 0.1 }), {}, 30, {
      record: { dual: (s) => s.dual },
    })
    const d = toFlat(t.series.dual)
    for (let k = 2; k < d.length; k++) expect(d[k]).toBeGreaterThanOrEqual(d[k - 1] - 1e-12)
  })

  it('follows the Algorithm protocol', () => {
    checkProtocol(alg(), {}, { steps: 12, record: { err: (s) => s.marginalError, dual: (s) => s.dual } })
  })

  it('warm-starts from a given g', () => {
    const s = run(alg(), {}, 40)
    const warm = run(alg(), { g: s.g }, 1)
    expect(warm.marginalError).toBeLessThan(run(alg(), {}, 1).marginalError)
  })
})

describe('one-dimensional transport', () => {
  it('W1 matches scipy, plain and weighted', () => {
    const w = F.w1
    expect(wasserstein1d(w.u, w.v)).toBeCloseTo(w.plain, 12)
    expect(wasserstein1d(w.u, w.v, { uWeights: w.uw, vWeights: w.vw })).toBeCloseTo(w.weighted, 12)
  })

  it('W2 between equal-size samples pairs sorted values', () => {
    const x = [3, 1, 2]
    const y = [10, 30, 20]
    expect(wasserstein1d(x, y, { p: 2 })).toBeCloseTo(Math.sqrt((81 + 324 + 729) / 3), 12)
  })

  it('the monotone plan moves every unit of mass and matches W1', () => {
    const p = monotonePlan([0.5, 0.5, 0], [0, 0.25, 0.75])
    expect(toFlat(p.mass).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const cost = toFlat(p.mass).reduce((s, m, k) => s + m * Math.abs(toFlat(p.i)[k] - toFlat(p.j)[k]), 0)
    expect(cost).toBeCloseTo(
      wasserstein1d([0, 1, 2], [0, 1, 2], { uWeights: [0.5, 0.5, 1e-300], vWeights: [1e-300, 0.25, 0.75] }),
      9,
    )
  })

  it('the barycentre of two shifted samples sits halfway', () => {
    const a = Array.from({ length: 200 }, (_, i) => i / 199)
    const b = a.map((v) => v + 4)
    const bar = barycenter1d([a, b], { levels: 50 })
    const q = toFlat(bar.quantiles)
    const qa = toFlat(barycenter1d([a], { levels: 50 }).quantiles)
    q.forEach((v, i) => expect(v).toBeCloseTo(qa[i] + 2, 10))
  })

  it('sliced Wasserstein is zero for identical clouds and grows with a shift', () => {
    const x = F.assignment.x as number[][]
    expect(slicedWasserstein(stream(1), x, x).distance).toBeCloseTo(0, 12)
    const shifted = x.map(([a, b]) => [a + 3, b])
    const d = slicedWasserstein(stream(1), x, shifted, { projections: 200 }).distance
    // For a pure shift t, SW₂² = E[(θ·t)²] = |t|²/2 in two dimensions.
    expect(d).toBeCloseTo(3 / Math.SQRT2, 0)
  })
})

describe('Gromov–Wasserstein', () => {
  it('recovers the matching between a point set and its rotated copy', () => {
    const x = F.assignment.x as number[][]
    const rot = x.map(([a, b]) => [b, -a])
    const cx = costMatrix(x, x, { p: 1 })
    const cy = costMatrix(rot, rot, { p: 1 })
    const r = gromovWasserstein(
      { cx, cy, a: uniformWeights(6), b: uniformWeights(6) },
      { epsilon: 0.01, maxSteps: 100 },
    )
    const P = toFlat(r.plan)
    for (let i = 0; i < 6; i++) {
      const row = P.slice(i * 6, i * 6 + 6)
      expect(row.indexOf(Math.max(...row))).toBe(i)
    }
  })

  it('follows the Algorithm protocol', () => {
    const x = F.assignment.x as number[][]
    const cx = costMatrix(x, x, { p: 1 })
    const alg = gromovWassersteinSteps({ cx, cy: cx, a: uniformWeights(6), b: uniformWeights(6) }, { epsilon: 0.05 })
    checkProtocol(alg, undefined, { steps: 6, record: { loss: (s) => s.loss } })
  })
})

describe('against POT (Python Optimal Transport)', () => {
  const P = F.pot as {
    sinkhorn: { a: number[]; b: number[]; cost: number[][]; eps: number; emd: number; plan: number[][] }[]
    gromov: { cx: number[][]; cy: number[][]; eps: number; plan: number[][]; exactLoss: number }
  }
  it.each(P.sinkhorn.map((c) => [`${c.a.length}×${c.b.length}, ε = ${c.eps}`, c] as const))(
    'exact cost (ot.emd2) and the converged entropic plan (ot.sinkhorn), %s',
    (_, c) => {
      expect(exactTransport(c.a, c.b, c.cost).cost).toBeCloseTo(c.emd, 10)
      const s = sinkhorn(c.a, c.b, c.cost, { epsilon: c.eps, tolerance: 1e-14, maxSteps: 100000 })
      close(Array.from(toFlat(s.plan)), c.plan.flat(), 1e-10)
    },
  )
  it('entropic Gromov–Wasserstein reaches the plan of ot.gromov (its ε is twice aifn’s)', () => {
    const G = P.gromov
    const n = G.cx.length
    const r = gromovWasserstein(
      { cx: G.cx, cy: G.cy, a: uniformWeights(n), b: uniformWeights(n) },
      { epsilon: G.eps, maxSteps: 1000, innerSteps: 5000, tolerance: 1e-12 },
    )
    close(Array.from(toFlat(r.plan)), G.plan.flat(), 1e-6)
  })
})
