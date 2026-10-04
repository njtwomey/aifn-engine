import { describe, expect, test } from 'vitest'
import { dlqr, lqg, lqgSimulation, mpcController, recedingHorizon } from 'aifn-compute/dynamics/control'
import { stream } from 'aifn-compute/foundation/random'
import { toComplexFlat, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

type Mat = number[][]
const F = fixture<{
  lqg: Record<'A' | 'B' | 'C' | 'Q' | 'R' | 'W' | 'V' | 'K' | 'L' | 'S' | 'Ad' | 'Bd' | 'Kd' | 'Ld' | 'Sd', Mat>
}>('dynamics/control')

const close = (a: Tensor, b: Mat, tol: number) => {
  const x = toFlat(a)
  const y = b.flat()
  expect(x.length).toBe(y.length)
  const scale = Math.max(1, ...y.map(Math.abs))
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol * scale))
}

// The sampled double integrator, dt = 0.1: position and velocity, force input.
const dt = 0.1
const A = [
  [1, dt],
  [0, 1],
]
const B = [[0.5 * dt * dt], [dt]]
const Q = [
  [1, 0],
  [0, 0.1],
]
const R = [[0.1]]

describe('linear MPC', () => {
  test('unconstrained with the DARE terminal cost gives the LQR input for every horizon', () => {
    const K = toFlat(dlqr({ A, B }, Q, R).K)
    const x0 = [1, -0.5]
    for (const horizon of [1, 3, 10, 25]) {
      const plan = mpcController({ A, B, Q, R, horizon }).plan(x0)
      const u0 = toFlat(plan.u)[0]
      expect(u0).toBeCloseTo(-(K[0] * x0[0] + K[1] * x0[1]), 6)
      // The predicted states follow the model under the planned inputs.
      const xs = toRows(plan.x)
      const us = toFlat(plan.u)
      for (let k = 0; k < horizon; k++) {
        expect(xs[k + 1][0]).toBeCloseTo(xs[k][0] + dt * xs[k][1] + 0.5 * dt * dt * us[k], 10)
        expect(xs[k + 1][1]).toBeCloseTo(xs[k][1] + dt * us[k], 10)
      }
    }
  })

  test('input and state bounds hold, and a bounded plan costs at least the unbounded one', () => {
    const free = mpcController({ A, B, Q, R, horizon: 20 })
    const boxed = mpcController({ A, B, Q, R, horizon: 20, uMin: -1, uMax: 1 })
    const both = mpcController({
      A,
      B,
      Q,
      R,
      horizon: 20,
      uMin: -1,
      uMax: 1,
      xMin: [-Infinity, -0.6],
      xMax: [Infinity, 0.6],
    })
    const x0 = [3, 0]
    const pf = free.plan(x0)
    const pb = boxed.plan(x0)
    const ps = both.plan(x0)
    expect(Math.max(...toFlat(pb.u).map(Math.abs))).toBeLessThanOrEqual(1 + 1e-6)
    expect(pb.active).toBeGreaterThan(0)
    expect(pb.cost).toBeGreaterThanOrEqual(pf.cost - 1e-9)
    expect(ps.status).toBe('optimal')
    expect(Math.max(...toFlat(ps.u).map(Math.abs))).toBeLessThanOrEqual(1 + 1e-6)
    expect(Math.max(...toRows(ps.x).map((r) => Math.abs(r[1])))).toBeLessThanOrEqual(0.6 + 1e-6)
    expect(ps.cost).toBeGreaterThanOrEqual(pb.cost - 1e-9)
  })

  test('soft state bounds: exact penalty equals the hard plan when feasible, and stays feasible when not', () => {
    const bounds = { uMin: -1, uMax: 1, xMin: [-Infinity, -0.6], xMax: [Infinity, 0.6] }
    const hard = mpcController({ A, B, Q, R, horizon: 20, ...bounds })
    const exact = mpcController({ A, B, Q, R, horizon: 20, ...bounds, soft: { quadratic: 10, linear: 1e4 } })
    const ph = hard.plan([3, 0])
    const pe = exact.plan([3, 0])
    expect(pe.status).toBe('optimal')
    toFlat(pe.u).forEach((v, i) => expect(v).toBeCloseTo(toFlat(ph.u)[i], 4))
    expect(Math.max(...toFlat(pe.slack!))).toBeLessThan(1e-5)
    // From a velocity of 1.2 no input in [−1, 1] gets back under 0.6 at once: the hard plan drops the state bounds, the
    // soft one keeps them and pays for the excess, less of it as the quadratic weight grows.
    expect(hard.plan([0, 1.2]).status).not.toBe('optimal')
    const excess = (rho: number) => {
      const p = mpcController({ A, B, Q, R, horizon: 20, ...bounds, soft: { quadratic: rho, linear: 0 } }).plan([
        0, 1.2,
      ])
      expect(p.status).toBe('optimal')
      const s = toFlat(p.slack!)
      // The slack is exactly the bound violation of the predicted velocity.
      toRows(p.x)
        .slice(1)
        .forEach((x, k) => expect(s[2 * k + 1]).toBeCloseTo(Math.max(0, Math.abs(x[1]) - 0.6), 3))
      return s.reduce((a, v) => a + v, 0)
    }
    const [low, high] = [excess(1), excess(1e4)]
    expect(low).toBeGreaterThan(high)
    expect(high).toBeGreaterThan(0)
  })

  test('the receding-horizon loop drives the state to the origin within the bounds', () => {
    const c = mpcController({ A, B, Q, R, horizon: 15, uMin: -1, uMax: 1 })
    const alg = recedingHorizon(c, { steps: 120 })
    const s = run(alg, { x0: [3, 0] }, 200)
    expect(s.terminated).toBe(true)
    expect(Math.hypot(...toFlat(s.x))).toBeLessThan(1e-2)
    const tr = trace(alg, { x0: [3, 0] }, 120, { record: { u: (st) => toFlat(st.u)[0] } })
    expect(Math.max(...toFlat(tr.series.u).map(Math.abs))).toBeLessThanOrEqual(1 + 1e-6)
    checkProtocol(recedingHorizon(c), { x0: [1, 0] }, { steps: 5 })
  })

  test('the receding horizon simulates the controller model by default, or a given plant', () => {
    const c = mpcController({ A, B, Q, R, horizon: 5 })
    expect(toRows(c.model.B)).toEqual([[0.5 * dt * dt], [dt]])
    const s = run(recedingHorizon(c), { x0: [1, 0] }, 1)
    const u = toFlat(run(recedingHorizon(c), { x0: [1, 0] }, 0).u)[0]
    expect(toFlat(s.x)[0]).toBeCloseTo(1 + 0.5 * dt * dt * u, 12)
    expect(toFlat(s.x)[1]).toBeCloseTo(dt * u, 12)
    // A plant with twice the input gain moves twice as far under the same first input.
    const p = run(recedingHorizon(c, { plant: { A, B: [[dt * dt], [2 * dt]] } }), { x0: [1, 0] }, 1)
    expect(toFlat(p.x)[1]).toBeCloseTo(2 * dt * u, 12)
  })
})

describe('LQG', () => {
  test('continuous design matches python-control lqr and lqe', () => {
    const c = F.lqg
    const d = lqg({ A: c.A, B: c.B, C: c.C }, { Q: c.Q, R: c.R, W: c.W, V: c.V })
    close(d.K, c.K, 1e-8)
    close(d.L, c.L, 1e-8)
    close(d.S, c.S, 1e-8)
    for (const p of [...toComplexFlat(d.regulatorPoles), ...toComplexFlat(d.estimatorPoles)])
      expect(p.re).toBeLessThan(0)
  })

  test('discrete design matches python-control dlqr and dlqe (the predictor gain)', () => {
    const c = F.lqg
    const d = lqg({ A: c.Ad, B: c.Bd, C: c.C }, { Q: c.Q, R: c.R, W: c.W, V: c.V }, { discrete: true })
    close(d.K, c.Kd, 1e-8)
    close(d.L, c.Ld, 1e-8)
    close(d.S, c.Sd, 1e-8)
    // Separation principle: the compensator loop has the regulator and estimator poles.
    for (const p of [...toComplexFlat(d.regulatorPoles), ...toComplexFlat(d.estimatorPoles)])
      expect(Math.hypot(p.re, p.im)).toBeLessThan(1)
  })

  test('the noisy loop is stable; full state feedback costs less than output feedback on average', () => {
    const c = F.lqg
    const plant = { A: c.Ad, B: c.Bd, C: c.C }
    const weights = { Q: c.Q, R: c.R, W: c.W, V: c.V }
    const d = lqg(plant, weights, { discrete: true })
    const cost = (feedback: 'estimate' | 'state') =>
      run(lqgSimulation(plant, weights, d, { feedback }), { x0: [1, 0] }, 3000, { stream: stream('lqg') }).cost
    const out = cost('estimate')
    const full = cost('state')
    expect(Number.isFinite(out)).toBe(true)
    expect(full).toBeLessThan(out)
    checkProtocol(lqgSimulation(plant, weights, d), { x0: [1, 0] }, { steps: 5 })
  })
})
