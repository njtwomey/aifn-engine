import { describe, expect, it } from 'vitest'
import {
  eulerMaruyama,
  geometricBrownianMotion,
  milstein,
  ornsteinUhlenbeck,
  paths,
  stochasticRungeKutta,
} from 'aifn-compute/dynamics/sde'
import { stream } from 'aifn-compute/foundation/random'
import { mul, sin, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

describe('Euler–Maruyama', () => {
  it('has weak order 1 on geometric Brownian motion', () => {
    const [mu, sigma, T, x0] = [1.5, 0.2, 1, 1]
    const gbm = geometricBrownianMotion({ mu, sigma })
    const weak = (h: number) => {
      const s = run(eulerMaruyama(gbm.sde, { stepSize: h, tEnd: T }), { x0, paths: 40_000 }, 1000, {
        stream: stream('weak'),
      })
      return Math.abs(mean(toFlat(s.x)) - gbm.mean(T, x0))
    }
    const errors = [0.25, 0.125, 0.0625].map(weak)
    // The bias of E[X_T] is x₀(e^{μT} − (1 + μh)^{T/h}) = O(h); Monte Carlo error ≈ 0.005 is far smaller.
    expect(Math.log2(errors[0] / errors[1])).toBeCloseTo(1, 0)
    expect(Math.log2(errors[1] / errors[2])).toBeCloseTo(1, 0)
    expect(errors[0]).toBeCloseTo(Math.exp(1.5) - 1.375 ** 4, 1)
  })

  it('matches the Ornstein–Uhlenbeck moments', () => {
    const ou = ornsteinUhlenbeck({ theta: 2, mu: 1, sigma: 0.5 })
    const s = run(eulerMaruyama(ou.sde, { stepSize: 0.001, tEnd: 0.5 }), { x0: 3, paths: 20_000 }, 1000, {
      stream: stream(1),
    })
    const x = toFlat(s.x)
    const m = mean(x)
    const v = mean(x.map((a) => (a - m) ** 2))
    // Within four standard errors.
    expect(Math.abs(m - ou.mean(0.5, 3))).toBeLessThan(4 * Math.sqrt(ou.variance(0.5, 3) / x.length))
    expect(v / ou.variance(0.5, 3)).toBeCloseTo(1, 1)
    // Seeded, so deterministic: 10⁷ path-steps take about 2.5 s alone, which passes the 5 s default under a full,
    // parallel suite.
  }, 30_000)

  it('adds paths without changing existing ones', () => {
    const ou = ornsteinUhlenbeck({ theta: 1, sigma: 1 })
    const a = run(eulerMaruyama(ou.sde, { stepSize: 0.1 }), { x0: 0, paths: 5 }, 20, { stream: stream(9) })
    const b = run(eulerMaruyama(ou.sde, { stepSize: 0.1 }), { x0: 0, paths: 50 }, 20, { stream: stream(9) })
    expect(toFlat(b.x).slice(0, 5)).toEqual(toFlat(a.x))
  })

  it('runs vector SDEs with diagonal noise', () => {
    const s = run(
      eulerMaruyama({ drift: (_t, x) => mul(-1, x), diffusion: () => 0.1 }, { stepSize: 0.01 }),
      { x0: [1, 2, 3], paths: 4 },
      10,
      { stream: stream(2) },
    )
    expect(s.x.shape).toEqual([4, 3])
  })
})

describe('strong order', () => {
  const [mu, sigma, T, x0] = [0.5, 0.8, 1, 1]
  const gbm = geometricBrownianMotion({ mu, sigma })
  const strong = (make: typeof eulerMaruyama, h: number) => {
    const opts = { x0, paths: 4000 }
    const approx = toFlat(run(make(gbm.sde, { stepSize: h, tEnd: T }), opts, 1e4, { stream: stream('strong') }).x)
    const exact = toFlat(run(gbm.exact({ stepSize: h, tEnd: T }), opts, 1e4, { stream: stream('strong') }).x)
    return mean(approx.map((v, i) => Math.abs(v - exact[i])))
  }
  const order = (make: typeof eulerMaruyama) => Math.log2(strong(make, 0.02) / strong(make, 0.005)) / 2
  it('Euler–Maruyama ½, Milstein and stochastic Runge–Kutta 1', () => {
    expect(order(eulerMaruyama)).toBeCloseTo(0.5, 0)
    expect(order(milstein)).toBeCloseTo(1, 0)
    expect(order(stochasticRungeKutta)).toBeCloseTo(1, 0)
    expect(strong(milstein, 0.01)).toBeLessThan(strong(eulerMaruyama, 0.01) / 3)
  }, 30_000)

  it('Milstein by autodiff equals Milstein with the given derivative', () => {
    const sde = { drift: (_t: number, x: Tensor) => mul(0.1, x), diffusion: (_t: number, x: Tensor) => sin(x) }
    const withGiven = { ...sde, diffusionDerivative: (_t: number, x: Tensor) => toFlat(x).map(Math.cos) as never }
    const a = run(milstein(sde, { stepSize: 0.01 }), { x0: 1, paths: 10 }, 50, { stream: stream(4) })
    const b = run(milstein(withGiven, { stepSize: 0.01 }), { x0: 1, paths: 10 }, 50, { stream: stream(4) })
    toFlat(a.x).forEach((v, i) => expect(v).toBeCloseTo(toFlat(b.x)[i], 12))
  })
})

describe('exact solutions and densities', () => {
  it('OU exact sampler has the exact moments at any step', () => {
    const ou = ornsteinUhlenbeck({ theta: 1, mu: 0, sigma: 1 })
    const s = run(ou.exact({ stepSize: 0.5, tEnd: 1 }), { x0: 2, paths: 40_000 }, 10, { stream: stream(3) })
    const x = toFlat(s.x)
    const m = mean(x)
    expect(m).toBeCloseTo(ou.mean(1, 2), 1)
    expect(mean(x.map((v) => (v - m) ** 2)) / ou.variance(1, 2)).toBeCloseTo(1, 1)
  })

  it('paths extracts a path matrix', () => {
    const tr = trace(
      eulerMaruyama(ornsteinUhlenbeck({ theta: 1, sigma: 1 }).sde, { stepSize: 0.1 }),
      { x0: 0, paths: 3 },
      10,
      {
        stream: stream(5),
      },
    )
    const p = paths(tr)
    expect(p.values.shape).toEqual([11, 3])
    expect(toRows(p.values)[0]).toEqual([0, 0, 0])
  })
})

describe('trace protocol', () => {
  const gbm = geometricBrownianMotion({ mu: 0.1, sigma: 0.3 })
  const start = { x0: 1, paths: 8 }
  it.each([
    ['eulerMaruyama', eulerMaruyama(gbm.sde, { stepSize: 0.05 })],
    ['milstein', milstein(gbm.sde, { stepSize: 0.05 })],
    ['stochasticRungeKutta', stochasticRungeKutta(gbm.sde, { stepSize: 0.05 })],
    ['exact', gbm.exact({ stepSize: 0.05 })],
  ])('%s satisfies the Algorithm protocol', (_name, alg) => {
    checkProtocol(alg, start, { steps: 12, record: { t: (s) => s.t, time: (s) => s.time, x0: (s) => toFlat(s.x)[0] } })
  })

  it('t counts steps and time is the time', () => {
    const s = run(eulerMaruyama(gbm.sde, { stepSize: 0.25 }), { x0: 1, t0: 1 }, 3, { stream: stream(1) })
    expect(s.t).toBe(3)
    expect(s.time).toBeCloseTo(1.75, 12)
  })

  it('stops at tEnd with a shortened last step', () => {
    const s = run(eulerMaruyama(gbm.sde, { stepSize: 0.3, tEnd: 1 }), { x0: 1 }, 100, { stream: stream(1) })
    expect(s.time).toBeCloseTo(1, 12)
    expect(s.stepSize).toBeCloseTo(0.1, 12)
  })
})
