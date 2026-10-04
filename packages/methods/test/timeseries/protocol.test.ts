/** Every fitting algorithm of `aifn-methods/timeseries` follows the Algorithm protocol. */
import { describe, expect, it } from 'vitest'
import {
  armaFitSteps,
  constantVelocityModel,
  exponentialSmoothingFitSteps,
  garchFitSteps,
  simulateArma,
  simulateGarch,
  stateSpaceEm,
} from 'aifn-methods/timeseries'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../protocol'

const ar = simulateArma(stream('arma'), { ar: [0.6], ma: [0.3], sigma: 1 }, 300)
const series = Array.from(toFlat(ar.x))

describe('fitting algorithms', () => {
  it('ARMA (exact and CSS) and GARCH', () => {
    for (const method of ['exact', 'css'] as const)
      expectProtocol(armaFitSteps(series, { p: 1, q: 1, method }), undefined, {
        n: 10,
        record: { v: (s) => s.objective },
      })
    const g = simulateGarch(stream('garch'), { omega: 0.1, alpha: 0.1, beta: 0.8 }, 300)
    expectProtocol(garchFitSteps(g.returns), undefined, { n: 10, record: { v: (s) => s.objective } })
  })

  it('exponential smoothing', () => {
    const trended = series.map((v, i) => v + 0.05 * i + 3 * Math.sin((2 * Math.PI * i) / 12))
    expectProtocol(exponentialSmoothingFitSteps(trended, { trend: 'additive' }), undefined, { n: 10 })
    expectProtocol(
      exponentialSmoothingFitSteps(trended, { trend: 'damped', seasonal: 'additive', period: 12 }),
      undefined,
      {
        n: 10,
      },
    )
  })

  it('state-space EM raises the log-likelihood and follows the protocol', () => {
    const y = Array.from({ length: 60 }, (_, t) => [
      Math.cos(t / 8) * 10 + Math.sin(t) * 0.5,
      t * 0.3 + Math.cos(3 * t) * 0.5,
    ])
    const alg = stateSpaceEm(y, constantVelocityModel(), { estimate: { Q: true, R: true } })
    const s1 = run(alg, undefined, 1)
    const s5 = run(alg, undefined, 5)
    expect(s5.logLikelihood).toBeGreaterThanOrEqual(s1.logLikelihood - 1e-8)
    expectProtocol(alg, undefined, { n: 5, record: { ll: (s) => s.logLikelihood } })
  })
})
