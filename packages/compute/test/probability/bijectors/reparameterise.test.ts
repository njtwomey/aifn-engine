/** transformLogDensity: a log-density reparameterised through a change of variables. */
import { describe, expect, it } from 'vitest'
import type { LogDensity } from 'aifn-compute/foundation/contracts'
import {
  add,
  concat,
  exp,
  get,
  mul,
  slice,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { expBijector, transformLogDensity } from 'aifn-compute/probability/bijectors'
import { Gamma, Normal } from 'aifn-compute/probability/distributions'
import { hmc } from 'aifn-compute/inference/stochastic'
import { run } from 'aifn-compute/foundation/trace'
import { stream } from 'aifn-compute/foundation/random'

const num = (v: Value) => unwrap(v) as number
const d = 3
const s = 3
/** Neal's funnel, centred: v ~ N(0, s²), xᵢ | v ~ N(0, eᵛ). */
const funnel: LogDensity = {
  kind: 'log-density',
  name: 'funnel',
  dim: d,
  normalised: true,
  logDensity: (t: Value) =>
    add(Normal(0, s).logProb(get(t, 0)), sum(Normal(0, exp(mul(0.5, get(t, 0)))).logProb(slice(t, [1, null])))),
}
const centring = (u: Value) => concat([slice(u, [0, 1]), mul(slice(u, [1, null]), exp(mul(0.5, get(u, 0))))])

describe('transformLogDensity', () => {
  it('the non-centred funnel is N(0, s²) × N(0, 1)^(d − 1), with an explicit or an autodiff log-Jacobian', () => {
    const explicit = transformLogDensity(funnel, {
      name: 'non-centring',
      forward: centring,
      logAbsDetJacobian: (u) => mul((d - 1) / 2, get(u, 0)),
    })
    const auto = transformLogDensity(funnel, { forward: centring })
    for (const u of [
      [0.3, -1, 2],
      [-4, 0.5, 0.1],
      [2.5, 1, -1],
    ]) {
      const want = num(Normal(0, s).logProb(u[0])) + num(sum(Normal(0, 1).logProb(tensor(u.slice(1)))))
      expect(num(explicit.logDensity(tensor(u)))).toBeCloseTo(want, 12)
      expect(num(auto.logDensity(tensor(u)))).toBeCloseTo(want, 10)
    }
    expect(toFlat(explicit.toOriginal(tensor([2, 1, -1])) as never)).toEqual([2, Math.E, -Math.E])
    expect(explicit.dim).toBe(d)
    expect(explicit.normalised).toBe(true)
  })
  it('an elementwise bijector: a Gamma target on log scale', () => {
    const g = Gamma(2, 3)
    const positive: LogDensity = {
      kind: 'log-density',
      dim: 2,
      normalised: true,
      logDensity: (t: Value) => sum(g.logProb(t)),
    }
    const logScale = transformLogDensity(positive, expBijector)
    const u = [0.2, -0.7]
    // log p(e^u) + Σ u: the density of log X for X ~ Gamma(2, 3).
    const want = u.reduce((a, ui) => a + num(g.logProb(Math.exp(ui))) + ui, 0)
    expect(num(logScale.logDensity(tensor(u)))).toBeCloseTo(want, 12)
    expect(toFlat(logScale.fromOriginal!(tensor([1, Math.E])) as never)).toEqual([0, 1])
  })
  it('HMC on the non-centred funnel does not diverge where the centred one does', () => {
    const nc = transformLogDensity(funnel, { forward: centring, logAbsDetJacobian: (u) => mul((d - 1) / 2, get(u, 0)) })
    const o = { stepSize: 0.4, steps: 15, divergenceThreshold: { relative: 2 } } as const
    const centred = run(hmc(funnel, o), { x0: [-3, 0.1, 0.1] }, 80, { stream: stream(5) })
    const nonCentred = run(hmc(nc, o), { x0: [-3, 0.1, 0.1] }, 80, { stream: stream(5) })
    expect(centred.divergentCount).toBeGreaterThan(0)
    expect(nonCentred.divergentCount).toBe(0)
  })
})
