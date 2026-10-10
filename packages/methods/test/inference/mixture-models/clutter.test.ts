/** Minka's clutter problem: the tilted moments, EP and ADF against adaptive quadrature. */
import { describe, expect, it } from 'vitest'
import {
  assumedDensityFiltering,
  epLogEvidence,
  expectationPropagation,
} from 'aifn-compute/inference/expectation-propagation'
import {
  clutterEp,
  clutterLogLikelihood,
  clutterPosterior,
  clutterTilted,
  sampleClutter,
} from 'aifn-methods/inference/mixture-models'
import { integrate } from 'aifn-compute/numerics/quadrature'
import { stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'

/**
 * ∫ f over [a, b] as a sum over unit panels: a single adaptive rule over a wide range can miss a narrow peak (its
 * first Kronrod estimate sees nothing there and reports convergence).
 */
function panels(f: (t: number) => number, a: number, b: number): number {
  let total = 0
  for (let lo = a; lo < b; lo += 1) total += integrate(f, lo, Math.min(lo + 1, b), { rtol: 1e-13 }).value
  return total
}

/** Moments of N(θ; m, v) f(θ) on [lower, upper] (default ±12 sd) by quadrature: log Z, mean, variance. */
function quadTilted(m: number, v: number, f: (t: number) => number, bounds: [number, number] = [-Infinity, Infinity]) {
  const s = Math.sqrt(v)
  const w = (t: number) => (Math.exp((-0.5 * (t - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)) * f(t)
  const lo = Math.max(m - 12 * s, bounds[0])
  const hi = Math.min(m + 12 * s, bounds[1])
  const z = panels(w, lo, hi)
  const mean = panels((t) => t * w(t), lo, hi) / z
  const second = panels((t) => (t - mean) ** 2 * w(t), lo, hi) / z
  return { logZ: Math.log(z), mean, variance: second }
}

describe('the clutter factor’s tilted moments against quadrature', () => {
  it.each([
    [0.3, 1.2],
    [-2, 0.5],
    [4, 3],
  ])('at N(%d, %d)', (m, v) => {
    const closed = clutterTilted(1.5, m, v, { weight: 0.3 })
    const q = quadTilted(m, v, (t: number) => Math.exp(clutterLogLikelihood(t, [1.5], { weight: 0.3 })))
    expect(closed.logZ).toBeCloseTo(q.logZ, 6)
    expect(closed.mean).toBeCloseTo(q.mean, 6)
    expect(closed.variance).toBeCloseTo(q.variance, 6)
  })
})

describe('EP on the clutter problem against quadrature', () => {
  const problem = { weight: 0.25 }
  const x = Array.from(sampleClutter(stream(7), 20, 2, problem).data)
  // The exact posterior moments by adaptive quadrature, independent of clutterPosterior's grid.
  const logPost = (t: number) =>
    -0.5 * Math.log(2 * Math.PI * 100) - (t * t) / 200 + clutterLogLikelihood(t, x, problem)
  const top = logPost(2)
  const f = (t: number) => Math.exp(logPost(t) - top)
  const Z = panels(f, -40, 40)
  const mean = panels((t) => t * f(t), -40, 40) / Z
  const variance = panels((t) => (t - mean) ** 2 * f(t), -40, 40) / Z
  const logEvidence = top + Math.log(Z)
  it('the grid posterior matches adaptive quadrature', () => {
    const p = clutterPosterior(x, problem)
    expect(p.mean).toBeCloseTo(mean, 8)
    expect(p.variance).toBeCloseTo(variance, 8)
    expect(p.logEvidence).toBeCloseTo(logEvidence, 8)
  })
  it('EP converges close to the exact moments and evidence', () => {
    const s = run(expectationPropagation(clutterEp(x, problem)), undefined, 2000)
    expect(s.converged).toBe(true)
    expect(Math.abs(s.posterior.mean - mean)).toBeLessThan(0.05 * Math.sqrt(variance))
    expect(Math.abs(s.posterior.variance / variance - 1)).toBeLessThan(0.1)
    expect(Math.abs(epLogEvidence(s) - logEvidence)).toBeLessThan(0.05)
  })
  it('the first EP sweep is ADF', () => {
    const opts = clutterEp(x, problem)
    const ep = run(expectationPropagation(opts), undefined, x.length)
    const adf = run(assumedDensityFiltering(opts), undefined, x.length)
    expect(ep.posterior.mean).toBeCloseTo(adf.posterior.mean, 12)
    expect(ep.posterior.variance).toBeCloseTo(adf.posterior.variance, 12)
  })
  it('damped and power EP converge', () => {
    const damped = run(expectationPropagation(clutterEp(x, problem, { damping: 0.5 })), undefined, 5000)
    expect(damped.converged).toBe(true)
    const power = run(expectationPropagation(clutterEp(x, problem, { power: 0.5 })), undefined, 5000)
    expect(power.converged).toBe(true)
    expect(Math.abs(power.posterior.mean - mean)).toBeLessThan(0.2)
  })
})

describe('the clutter functions take sampleClutter’s tensor as it is', () => {
  it('a tensor gives the same answers as its array of values', () => {
    const problem = { weight: 0.3 }
    const x = sampleClutter(stream(5), 20, 2, problem)
    const values = Array.from(x.data as Float64Array)
    expect(clutterLogLikelihood(1.5, x, problem)).toBeCloseTo(clutterLogLikelihood(1.5, values, problem), 12)
    expect(clutterLogLikelihood(1.5, x, problem)).not.toBe(0)
    expect(clutterEp(x, problem).factors).toBe(20)
    const fromTensor = clutterPosterior(x, problem)
    const fromArray = clutterPosterior(values, problem)
    expect(fromTensor.mean).toBeCloseTo(fromArray.mean, 12)
    expect(fromTensor.logEvidence).toBeCloseTo(fromArray.logEvidence, 12)
    expect(fromTensor.logEvidence).not.toBe(0)
  })
})
