import { describe, expect, it } from 'vitest'
import { factorGraphGibbs, gibbsMarginals, modelGibbs, type ModelGibbsState } from 'aifn-compute/inference/stochastic'
import { enumerate } from 'aifn-compute/inference/exact'
import { dist, model } from 'aifn-compute/inference/model'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'
import { isingGrid, sprinkler, sprinklerBrute } from '../graphs'

describe('Gibbs on a discrete factor graph', () => {
  it('estimates the marginals of a 3 × 3 Ising grid', () => {
    const g = isingGrid(3, 3, 0.3, 0.1)
    const exact = enumerate(g)
    const s = run(factorGraphGibbs(g), undefined, 5000, { stream: stream(1) })
    gibbsMarginals(s).forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.04))
  })

  it('satisfies the Algorithm protocol, by sweep and by variable, with evidence', () => {
    const g = isingGrid(2, 2, 0.4, 0.1)
    const record = { a0: (s: { assignment: { data: ArrayLike<number> } }) => s.assignment.data[0] }
    checkProtocol(factorGraphGibbs(g), undefined, { steps: 10, random: true, record })
    checkProtocol(factorGraphGibbs(g, { granularity: 'variable', evidence: { 3: 1 } }), undefined, {
      steps: 10,
      random: true,
    })
  })
})

describe('Gibbs on a model description', () => {
  it('matches the exact sprinkler posterior', () => {
    const brute = sprinklerBrute()
    const t = trace(modelGibbs(sprinkler, { data: { wet: 1 } }), undefined, 4000, {
      stream: stream(3),
      record: { rain: (st: ModelGibbsState) => st.values.rain as number },
    })
    expect(t.final.sweep).toBe(4000)
    const mean =
      toFlat(t.series.rain)
        .slice(1)
        .reduce((a, b) => a + b, 0) / 4000
    expect(Math.abs(mean - brute.rain)).toBeLessThan(0.03)
  })

  it('Beta–Bernoulli: the chain mean approaches the posterior mean', () => {
    const coin = model('coin', (m) => {
      const p = m.variable('p', dist.Beta(2, 2))
      m.plate('flips', 'n').observed('x', dist.Bernoulli(p))
    })
    const x = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    const t = trace(modelGibbs(coin, { data: { x } }), undefined, 3000, {
      stream: stream(9),
      record: { p: (s: ModelGibbsState) => s.values.p as number },
    })
    expect(t.final.kinds.p).toBe('beta')
    const mean =
      toFlat(t.series.p)
        .slice(1)
        .reduce((a, b) => a + b, 0) / 3000
    expect(Math.abs(mean - (2 + 8) / (4 + 10))).toBeLessThan(0.01)
  })

  it('Normal means with a mixture indicator: enumerated and conjugate conditionals together', () => {
    const mix = model('two means', (m) => {
      const mu = m.plate('components', 2).variable('μ', dist.Normal(0, 10))
      const points = m.plate('points', 'n')
      const z = points.variable('z', dist.Categorical([0.5, 0.5]))
      points.observed('x', dist.Normal(mu.at(z), 1))
    })
    const x = [-5.1, -4.8, -5.3, 4.9, 5.2, 5.0]
    const s = run(modelGibbs(mix, { data: { x } }), undefined, 200, { stream: stream(2) })
    const means = [s.values['μ[0]'] as number, s.values['μ[1]'] as number].sort((a, b) => a - b)
    expect(Math.abs(means[0] + 5)).toBeLessThan(1.5)
    expect(Math.abs(means[1] - 5)).toBeLessThan(1.5)
    expect(s.kinds['μ[0]']).toBe('normal')
    expect(s.kinds['z[0]']).toBe('enumerate')
  })

  it('satisfies the Algorithm protocol', () => {
    checkProtocol(modelGibbs(sprinkler, { data: { wet: 1 } }), undefined, {
      steps: 8,
      random: true,
      record: { lj: (s) => s.logJoint },
    })
  })
})
