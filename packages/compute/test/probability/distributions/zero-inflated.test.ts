/**
 * ZeroInflated: hand-computed masses (P(0) = π + (1 − π)p(0), P(k) = (1 − π)p(k)), laws (π = 0 is the base, the mass
 * sums to one, moments by summation, logits agree with probabilities, draws match the mass) and gradients of the
 * log-mass against central finite differences.
 */

import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { Bernoulli, NegativeBinomial, Normal, Poisson, ZeroInflated } from 'aifn-compute/probability/distributions'
import { checkGradient } from '../../foundation/tensor/check-gradient'

const flat = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const num = (v: Value) => flat(v)[0]
const poissonPmf = (k: number, l: number) => {
  let f = 1
  for (let j = 2; j <= k; j++) f *= j
  return (Math.exp(-l) * l ** k) / f
}

describe('ZeroInflated', () => {
  it('has the zero-inflated mass, by hand', () => {
    const d = ZeroInflated(0.25, Poisson(2))
    const lp = flat(d.logProb(tensor([0, 1, 3])))
    expect(Math.exp(lp[0])).toBeCloseTo(0.25 + 0.75 * Math.exp(-2), 14)
    expect(Math.exp(lp[1])).toBeCloseTo(0.75 * poissonPmf(1, 2), 14)
    expect(Math.exp(lp[2])).toBeCloseTo(0.75 * poissonPmf(3, 2), 14)
    expect(num(d.logProb(-1))).toBe(-Infinity)
    expect(num(d.logProb(0.5))).toBe(-Infinity)
  })

  it('is the base when π = 0', () => {
    const base = NegativeBinomial(3, 0.4)
    const d = ZeroInflated(0, base)
    const k = tensor([0, 1, 2, 7])
    flat(d.logProb(k)).forEach((v, i) => expect(v).toBeCloseTo(flat(base.logProb(k))[i], 13))
    expect(num(d.mean())).toBeCloseTo(num(base.mean()), 13)
    expect(num(d.variance())).toBeCloseTo(num(base.variance()), 13)
  })

  it('is a point mass at zero when π = 1', () => {
    const d = ZeroInflated(1, Poisson(4))
    expect(num(d.prob(0))).toBeCloseTo(1, 14)
    expect(num(d.mean())).toBe(0)
  })

  it('sums to one and has the stated moments', () => {
    const pi = 0.35
    const d = ZeroInflated(pi, Poisson(3.5))
    const ks = Array.from({ length: 80 }, (_, k) => k)
    const p = flat(d.prob(tensor(ks)))
    const total = p.reduce((a, b) => a + b, 0)
    const m = p.reduce((a, q, k) => a + q * k, 0)
    const m2 = p.reduce((a, q, k) => a + q * k * k, 0)
    expect(total).toBeCloseTo(1, 12)
    expect(num(d.mean())).toBeCloseTo(m, 10)
    expect(num(d.variance())).toBeCloseTo(m2 - m * m, 10)
    expect(num(d.mean())).toBeCloseTo((1 - pi) * 3.5, 12)
    // The cdf is the running sum of the mass.
    const F = flat(d.cdf(tensor([0, 2, 5])))
    expect(F[0]).toBeCloseTo(p[0], 12)
    expect(F[1]).toBeCloseTo(p[0] + p[1] + p[2], 12)
    expect(F[2]).toBeCloseTo(
      p.slice(0, 6).reduce((a, b) => a + b, 0),
      12,
    )
  })

  it('takes logits', () => {
    const eta = -0.7
    const a = ZeroInflated({ logits: eta }, Poisson(1.5))
    const b = ZeroInflated(1 / (1 + Math.exp(-eta)), Poisson(1.5))
    const k = tensor([0, 1, 4])
    flat(a.logProb(k)).forEach((v, i) => expect(v).toBeCloseTo(flat(b.logProb(k))[i], 13))
  })

  it('zero-inflates a Bernoulli: P(1) = (1 − π)p', () => {
    const d = ZeroInflated(0.2, Bernoulli(0.7))
    expect(num(d.prob(1))).toBeCloseTo(0.8 * 0.7, 14)
    expect(num(d.prob(0))).toBeCloseTo(0.2 + 0.8 * 0.3, 14)
  })

  it('broadcasts a batch of π against the base', () => {
    const d = ZeroInflated(tensor([0, 0.5]), Poisson(2))
    expect(d.batchShape).toEqual([2])
    const p0 = flat(d.prob(0))
    expect(p0[0]).toBeCloseTo(Math.exp(-2), 14)
    expect(p0[1]).toBeCloseTo(0.5 + 0.5 * Math.exp(-2), 14)
  })

  it('draws structural zeros at the right rate', () => {
    const d = ZeroInflated(0.4, Poisson(5))
    const x = flat(d.sample(stream('zi/draws'), { shape: [20000] }) as Value)
    const zeros = x.filter((v) => v === 0).length / x.length
    expect(Math.abs(zeros - num(d.prob(0)))).toBeLessThan(0.015)
    const mean = x.reduce((a, b) => a + b, 0) / x.length
    expect(Math.abs(mean - 3)).toBeLessThan(0.06)
  })

  it('differentiates the log-mass in π, the logits and the base parameters', () => {
    checkGradient((pi, rate) => ZeroInflated(pi, Poisson(rate)).logProb(tensor([0, 1, 3])), [0.3, 2.2], { tol: 1e-6 })
    checkGradient((eta, p) => ZeroInflated({ logits: eta }, Bernoulli(p)).logProb(tensor([0, 1])), [0.4, 0.6], {
      tol: 1e-6,
    })
  })

  it('rejects a continuous base or one without zero', () => {
    expect(() => ZeroInflated(0.2, Normal(0, 1))).toThrow(DomainError)
    expect(() => ZeroInflated(1.2, Poisson(1))).toThrow(DomainError)
  })
})
