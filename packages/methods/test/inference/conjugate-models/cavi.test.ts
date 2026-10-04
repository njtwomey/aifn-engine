import { describe, expect, it } from 'vitest'
import { caviNormalGamma, normalGammaPosterior, type NormalGammaPrior } from 'aifn-methods/inference/conjugate-models'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

describe('CAVI normal–gamma', () => {
  const c = fixture<{ normal_gamma: { x: number[]; prior: NormalGammaPrior; log_evidence: number } }>(
    'inference/conjugate-models',
  ).normal_gamma

  it('normal-gamma exact posterior evidence matches quadrature', () => {
    expect(normalGammaPosterior(c.x, c.prior).logEvidence).toBeCloseTo(c.log_evidence, 7)
  })

  it('CAVI normal-gamma: ELBO rises, stays below log p(x), and the means match the truth', () => {
    const tr = trace(caviNormalGamma(c.x, c.prior), { expectedTau0: 0.1 }, 200, {
      record: { elbo: (s) => s.elbo },
    })
    const e = toFlat(tr.series.elbo)
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThanOrEqual(e[i - 1] - 1e-12)
    const last = tr.steps.at(-1)!
    expect(tr.meta.stopped).toBe('done')
    expect(last.kl).toBeGreaterThan(0)
    expect(last.kl).toBeLessThan(0.1)
    const exact = normalGammaPosterior(c.x, c.prior)
    expect(last.muMean).toBeCloseTo(exact.meanOfMu, 10)
    expect(1 / last.muPrecision).toBeLessThan(exact.varianceOfMu)
    expect(last.expectedTau).toBeCloseTo(exact.meanOfTau, 1)
  })
})
