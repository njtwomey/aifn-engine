/**
 * Threshold problems on the line (the site's foundations self-check): the clusters' Bayes risk $\Phi(-2) = 0.0227501$
 * equals $R(0)$, the quantile inverts the distribution function, the uniform risk, and the learning curves: unlabelled
 * data help on the clusters at one label per class and not on the uniform problem.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  clustersProblem,
  drawLine,
  firstLabelled,
  fitBoth,
  lineProblem,
  thresholdLearningCurve,
  uniformProblem,
} from 'aifn-methods/data/synthetic'

describe('line problems', () => {
  it('clusters with s = 4: Bayes risk Φ(-2) = 0.0227501 = R(0), and the quantile inverts the cdf', () => {
    const p = clustersProblem(4)
    expect(Math.abs(p.bayesRisk - 0.0227501)).toBeLessThan(1e-6)
    expect(Math.abs(p.risk(0) - p.bayesRisk)).toBeLessThan(1e-12)
    expect(Math.abs(p.cdf(p.quantile(0.3)) - 0.3)).toBeLessThan(1e-9)
    expect(lineProblem({ kind: 'clusters', separation: 4 }).bayesRisk).toBe(p.bayesRisk)
  })

  it('uniform with t* = 0.3: R(0.45) = 0.15', () => {
    expect(Math.abs(uniformProblem(0.3).risk(0.45) - 0.15)).toBeLessThan(1e-12)
  })

  it('draws nested samples: the first m labelled points of each class, class 0 first', () => {
    const d = drawLine(stream(1), clustersProblem(4), 10, 4)
    expect([d.unlabelled.length, d.class0.length, d.class1.length]).toEqual([10, 4, 4])
    const s = firstLabelled(d, 2)
    expect(s.x).toEqual([...d.class0.slice(0, 2), ...d.class1.slice(0, 2)])
    expect(s.y).toEqual([0, 0, 1, 1])
    const fit = fitBoth(clustersProblem(4), d, 2)
    expect(Number.isFinite(fit.supervised) && Number.isFinite(fit.semiSupervised)).toBe(true)
  })
})

describe('thresholdLearningCurve', () => {
  const sizes = { perClass: [1, 2, 4, 8, 16, 32, 64], unlabelled: 200, repeats: 100 }

  it('semi-supervised beats supervised at m = 1 on the clusters', () => {
    const c = thresholdLearningCurve(stream(1), clustersProblem(4), sizes)
    expect(c.semiSupervised[0].mean).toBeLessThan(c.supervised[0].mean)
  })

  it('brings no gain on the uniform problem, and the spec form gives the same curves', () => {
    const u = thresholdLearningCurve(stream(1), uniformProblem(0.3), sizes)
    expect(u.semiSupervised[3].mean).toBeGreaterThanOrEqual(0.8 * u.supervised[3].mean)
    const spec = thresholdLearningCurve(stream(1), { kind: 'uniform', boundary: 0.3 }, sizes)
    expect(spec).toEqual(u)
  })
})
