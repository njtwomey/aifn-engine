/**
 * The evidence of Bayesian linear regression and MacKay's updates (the `evidence` cases of fixture `learning/linear`):
 * the log evidence against the log density of the marginal
 * $\Gauss(\zeros, \beta^{-1}\Imat + \alpha^{-1}\Xmat\Xmat^\top)$ computed by scipy, and the updates state by state
 * against scikit-learn's `BayesianRidge` with its hyperpriors switched off, on the standardised diabetes data and on
 * a wide slice of it ($n < d$).
 */
import { describe, expect, it } from 'vitest'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import {
  bayesianLinearEvidence,
  evidenceMaximisation,
  evidenceMaximisationSteps,
  linearAlgorithms,
  linearFunctions,
  linearRegression,
  type EvidenceState,
} from 'aifn-methods/learning/linear'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { inverse } from 'aifn-compute/numerics/linalg'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dataset } from 'aifn-compute/learning/estimators'
import { fixture } from '../../fixtures'

type Fixture = {
  runs: {
    name: string
    rows: number
    scores: number[]
    n_iter: number
    alpha: number
    beta: number
    coef: number[]
    intercept: number
    sigma: number[][]
  }[]
  polynomial: { t: number[]; y: number[] }
  fixed: { case: 'polynomial' | 'diabetes'; degree: number; alpha: number; beta: number; log_evidence: number }[]
}
const fx = fixture<{ evidence: Fixture }>('learning/linear').evidence

const data = diabetes()
const x = standardScaler().fit({ x: data.x }).transform(data.x)
const y = data.y!

const rowsOf = (t: Tensor, count: number): Tensor => {
  const [, d] = t.shape
  return fromData(Float64Array.from(toFlat(t).slice(0, count * d)), [count, d])
}

const expectClose = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, what: string): void => {
  expect(actual.length, what).toBe(expected.length)
  for (let i = 0; i < expected.length; i++)
    expect(Math.abs(actual[i] - expected[i]), `${what}[${i}]: ${actual[i]} vs ${expected[i]}`).toBeLessThan(tol)
}

describe('bayesianLinearEvidence', () => {
  it('equals the log density of the marginal of y (scipy), for polynomial designs and centred diabetes data', () => {
    const { t, y: target } = fx.polynomial
    for (const c of fx.fixed) {
      const e =
        c.case === 'polynomial'
          ? bayesianLinearEvidence(
              tensor(t.map((v) => Array.from({ length: c.degree + 1 }, (_, k) => v ** k))),
              tensor(target),
              { alpha: c.alpha, beta: c.beta },
            )
          : bayesianLinearEvidence(x, y, { alpha: c.alpha, beta: c.beta, intercept: true })
      expect(Math.abs(e.logEvidence - c.log_evidence), `${c.case} ${c.degree}`).toBeLessThan(
        1e-10 * Math.abs(c.log_evidence),
      )
    }
  })

  it('splits into terms that sum to it, and a posterior that is ridge with penalty alpha / beta', () => {
    const e = bayesianLinearEvidence(x, y, { alpha: 0.01, beta: 3e-4, intercept: true })
    const terms = Object.values(e.terms).reduce((a, b) => a + b, 0)
    expect(terms).toBeCloseTo(e.logEvidence, 9)
    expect(e.bestFitLogLikelihood + e.logOccamFactor).toBeCloseTo(e.logEvidence, 9)
    expect(e.logOccamFactor).toBeLessThan(0)
    const ridge = linearRegression({ l2: 0.01 / 3e-4 }).fit(dataset(x, y))
    expectClose(toFlat(e.mean), toFlat(ridge.weights), 1e-9, 'mean')
    expect(e.intercept).toBeCloseTo(ridge.intercept, 9)
    // The covariance is the inverse of A = alpha I + beta XᵀX on the centred design.
    const [n, d] = x.shape
    const X = toRows(x)
    const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n)
    const A = Array.from({ length: d }, (_, j) =>
      Array.from({ length: d }, (_, k) => {
        let s = j === k ? 0.01 : 0
        for (const r of X) s += 3e-4 * (r[j] - mu[j]) * (r[k] - mu[k])
        return s
      }),
    )
    expectClose(toFlat(e.covariance), toFlat(inverse(tensor(A))), 1e-12, 'covariance')
    // gamma counts the well-determined parameters: between 0 and d.
    expect(e.gamma).toBeGreaterThan(0)
    expect(e.gamma).toBeLessThan(d)
  })

  it('throws for a precision that is not positive', () => {
    expect(() => bayesianLinearEvidence(x, y, { alpha: 0, beta: 1 })).toThrow(DomainError)
    expect(() => bayesianLinearEvidence(x, y, { alpha: 1, beta: NaN })).toThrow(DomainError)
  })
})

describe('evidenceMaximisationSteps', () => {
  it("matches BayesianRidge's scores_ state by state, and its fit at the end", () => {
    for (const run of fx.runs) {
      const design = run.name === 'wide' ? rowsOf(x, run.rows) : x
      const target = fromData(Float64Array.from(toFlat(y).slice(0, run.rows)))
      // Tolerance 0: never flagged converged, so the run can be stepped as far as scikit-learn went.
      const alg = evidenceMaximisationSteps({ x: design, y: target, intercept: true, tolerance: 0 })
      const states: EvidenceState[] = [alg.init({}, stream(0))]
      while (states.length <= run.n_iter)
        states.push(alg.step(states.at(-1)!, { t: states.length - 1, stream: stream(0) }))
      expectClose(
        states.map((s) => s.logEvidence),
        run.scores,
        1e-9 * Math.max(...run.scores.map(Math.abs)),
        `${run.name} scores`,
      )
      const last = states[run.n_iter]
      expect(Math.abs(last.alpha / run.alpha - 1), `${run.name} alpha`).toBeLessThan(1e-9)
      expect(Math.abs(last.beta / run.beta - 1), `${run.name} beta`).toBeLessThan(1e-9)
      expectClose(toFlat(last.mean), run.coef, 1e-8, `${run.name} coef`)
      // The evidence never fell along these runs.
      for (let t = 1; t < states.length; t++)
        expect(states[t].logEvidence).toBeGreaterThan(states[t - 1].logEvidence - 1e-9)
    }
  })

  it('evidenceMaximisation stops on the same test and returns the posterior at the end', () => {
    const run = fx.runs[0]
    const fit = evidenceMaximisation(x, y, { intercept: true, tolerance: 1e-8 })
    expect(fit.converged).toBe(true)
    // scikit-learn stops one update after the state whose mean moved less than the tolerance.
    expect(fit.steps).toBe(run.n_iter - 1)
    expect(fit.history.logEvidence).toHaveLength(fit.steps + 1)
    expectClose(fit.history.logEvidence, run.scores.slice(0, run.n_iter), 1e-6, 'history')
    expect(Math.abs(fit.alpha / run.alpha - 1)).toBeLessThan(1e-7)
    expect(Math.abs(fit.beta / run.beta - 1)).toBeLessThan(1e-7)
    expectClose(toFlat(fit.mean), run.coef, 1e-6, 'coef')
    expect(fit.intercept).toBeCloseTo(run.intercept, 6)
    expectClose(toFlat(fit.covariance), run.sigma.flat(), 1e-8, 'sigma')
  })

  it('recovers the noise precision of synthetic data, and flags a design with no signal as diverged', () => {
    const z = toRows(rowsOf(x, 400))
    const noise = toFlat(y)
      .slice(0, 400)
      .map((v, i) => Math.sin(7 * i + v))
    const target = tensor(z.map((r, i) => 3 * r[2] - 2 * r[8] + 0.5 * noise[i]))
    const fit = evidenceMaximisation(fromData(Float64Array.from(z.flat()), [400, 10]), target, { intercept: true })
    // sin(...) has variance near 1/2, so the noise variance is near 1/8 and beta near 8.
    expect(fit.beta).toBeGreaterThan(6)
    expect(fit.beta).toBeLessThan(10)
    // A target that is exactly zero leaves the weights at zero: alpha = gamma / 0 is infinite.
    const alg = evidenceMaximisationSteps({ x, y: fromData(new Float64Array(442)) })
    const start = alg.init({ beta: 1 }, stream(0))
    expect(alg.step(start, { t: 0, stream: stream(0) }).diverged).toBe(true)
    expect(() => alg.init({ alpha: -1 }, stream(0))).toThrow(DomainError)
  })

  it('is registered', () => {
    expect(linearAlgorithms.evidenceMaximisationSteps).toBe(evidenceMaximisationSteps)
    expect(linearFunctions.bayesianLinearEvidence).toBe(bayesianLinearEvidence)
    expect(linearFunctions.evidenceMaximisation).toBe(evidenceMaximisation)
  })
})
