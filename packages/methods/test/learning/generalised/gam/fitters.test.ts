import { describe, expect, it } from 'vitest'
import {
  gam,
  gamFitter,
  gamFitters,
  gamModel,
  gamProblem,
  gamProblemChoices,
  gamTrainingRun,
  s,
  smoothingProfile,
  type GamFitMethod,
  type GamProblem,
} from 'aifn-methods/learning/generalised/gam'
import { additiveData, ADDITIVE_SHAPES, type AdditiveFamily } from 'aifn-methods/data/synthetic'
import { datasetRegistry, type AdditiveTruth } from 'aifn-methods/data'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { LinkName } from 'aifn-compute/probability/likelihoods'
import { expectProtocol } from '../../../protocol'

const flat = (t: Tensor | Value) =>
  typeof t === 'number' ? Float64Array.of(t) : Float64Array.from(toFlat(t as Tensor))
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)

/** A small two-term problem at a fixed λ, well enough conditioned for first-order methods to converge in a test. */
function smallProblem(family: AdditiveFamily, link?: LinkName, lambda = 10): GamProblem {
  const d = additiveData(stream(`fitters-${family}`), { family, link, n: 150, shapes: ['periodic', 'monotone'] })
  return gamProblem(
    { terms: [s(0, { k: 8 }), s(1, { k: 8 })], family, link, method: 'fixed', lambda },
    { x: d.x as Tensor, y: d.y as Tensor },
  )
}

describe('GAM fitters', () => {
  const cases: [AdditiveFamily, LinkName | undefined][] = [
    ['gaussian', undefined],
    ['binomial', 'logit'],
    ['binomial', 'probit'],
    ['poisson', 'log'],
    ['poisson', 'sqrt'],
  ]

  it.each(cases)(
    '%s (%s link): every deterministic fitter reaches the P-IRLS optimum at the same λ',
    (family, link) => {
      const problem = smallProblem(family, link)
      const opt = problem.optimum.beta
      const J = problem.evaluate(opt).objective
      const steps: Record<Exclude<GamFitMethod, 'sgd'>, number> = {
        'p-irls': 50,
        backfitting: 500,
        'gradient-descent': 20000,
        adam: 20000,
        lbfgs: 500,
      }
      for (const [method, n] of Object.entries(steps) as [GamFitMethod, number][]) {
        const options = method === 'adam' ? { stepSize: 0.02 } : {}
        const final = run(gamFitter(problem, method, options), undefined, n)
        const beta = flat(final.coefficients)
        expect(maxDiff(beta, opt), `${family} ${method}: β`).toBeLessThan(1e-6)
        expect(Math.abs(final.objective - J) / J, `${family} ${method}: J`).toBeLessThan(1e-10)
        expect(final.diverged).toBe(false)
      }
    },
  )

  it('SGD reaches the optimum approximately, with a decaying step', () => {
    const problem = smallProblem('gaussian')
    const opt = problem.optimum.beta
    const J = problem.evaluate(opt).objective
    const eta0 = 0.5 / problem.curvature
    const final = run(
      gamFitter(problem, 'sgd', { batchSize: 30, stepSize: (t) => eta0 / (1 + t / 2000) }),
      undefined,
      6000,
    )
    expect(final.epochs).toBeCloseTo((6000 * 30) / 150, 10)
    expect((final.objective - J) / J).toBeLessThan(2e-3)
    expect(maxDiff(flat(final.coefficients), opt)).toBeLessThan(0.2)
  })

  it('all fitters start from the same β₀ and follow the algorithm protocol', () => {
    const problem = smallProblem('poisson')
    const methods: GamFitMethod[] = ['p-irls', 'backfitting', 'gradient-descent', 'sgd', 'adam', 'lbfgs']
    for (const m of methods) {
      const s0 = run(gamFitter(problem, m), undefined, 0)
      expect(maxDiff(flat(s0.coefficients), problem.start), m).toBe(0)
      expectProtocol(gamFitter(problem, m), undefined, { n: 5, record: { objective: (st) => st.objective } })
    }
    // P-IRLS states carry the working response and weights it regresses on next.
    const p1 = run(gamFitter(problem, 'p-irls'), undefined, 1)
    const { z, W } = problem.working(flat(p1.coefficients))
    expect(maxDiff(flat(p1.working!), z)).toBeLessThan(1e-12)
    expect(maxDiff(flat(p1.workingWeights!), W)).toBeLessThan(1e-12)
  })

  it('the closed-form gradient is the autodiff gradient, and minibatches estimate J without bias', () => {
    const problem = smallProblem('binomial', 'cloglog', 3)
    const beta = Float64Array.from(problem.optimum.beta, (v, i) => v + 0.1 * Math.sin(i + 1))
    const { value, grad } = valueAndGrad((b: Value) => problem.objective(b))(fromData(beta, [beta.length]))
    expect(maxDiff(flat(grad as Tensor), problem.gradient(beta))).toBeLessThan(1e-12)
    expect(flat(value as Tensor)[0]).toBeCloseTo(problem.evaluate(beta).objective, 12)
    // A partition of the rows into 5 minibatches: the mean of the estimates is J.
    const n = problem.design.n
    const estimates = [0, 1, 2, 3, 4].map((k) => {
      const rows = Array.from({ length: n / 5 }, (_, r) => k * (n / 5) + r)
      return flat(problem.objective(fromData(beta, [beta.length]), rows) as Tensor)[0]
    })
    expect(estimates.reduce((a, b) => a + b, 0) / 5).toBeCloseTo(problem.evaluate(beta).objective, 12)
  })

  it('gamFitters registers every fitter as an algorithm', () => {
    expect(Object.keys(gamFitters).sort()).toEqual(
      ['gamAdam', 'gamBackfitting', 'gamGradientDescent', 'gamLbfgs', 'gamPirls', 'gamSgd'].sort(),
    )
    for (const e of Object.values(gamFitters)) expect(e.info.state.objective).toBe('objective')
  })
})

describe('GAM problems and models', () => {
  it('a training run is plain data, and the problem rebuilt from its choices skips the search', () => {
    const d = additiveData(stream('run'), { family: 'poisson', n: 120, shapes: ['periodic', 'smooth'] })
    const data = { x: d.x as Tensor, y: d.y as Tensor }
    const spec = { terms: [s(0, { k: 8 }), s(1, { k: 8, constraint: 'convex' as const })], family: 'poisson' as const }
    const run = gamTrainingRun(spec, data, [
      { method: 'p-irls', steps: 20 },
      { method: 'sgd', options: { batchSize: 20 }, steps: 12, seed: 3 },
    ])
    expect(run.runs.map((r) => r.states.length)).toEqual([run.runs[0].states.length, 13])
    expect('inner' in run.runs[1].states[0]).toBe(false)
    expect(JSON.parse(JSON.stringify(run.choices))).toEqual(run.choices)
    const choices = gamProblemChoices(spec, data)
    expect(choices.lambdas).toEqual(run.choices.lambdas)
    const rebuilt = gamProblem({ ...spec, ...choices }, data)
    expect(rebuilt.smoothing.evaluations).toBe(0)
    expect(maxDiff(rebuilt.optimum.beta, flat(run.optimum.coefficients))).toBeLessThan(1e-9)
    expect(maxDiff(flat(run.runs[0].states.at(-1)!.coefficients), rebuilt.optimum.beta)).toBeLessThan(1e-6)
  })

  it('rejects a link the family does not take, naming the valid ones', () => {
    const d = additiveData(stream('links'), { family: 'poisson', n: 50 })
    expect(() =>
      gam({ terms: [s(0)], family: 'poisson', link: 'logit' }).fit({ x: d.x as Tensor, y: d.y as Tensor }),
    ).toThrow('gam: the poisson family does not take the logit link; use one of log, identity, sqrt')
    expect(() => additiveData(stream('links'), { family: 'binomial', link: 'identity' })).toThrow(/does not take/)
  })

  it('the model at the optimum is the fitted model; at another β it describes that β', () => {
    const d = additiveData(stream('model'), { family: 'gamma', link: 'log', n: 200 })
    const spec = { terms: [s(0), s(1), s(2)], family: 'gamma' as const, link: 'log' as const }
    const data = { x: d.x as Tensor, y: d.y as Tensor }
    const fitted = gam(spec).fit(data)
    const problem = gamProblem(spec, data)
    const again = gamModel(problem, problem.optimum.beta)
    expect(maxDiff(flat(again.fitted), flat(fitted.fitted))).toBeLessThan(1e-12)
    expect(again.edf).toBeCloseTo(fitted.edf, 10)
    expect(fitted.devianceExplained).toBeGreaterThan(0.2)
    expect(fitted.devianceExplained).toBeLessThan(1)
    expect(Number.isFinite(fitted.aic)).toBe(true)
    expect(fitted.termPenalties.length).toBe(3)
    const start = gamModel(problem, problem.start)
    expect(start.devianceExplained).toBeCloseTo(1 - start.deviance / problem.nullDeviance, 12)
    expect(start.deviance).toBeGreaterThan(fitted.deviance)
  })

  it("a term's basis: raw · Z is the constrained basis, ZᵀSZ the penalty, and the weighted columns sum to fⱼ", () => {
    const d = additiveData(stream('basis'), { n: 200 })
    const m = gam({ terms: [s(0, { k: 9, degree: 2, order: 1 }), s(1)] }).fit({ x: d.x as Tensor, y: d.y as Tensor })
    const grid = linspace(0, 1, 41)
    const b = m.basis(0, grid)
    expect(b.raw.shape).toEqual([41, 9])
    expect(b.constrained.shape).toEqual([41, 8])
    const raw = flat(b.raw)
    const Z = flat(b.Z)
    const con = flat(b.constrained)
    for (let i = 0; i < 41; i++)
      for (let c = 0; c < 8; c++) {
        let v = 0
        for (let r = 0; r < 9; r++) v += raw[i * 9 + r] * Z[r * 8 + c]
        expect(v).toBeCloseTo(con[i * 8 + c], 12)
      }
    // A degree-2 B-spline basis is a partition of unity inside the range.
    for (let i = 0; i < 41; i++) expect(raw.slice(i * 9, i * 9 + 9).reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    const S = flat(b.rawPenalties[0])
    const Sc = flat(b.penalties[0])
    for (let a = 0; a < 8; a++)
      for (let c = 0; c < 8; c++) {
        let v = 0
        for (let r = 0; r < 9; r++) for (let q = 0; q < 9; q++) v += Z[r * 8 + a] * S[r * 9 + q] * Z[q * 8 + c]
        expect(v).toBeCloseTo(Sc[a * 8 + c], 10)
      }
    expect(maxDiff(flat(b.sum), flat(m.partial(0, grid).fit))).toBeLessThan(1e-12)
    const w = flat(b.weighted)
    for (let i = 0; i < 41; i++)
      expect(w.slice(i * 9, i * 9 + 9).reduce((a, v) => a + v, 0)).toBeCloseTo(flat(b.sum)[i], 12)
  })

  it('the REML profile over a common log λ is lowest near the λ REML selects for a single smooth', () => {
    const d = additiveData(stream('profile'), { n: 200, shapes: ['wiggly'] })
    const problem = gamProblem({ terms: [s(0, { k: 15 })] }, { x: d.x as Tensor, y: d.y as Tensor })
    const logs = Array.from({ length: 61 }, (_, i) => -12 + 0.3 * i)
    const profile = smoothingProfile(problem, logs)
    const best = logs[profile.indexOf(Math.min(...profile))]
    expect(Math.abs(best - Math.log(problem.lambdas[0]))).toBeLessThan(0.3)
    expect(Math.min(...profile)).toBeCloseTo(problem.smoothing.value, 1)
  })
})

describe('additiveData', () => {
  it('draws from the family with centred true effects, and is registered with a truth', () => {
    for (const [family, link] of [
      ['gaussian', 'identity'],
      ['binomial', 'logit'],
      ['binomial', 'cloglog'],
      ['poisson', 'identity'],
      ['gamma', 'inverse'],
    ] as const) {
      const d = additiveData(stream(`data-${family}-${link}`), { family, link, n: 400 })
      const t = d.meta!.truth as AdditiveTruth
      const y = flat(d.y as Tensor)
      if (family === 'binomial') expect(y.every((v) => v === 0 || v === 1)).toBe(true)
      if (family === 'poisson') expect(y.every((v) => Number.isInteger(v) && v >= 0)).toBe(true)
      if (family === 'gamma') expect(y.every((v) => v > 0)).toBe(true)
      expect(maxDiff(flat(t.mean(d.x as Tensor)), flat(d.f!))).toBe(0)
      // Mean of y close to the mean of μ.
      const mu = flat(d.f!)
      const mean = (a: Float64Array) => a.reduce((p, v) => p + v, 0) / a.length
      expect(Math.abs(mean(y) - mean(mu))).toBeLessThan(4 * Math.sqrt(t.bayesRisk / 400) + 1e-9)
      // Each partial effect has mean zero over U(0, 1), and over the data when centred on it.
      const grid = linspace(0.0005, 0.9995, 1000)
      for (let j = 0; j < 3; j++) {
        expect(Math.abs(mean(flat(t.partial(j, grid))))).toBeLessThan(1e-5)
        const xj = fromData(
          Float64Array.from({ length: 400 }, (_, i) => flat(d.x as Tensor)[i * 3 + j]),
          [400],
        )
        expect(Math.abs(mean(flat(t.partial(j, xj, xj))))).toBeLessThan(1e-12)
      }
    }
    expect(datasetRegistry.additiveData.info.truth).toBe(true)
    expect(Object.keys(ADDITIVE_SHAPES)).toContain('periodic')
  })
})
