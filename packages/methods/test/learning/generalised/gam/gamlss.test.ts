import { describe, expect, it } from 'vitest'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { growthChart, GROWTH_TRUTH } from 'aifn-methods/data/synthetic'
import {
  gamlssModel,
  gamlssProblem,
  gamlssTrace,
  gamProblem,
  s,
  type GamlssSpec,
} from 'aifn-methods/learning/generalised/gam'

const flat = (t: Tensor) => Float64Array.from(toFlat(t))
const AGES = [1, 3, 6, 9, 12, 15, 17]
const ageGrid = fromData(Float64Array.from(AGES), [AGES.length, 1])

/** y ~ N(μ(x), σ(x)²) with x uniform on [0, 1]. */
const muX = (x: number) => Math.sin(2 * Math.PI * x)
const sigmaX = (x: number) => 0.2 + 0.5 * x * x
function normalData(n: number, seed: number) {
  const r = stream(seed)
  const x = flat(uniform(child(r, 'x'), 0, 1, { shape: [n] }) as Tensor)
  const e = flat(normals(child(r, 'e'), [n]) as Tensor)
  const y = Float64Array.from(x, (xi, i) => muX(xi) + sigmaX(xi) * e[i])
  return { x: fromData(x, [n, 1]), y: fromData(y, [n]) }
}

const growth = growthChart(stream(7), { n: 2000 })
const growthData = { x: growth.x, y: growth.y! }
const smooth = (lambda?: number) => ({ terms: [s(0, { k: 12, ...(lambda ? { lambda } : {}) })] })

describe('gamlss', () => {
  it('recovers smooth μ(x), σ(x), ν(x) of a BCCG growth chart', () => {
    const problem = gamlssProblem(
      { family: 'box-cox-cole-green', parameters: { mu: smooth(), sigma: smooth(), nu: smooth() } },
      growthData,
    )
    const model = gamlssModel(problem)
    expect(model.state.converged).toBe(true)
    const [mu, sigma, nu] = model.parameters(ageGrid)
    AGES.forEach((a, i) => {
      expect(Math.abs(mu[i] / GROWTH_TRUTH.mu(a) - 1)).toBeLessThan(0.03)
      expect(Math.abs(sigma[i] - GROWTH_TRUTH.sigma(a))).toBeLessThan(0.025)
      expect(Math.abs(nu[i] - GROWTH_TRUTH.nu(a))).toBeLessThan(0.5)
    })
  })

  it('recovers a heteroscedastic normal location-scale model', () => {
    const data = normalData(1500, 3)
    const model = gamlssModel(gamlssProblem({ family: 'normal', parameters: { mu: smooth(), sigma: smooth() } }, data))
    const grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    const [mu, sigma] = model.parameters(fromData(Float64Array.from(grid), [grid.length, 1]))
    grid.forEach((x, i) => {
      expect(Math.abs(mu[i] - muX(x))).toBeLessThan(0.25 * sigmaX(x))
      expect(Math.abs(sigma[i] / sigmaX(x) - 1)).toBeLessThan(0.15)
    })
  })

  it('never raises the penalised global deviance across RS cycles at fixed λ (law)', () => {
    const specs: GamlssSpec[] = [
      { family: 'normal', parameters: { mu: smooth(10), sigma: smooth(10) } },
      { family: 'student-t', parameters: { mu: smooth(10), sigma: smooth(10) } },
      { family: 'box-cox-cole-green', parameters: { mu: smooth(10), sigma: smooth(10), nu: smooth(10) } },
    ]
    for (const spec of specs) {
      const tr = gamlssTrace(gamlssProblem({ ...spec, smoothing: 'fixed', tolerance: 1e-6 }, growthData), 40)
      const pd = flat(tr.series.penalisedDeviance)
      for (let t = 1; t < pd.length; t++) expect(pd[t]).toBeLessThanOrEqual(pd[t - 1] * (1 + 1e-12))
      expect(tr.final.converged).toBe(true)
    }
  })

  it('with σ constant, the normal GAMLSS location fit is the Gaussian GAM at λσ̂²', () => {
    const data = normalData(400, 5)
    const lambda = 2
    const model = gamlssModel(
      gamlssProblem(
        { family: 'normal', parameters: { mu: smooth(lambda) }, smoothing: 'fixed', tolerance: 1e-10 },
        data,
      ),
    )
    const sigma = model.state.theta[1][0]
    const gam = gamProblem({ terms: [s(0, { k: 12 })], method: 'fixed', lambda: lambda * sigma * sigma }, data)
    const muGam = gam.evaluate(gam.optimum.beta).mu
    const muLss = model.state.theta[0]
    for (let i = 0; i < muGam.length; i++) expect(muLss[i]).toBeCloseTo(muGam[i], 5)
    // σ̂ is the maximum-likelihood σ: √(RSS/n).
    let rss = 0
    const y = flat(data.y)
    for (let i = 0; i < y.length; i++) rss += (y[i] - muLss[i]) ** 2
    expect(sigma).toBeCloseTo(Math.sqrt(rss / y.length), 5)
  })

  it('gives approximately standard normal quantile residuals under the true model', () => {
    const data = normalData(2000, 11)
    const r = Float64Array.from(
      gamlssModel(gamlssProblem({ family: 'normal', parameters: { mu: smooth(), sigma: smooth() } }, data)).residuals(),
    ).sort()
    const n = r.length
    const mean = r.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
    const skew = r.reduce((a, b) => a + ((b - mean) / sd) ** 3, 0) / n
    expect(Math.abs(mean)).toBeLessThan(0.05)
    expect(Math.abs(sd - 1)).toBeLessThan(0.05)
    expect(Math.abs(skew)).toBeLessThan(0.15)
    // Kolmogorov–Smirnov distance to Φ, against the 1% critical value 1.63/√n.
    const Phi = (z: number) => 0.5 * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
    let ks = 0
    r.forEach((v, i) => (ks = Math.max(ks, Math.abs(Phi(v) - (i + 0.5) / n))))
    expect(ks).toBeLessThan(1.63 / Math.sqrt(n))
  })

  it('a constant-σ normal follows the centre but misses the centiles of the growth chart; BCCG gets both', () => {
    const ages = flat(growth.x)
    const y = flat(growthData.y)
    const share = (spec: GamlssSpec) => {
      const c = gamlssModel(gamlssProblem(spec, growthData)).centiles(growth.x, [0.03, 0.5, 0.97])
      const older = Array.from(ages).flatMap((a, i) => (a > 12 ? [i] : []))
      const frac = (j: number) => older.filter((i) => y[i] < c[j][i]).length / older.length
      return [frac(0), frac(1), frac(2)]
    }
    const constant = share({ family: 'normal', parameters: { mu: smooth() } })
    const lms = share({ family: 'box-cox-cole-green', parameters: { mu: smooth(), sigma: smooth(), nu: smooth() } })
    expect(Math.abs(constant[1] - 0.5)).toBeLessThan(0.08)
    expect(Math.abs(constant[2] - 0.97)).toBeGreaterThan(0.04)
    lms.forEach((v, j) => expect(Math.abs(v - [0.03, 0.5, 0.97][j])).toBeLessThan(0.025))
    // The fit with ν reports a lower GAIC than the one without.
    const aic = (spec: GamlssSpec) => gamlssModel(gamlssProblem(spec, growthData)).gaic()
    expect(
      aic({ family: 'box-cox-cole-green', parameters: { mu: smooth(), sigma: smooth(), nu: smooth() } }),
    ).toBeLessThan(aic({ family: 'normal', parameters: { mu: smooth(), sigma: smooth() } }))
  })
})
