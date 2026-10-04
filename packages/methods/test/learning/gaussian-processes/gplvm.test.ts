/** The GPLVM (`gplvm.ts`) on the fonts data: gradients, the fit as an Algorithm, the latent-to-outline map. */
import { describe, expect, it } from 'vitest'
import { fonts } from 'aifn-methods/data/real/fonts'
import { fitGplvm, gplvmFitSteps, gplvmModel, gplvmProblem } from 'aifn-methods/learning/gaussian-processes'
import { pca } from 'aifn-methods/unsupervised/embedding/linear'
import { fromData, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'
import { matern52 } from 'aifn-compute/learning/kernels'
import { expectProtocol } from '../../protocol'

const small = fonts({ chars: 'GPL', rows: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] })
const rms = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
  return Math.sqrt(s / a.length)
}

describe('the GPLVM problem', () => {
  const problem = gplvmProblem(small.x)

  it('summarises Y by YYᵀ = MMᵀ and starts at the principal-component scores', () => {
    expect([problem.n, problem.d, problem.q]).toEqual([12, small.x.shape[1], 2])
    const y = toRows(problem.y)
    const m = toRows(problem.m)
    for (const [i, j] of [
      [0, 0],
      [3, 7],
      [11, 2],
    ]) {
      const yy = y[i].reduce((s, v, k) => s + v * y[j][k], 0)
      const mm = m[i].reduce((s, v, k) => s + v * m[j][k], 0)
      expect(mm).toBeCloseTo(yy, 6)
    }
    // The explained fractions are PCA's.
    const p = pca({ components: 2 }).fit(dataset(small.x))
    const ratio = toFlat(p.explainedVarianceRatio)
    problem.explained.forEach((e, k) => expect(e).toBeCloseTo(ratio[k], 8))
    // Latent start = PCA scores up to the sign of each axis (and a common scale).
    const z0 = toRows(problem.unpack(problem.theta0).latent)
    const s = toRows(p.transform(small.x))
    for (let c = 0; c < 2; c++) {
      const a = z0.map((r) => r[c])
      const b = s.map((r) => r[c])
      const dot = a.reduce((acc, v, i) => acc + v * b[i], 0)
      const cos = dot / Math.hypot(...a) / Math.hypot(...b)
      expect(Math.abs(cos)).toBeCloseTo(1, 8)
    }
  })

  it('the gradient of the negative log posterior matches central differences', () => {
    const theta = toFlat(problem.theta0)
    const { value, grad } = problem.objective(problem.theta0)
    expect(Number.isFinite(value)).toBe(true)
    const g = toFlat(grad)
    const h = 1e-5
    for (const k of [0, 5, 13, theta.length - 3, theta.length - 2, theta.length - 1]) {
      const plus = Float64Array.from(theta)
      const minus = Float64Array.from(theta)
      plus[k] += h
      minus[k] -= h
      const fd = (problem.objective(fromData(plus)).value - problem.objective(fromData(minus)).value) / (2 * h)
      expect(Math.abs(g[k] - fd)).toBeLessThan(1e-5 * (1 + Math.abs(fd)))
    }
    expect(problem.negLogPosterior(problem.theta0) as number).toBeCloseTo(value, 8)
  })

  it('the fit raises the log posterior at every step and follows the trace protocol', () => {
    const tr = trace(gplvmFitSteps(problem), undefined, 60, { record: { lp: (s) => s.logPosterior } })
    const lp = toFlat(tr.series.lp)
    for (let i = 1; i < lp.length; i++) expect(lp[i]).toBeGreaterThanOrEqual(lp[i - 1] - 1e-8)
    expect(lp.at(-1)!).toBeGreaterThan(lp[0])
    expect(tr.final.noiseVariance).toBeGreaterThanOrEqual(1e-4)
    expectProtocol(gplvmFitSteps(problem), undefined, { n: 6, record: { lp: (s) => s.logPosterior } })
  })

  it('project maps latent points back to outlines; the variance is small at the data and σ_f² far away', () => {
    const state = run(gplvmFitSteps(problem), undefined, 200)
    const model = gplvmModel(problem, state)
    expect(model.kind).toBe('model')
    const recon = toRows(model.project(model.latent))
    const data = toRows(small.x)
    // Within 1% of the letters' 700-unit cap height, on average.
    data.forEach((row, i) => expect(rms(recon[i], row)).toBeLessThan(7))
    const one = model.project(toRows(model.latent)[3])
    expect(one.shape).toEqual([small.x.shape[1]])
    expect(Array.from(toFlat(one))).toEqual(recon[3])
    const atData = toFlat(model.variance(model.latent))
    const far = toFlat(model.variance(tensor([[50, -50]])))[0]
    const scale2 = problem.scale ** 2
    expect(far).toBeCloseTo(model.signalVariance * scale2, 6)
    for (const v of atData) expect(v).toBeLessThan(0.05 * far)
    expect(toFlat(model.variance(tensor([[50, -50]]), { noise: true }))[0]).toBeCloseTo(
      (model.signalVariance + model.noiseVariance) * scale2,
      6,
    )
  })

  it('takes another stationary kernel and a maximum-likelihood objective', () => {
    const ml = gplvmProblem(small.x, { kernel: matern52, prior: false, latentDim: 1 })
    expect(ml.q).toBe(1)
    const s = run(gplvmFitSteps(ml), undefined, 20)
    expect(s.latent.shape).toEqual([12, 1])
    expect(Number.isFinite(s.logPosterior)).toBe(true)
  })
})

describe('a manifold of fonts', () => {
  it('fitGplvm on every font lays out the design classes: most fonts have a nearest latent neighbour of their class', () => {
    const d = fonts({ chars: 'GPLVM' })
    const fit = fitGplvm(d.x, { maxSteps: 150 })
    const lp = toFlat(fit.training.series.logPosterior)
    expect(lp.at(-1)!).toBeGreaterThan(lp[0])
    const z = toRows(fit.model.latent)
    const y = toFlat(d.y!)
    let agree = 0
    z.forEach((p, i) => {
      let best = -1
      let bestD = Infinity
      z.forEach((q, j) => {
        const dist = Math.hypot(p[0] - q[0], p[1] - q[1])
        if (j !== i && dist < bestD) [best, bestD] = [j, dist]
      })
      if (y[best] === y[i]) agree++
    })
    // Six classes: chance agreement is about one in five.
    expect(agree / z.length).toBeGreaterThan(0.5)
  })
})
