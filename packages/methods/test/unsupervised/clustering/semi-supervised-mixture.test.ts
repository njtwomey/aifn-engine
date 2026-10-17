/**
 * Semi-supervised EM for Gaussian classes, the merge of the site's two candidates. Both are inlined below as they were
 * written (the diagonal EM of the semi-supervised-in-practice notes and the tied/full mixture of the generative
 * notes), and the merged export reproduces each to rounding over many steps. Then the numbers of the two self-checks:
 * the labelled-only state 0 (means 1 and 12, variances 1 and 4), the monotone objective, the $\lambda = 0$ fixed point,
 * the unsupervised limit against `gaussianMixtureSteps`, and the posterior rows.
 */
import { describe, expect, it } from 'vitest'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { float64Data, fromData, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { cholesky, choleskyLogDet, choleskySolve } from 'aifn-compute/numerics/linalg'
import { normalCdf } from 'aifn-compute/numerics/special'
import { multivariateNormal } from 'aifn-compute/probability/samplers'
import {
  classPosterior,
  gaussianMixtureSteps,
  labelledOnlyFit,
  semiSupervisedMixtureSteps,
  type SemiSupervisedMixtureOptions,
} from 'aifn-methods/unsupervised/clustering'

// ── The site's diagonal EM (semi-supervised-em.ts), inlined ─────────────────────────────────────────────────────────

type EmState = { priors: number[]; means: number[][]; variances: number[][]; r: Float64Array; ll: number }

function candidateEm(rows: number[][], labels: number[], K: number, lambda: number, floor: number) {
  const n = rows.length
  const d = rows[0].length
  const X = Float64Array.from(rows.flat())
  const mStep = (r: Float64Array) => {
    const w = new Float64Array(K)
    const mu = new Float64Array(K * d)
    const va = new Float64Array(K * d)
    for (let i = 0; i < n; i++) {
      const a = labels[i] >= 0 ? 1 : lambda
      for (let k = 0; k < K; k++) {
        const q = a * r[i * K + k]
        if (q === 0) continue
        w[k] += q
        for (let j = 0; j < d; j++) mu[k * d + j] += q * X[i * d + j]
      }
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) mu[k * d + j] /= w[k]
    for (let i = 0; i < n; i++) {
      const a = labels[i] >= 0 ? 1 : lambda
      for (let k = 0; k < K; k++) {
        const q = a * r[i * K + k]
        if (q === 0) continue
        for (let j = 0; j < d; j++) va[k * d + j] += q * (X[i * d + j] - mu[k * d + j]) ** 2
      }
    }
    const total = w.reduce((a, b) => a + b, 0)
    return {
      priors: Array.from(w, (v) => v / total),
      means: Array.from({ length: K }, (_, k) => Array.from(mu.subarray(k * d, (k + 1) * d))),
      variances: Array.from({ length: K }, (_, k) =>
        Array.from(va.subarray(k * d, (k + 1) * d), (v) => Math.max(floor, v / w[k])),
      ),
    }
  }
  const eStep = (p: ReturnType<typeof mStep>): EmState => {
    const r = new Float64Array(n * K)
    let ll = 0
    for (let i = 0; i < n; i++) {
      const l = Array.from({ length: K }, (_, k) => {
        let s = Math.log(p.priors[k])
        for (let j = 0; j < d; j++)
          s -=
            0.5 * (Math.log(2 * Math.PI * p.variances[k][j]) + (X[i * d + j] - p.means[k][j]) ** 2 / p.variances[k][j])
        return s
      })
      if (labels[i] >= 0) {
        r[i * K + labels[i]] = 1
        ll += l[labels[i]]
        continue
      }
      const best = Math.max(...l)
      const z = l.reduce((a, v) => a + Math.exp(v - best), 0)
      l.forEach((v, k) => (r[i * K + k] = Math.exp(v - best) / z))
      ll += lambda * (best + Math.log(z))
    }
    return { ...p, r, ll: ll / n }
  }
  const clamped = new Float64Array(n * K)
  labels.forEach((c, i) => c >= 0 && (clamped[i * K + c] = 1))
  return { init: () => eStep(mStep(clamped)), step: (s: EmState) => eStep(mStep(s.r)) }
}

/** The options under which the merged export is the site's diagonal EM. */
const emOptions = (lambda: number, floor: number): SemiSupervisedMixtureOptions => ({
  covariance: 'diagonal',
  start: 'per-class',
  regularisation: 0,
  varianceFloor: floor,
  objective: 'per-row',
  unlabelledWeight: lambda,
})

// ── The site's tied and full mixture (semi-supervised-mixture.ts), inlined ─────────────────────────────────────────

type Params = { weights: Float64Array; means: Float64Array; covariances: Float64Array }

function candidateMixture(
  X: Float64Array,
  n: number,
  d: number,
  labels: number[],
  K: number,
  full: boolean,
  lambda: number,
) {
  const mStep = (r: Float64Array, w: Float64Array, tied: boolean, reg: number): Params => {
    const Nk = new Float64Array(K)
    const means = new Float64Array(K * d)
    let total = 0
    for (let i = 0; i < n; i++) {
      total += w[i]
      for (let k = 0; k < K; k++) {
        const a = w[i] * r[i * K + k]
        Nk[k] += a
        for (let j = 0; j < d; j++) means[k * d + j] += a * X[i * d + j]
      }
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) means[k * d + j] /= Nk[k] || 1
    const scatter = new Float64Array(K * d * d)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        const a = w[i] * r[i * K + k]
        if (a === 0) continue
        for (let p = 0; p < d; p++)
          for (let q = 0; q < d; q++)
            scatter[k * d * d + p * d + q] += a * (X[i * d + p] - means[k * d + p]) * (X[i * d + q] - means[k * d + q])
      }
    const covariances = new Float64Array(K * d * d)
    if (tied) {
      const shared = new Float64Array(d * d)
      for (let k = 0; k < K; k++) for (let e = 0; e < d * d; e++) shared[e] += scatter[k * d * d + e] / total
      for (let p = 0; p < d; p++) shared[p * d + p] += reg
      for (let k = 0; k < K; k++) covariances.set(shared, k * d * d)
    } else
      for (let k = 0; k < K; k++) {
        for (let e = 0; e < d * d; e++) covariances[k * d * d + e] = scatter[k * d * d + e] / (Nk[k] || 1)
        for (let p = 0; p < d; p++) covariances[k * d * d + p * d + p] += reg
      }
    return { weights: Float64Array.from(Nk, (v) => v / total), means, covariances }
  }
  const evaluate = (p: Params) => {
    const r = new Float64Array(n * K)
    let sumL = 0
    let sumU = 0
    const lj = new Float64Array(n * K)
    for (let k = 0; k < K; k++) {
      const { L } = cholesky(fromData(p.covariances.slice(k * d * d, (k + 1) * d * d), [d, d]))
      const base = Math.log(p.weights[k]) - 0.5 * (d * Math.log(2 * Math.PI) + (choleskyLogDet(L) as number))
      for (let i = 0; i < n; i++) {
        const dev = Array.from({ length: d }, (_, j) => X[i * d + j] - p.means[k * d + j])
        const sol = toFlat(choleskySolve(L, tensor(dev)))
        lj[i * K + k] = base - 0.5 * dev.reduce((a, v, j) => a + v * sol[j], 0)
      }
    }
    let nL = 0
    for (let i = 0; i < n; i++) {
      if (labels[i] >= 0) {
        nL++
        r[i * K + labels[i]] = 1
        sumL += lj[i * K + labels[i]]
        continue
      }
      const top = Math.max(lj[i * K], lj[i * K + 1])
      const z = Math.exp(lj[i * K] - top) + Math.exp(lj[i * K + 1] - top)
      for (let k = 0; k < K; k++) r[i * K + k] = Math.exp(lj[i * K + k] - top) / z
      sumU += top + Math.log(z)
    }
    return { ...p, r, objective: (sumL + lambda * sumU) / (nL + lambda * (n - nL)) }
  }
  const one = new Float64Array(n * K)
  const lw = new Float64Array(n)
  labels.forEach((c, i) => c >= 0 && ((one[i * K + c] = 1), (lw[i] = 1)))
  const w = Float64Array.from(labels, (c) => (c >= 0 ? 1 : lambda))
  return {
    init: () => evaluate(mStep(one, lw, true, 0.1)),
    step: (s: ReturnType<typeof evaluate>) => evaluate(mStep(s.r, w, !full, 1e-6)),
  }
}

// ── Data ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Two Gaussian classes in 2-d, 60 points each, 3 labels per class (the mixture self-check's data). */
const s7 = stream(7)
const cov = {
  covariance: tensor([
    [1, 0.4],
    [0.4, 0.6],
  ]),
}
const a = float64Data(multivariateNormal(child(s7, 'a'), [0, 0], cov, { shape: [60] }))
const b = float64Data(multivariateNormal(child(s7, 'b'), [2.5, 0.5], cov, { shape: [60] }))
const X2 = new Float64Array([...a, ...b])
const x2 = fromData(X2, [120, 2])
const truth2 = Array.from({ length: 120 }, (_, i) => (i < 60 ? 0 : 1))
const labels2 = truth2.map((c, i) => (i % 20 === 0 ? c : -1))

/** A 1-d two-class sample with 6 labels per class and 200 unlabelled points from N(-1, 1) and N(1.5, 1). */
function lineSample(seed: number) {
  const s = stream(seed)
  const z = toFlat(normals(child(s, 'z'), 212))
  const u = toFlat(uniform(child(s, 'c'), 0, 1, { shape: [212] }))
  const cls = u.map((v, i) => (i < 6 ? 0 : i < 12 ? 1 : v < 0.5 ? 0 : 1))
  return { x: z.map((v, i) => [(cls[i] === 0 ? -1 : 1.5) + v]), labels: cls.map((c, i) => (i < 12 ? c : -1)) }
}

const maxDiff = (p: ArrayLike<number>, q: ArrayLike<number>) => Math.max(...Array.from(p, (v, i) => Math.abs(v - q[i])))

describe('semiSupervisedMixtureSteps reproduces the site’s diagonal EM', () => {
  it('state 0 is the labelled maximum-likelihood fit: means 1 and 12, variances 1 and 4, priors 1/2', () => {
    const problem = { x: [[0], [2], [10], [14], [5], [6]], labels: [0, 0, 1, 1, -1, -1], classes: 2 }
    const s0 = run(semiSupervisedMixtureSteps(problem, emOptions(1, 1e-3)), undefined, 0)
    expect(Array.from(s0.means)).toEqual([1, 12])
    expect(Array.from(s0.covariances)).toEqual([1, 4])
    expect(Array.from(s0.weights)).toEqual([0.5, 0.5])
    // λ = 0 is converged at state 0.
    expect(run(semiSupervisedMixtureSteps(problem, emOptions(0, 1e-3)), undefined, 0).converged).toBe(true)
  })

  it('matches the candidate to rounding over 50 steps, and its objective never falls', () => {
    for (const [seed, lambda, floor] of [
      [1, 1, 0.01],
      [2, 0.3, 1e-3],
    ] as const) {
      const { x, labels } = lineSample(seed)
      const ours = semiSupervisedMixtureSteps({ x, labels, classes: 2 }, emOptions(lambda, floor))
      const theirs = candidateEm(x, labels, 2, lambda, floor)
      let p = ours.init(undefined, stream(0))
      let q = theirs.init()
      for (let t = 0; t <= 50; t++) {
        expect(maxDiff(p.means, q.means.flat())).toBeLessThan(1e-9)
        expect(maxDiff(p.covariances, q.variances.flat())).toBeLessThan(1e-9)
        expect(maxDiff(p.weights, q.priors)).toBeLessThan(1e-12)
        expect(Math.abs(p.objective - q.ll)).toBeLessThan(1e-9)
        const next = ours.step(p, { t, stream: stream(0) })
        expect(next.objective).toBeGreaterThanOrEqual(p.objective - 1e-12)
        p = next
        q = theirs.step(q)
      }
    }
  })

  it('stops at the same step as the candidate under the open-set study’s options', () => {
    const { x, labels } = lineSample(3)
    const options = { ...emOptions(1, 0.01), tolerance: 1e-7 }
    const final = run(semiSupervisedMixtureSteps({ x, labels, classes: 2 }, options), undefined, 300)
    const theirs = candidateEm(x, labels, 2, 1, 0.01)
    let q = theirs.init()
    let t = 0
    for (; t < 300; t++) {
      const next = theirs.step(q)
      const done = Math.abs(next.ll - q.ll) < 1e-7
      q = next
      if (done) break
    }
    expect(final.t).toBe(t + 1)
    expect(maxDiff(final.means, q.means.flat())).toBeLessThan(1e-9)
  })

  it('gives the supervised-limit error 0.1044 of the bimodal note', () => {
    // The exact error of the rule (priors 1/2, means -0.25 and 0.5, variances 5.4225 and 0.36) on the problem whose
    // class 0 is 0.5 N(-2.5, 0.6²) + 0.5 N(2, 0.6²) and class 1 is N(0.5, 0.6²).
    const v0 = 5.4225
    const v1 = 0.36
    const [m0, m1] = [-0.25, 0.5]
    const qa = 0.5 / v0 - 0.5 / v1
    const qb = m1 / v1 - m0 / v0
    const qc = -0.5 * Math.log(v1 / v0) - (m1 * m1) / (2 * v1) + (m0 * m0) / (2 * v0)
    const r = Math.sqrt(qb * qb - 4 * qa * qc)
    const [lo, hi] = [(-qb - r) / (2 * qa), (-qb + r) / (2 * qa)].sort((u, v) => u - v)
    const mass = (m: number, s: number, x0: number, x1: number) => normalCdf((x1 - m) / s) - normalCdf((x0 - m) / s)
    const class0 = (x0: number, x1: number) => 0.5 * mass(-2.5, 0.6, x0, x1) + 0.5 * mass(2, 0.6, x0, x1)
    // Class 1 is predicted between the roots.
    const error = 0.5 * class0(lo, hi) + 0.5 * (mass(0.5, 0.6, -Infinity, lo) + mass(0.5, 0.6, hi, Infinity))
    expect(Math.abs(error - 0.1044)).toBeLessThan(5e-4)
  })
})

describe('semiSupervisedMixtureSteps reproduces the site’s tied and full mixture', () => {
  it('matches the candidate to rounding over 15 steps (tied and full, λ = 1 and 0.3)', () => {
    for (const covariance of ['tied', 'full'] as const)
      for (const lambda of [1, 0.3]) {
        const ours = semiSupervisedMixtureSteps(
          { x: x2, labels: labels2, classes: 2 },
          { covariance, unlabelledWeight: lambda },
        )
        const theirs = candidateMixture(X2, 120, 2, labels2, 2, covariance === 'full', lambda)
        let p = ours.init(undefined, stream(0))
        let q = theirs.init()
        for (let t = 0; t <= 15; t++) {
          expect(maxDiff(p.means, q.means)).toBeLessThan(1e-9)
          expect(maxDiff(p.covariances, q.covariances)).toBeLessThan(1e-9)
          expect(Math.abs(p.objective - q.objective)).toBeLessThan(1e-9)
          p = ours.step(p, { t, stream: stream(0) })
          q = theirs.step(q)
        }
      }
  })

  it('with every label, one step gives the class sample means and the class frequencies', () => {
    const alg = semiSupervisedMixtureSteps({ x: x2, labels: truth2, classes: 2 }, { regularisation: 0 })
    const st = run(alg, undefined, 1)
    const m0 = [0, 1].map((j) => a.filter((_, e) => e % 2 === j).reduce((p, q) => p + q, 0) / 60)
    expect(Math.abs(st.means[0] - m0[0])).toBeLessThan(1e-9)
    expect(Math.abs(st.means[1] - m0[1])).toBeLessThan(1e-9)
    expect(st.weights[0]).toBeCloseTo(0.5, 12)
  })

  it('with λ = 0 and a tied covariance the labelled-only fit is a fixed point', () => {
    const alg = semiSupervisedMixtureSteps({ x: x2, labels: labels2, classes: 2 }, { unlabelledWeight: 0, ridge: 1e-6 })
    expect(maxDiff(run(alg, undefined, 0).means, run(alg, undefined, 5).means)).toBeLessThan(1e-12)
  })

  it('with no labels and full covariances, 3 steps are gaussianMixtureSteps from the same start', () => {
    const start = labelledOnlyFit({ x: x2, labels: labels2, classes: 2 })
    const ours = run(
      semiSupervisedMixtureSteps({ x: x2, labels: truth2.map(() => -1), classes: 2 }, { covariance: 'full' }),
      start,
      3,
    )
    const engine = run(
      gaussianMixtureSteps(x2, { k: 2, covariance: 'full', regularisation: 1e-6 }),
      {
        weights: fromData(start.weights, [2]),
        means: fromData(start.means, [2, 2]),
        covariances: fromData(start.covariances, [2, 2, 2]),
      },
      3,
    )
    expect(maxDiff(ours.means, float64Data(engine.means))).toBeLessThan(1e-9)
    expect(Math.abs(ours.objective - engine.logLikelihood)).toBeLessThan(1e-9)
  })

  it('the objective never falls over 100 steps, posteriors sum to one and labelled rows stay one-hot', () => {
    for (const covariance of ['tied', 'full'] as const)
      for (const unlabelledWeight of [1, 0.3]) {
        const alg = semiSupervisedMixtureSteps({ x: x2, labels: labels2, classes: 2 }, { covariance, unlabelledWeight })
        let st = alg.init(undefined, stream(0))
        for (let t = 0; t < 100 && !st.converged; t++) {
          const next = alg.step(st, { t, stream: stream(0) })
          expect(next.objective - st.objective).toBeGreaterThan(-1e-10)
          st = next
        }
      }
    const st = run(semiSupervisedMixtureSteps({ x: x2, labels: labels2, classes: 2 }), undefined, 10)
    const p = classPosterior(st, x2)
    for (let i = 0; i < 120; i++) expect(p[2 * i] + p[2 * i + 1]).toBeCloseTo(1, 12)
    labels2.forEach((c, i) => c >= 0 && expect(st.responsibilities[2 * i + c]).toBe(1))
  })

  it('refuses a class with no label, a bad λ and mismatched rows', () => {
    expect(() => labelledOnlyFit({ x: x2, labels: truth2.map(() => -1), classes: 2 })).toThrow(/no labelled/)
    expect(() => semiSupervisedMixtureSteps({ x: x2, labels: labels2, classes: 2 }, { unlabelledWeight: 2 })).toThrow(
      /unlabelledWeight/,
    )
    expect(() => semiSupervisedMixtureSteps({ x: x2, labels: [0, 1], classes: 2 })).toThrow(/rows/)
  })
})

describe('gaussianMixtureSteps with a tied covariance', () => {
  it('shares one covariance between the components', () => {
    const final = run(gaussianMixtureSteps(x2, { k: 2, covariance: 'tied' }), {}, 20, { stream: stream(1) })
    const c = float64Data(final.covariances)
    expect(maxDiff(c.slice(0, 4), c.slice(4, 8))).toBe(0)
  })
})
