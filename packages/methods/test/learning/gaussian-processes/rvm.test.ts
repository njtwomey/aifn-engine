/** The relevance vector machine (`rvm.ts`): the posterior against a direct evidence, both fits on noisy sinc. */
import { describe, expect, it } from 'vitest'
import {
  relevanceVectorMachine,
  rvmFastSteps,
  rvmModel,
  rvmPosterior,
  rvmProblem,
  rvmReestimationSteps,
  type RvmState,
} from 'aifn-methods/learning/gaussian-processes'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { isEntry } from 'aifn-compute/foundation/registry'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { cholesky, choleskyLogDet, solveTriangular } from 'aifn-compute/numerics/linalg'

const sinc = (x: number) => (x === 0 ? 1 : Math.sin(x) / x)

/** Tipping's (2001, §5.1) sinc data: n inputs uniform on [−10, 10], Gaussian noise of sd `sd`. */
function sincData(n: number, sd: number, seed: string) {
  const s = stream(seed)
  const x = toFlat(uniform(child(s, 'x'), -10, 10, { shape: [n] }) as Tensor)
  const e = toFlat(normals(child(s, 'e'), n, 0, sd))
  return {
    x: fromData(Float64Array.from(x), [n]),
    y: fromData(
      Float64Array.from(x, (v, i) => sinc(v) + e[i]),
      [n],
    ),
  }
}

/** log N(y | 0, β⁻¹I + Φ_M A⁻¹ Φ_Mᵀ), computed directly from the N × N covariance. */
function directEvidence(phi: Tensor, y: Tensor, active: number[], alpha: ArrayLike<number>, beta: number) {
  const P = toFlat(phi)
  const [n, B] = phi.shape
  const C = new Float64Array(n * n)
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      let v = r === c ? 1 / beta : 0
      for (const i of active) v += (P[r * B + i] * P[c * B + i]) / alpha[i]
      C[r * n + c] = v
    }
  const { L } = cholesky(fromData(C, [n, n]), { jitter: false })
  const z = toFlat(solveTriangular(L, y) as Tensor)
  return -0.5 * (n * Math.log(2 * Math.PI) + (choleskyLogDet(L) as number) + z.reduce((a, v) => a + v * v, 0))
}

const data = sincData(100, 0.1, 'rvm-sinc')
const kernel = rbf({ lengthscale: 2 })
const problem = rvmProblem(kernel, data.x, data.y)

describe('the RVM posterior', () => {
  it('gives the log marginal likelihood of N(0, β⁻¹I + ΦA⁻¹Φᵀ)', () => {
    const active = [0, 4, 17, 50, 81]
    const alpha = new Float64Array(problem.size).fill(Infinity)
    active.forEach((i, k) => (alpha[i] = 0.5 + k))
    const post = rvmPosterior(problem, active, alpha, 80)
    const phi = problem.basis(data.x)
    expect(post.logMarginal).toBeCloseTo(directEvidence(phi, data.y, active, alpha, 80), 6)
  })

  it('has sᵢ, qᵢ consistent with the one-precision form of L', () => {
    // For an inactive φᵢ, L(αᵢ) − L(∞) = ½[ln αᵢ − ln(αᵢ + sᵢ) + qᵢ²/(αᵢ + sᵢ)].
    const active = [0, 10, 30]
    const alpha = new Float64Array(problem.size).fill(Infinity)
    for (const i of active) alpha[i] = 2
    const post = rvmPosterior(problem, active, alpha, 50)
    const i = 61
    const a = 3
    const s = post.S[i]
    const q = post.Q[i]
    const withI = Float64Array.from(alpha)
    withI[i] = a
    const grown = rvmPosterior(
      problem,
      [...active, i].sort((p, r) => p - r),
      withI,
      50,
    )
    const predicted = 0.5 * (Math.log(a) - Math.log(a + s) + (q * q) / (a + s))
    expect(grown.logMarginal - post.logMarginal).toBeCloseTo(predicted, 8)
  })
})

describe('the fast RVM (Tipping and Faul, 2003)', () => {
  const tr = trace(rvmFastSteps(problem), undefined, 400)
  const states = tr.steps as RvmState[]

  it('never decreases the log marginal likelihood, and each gain is the change in L', () => {
    for (let t = 1; t < states.length; t++) {
      expect(states[t].logMarginal).toBeGreaterThanOrEqual(states[t - 1].logMarginal - 1e-9)
      expect(states[t].gain).toBeCloseTo(states[t].logMarginal - states[t - 1].logMarginal, 9)
    }
    expect(tr.meta.stopped).toBe('done')
  })

  it('finds a handful of relevance vectors on sinc at noise 0.1 (Tipping, 2001: 6 to 10)', () => {
    const final = tr.final
    const model = rvmModel(problem, final)
    expect(model.relevanceVectors.length).toBeGreaterThanOrEqual(4)
    expect(model.relevanceVectors.length).toBeLessThanOrEqual(12)
    // The noise sd is recovered and the fit is close to sinc.
    expect(Math.sqrt(1 / final.beta)).toBeGreaterThan(0.07)
    expect(Math.sqrt(1 / final.beta)).toBeLessThan(0.14)
    const xs = linspace(-9, 9, 61)
    const mean = toFlat(model.predict(xs).mean)
    const rms = Math.sqrt(toFlat(xs).reduce((a, x, k) => a + (mean[k] - sinc(x)) ** 2, 0) / 61)
    expect(rms).toBeLessThan(0.06)
    // The evidence matches the direct computation at the final state.
    expect(final.logMarginal).toBeCloseTo(
      directEvidence(problem.basis(data.x), data.y, final.active, final.alpha, final.beta),
      5,
    )
  })

  it('takes every kind of action and records the basis function acted on', () => {
    const actions = new Set(states.map((s) => s.action))
    expect(actions.has('add')).toBe(true)
    expect(actions.has('re-estimate')).toBe(true)
    expect(actions.has('noise')).toBe(true)
    for (const s of states.slice(1))
      if (s.action === 'add') expect(s.active).toContain(s.index)
      else if (s.action === 'delete') expect(s.active).not.toContain(s.index)
  })
})

describe('RVM re-estimation (Tipping, 2001)', () => {
  const tr = trace(rvmReestimationSteps(problem), undefined, 3000)

  it('prunes to a sparse model at which no single fast-algorithm change gains anything', () => {
    expect(tr.meta.stopped).toBe('done')
    expect(tr.final.active.length).toBeLessThan(15)
    // The two fits reach local maxima of L (not necessarily the same one); each is a fixed point of the other's moves.
    const next = rvmFastSteps(problem).step({ ...tr.final, converged: false }, { t: 0, stream: stream('unused') })
    expect(next.gain).toBeLessThan(1e-3)
    expect(
      Math.abs(tr.final.logMarginal - trace(rvmFastSteps(problem), undefined, 400).final.logMarginal),
    ).toBeLessThan(5)
  })
})

describe('the RVM estimator', () => {
  it('is a registered model with a predictive distribution', () => {
    expect(isEntry(relevanceVectorMachine)).toBe(true)
    expect(isEntry(rvmFastSteps, 'algorithm')).toBe(true)
    expect(isEntry(rvmReestimationSteps, 'algorithm')).toBe(true)
    const model = relevanceVectorMachine({ kernel }).fit(dataset(data.x, data.y))
    const xs = linspace(-5, 5, 11)
    expect(toFlat(model.expect(xs))).toEqual(toFlat(model.forward(xs)))
    const p = model.predictive(xs)
    expect(p).toBeDefined()
    expect(model.rvm.relevanceVectors.length).toBeGreaterThan(2)
  })
})
