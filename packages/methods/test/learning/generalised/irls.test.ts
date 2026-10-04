import { describe, expect, it } from 'vitest'
import { irls } from 'aifn-methods/learning/generalised'
import { binomialFamily, likelihood, link } from 'aifn-compute/probability/likelihoods'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

/**
 * A nearly separable logistic problem (the lab's pipeline figure): standardised x₁, x₂, their squares and product,
 * standardised again, and an intercept column. Some points reach |η| ≈ 37, where σ(η) rounds to 1.
 */
function parabolaDesign(n: number) {
  const s = stream('pipeline-steps')
  const u = toFlat(normals(child(s, 'u'), [n]))
  const v = toFlat(normals(child(s, 'v'), [n]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const std = (cols: number[][]) =>
    cols.map((c) => {
      const m = c.reduce((a, b) => a + b, 0) / c.length
      const sd = Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length)
      return c.map((x) => (x - m) / sd)
    })
  const [a, b] = std([Array.from(u, (x) => 5 + 3 * x), Array.from(v, (x) => 10 + 0.5 * x)])
  const cols = std([a, b, a.map((x) => x * x), a.map((x, i) => x * b[i]), b.map((x) => x * x)])
  const p = cols.length + 1
  const X = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    cols.forEach((c, j) => (X[i * p + j] = c[i]))
    X[i * p + p - 1] = 1
  }
  const y = Float64Array.from(u, (x, i) => (x * x + v[i] + 0.4 * e[i] > 0.8 ? 1 : 0))
  return { X, y, n, p }
}

describe('irls', () => {
  it('converges quadratically on a nearly separable penalised logistic problem, with the KKT gradient ≈ 0', () => {
    const { X, y, n, p } = parabolaDesign(150)
    const P = new Float64Array(p * p)
    for (let j = 0; j < p - 1; j++) P[j * p + j] = 0.1
    const run = trace(
      irls({
        design: fromData(X, [n, p]),
        y: fromData(y, [n]),
        family: binomialFamily(),
        link: link('logit'),
        penalty: fromData(P, [p, p]),
        tolerance: 1e-12,
      }),
      {},
      50,
    )
    const final = run.final
    expect(final.converged).toBe(true)
    expect(final.diverged).toBe(false)
    expect(final.t).toBeLessThan(15)
    // ∇(D + βᵀPβ)/2 = −Xᵀ(y − μ) + Pβ.
    const beta = toFlat(final.coefficients as Tensor)
    const mu = toFlat(final.mu)
    const g = new Float64Array(p)
    for (let i = 0; i < n; i++) for (let a = 0; a < p; a++) g[a] -= X[i * p + a] * (y[i] - mu[i])
    for (let a = 0; a < p; a++) g[a] += P[a * p + a] * beta[a]
    expect(Math.hypot(...g)).toBeLessThan(1e-6)
  })

  it('the binomial deviance from η stays finite and exact where μ rounds to 0 or 1', () => {
    for (const name of ['logit', 'cloglog'] as const) {
      const lik = likelihood(binomialFamily(), name)
      const eta = fromData(Float64Array.of(-3, 0.5, 2), [3])
      const y = fromData(Float64Array.of(0, 1, 1), [3])
      const direct = toFlat(binomialFamily().unitDeviance(y, link(name).inverse(eta)) as Tensor)
      toFlat(lik.unitDeviance(y, eta) as Tensor).forEach((d, i) => expect(d).toBeCloseTo(direct[i], 10))
    }
    const far = toFlat(
      likelihood(binomialFamily(), 'logit').unitDeviance(
        fromData(Float64Array.of(0), [1]),
        fromData(Float64Array.of(40), [1]),
      ) as Tensor,
    )
    expect(far[0]).toBeCloseTo(2 * 40, 8)
  })
})
