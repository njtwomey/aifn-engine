/**
 * Data attribution by its laws: KNN-Shapley equals exact Shapley values of the k-NN utility by enumeration; TMC data
 * Shapley is efficient per permutation and converges to exact Shapley values; influence functions predict the change
 * of the test loss on leave-one-out retraining of an L2 logistic regression, LiSSA approaches the exact H⁻¹v, and
 * self-influence is non-negative; TracIn is the rate-weighted sum of per-example gradient products.
 */
import { describe, expect, it } from 'vitest'
import { normal, stream } from 'aifn-compute/foundation/random'
import { add, fromData, mean, mul, sub, sum, square, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { correlation } from 'aifn-compute/probability/stats'
import {
  dataShapley,
  exactShapley,
  exampleGradients,
  influenceFunctions,
  knnShapley,
  tracIn,
} from 'aifn-compute/learning/explain'

describe('KNN-Shapley', () => {
  it('equals exact Shapley values of the k-NN utility', () => {
    const z = toFlat(normal(stream('knn'), 0, 1, { shape: [9 * 2] }))
    const X = Array.from({ length: 8 }, (_, i) => [z[2 * i], z[2 * i + 1]])
    const y = [0, 1, 1, 0, 1, 0, 0, 1]
    const tests = [
      { x: [z[16], z[17]], y: 1 },
      { x: [0.1, -0.2], y: 0 },
    ]
    const K = 3
    const r = knnShapley({ x: X, y }, { x: tests.map((t) => t.x), y: tests.map((t) => t.y) }, K)
    const exact = new Float64Array(8)
    for (const t of tests) {
      const dist = X.map((p) => (p[0] - t.x[0]) ** 2 + (p[1] - t.x[1]) ** 2)
      const v = (mask: readonly boolean[]) => {
        const S = mask
          .map((b, i) => (b ? i : -1))
          .filter((i) => i >= 0)
          .sort((a, b) => dist[a] - dist[b] || a - b)
        return S.slice(0, K).reduce((a, i) => a + (y[i] === t.y ? 1 : 0), 0) / K
      }
      exactShapley(v, 8).values.forEach((s, i) => (exact[i] += s / tests.length))
    }
    exact.forEach((s, i) => expect(r.values[i]).toBeCloseTo(s, 12))
  })
})

describe('TMC data Shapley', () => {
  const a = [1, 2, -1, 0.5, 3, 0]
  const v = (S: readonly number[]) => S.reduce((s, i) => s + a[i], 0) ** 2 / 10
  it('converges to exact Shapley values and is efficient', () => {
    const exact = exactShapley((m) => v(m.map((b, i) => (b ? i : -1)).filter((i) => i >= 0)), 6).values
    const r = dataShapley(v, 6, stream('tmc'), { permutations: 3000 })
    exact.forEach((s, i) => expect(Math.abs(r.values[i] - s)).toBeLessThan(4 * r.standardError[i] + 1e-9))
    expect(r.values.reduce((x, y) => x + y, 0)).toBeCloseTo(v([0, 1, 2, 3, 4, 5]) - v([]), 10)
    expect(r.history.shape).toEqual([300, 6])
  })
  it('truncates once the score is within tolerance', () => {
    const r = dataShapley((S) => Math.min(1, S.length / 2), 6, stream('tmc-t'), { permutations: 20, tolerance: 1e-9 })
    expect(r.truncated).toBeGreaterThan(0.5)
    expect(r.values.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 10)
  })
})

// L2 logistic regression with a bias feature: ℓ(θ, x, y) = softplus(θᵀx) − y θᵀx.
const loss = (theta: Tensor, x: Tensor, y: Tensor) => {
  const z = sum(mul(theta, x))
  return sub(softplus(z), mul(y, z))
}
function logisticData(seed: string, n: number) {
  const z = toFlat(normal(stream(seed), 0, 1, { shape: [n * 3] }))
  const x = Array.from({ length: n }, (_, i) => [z[3 * i], z[3 * i + 1], 1])
  const y = x.map((r, i) => (r[0] - 0.5 * r[1] + 0.8 * z[3 * i + 2] > 0 ? 1 : 0))
  return { x, y }
}
function fit(data: { x: number[][]; y: number[] }, l2: number) {
  const X = fromData(Float64Array.from(data.x.flat()), [data.x.length, 3])
  const Y = fromData(Float64Array.from(data.y), [data.y.length])
  const objective = {
    kind: 'objective' as const,
    name: 'logistic',
    dim: 3,
    value: (t: Tensor) => {
      const margin = sum(mul(X, t), -1)
      return add(mean(sub(softplus(margin), mul(Y, margin))), mul(l2 / 2, sum(square(t))))
    },
  }
  return Float64Array.from(
    toFlat(minimize(objective, [0, 0, 0], { method: 'lbfgs', tolerance: 1e-12, maxSteps: 500 } as never).x),
  )
}

describe('influence functions', () => {
  const l2 = 0.05
  const train = logisticData('inf-train', 40)
  const test = logisticData('inf-test', 15)
  const theta = fit(train, l2)
  const testLoss = (t: Float64Array) =>
    test.x.reduce((a, r, i) => {
      const z = r[0] * t[0] + r[1] * t[1] + r[2] * t[2]
      return a + Math.log1p(Math.exp(z)) - test.y[i] * z
    }, 0)
  it('predict leave-one-out changes of the test loss', () => {
    const r = influenceFunctions(loss, theta, train, test, { l2 })
    const base = testLoss(theta)
    const actual = Array.from({ length: 12 }, (_, i) => {
      const keep = train.x.map((_, k) => k).filter((k) => k !== i)
      const t = fit({ x: keep.map((k) => train.x[k]), y: keep.map((k) => train.y[k]) }, l2)
      return testLoss(t) - base
    })
    const predicted = Array.from(r.removal.slice(0, 12))
    expect(correlation(predicted, actual)).toBeGreaterThan(0.95)
    for (const s of r.selfInfluence) expect(s).toBeGreaterThanOrEqual(0)
  })
  it('LiSSA approaches the exact inverse Hessian-vector product', () => {
    const exact = influenceFunctions(loss, theta, train, test, { l2 })
    const lissa = influenceFunctions(loss, theta, train, test, {
      l2,
      method: 'lissa',
      depth: 300,
      scale: 2,
      damping: 0,
      batch: 40,
      stream: stream('lissa'),
    })
    const err = Math.hypot(...exact.inverseHvp.map((v, i) => v - lissa.inverseHvp[i]))
    expect(err / Math.hypot(...exact.inverseHvp)).toBeLessThan(0.1)
  })
})

describe('TracIn', () => {
  it('is the rate-weighted sum of gradient products', () => {
    const train = logisticData('tracin', 10)
    const test = logisticData('tracin-test', 3)
    const cps = [Float64Array.of(0.1, -0.2, 0), Float64Array.of(0.5, 0.1, -0.1)]
    const r = tracIn(loss, cps, [0.3, 0.1], train, test)
    const want = new Float64Array(30)
    const self = new Float64Array(10)
    cps.forEach((th, t) => {
      const eta = [0.3, 0.1][t]
      const G = toFlat(exampleGradients(loss, th, train))
      const Gt = toFlat(exampleGradients(loss, th, test))
      for (let i = 0; i < 10; i++) {
        for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) want[i * 3 + k] += eta * G[i * 3 + j] * Gt[k * 3 + j]
        for (let j = 0; j < 3; j++) self[i] += eta * G[i * 3 + j] ** 2
      }
    })
    const got = toFlat(r.influence as Tensor)
    want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, 12))
    self.forEach((w, i) => expect(r.selfInfluence[i]).toBeCloseTo(w, 12))
    // The gradient of the logistic loss is (σ(θᵀx) − y) x.
    const G = toFlat(exampleGradients(loss, cps[0], train))
    const x0 = train.x[0]
    const z = x0[0] * 0.1 - 0.2 * x0[1]
    const s = 1 / (1 + Math.exp(-z)) - train.y[0]
    expect(G[0]).toBeCloseTo(s * x0[0], 12)
  })
})
