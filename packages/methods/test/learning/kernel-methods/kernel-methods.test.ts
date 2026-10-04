import { describe, expect, it } from 'vitest'
import {
  crammerSinger,
  crammerSingerSteps,
  dualCoordinateSteps,
  linearSvm,
  pegasosSteps,
  smoSteps,
  supportVectorMachine,
} from 'aifn-methods/learning/kernel-methods'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { plattScaling } from 'aifn-compute/learning/calibration'
import { dataset } from 'aifn-compute/learning/estimators'
import { linearKernel, rbf } from 'aifn-compute/learning/kernels'
import { expectProtocol } from '../../protocol'
import { close, fx, X2, X3, XQ, Y2, Y3 } from '../shared'

describe('Platt scaling', () => {
  // The regularised cross-entropy Platt minimises, for a finite-difference check of its stationarity.
  const objective = (f: number[], y: number[], A: number, B: number) => {
    const pos = y.filter((v) => v === 1).length
    const [hi, lo] = [(pos + 1) / (pos + 2), 1 / (y.length - pos + 2)]
    return f.reduce((s, fi, i) => {
      const z = A * fi + B
      return s + (y[i] === 1 ? hi : lo) * z + Math.log1p(Math.exp(-z))
    }, 0)
  }
  const f = [-2.1, -1.4, -0.9, -0.3, 0.2, -0.1, 0.4, 0.8, 1.3, 2.2, 0.6, -0.6]
  const y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1]

  it('reaches a stationary point of the regularised cross-entropy', () => {
    const p = plattScaling(tensor(f), tensor(y))
    expect(p.converged).toBe(true)
    expect(p.A).toBeLessThan(0)
    const h = 1e-6
    const dA = (objective(f, y, p.A + h, p.B) - objective(f, y, p.A - h, p.B)) / (2 * h)
    const dB = (objective(f, y, p.A, p.B + h) - objective(f, y, p.A, p.B - h)) / (2 * h)
    expect(Math.abs(dA)).toBeLessThan(1e-4)
    expect(Math.abs(dB)).toBeLessThan(1e-4)
    const probs = toFlat(p.probability(tensor([-3, 0, 3])))
    expect(probs[0]).toBeLessThan(probs[1])
    expect(probs[1]).toBeLessThan(probs[2])
    expect(probs[1]).toBeCloseTo(1 / (1 + Math.exp(p.B)), 12)
  })

  it('is symmetric for mirrored scores and stays finite on separable scores', () => {
    const sym = plattScaling(tensor([-2, -1, 1, 2]), tensor([0, 0, 1, 1]))
    expect(sym.B).toBeCloseTo(0, 8)
    const sep = plattScaling(tensor([-3, -2, -1, 1, 2, 3]), tensor([0, 0, 0, 1, 1, 1]))
    expect(Number.isFinite(sep.A)).toBe(true)
    // The targets cap the probabilities: at most (N₊ + 1)/(N₊ + 2) = 0.8 is reachable in the mean.
    expect(toFlat(sep.probability(tensor([3])))[0]).toBeLessThan(1)
  })

  it('gives the kernel SVM a predictive probability', () => {
    const m = supportVectorMachine({ probability: true }).fit(dataset(X2, Y2))
    expect(m.platt).not.toBeNull()
    const scores = toFlat(m.score(X2))
    const p = toFlat(m.predictive!(X2).mean() as never)
    // P(y = 1 | x) is increasing in the decision value.
    const order = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b])
    for (let r = 1; r < order.length; r++) expect(p[order[r]]).toBeGreaterThanOrEqual(p[order[r - 1]] - 1e-12)
    expect(toFlat(m.platt!.probability(m.score(X2)))).toEqual(p)
    expect(supportVectorMachine().fit(dataset(X2, Y2)).platt).toBeNull()
  })
})

describe('support vector machines', () => {
  it('SMO matches scikit-learn SVC (RBF and linear kernels)', () => {
    for (const selection of ['second-order', 'maximal-violating'] as const) {
      const m = supportVectorMachine({ C: 1, kernel: rbf({ lengthscale: 1 }), tolerance: 1e-6, selection }).fit(
        dataset(X2, Y2),
      )
      close(m.score(XQ), fx.svc.decision, 1e-4)
      expect(toFlat(m.supportVectors)).toEqual(fx.svc.support.slice().sort((a, b) => a - b))
    }
    const lin = supportVectorMachine({ C: 0.5, kernel: linearKernel(), tolerance: 1e-6 }).fit(dataset(X2, Y2))
    close(lin.score(XQ), fx.svc_linear.decision, 1e-4)
    close(lin.weights!, fx.svc_linear.coef, 1e-4)
  })
  it('SMO states are feasible and the dual objective rises', () => {
    const signs = tensor(fx.y3.slice(0, 40).map((c) => (c ? 1 : -1)))
    const t = trace(smoSteps({ x: X2, y: signs, C: 1 }), {}, 500)
    let prev = -Infinity
    for (const s of t.steps) {
      const a = toFlat(s.alpha)
      expect(Math.abs(a.reduce((acc, v, i) => acc + v * toFlat(signs)[i], 0))).toBeLessThan(1e-9)
      for (const v of a) expect(v >= 0 && v <= 1).toBe(true)
      expect(s.dualObjective).toBeGreaterThanOrEqual(prev - 1e-12)
      prev = s.dualObjective
    }
    expect(t.meta.stopped).toBe('done')
    expectProtocol(smoSteps({ x: X2, y: signs, C: 1 }), {}, { n: 30, record: { k: (s) => s.dualObjective } })
  })
  it('linear SVM by dual coordinate descent matches LinearSVC; Pegasos approaches it', () => {
    const m = linearSvm({ C: 0.5, tolerance: 1e-10, maxSteps: 20000 }).fit(dataset(X2, Y2))
    close(m.weights, fx.lsvc.coef, 1e-4)
    expect(m.bias).toBeCloseTo(fx.lsvc.intercept, 4)
    const p = linearSvm({ C: 0.5, method: 'pegasos', maxSteps: 20000 }).fit(dataset(X2, Y2), { stream: stream(1) })
    const gap = p.training.steps.at(-1)!.primalObjective - m.training.steps.at(-1)!.primalObjective
    expect(gap).toBeGreaterThanOrEqual(-1e-9)
    expect(gap).toBeLessThan(0.5)
  })
  it('Crammer–Singer matches LinearSVC(multi_class="crammer_singer")', () => {
    const m = crammerSinger({ C: 0.5, tolerance: 1e-10, maxSteps: 20000 }).fit(dataset(X3, Y3))
    close(m.weights, fx.cs.coef, 1e-4)
    close(m.bias, fx.cs.intercept, 1e-4)
  })
})

describe('trace protocol', () => {
  it('dual coordinate descent, Pegasos and Crammer–Singer follow it', () => {
    const signs = tensor(fx.y3.slice(0, 40).map((c) => (c ? 1 : -1)))
    expectProtocol(dualCoordinateSteps({ x: X2, y: signs, C: 0.5 }), undefined, { n: 8 })
    expectProtocol(dualCoordinateSteps({ x: X2, y: signs, C: 0.5, shuffle: false }), undefined, { n: 8 })
    expectProtocol(pegasosSteps({ x: X2, y: signs, C: 0.5 }), undefined, { n: 30 })
    expectProtocol(crammerSingerSteps({ x: X3, y: Y3, C: 0.5 }), undefined, { n: 8 })
  })
})
