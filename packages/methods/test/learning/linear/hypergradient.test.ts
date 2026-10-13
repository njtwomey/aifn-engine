/**
 * The ridge hypergradient on the standardised diabetes data (300 rows to train on and 142 to validate, or 20 and 422
 * for the descent): against central finite differences, against reverse-mode autodiff through a linear solve, and the
 * descent on the log penalty against a fine grid.
 */
import { describe, expect, it } from 'vitest'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import {
  linearAlgorithms,
  linearFunctions,
  ridgeHyperDescentSteps,
  ridgeHypergradient,
  ridgePath,
} from 'aifn-methods/learning/linear'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  eye,
  fromData,
  matmul,
  mean,
  mul,
  square,
  sub,
  toFlat,
  toRows,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import { trace } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'

const data = diabetes()
const x = standardScaler().fit({ x: data.x }).transform(data.x)
const rows = toRows(x)
const target = toFlat(data.y!)
// Targets in units of their standard deviation, so that losses are of order one.
const average = target.reduce((a, v) => a + v, 0) / target.length
const sd = Math.sqrt(target.reduce((a, v) => a + (v - average) ** 2, 0) / target.length)
const yAll = target.map((v) => v / sd)
const matrix = (r: number[][]): Tensor => fromData(Float64Array.from(r.flat()), [r.length, r[0].length])
const vector = (v: number[]): Tensor => fromData(Float64Array.from(v))
const xTrain = matrix(rows.slice(0, 300))
const yTrain = vector(yAll.slice(0, 300))
const xVal = matrix(rows.slice(300))
const yVal = vector(yAll.slice(300))

describe('ridgeHypergradient', () => {
  it('agrees with central finite differences', () => {
    for (const lambda of [0.1, 3, 50, 1000]) {
      const h = ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda)
      const step = 1e-4 * lambda
      const up = ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda + step).loss
      const down = ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda - step).loss
      const fd = (up - down) / (2 * step)
      expect(Math.abs(h.gradient - fd), `lambda ${lambda}`).toBeLessThan(1e-6 * Math.abs(fd) + 1e-12)
      expect(h.logGradient).toBeCloseTo(lambda * h.gradient, 14)
      // dw/dλ against the finite difference of the weights.
      const wUp = toFlat(ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda + step).weights)
      const wDown = toFlat(ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda - step).weights)
      toFlat(h.weightsGradient).forEach((v, j) =>
        expect(Math.abs(v - (wUp[j] - wDown[j]) / (2 * step))).toBeLessThan(1e-6 * Math.abs(v) + 1e-10),
      )
    }
  })

  it('agrees with reverse-mode autodiff through the solve', () => {
    // Centre by the training means, as ridgeHypergradient does.
    const n = 300
    const xm = Array.from({ length: 10 }, (_, j) => rows.slice(0, n).reduce((s, r) => s + r[j], 0) / n)
    const ym = yAll.slice(0, n).reduce((a, b) => a + b, 0) / n
    const Xc = matrix(rows.slice(0, n).map((r) => r.map((v, j) => v - xm[j])))
    const Vc = matrix(rows.slice(n).map((r) => r.map((v, j) => v - xm[j])))
    const yc = fromData(
      Float64Array.from(yAll.slice(0, n), (v) => v - ym),
      [n, 1],
    )
    const yv = fromData(
      Float64Array.from(yAll.slice(n), (v) => v - ym),
      [yAll.length - n, 1],
    )
    const H = matmul(transpose(Xc), Xc)
    const loss = (lambda: Value): Value => {
      const w = solve(add(H, mul(lambda, eye(10))), matmul(transpose(Xc), yc))
      return mean(square(sub(yv, matmul(Vc, w))))
    }
    for (const lambda of [0.5, 20]) {
      const h = ridgeHypergradient(xTrain, yTrain, xVal, yVal, lambda)
      expect(Number(loss(lambda))).toBeCloseTo(h.loss, 12)
      const g = Number(grad(loss)(lambda))
      expect(Math.abs(g - h.gradient)).toBeLessThan(1e-10 * Math.abs(g))
    }
  })

  it('has the weights of the ridge path', () => {
    const h = ridgeHypergradient(xTrain, yTrain, xVal, yVal, 7)
    const path = ridgePath(xTrain, yTrain, [7])
    toFlat(h.weights).forEach((v, j) => expect(v).toBeCloseTo(toFlat(path.coef)[j], 12))
    expect(h.intercept).toBeCloseTo(path.intercept[0], 12)
    expect(() => ridgeHypergradient(xTrain, yTrain, xVal, yVal, 0)).toThrow(DomainError)
  })
})

describe('ridgeHyperDescentSteps', () => {
  it('descends to the minimum of the validation loss over a fine grid of penalties', () => {
    // With 20 rows to train on, the validation loss has its minimum inside, near lambda = 10.
    const small = { xTrain: matrix(rows.slice(0, 20)), yTrain: vector(yAll.slice(0, 20)) }
    const split = { ...small, xVal: matrix(rows.slice(20)), yVal: vector(yAll.slice(20)) }
    const alg = ridgeHyperDescentSteps({ ...split, eta: 2, tolerance: 1e-10 })
    const tr = trace(alg, { lambda: 1 }, 2000, { keep: 'all' })
    const last = tr.steps.at(-1)!
    expect(last.converged).toBe(true)
    for (let t = 1; t < tr.steps.length; t++) expect(tr.steps[t].loss).toBeLessThanOrEqual(tr.steps[t - 1].loss + 1e-13)
    let best = Infinity
    let bestLambda = NaN
    for (let k = 0; k <= 4000; k++) {
      const lambda = 10 ** (-2 + (5 * k) / 4000)
      const l = ridgeHypergradient(split.xTrain, split.yTrain, split.xVal, split.yVal, lambda).loss
      if (l < best) {
        best = l
        bestLambda = lambda
      }
    }
    expect(last.loss).toBeLessThanOrEqual(best + 1e-12)
    expect(Math.abs(Math.log(last.lambda / bestLambda))).toBeLessThan(0.01)
    expect(Math.abs(last.gradient)).toBeLessThan(1e-10)
  })

  it('flags a step that overflows as diverged, throws for a bad step size, and is registered', () => {
    const alg = ridgeHyperDescentSteps({ xTrain, yTrain, xVal, yVal, eta: 1e6 })
    const tr = trace(alg, { lambda: 1 }, 50)
    expect(tr.meta.stopped).toBe('diverged')
    expect(() => ridgeHyperDescentSteps({ xTrain, yTrain, xVal, yVal, eta: 0 })).toThrow(DomainError)
    expect(linearAlgorithms.ridgeHyperDescentSteps).toBe(ridgeHyperDescentSteps)
    expect(linearFunctions.ridgeHypergradient).toBe(ridgeHypergradient)
  })
})
