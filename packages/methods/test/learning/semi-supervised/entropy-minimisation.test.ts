/** aifn-methods/learning/semi-supervised: entropy minimisation, against the note's worked line and the figure. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { blobs } from 'aifn-methods/data/synthetic'
import {
  entropyMinimisationFit,
  entropyMinimisationObjective,
  entropyMinimisationPath,
  entropyMinimisationProbabilities,
} from 'aifn-methods/learning/semi-supervised'

const LINE = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.4, 3.8, 4.2]

describe('entropyMinimisationObjective', () => {
  it('has the gradient of central differences (3 classes)', () => {
    const problem = {
      labelled: [
        [0.1, 0.2, 1],
        [0.5, -1, 0.3],
        [1, 1, 1],
      ],
      labels: [0, 2, 1],
      unlabelled: [
        [0.3, 0.3, 0.1],
        [-1, 0.5, 2],
        [0.2, 0.9, -0.4],
        [0, 0, 1],
      ],
      classes: 3,
    }
    const f = entropyMinimisationObjective(problem, { lambda: 0.7, l2: 0.3 })
    expect(f.size).toBe(12)
    const x = Array.from({ length: f.size }, (_, i) => Math.sin(3 * i + 1))
    const g = f.value(x).grad
    for (let q = 0; q < f.size; q++) {
      const a = [...x]
      const b = [...x]
      a[q] += 1e-6
      b[q] -= 1e-6
      expect(Math.abs((f.value(a).value - f.value(b).value) / 2e-6 - g[q])).toBeLessThan(1e-7)
    }
  })

  it('at zero parameters gives cross-entropy and entropy log K', () => {
    const f = entropyMinimisationObjective(
      { labelled: [[0], [1]], labels: [0, 1], unlabelled: [[0.5]], classes: 2 },
      { lambda: 1, l2: 0.1 },
    )
    const parts = f.parts([0, 0, 0, 0])
    expect(parts.crossEntropy).toBeCloseTo(Math.log(2), 15)
    expect(parts.entropy).toBeCloseTo(Math.log(2), 15)
    const bad = { labelled: [[0]], labels: [2], unlabelled: [], classes: 2 }
    expect(() => entropyMinimisationObjective(bad, { lambda: 1, l2: 0 })).toThrow(/not a class/)
  })
})

describe('entropyMinimisationFit', () => {
  it('with λ = 0 and no unlabelled points is logistic regression: P = 1/2 at the centre', () => {
    const fit = entropyMinimisationFit(
      { labelled: [[0], [1], [3], [4]], labels: [0, 0, 1, 1], unlabelled: [], classes: 2 },
      { lambda: 0, l2: 0.1 },
    )
    const [w0, w1] = fit.params
    expect(Math.abs(w0 + w1)).toBeLessThan(1e-5)
    expect(Math.abs(entropyMinimisationProbabilities([[2]], fit)[1] - 0.5)).toBeLessThan(1e-5)
  })

  it('moves the threshold of the line into the gap: 2.100, 2.377 and 3.044 at λ = 0, 0.3 and 1', () => {
    const problem = { labelled: [[0], [4.2]], labels: [0, 1], unlabelled: LINE.slice(1, 9).map((x) => [x]), classes: 2 }
    const thresholds = [0, 0.3, 1].map((lambda) => {
      const fit = entropyMinimisationFit(problem, { lambda, l2: 0.2, tolerance: 1e-10, maxSteps: 2000 })
      return -(fit.params[3] - fit.params[2]) / (fit.params[1] - fit.params[0])
    })
    expect(Math.abs(thresholds[0] - 2.1)).toBeLessThan(0.02)
    expect(Math.abs(thresholds[1] - 2.38)).toBeLessThan(0.02)
    expect(Math.abs(thresholds[2] - 3.045)).toBeLessThan(0.02)
    expect(thresholds[1]).toBeCloseTo(2.3770959279338904, 6)
    expect(thresholds[2]).toBeCloseTo(3.0438607279006518, 6)
  })

  it('the mean entropy of the note at w = 5: 0.1867 at t = 2.1 and 0.084 at t = 2.9', () => {
    const U = LINE.slice(1, 9)
    const problem = { labelled: [[0]], labels: [0], unlabelled: U.map((x) => [x]), classes: 2 }
    const f = entropyMinimisationObjective(problem, { lambda: 1, l2: 0 })
    // Two-class softmax with W = (-w/2, w/2) and b = (wt/2, -wt/2) gives logit w (x - t).
    const at = (w: number, t: number) => f.parts([-w / 2, w / 2, (w * t) / 2, (-w * t) / 2]).entropy
    expect(Math.abs(at(5, 2.1) - 0.1867)).toBeLessThan(5e-4)
    expect(Math.abs(at(5, 2.9) - 0.084)).toBeLessThan(5e-4)
  })
})

describe('entropyMinimisationPath', () => {
  it('λ = 1 beats λ = 0 on the figure’s clusters (seed 3, sd 0.55): 0.974 against 0.862', () => {
    const d = blobs(stream(3), {
      n: [100, 100],
      centers: [
        [-1.1, 0],
        [1.1, 0],
      ],
      sd: 0.55,
    })
    const rows = toArray(d.x) as number[][]
    const y = Array.from(toFlat(d.y))
    // The figure labels the point of each class nearest each of its anchors.
    const anchors = [
      [
        [-1.0, 0.9],
        [-0.5, 0.8],
      ],
      [
        [1.0, -0.9],
        [0.5, -0.8],
      ],
    ]
    const lab: number[] = []
    const cls: number[] = []
    anchors.forEach((list, c) =>
      list.forEach(([ax, ay]) => {
        let best = -1
        let bd = Infinity
        rows.forEach(([a, b], i) => {
          const d2 = (a - ax) ** 2 + (b - ay) ** 2
          if (y[i] === c && d2 < bd) {
            best = i
            bd = d2
          }
        })
        if (!lab.includes(best)) {
          lab.push(best)
          cls.push(c)
        }
      }),
    )
    const unl = rows.map((_, i) => i).filter((i) => !lab.includes(i))
    const lambdas = Array.from({ length: 31 }, (_, i) => i * 0.1)
    const path = entropyMinimisationPath(
      { labelled: lab.map((i) => rows[i]), labels: cls, unlabelled: unl.map((i) => rows[i]), classes: 2 },
      { lambdas, l2: 0.01, maxSteps: 400 },
    )
    expect(path.map((f) => f.lambda)).toEqual(lambdas)
    const acc = (k: number) => {
      const p = entropyMinimisationProbabilities(
        unl.map((i) => rows[i]),
        path[k],
      )
      return unl.filter((i, j) => (p[2 * j + 1] > 0.5 ? 1 : 0) === y[i]).length / unl.length
    }
    expect(acc(0)).toBeCloseTo(0.8622448979591837, 9)
    expect(acc(10)).toBeCloseTo(0.9744897959183674, 9)
    expect(acc(30)).toBeCloseTo(0.9693877551020408, 9)
  })
})
