/** aifn-methods/learning/semi-supervised: the transductive SVM in the primal (∇TSVM), against the note's numbers. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { blobs } from 'aifn-methods/data/synthetic'
import {
  semiSupervisedFunctions,
  symmetricHinge,
  symmetricRamp,
  symmetricSigmoid,
  transductiveSvmDecision,
  transductiveSvmFit,
  transductiveSvmObjective,
  transductiveSvmStages,
} from 'aifn-methods/learning/semi-supervised'

const LINE = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.4, 3.8, 4.2]

describe('unlabelled losses', () => {
  it('hinge, sigmoid and ramp at a few outputs', () => {
    expect([-2, -0.5, 0, 0.5, 2].map((t) => symmetricHinge(t))).toEqual([0, 0.5, 1, 0.5, 0])
    expect(symmetricSigmoid(0)).toBe(1)
    expect(symmetricSigmoid(0.5)).toBeCloseTo(Math.exp(-0.75), 15)
    expect([0, 0.2, 0.5, 1].map((t) => symmetricRamp(t))).toEqual([0.7, 0.7, 0.5, 0])
  })
})

describe('transductiveSvmObjective', () => {
  for (const balance of [undefined, null] as const)
    it(`has the gradient of central differences (${balance === null ? 'free bias' : 'constraint'})`, () => {
      const problem = {
        labelled: [
          [0.1, 0.2],
          [0.5, -1],
          [1, 1],
        ],
        labels: [-1, 1, 1],
        unlabelled: [
          [0.3, 0.3],
          [-1, 0.5],
          [0.2, 0.9],
          [0, 0],
        ],
      }
      const f = transductiveSvmObjective(problem, { C: 2, Cstar: 0.7, balance })
      expect(f.size).toBe(balance === null ? 3 : 2)
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

  it('refuses labels other than ±1 and a constraint without unlabelled points', () => {
    const weights = { C: 1, Cstar: 1 }
    expect(() => transductiveSvmObjective({ labelled: [[0]], labels: [0], unlabelled: [[1]] }, weights)).toThrow(/±1/)
    expect(() => transductiveSvmObjective({ labelled: [[0]], labels: [1], unlabelled: [] }, weights)).toThrow(
      /unlabelled/,
    )
  })
})

describe('transductiveSvmFit', () => {
  it('holds the balancing constraint exactly: mean unlabelled output = mean label = −1/3', () => {
    const d = blobs(stream(1), {
      n: [60, 60],
      centers: [
        [-1, 0],
        [1, 0],
      ],
      sd: 0.5,
    })
    const rows = toArray(d.x) as number[][]
    const sol = transductiveSvmFit(
      { labelled: [rows[0], rows[1], rows[60]], labels: [-1, -1, 1], unlabelled: rows.slice(2) },
      { C: 10, Cstar: 1 },
    )
    const mean = sol.unlabelledOutputs.reduce((a, v) => a + v, 0) / sol.unlabelledOutputs.length
    expect(Math.abs(mean + 1 / 3)).toBeLessThan(1e-9)
    // The outputs are the decision values of the unlabelled rows.
    const f = transductiveSvmDecision(rows.slice(2), sol)
    f.forEach((v, j) => expect(v).toBeCloseTo(sol.unlabelledOutputs[j], 10))
  })

  it('the hand results of the line: hard-margin threshold 2.9, the supervised hinge total 30/7', () => {
    // Hard-margin TSVM by enumeration: the threshold in the widest gap, ½w² = 2 / gap².
    const gaps = LINE.slice(1).map((v, i) => ({ t: (v + LINE[i]) / 2, g: v - LINE[i] }))
    const best = gaps.reduce((a, b) => (2 / b.g ** 2 < 2 / a.g ** 2 ? b : a))
    expect(best.t).toBeCloseTo(2.9, 12)
    expect(2 / best.g ** 2).toBeCloseTo(2, 12)
    const w = 2 / 4.2
    const total = LINE.slice(1, 9).reduce((a, x) => a + symmetricHinge(w * (x - 2.1)), 0)
    expect(total).toBeCloseTo(30 / 7, 12)
    // The supervised squared-hinge SVM with a free bias puts its threshold at the labelled midpoint 2.1.
    const sup = transductiveSvmFit(
      { labelled: [[0], [4.2]], labels: [-1, 1], unlabelled: LINE.slice(1, 9).map((x) => [x]) },
      { C: 10, Cstar: 0, balance: null },
    )
    expect(-sup.bias / sup.weights[0]).toBeCloseTo(2.1, 4)
  })
})

describe('transductiveSvmStages', () => {
  it('beats the supervised SVM on the figure’s clusters (seed 1, sd 0.45): 0.990 against 0.918', () => {
    const d = blobs(stream(1), {
      n: [100, 100],
      centers: [
        [-1.1, 0],
        [1.1, 0],
      ],
      sd: 0.45,
    })
    const rows = toArray(d.x) as number[][]
    const y = Array.from(toFlat(d.y))
    // The labelled points the figure picks nearest its anchors.
    const lab = [7, 93, 105, 183]
    const cls = [0, 0, 1, 1]
    const unl = rows.map((_, i) => i).filter((i) => !lab.includes(i))
    const P = {
      labelled: lab.map((i) => rows[i]),
      labels: cls.map((c) => (c ? 1 : -1)),
      unlabelled: unl.map((i) => rows[i]),
    }
    const acc = (sol: { weights: Float64Array; bias: number }) => {
      const f = transductiveSvmDecision(
        unl.map((i) => rows[i]),
        sol,
      )
      return unl.filter((i, j) => (f[j] > 0 ? 1 : 0) === y[i]).length / unl.length
    }
    const sup = transductiveSvmFit(P, { C: 10, Cstar: 0, balance: null })
    const stages = transductiveSvmStages(P, { C: 10, Cstar: 10 })
    expect(stages.length).toBe(11)
    expect(stages.map((s) => s.Cstar)).toEqual(Array.from({ length: 11 }, (_, k) => 10 * 2 ** (k - 10)))
    expect(acc(sup)).toBeCloseTo(0.9183673469387755, 12)
    expect(acc(stages[10])).toBeCloseTo(0.9897959183673469, 12)
    expect(stages[10].weights[0]).toBeCloseTo(4.948579000006311, 6)
    expect(stages[10].bias).toBeCloseTo(-0.046224317287926, 6)
  })

  it('is registered', () => {
    expect(semiSupervisedFunctions.transductiveSvmStages.info.cite).toContain('chapelle2005lds')
  })
})
