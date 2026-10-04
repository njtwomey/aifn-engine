/**
 * Label propagation and label spreading against scikit-learn on two moons (RBF affinity), the stepped forms against
 * their closed-form limits, and the harmonic property on a path.
 */
import { describe, expect, it } from 'vitest'
import { fromEdges } from 'aifn-compute/graph'
import {
  graphAffinity,
  harmonicLabels,
  labelMatrix,
  labelPropagationSteps,
  labelSpreading,
  labelSpreadingSteps,
  normaliseScores,
  spreadingMatrix,
} from 'aifn-compute/graph/propagation'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

type Result = { scores: number[][]; labels: number[] }
const F = fixture('graph/propagation') as {
  labels: number[]
  affinity: number[][]
  propagation: Result
  'spreading0.2': Result
  'spreading0.9': Result
}

const close = (a: number[][], b: number[][], digits: number) =>
  a.forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(b[i][j], digits)))

describe('label propagation against scikit-learn', () => {
  it('the harmonic solution equals LabelPropagation at convergence', () => {
    const got = harmonicLabels(F.affinity, F.labels)
    close(toRows(got.scores) as number[][], F.propagation.scores, 8)
    expect(Array.from(toFlat(got.labels))).toEqual(F.propagation.labels)
  })
  it('the steps converge to the harmonic solution', () => {
    const steps = labelPropagationSteps(F.affinity, F.labels, { tolerance: 1e-13 })
    const final = run(steps, undefined, 100000)
    close(toRows(normaliseScores(final.scores).scores) as number[][], F.propagation.scores, 7)
  })
  for (const alpha of [0.2, 0.9] as const) {
    it(`label spreading (α = ${alpha}) equals LabelSpreading, closed form and steps`, () => {
      const want = F[`spreading${alpha}`]
      close(toRows(labelSpreading(F.affinity, F.labels, { alpha }).scores) as number[][], want.scores, 8)
      const final = run(labelSpreadingSteps(F.affinity, F.labels, { alpha, tolerance: 1e-14 }), undefined, 100000)
      close(toRows(normaliseScores(final.scores).scores) as number[][], want.scores, 7)
      expect(Array.from(toFlat(normaliseScores(final.scores).labels))).toEqual(want.labels)
    })
  }
})

describe('laws', () => {
  it('on a path, harmonic scores interpolate linearly between the labelled ends', () => {
    const g = fromEdges(
      5,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
      ],
      { directed: false },
    )
    const s = toRows(harmonicLabels(g, [0, -1, -1, -1, 1]).scores) as number[][]
    s.forEach((r, i) => expect(r[1]).toBeCloseTo(i / 4, 12))
  })
  it('a component without labels keeps zero scores and label −1 (the system is not singular)', () => {
    // Path 0 – 1 – 2 labelled at both ends, and a separate edge 3 – 4 with no labels.
    const g = fromEdges(
      5,
      [
        [0, 1],
        [1, 2],
        [3, 4],
      ],
      { directed: false },
    )
    const r = harmonicLabels(g, [0, -1, 1, -1, -1])
    expect(toRows(r.scores)).toEqual([
      [1, 0],
      [0.5, 0.5],
      [0, 1],
      [0, 0],
      [0, 0],
    ])
    expect(Array.from(toFlat(r.labels))).toEqual([0, 0, 1, -1, -1])
  })
  it('each unlabelled score is the weighted average of its neighbours', () => {
    const W = F.affinity
    const S = toRows(harmonicLabels(W, F.labels).scores) as number[][]
    for (let i = 0; i < W.length; i++) {
      if (F.labels[i] !== -1) continue
      let num = 0
      let den = 0
      for (let j = 0; j < W.length; j++)
        if (j !== i) {
          num += W[i][j] * S[j][1]
          den += W[i][j]
        }
      expect(S[i][1]).toBeCloseTo(num / den, 9)
    }
  })
  it('step 0 holds only the given labels; labelled rows stay clamped', () => {
    const t = trace(labelPropagationSteps(F.affinity, F.labels), undefined, 5, { keep: 'all' })
    const first = toRows(t.steps[0].scores) as number[][]
    expect(first).toEqual(toRows(labelMatrix(F.labels)))
    const last = toRows(t.final.scores) as number[][]
    F.labels.forEach((l, i) => l !== -1 && expect(last[i][l]).toBe(1))
  })
  it('S is symmetric with a zero diagonal; heat affinities shrink with distance', () => {
    const S = toRows(spreadingMatrix(F.affinity)) as number[][]
    S.forEach((r, i) => {
      expect(r[i]).toBe(0)
      r.forEach((v, j) => expect(v).toBeCloseTo(S[j][i], 14))
    })
    const g = fromEdges(
      3,
      [
        [0, 1, 0.5],
        [1, 2, 2],
      ],
      { directed: false },
    )
    const A = toRows(graphAffinity(g, { kind: 'heat', sigma: 1 })) as number[][]
    expect(A[0][1]).toBeCloseTo(Math.exp(-0.125), 14)
    expect(A[1][2]).toBeLessThan(A[0][1])
    expect(toRows(graphAffinity(g, { kind: 'connectivity' }))).toEqual([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ])
  })
})
