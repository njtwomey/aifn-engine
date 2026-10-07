/**
 * Sparse representations: exact recovery of planted sparse vectors by OMP, basis pursuit and IHT on random Gaussian
 * dictionaries; the residual laws of the pursuits (orthogonality for OMP, monotone norm for MP); the lasso's optimality
 * conditions; batch coding matching one-at-a-time coding; dictionary learning recovering a planted dictionary and never
 * increasing its objective.
 */
import { describe, expect, it } from 'vitest'
import { normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  basisPursuit,
  basisPursuitDenoising,
  dictionaryLearning,
  dictionaryLearningSteps,
  hardThreshold,
  iterativeHardThresholding,
  matchingPursuit,
  matchingPursuitSteps,
  mutualCoherence,
  normaliseAtoms,
  orthogonalMatchingPursuit,
  sparseCode,
} from 'aifn-compute/signal/sparse'

/** A random $m \times k$ dictionary with unit-norm Gaussian atoms, as rows. */
function gaussianDictionary(seed: string, m: number, k: number): number[][] {
  const g = toFlat(normal(stream(seed), 0, 1, { shape: [m * k] }))
  return toRows(normaliseAtoms(Array.from({ length: m }, (_, i) => Array.from(g.slice(i * k, (i + 1) * k)))))
}

const times = (D: number[][], x: ArrayLike<number>) => D.map((row) => row.reduce((a, v, j) => a + v * x[j], 0))

/** A planted problem: a 3-sparse x over a 20 × 40 Gaussian dictionary, and y = Dx. */
const D = gaussianDictionary('sparse', 20, 40)
const planted = new Array<number>(40).fill(0)
planted[3] = 1.5
planted[17] = -2
planted[31] = 0.8
const y = times(D, planted)

const close = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 8) =>
  Array.from(a).forEach((v, i) => expect(v).toBeCloseTo(b[i], digits))

describe('dictionaries', () => {
  it('normalises atoms to unit norm and leaves a zero atom alone', () => {
    expect(
      toRows(
        normaliseAtoms([
          [3, 0],
          [4, 0],
        ]),
      ),
    ).toEqual([
      [0.6, 0],
      [0.8, 0],
    ])
  })
  it('measures coherence: 0 for orthogonal atoms, 1/sqrt(2) for spikes with the Haar basis', () => {
    expect(
      mutualCoherence([
        [1, 0],
        [0, 1],
      ]),
    ).toBe(0)
    const s = Math.SQRT1_2
    expect(
      mutualCoherence([
        [1, 0, s, s],
        [0, 1, s, -s],
      ]),
    ).toBeCloseTo(s, 12)
  })
  it('hard thresholds to the s largest magnitudes', () => {
    expect(Array.from(toFlat(hardThreshold([0.5, -3, 1, 2], 2)))).toEqual([0, -3, 0, 2])
    expect(Array.from(toFlat(hardThreshold([1, 2], 5)))).toEqual([1, 2])
  })
})

describe('greedy pursuits', () => {
  it('OMP recovers a planted sparse vector exactly', () => {
    const res = orthogonalMatchingPursuit(D, y, { sparsity: 3 })
    expect(res.support).toEqual([3, 17, 31])
    close(toFlat(res.x), planted)
    expect(res.residualNorm).toBeLessThan(1e-10)
  })
  it("OMP's residual is orthogonal to every chosen atom", () => {
    const signal = times(
      D,
      planted.map((v, j) => v + (j % 7 === 0 ? 0.3 : 0)),
    )
    const res = orthogonalMatchingPursuit(D, signal, { sparsity: 5 })
    expect(res.support).toHaveLength(5)
    const r = toFlat(res.residual)
    for (const j of res.support) expect(D.reduce((a, row, i) => a + row[j] * r[i], 0)).toBeCloseTo(0, 10)
  })
  it('OMP stops early at the tolerance', () => {
    expect(orthogonalMatchingPursuit(D, y).steps).toBe(3)
  })
  it("MP's residual norm never increases, and its coefficients reproduce y minus the residual", () => {
    const norms = toFlat(
      trace(matchingPursuitSteps(D, y), undefined, 60, { record: { norm: (s) => s.residualNorm } }).series.norm,
    )
    for (let t = 1; t < norms.length; t++) expect(norms[t]).toBeLessThanOrEqual(norms[t - 1] + 1e-12)
    const res = matchingPursuit(D, y, { maxSteps: 60 })
    close(
      times(D, toFlat(res.x)),
      y.map((v, i) => v - toFlat(res.residual)[i]),
      10,
    )
  })
  it('a dictionary that does not span the signal stalls the pursuit', () => {
    const s = run(
      matchingPursuitSteps(
        [
          [1, 1],
          [0, 0],
        ],
        [0, 1],
      ),
      undefined,
      10,
    )
    expect(s.stalled).toBe(true)
    expect(s.t).toBe(0)
  })
})

describe('basis pursuit and its relatives', () => {
  it('basis pursuit recovers the planted vector', () => {
    const res = basisPursuit(D, y)
    expect(res.status).toBe('optimal')
    expect(res.support).toEqual([3, 17, 31])
    close(toFlat(res.x), planted, 7)
    expect(res.norm1).toBeCloseTo(4.3, 7)
  })
  it('basis pursuit reports a signal outside the range as infeasible', () => {
    expect(
      basisPursuit(
        [
          [1, 2],
          [2, 4],
        ],
        [1, 0],
      ).status,
    ).toBe('infeasible')
  })
  it('the lasso meets its optimality conditions', () => {
    const lambda = 0.05
    const res = basisPursuitDenoising(D, y, { lambda, tolerance: 1e-10, maxSteps: 5000 })
    expect(res.converged).toBe(true)
    const x = toFlat(res.x)
    const r = toFlat(res.residual)
    // D^T (y − Dx) = λ sign(x) on the support, and at most λ in magnitude off it.
    for (let j = 0; j < 40; j++) {
      const g = D.reduce((a, row, i) => a + row[j] * r[i], 0)
      if (x[j] !== 0) expect(g).toBeCloseTo(lambda * Math.sign(x[j]), 6)
      else expect(Math.abs(g)).toBeLessThanOrEqual(lambda + 1e-6)
    }
    expect(res.support).toEqual(expect.arrayContaining([3, 17, 31]))
  })
  it('the lasso gives zero from the critical penalty up', () => {
    const critical = Math.max(...D[0].map((_, j) => Math.abs(D.reduce((a, row, i) => a + row[j] * y[i], 0))))
    expect(basisPursuitDenoising(D, y, { lambda: critical * 1.001 }).support).toEqual([])
    expect(basisPursuitDenoising(D, y, { lambda: critical * 0.9 }).support.length).toBeGreaterThan(0)
  })
  it('FISTA reaches a lower objective than ISTA in the same steps', () => {
    const fista = basisPursuitDenoising(D, y, { lambda: 0.05, maxSteps: 30, tolerance: 0 })
    const ista = basisPursuitDenoising(D, y, { lambda: 0.05, maxSteps: 30, tolerance: 0, accelerated: false })
    expect(fista.objective).toBeLessThan(ista.objective)
  })
  it('IHT recovers the planted vector and keeps at most s non-zeros', () => {
    const res = iterativeHardThresholding(D, y, { sparsity: 3, maxSteps: 3000, tolerance: 1e-10 })
    expect(res.support).toEqual([3, 17, 31])
    close(toFlat(res.x), planted, 6)
  })
})

describe('sparseCode', () => {
  it('codes each column as the single-signal solver does', () => {
    const other = times(
      D,
      planted.map((v, j) => (j === 5 ? 1 : v / 2)),
    )
    const Y = y.map((v, i) => [v, other[i]])
    const { X, residualNorms } = sparseCode(D, Y, { method: 'omp', sparsity: 4 })
    expect(X.shape).toEqual([40, 2])
    const rows = toRows(X)
    for (const [c, signal] of [y, other].entries()) {
      const one = orthogonalMatchingPursuit(D, signal, { sparsity: 4 })
      close(
        rows.map((row) => row[c]),
        toFlat(one.x),
        12,
      )
      expect(toFlat(residualNorms)[c]).toBeCloseTo(one.residualNorm, 12)
    }
  })
  it('rejects signals of the wrong length', () => {
    expect(() => sparseCode(D, [[1], [2]])).toThrow(/rows/)
  })
})

describe('dictionary learning', () => {
  // Four unit atoms in R^3 and signals that are each one atom times a coefficient.
  const D0 = toRows(
    normaliseAtoms([
      [1, 0, 1, 1],
      [0, 1, 1, -1],
      [0, 0, 1, 1],
    ]),
  )
  const coefs = [1, -2, 0.5, 3, -1.5]
  const oneSparse = D0.map((row) => row.flatMap((v) => coefs.map((c) => c * v)))
  const pairs = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
  ]
  const weights = [
    [1, 0.5],
    [-0.7, 1],
    [0.3, -1.2],
  ]
  const twoSparse = D0.map((row) => pairs.flatMap(([a, b]) => weights.map(([p, q]) => p * row[a] + q * row[b])))

  for (const update of ['ksvd', 'mod'] as const) {
    it(`${update} recovers a planted dictionary up to order and sign`, () => {
      const { D: learnt, objective } = dictionaryLearning(oneSparse, { atoms: 4, sparsity: 1, update })
      expect(objective).toBeLessThan(1e-20)
      const L = toRows(learnt)
      for (let a = 0; a < 4; a++) {
        const best = Math.max(...[0, 1, 2, 3].map((b) => Math.abs(D0.reduce((s, row, i) => s + row[a] * L[i][b], 0))))
        expect(best).toBeCloseTo(1, 10)
      }
    })
    it(`${update} never increases the objective, keeps codes s-sparse and atoms unit-norm`, () => {
      const objectives = toFlat(
        trace(dictionaryLearningSteps(twoSparse, { atoms: 4, sparsity: 2, update }), undefined, 20, {
          record: { objective: (s) => s.objective },
        }).series.objective,
      )
      for (let t = 1; t < objectives.length; t++)
        expect(objectives[t]).toBeLessThanOrEqual(objectives[t - 1] * (1 + 1e-12))
      const s = run(dictionaryLearningSteps(twoSparse, { atoms: 4, sparsity: 2, update }), undefined, 20)
      for (const column of toRows(s.X)[0].keys())
        expect(toRows(s.X).filter((row) => row[column] !== 0).length).toBeLessThanOrEqual(2)
      const atoms = toRows(s.D)
      for (let j = 0; j < 4; j++) expect(Math.hypot(...atoms.map((row) => row[j]))).toBeCloseTo(1, 12)
    })
  }
  it('replaces an unused atom', () => {
    // A duplicated initial atom: OMP never picks the copy, so it is replaced on the first step.
    const s = run(
      dictionaryLearningSteps(oneSparse, {
        atoms: 4,
        sparsity: 1,
        init: [
          [1, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ],
      }),
      undefined,
      1,
    )
    expect(s.replaced).toBeGreaterThan(0)
  })
})
