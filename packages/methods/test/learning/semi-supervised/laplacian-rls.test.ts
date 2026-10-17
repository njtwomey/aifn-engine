/**
 * Laplacian regularised least squares on the site's self-checks: $\gamma_I = 0$ is kernel ridge regression on the
 * labelled points, the gradient of the objective vanishes at the solution for $\gamma_I > 0$, and the six-point arc of
 * the semi-supervised regression note.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { laplacian } from 'aifn-compute/graph/matrices'
import { pointAffinity } from 'aifn-compute/graph/propagation'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { solve } from 'aifn-compute/numerics/linalg'
import { moons } from 'aifn-methods/data/synthetic'
import { laplacianRegularisedLeastSquares } from 'aifn-methods/learning/semi-supervised'

describe('laplacianRegularisedLeastSquares', () => {
  const d = moons(stream(2), { n: 40, noise: 0.1 })
  const y = toFlat(d.y).map((v) => (v === 1 ? 1 : -1))
  const labelled = [0, 5, 22, 30]
  const K = gram(rbf({ lengthscale: 0.5 }), d.x)
  const L = laplacian(pointAffinity(d.x, { gamma: 8, neighbours: 6 }))
  const gammaA = 0.01

  it('with gammaI = 0 the unlabelled coefficients vanish and the labelled ones are kernel ridge regression', () => {
    const a0 = toFlat(laplacianRegularisedLeastSquares(K, L, y, labelled, { gammaA, gammaI: 0 }).alpha)
    const Kr = toRows(K) as number[][]
    const KLL = labelled.map((i) => labelled.map((j) => Kr[i][j] + (i === j ? gammaA * labelled.length : 0)))
    const ridge = toFlat(solve(tensor(KLL), tensor(labelled.map((i) => y[i]))))
    expect(Math.max(...a0.filter((_, i) => !labelled.includes(i)).map(Math.abs))).toBeLessThan(1e-12)
    labelled.forEach((i, r) => expect(Math.abs(a0[i] - ridge[r])).toBeLessThan(1e-9))
  })

  it('with gammaI = 1 the gradient of the objective vanishes and the unlabelled coefficients are used', () => {
    const gammaI = 1
    const m1 = laplacianRegularisedLeastSquares(K, L, y, labelled, { gammaA, gammaI })
    const a = toFlat(m1.alpha)
    const f = toFlat(m1.fitted)
    const n = y.length
    const l = labelled.length
    const Kr = toRows(K) as number[][]
    const Lr = toRows(L) as number[][]
    const Lf = Lr.map((row) => row.reduce((s, v, j) => s + v * f[j], 0))
    const inner = Array.from(
      { length: n },
      (_, i) =>
        (labelled.includes(i) ? (-2 / l) * (y[i] - f[i]) : 0) + 2 * gammaA * a[i] + ((2 * gammaI) / (n * n)) * Lf[i],
    )
    const gradient = Kr.map((row) => row.reduce((s, v, j) => s + v * inner[j], 0))
    expect(Math.max(...gradient.map(Math.abs))).toBeLessThan(1e-9)
    expect(Math.max(...a.filter((_, i) => !labelled.includes(i)).map(Math.abs))).toBeGreaterThan(1e-3)
  })

  it('fills in the middle of the six-point arc of the regression note', () => {
    const pts = [0, 1, 2, 3, 4, 5].map((i) => [Math.cos((i * 50 * Math.PI) / 180), Math.sin((i * 50 * Math.PI) / 180)])
    const W = pts.map((_, i) => pts.map((__, j) => (Math.abs(i - j) === 1 ? 1 : 0)))
    const Lap = W.map((r, i) => r.map((v, j) => (i === j ? r.reduce<number>((a, b) => a + b, 0) : -v)))
    const Ka = gram(rbf({ lengthscale: 1 }), tensor(pts))
    const t = [0, 1, 2, 3, 4, 5]
    const fit = (gA: number, gI: number) =>
      toFlat(laplacianRegularisedLeastSquares(Ka, Lap, t, [0, 5], { gammaA: gA, gammaI: gI }).fitted).map(
        (v) => Math.round(v * 100) / 100,
      )
    expect(fit(1e-5, 0).join()).toBe('0,-0.21,0.4,1.44,3.55,5')
    expect(fit(1e-5, 36 * 0.001).join()).toBe('0,0.98,1.98,2.97,4,5')
    expect(fit(1e-5, 36 * 0.1).join()).toBe('0.19,1.11,2.04,2.96,3.89,4.81')
    // 'none' uses gammaI as given: the same fit as gammaI / n² with the default scale.
    const opts = { gammaA: 1e-5, gammaI: 0.1, scale: 'none' } as const
    const none = toFlat(laplacianRegularisedLeastSquares(Ka, Lap, t, [0, 5], opts).fitted)
    const scaled = toFlat(laplacianRegularisedLeastSquares(Ka, Lap, t, [0, 5], { gammaA: 1e-5, gammaI: 3.6 }).fitted)
    none.forEach((v, i) => expect(v).toBeCloseTo(scaled[i], 9))
  })

  it('refuses bad weights, shapes and an empty labelled set', () => {
    expect(() => laplacianRegularisedLeastSquares(K, L, y, labelled, { gammaA: 0, gammaI: 0 })).toThrow(/gammaA/)
    expect(() => laplacianRegularisedLeastSquares(K, L, y.slice(1), labelled, { gammaA, gammaI: 0 })).toThrow(/same n/)
    expect(() => laplacianRegularisedLeastSquares(K, L, y, [], { gammaA, gammaI: 0 })).toThrow(/labelled/)
  })
})
