import { describe, expect, it } from 'vitest'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  chebyshevScalarisation,
  expectedHypervolumeImprovement,
  hypervolume,
  hypervolumeImprovement,
  multiObjectiveFunctions,
  paretoFront,
  simplexLattice,
} from 'aifn-methods/optimisation/multi-objective'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type M = number[][]
type Ehvi = { front: M; reference: number[]; mean: M; sd: M; exact: number[]; monteCarlo?: number[] }
const F = fixture<{
  pareto: { points: M; indices: number[] }[]
  paretoMixed: { points: M; indices: number[] }
  hypervolume: { points: M; reference: number[]; hv: number }[]
  hypervolumeInclusionExclusion: { points: M; reference: number[]; hv: number }
  improvement: { front: M; reference: number[]; y: M; hvi: number[] }[]
  ehvi2: Ehvi
  ehvi2Empty: number
  ehvi3: Ehvi
  ehvi1: Ehvi
  chebyshev: { y: M; weights: number[]; ideal: number[]; nadir: number[]; plain: number[]; normalised: number[] }
}>('optimisation/multi-objective')

const flat = (t: Tensor) => Array.from(toFlat(t) as ArrayLike<number>)
const negate = (m: M) => m.map((r) => r.map((v) => -v))

describe('paretoFront', () => {
  it('matches the pairwise check in 2-D and 3-D, with ties and repeats, in lexicographic order', () => {
    for (const c of F.pareto) {
      const { indices, front } = paretoFront(c.points)
      expect(indices).toEqual(c.indices)
      expect(toArray(front)).toEqual(c.indices.map((i) => c.points[i]))
    }
  })

  it('maximises the objectives asked for', () => {
    const c = F.paretoMixed
    expect(paretoFront(c.points, { minimise: [true, false] }).indices).toEqual(c.indices)
    const all = paretoFront(negate(F.pareto[0].points), { minimise: false })
    expect(all.indices).toEqual(F.pareto[0].indices)
  })

  it('keeps every copy of a repeated point and handles an empty set', () => {
    expect(
      paretoFront([
        [1, 2],
        [1, 2],
        [2, 3],
      ]).indices,
    ).toEqual([0, 1])
    expect(paretoFront([]).indices).toEqual([])
    expect(() => paretoFront([[1, 2]], { minimise: [true] })).toThrow(ShapeError)
    expect(() => paretoFront([[1, NaN]])).toThrow(DomainError)
  })
})

describe('hypervolume', () => {
  it('matches the grid decomposition in 2, 3 and 4 objectives', () => {
    for (const c of F.hypervolume) expect(hypervolume(c.points, c.reference)).toBeCloseTo(c.hv, 12)
  })

  it('matches inclusion and exclusion', () => {
    const c = F.hypervolumeInclusionExclusion
    expect(hypervolume(c.points, c.reference)).toBeCloseTo(c.hv, 12)
  })

  it('is invariant to maximising negated objectives, and 0 for an empty set', () => {
    const c = F.hypervolume[1]
    const ref = c.reference.map((v) => -v)
    expect(hypervolume(negate(c.points), ref, { minimise: false })).toBeCloseTo(c.hv, 12)
    expect(hypervolume([], [1, 1])).toBe(0)
    expect(hypervolume([[2, 0.5]], [1, 1])).toBe(0)
    expect(hypervolume([[0.5], [0.2]], [1])).toBeCloseTo(0.8, 15)
    expect(() => hypervolume([[0, 0]], [1, 1, 1])).toThrow(ShapeError)
  })
})

describe('hypervolumeImprovement', () => {
  it('matches the difference of grid volumes in 2-D and 3-D', () => {
    for (const c of F.improvement) {
      const got = flat(hypervolumeImprovement(c.front, c.reference, c.y))
      got.forEach((v, i) => expect(v).toBeCloseTo(c.hvi[i], 12))
    }
  })

  it('is the hypervolume of the front with the point added, less the hypervolume of the front', () => {
    const c = F.improvement[1]
    const base = hypervolume(c.front, c.reference)
    const got = flat(hypervolumeImprovement(c.front, c.reference, c.y))
    c.y.forEach((y, i) => expect(got[i]).toBeCloseTo(hypervolume([...c.front, y], c.reference) - base, 12))
  })
})

describe('expectedHypervolumeImprovement', () => {
  it('is exact for two objectives (the cell integral), and agrees with 10^6 draws', () => {
    const c = F.ehvi2
    const got = flat(expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference))
    got.forEach((v, i) => {
      expect(v).toBeCloseTo(c.exact[i], 12)
      expect(Math.abs(v - (c.monteCarlo as number[])[i])).toBeLessThan(2e-3 * Math.max(1, v) + 1e-4)
    })
    const empty = expectedHypervolumeImprovement([c.mean[0]], [c.sd[0]], [], c.reference)
    expect(flat(empty)[0]).toBeCloseTo(F.ehvi2Empty, 12)
  })

  it('reduces to the improvement where the standard deviation is 0', () => {
    const c = F.ehvi2
    const hvi = flat(hypervolumeImprovement(c.front, c.reference, c.mean))
    flat(expectedHypervolumeImprovement(c.mean, 0, c.front, c.reference)).forEach((v, i) =>
      expect(v).toBeCloseTo(hvi[i], 12),
    )
  })

  it('is the expected improvement on min(r, best) for one objective', () => {
    const c = F.ehvi1
    const got = flat(expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference))
    got.forEach((v, i) => expect(v).toBeCloseTo(c.exact[i], 12))
  })

  it('estimates three objectives by Monte Carlo within a few standard errors of the exact cell integral', () => {
    const c = F.ehvi3
    const got = flat(
      expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference, { samples: 20000, stream: stream(3) }),
    )
    got.forEach((v, i) => expect(Math.abs(v - c.exact[i])).toBeLessThan(0.05 * c.exact[i] + 1e-4))
  })

  it('Monte Carlo for two objectives agrees with the exact value, and is reproducible from its stream', () => {
    const c = F.ehvi2
    const options = { method: 'monte-carlo' as const, samples: 20000, stream: stream(5) }
    const a = flat(expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference, options))
    a.forEach((v, i) => expect(Math.abs(v - c.exact[i])).toBeLessThan(0.05 * c.exact[i] + 2e-4))
    const b = flat(
      expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference, { ...options, stream: stream(5) }),
    )
    expect(b).toEqual(a)
  })

  it('reports bad input', () => {
    const c = F.ehvi3
    expect(() => expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference)).toThrow(DomainError)
    expect(() =>
      expectedHypervolumeImprovement(c.mean, c.sd, c.front, c.reference, { method: 'exact', stream: stream(0) }),
    ).toThrow(DomainError)
    expect(() => expectedHypervolumeImprovement([[0, 0]], -1, [], [1, 1])).toThrow(DomainError)
    expect(() => expectedHypervolumeImprovement([[0, 0]], [[1, 1, 1]], [], [1, 1])).toThrow(ShapeError)
  })
})

describe('chebyshevScalarisation', () => {
  it('matches numpy, plain and normalised', () => {
    const c = F.chebyshev
    flat(chebyshevScalarisation(c.y, c.weights)).forEach((v, i) => expect(v).toBeCloseTo(c.plain[i], 12))
    const normalised = chebyshevScalarisation(c.y, c.weights, { rho: 0.1, ideal: c.ideal, nadir: c.nadir })
    flat(normalised).forEach((v, i) => expect(v).toBeCloseTo(c.normalised[i], 12))
  })

  it('treats maximised objectives as negated ones', () => {
    const c = F.chebyshev
    const options = { rho: 0.1, ideal: c.ideal, nadir: c.nadir }
    const neg = (v: number[]) => v.map((x) => -x)
    const got = chebyshevScalarisation(negate(c.y), c.weights, {
      ...options,
      ideal: neg(c.ideal),
      nadir: neg(c.nadir),
      minimise: false,
    })
    flat(got).forEach((v, i) => expect(v).toBeCloseTo(c.normalised[i], 12))
    expect(() => chebyshevScalarisation(c.y, [0.5, 0.5])).toThrow(ShapeError)
    expect(() => chebyshevScalarisation(c.y, [0.5, -0.5, 1])).toThrow(DomainError)
    expect(() => chebyshevScalarisation(c.y, c.weights, { nadir: c.nadir })).toThrow(DomainError)
  })
})

describe('simplexLattice', () => {
  it('has binom(s + M - 1, M - 1) rows summing to 1, as ParEGO uses', () => {
    expect(simplexLattice(2).shape).toEqual([11, 2])
    expect(simplexLattice(3).shape).toEqual([66, 3])
    expect(simplexLattice(4, 3).shape).toEqual([20, 4])
    for (const row of toArray(simplexLattice(3)) as M) expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 14)
    expect(toArray(simplexLattice(2, 2))).toEqual([
      [0, 1],
      [0.5, 0.5],
      [1, 0],
    ])
    expect(() => simplexLattice(0)).toThrow(DomainError)
  })
})

it('registers its functions', () => {
  expectInfo(multiObjectiveFunctions, 'function')
  expect(Object.keys(multiObjectiveFunctions).sort()).toEqual([
    'chebyshevScalarisation',
    'expectedHypervolumeImprovement',
    'hypervolume',
    'hypervolumeImprovement',
    'paretoFront',
    'simplexLattice',
  ])
})
