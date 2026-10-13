/**
 * The group lasso on the standardised diabetes data (the `group_lasso` cases of fixture `learning/linear`): against a
 * proximal-gradient solver run to convergence in numpy, by its optimality conditions, and against `elasticNet`'s lasso
 * when every group is a single column.
 */
import { describe, expect, it } from 'vitest'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import { elasticNet, groupLasso, groupLassoPath, linearFunctions } from 'aifn-methods/learning/linear'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dataset } from 'aifn-compute/learning/estimators'
import { fixture } from '../../fixtures'

type Fixture = {
  cases: {
    groups: number[]
    alpha_max: number
    fits: { alpha: number; coef: number[]; intercept: number; objective: number }[]
  }[]
}
const fx = fixture<{ group_lasso: Fixture }>('learning/linear').group_lasso

const data = diabetes()
const x = standardScaler().fit({ x: data.x }).transform(data.x)
const y = data.y!
const X = toRows(x)
const Y = toFlat(y)
const n = X.length

/** The gradient of the loss, (1/n) Xᵀ(y − Xw − b), on the data as given. */
function correlations(w: ArrayLike<number>, b: number): number[] {
  const g = new Array<number>(10).fill(0)
  for (let i = 0; i < n; i++) {
    let r = Y[i] - b
    for (let j = 0; j < 10; j++) r -= X[i][j] * w[j]
    for (let j = 0; j < 10; j++) g[j] += (X[i][j] * r) / n
  }
  return g
}

/** The largest violation of the group lasso's optimality conditions. */
function kktViolation(w: ArrayLike<number>, b: number, groups: number[], alpha: number): number {
  const g = correlations(w, b)
  let worst = 0
  for (const label of new Set(groups)) {
    const cols = groups.flatMap((l, j) => (l === label ? [j] : []))
    const scale = alpha * Math.sqrt(cols.length)
    const norm = Math.hypot(...cols.map((j) => w[j]))
    if (norm === 0) worst = Math.max(worst, Math.hypot(...cols.map((j) => g[j])) - scale)
    // Active: the gradient equals the penalty's, alpha sqrt(p_g) w_g / |w_g|.
    else for (const j of cols) worst = Math.max(worst, Math.abs(g[j] - (scale * w[j]) / norm))
  }
  return worst
}

const flat = (t: Tensor): number[] => toFlat(t)

describe('groupLasso', () => {
  it('matches proximal gradient run to convergence, and meets the optimality conditions', () => {
    for (const c of fx.cases) {
      for (const f of c.fits) {
        const m = groupLasso({ alpha: f.alpha, groups: c.groups }).fit(dataset(x, y))
        const what = `groups ${c.groups.join('')} alpha ${f.alpha}`
        expect(m.converged, what).toBe(true)
        flat(m.weights).forEach((v, j) => expect(Math.abs(v - f.coef[j]), `${what} w${j}`).toBeLessThan(1e-7))
        expect(m.intercept).toBeCloseTo(f.intercept, 7)
        // Our objective is no worse than the reference's: both are at the optimum.
        expect(m.objective).toBeLessThan(f.objective * (1 + 1e-12))
        expect(m.objective).toBeGreaterThan(f.objective * (1 - 1e-10))
        expect(kktViolation(flat(m.weights), m.intercept, c.groups, f.alpha), what).toBeLessThan(1e-8)
        expect(m.gap).toBeGreaterThanOrEqual(-1e-12)
        // Whole groups are zero or not: a group's members are never zero alone in an active group.
        m.groupNorms.forEach((norm, gi) => {
          const cols = c.groups.flatMap((l, j) => (l === m.groups[gi] ? [j] : []))
          for (const j of cols) expect(flat(m.weights)[j] === 0).toBe(norm === 0)
        })
      }
    }
  })

  it('is the lasso when every group is one column', () => {
    for (const alpha of [10, 1, 0.1]) {
      const g = groupLasso({ alpha, groups: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }).fit(dataset(x, y))
      const l = elasticNet({ alpha }).fit(dataset(x, y))
      flat(g.weights).forEach((v, j) => expect(Math.abs(v - flat(l.weights)[j]), `alpha ${alpha}`).toBeLessThan(1e-8))
      expect(g.intercept).toBeCloseTo(l.intercept, 8)
      expect(g.objective).toBeCloseTo(l.objective, 10)
      // The default grouping is the same.
      expect(flat(groupLasso({ alpha }).fit(dataset(x, y)).weights)).toEqual(flat(g.weights))
    }
  })

  it('does not depend on how the groups are labelled or ordered', () => {
    const { groups, fits } = fx.cases[1]
    const relabelled = groups.map((l) => 10 * (3 - l) + 7)
    const m = groupLasso({ alpha: fits[1].alpha, groups: relabelled }).fit(dataset(x, y))
    flat(m.weights).forEach((v, j) => expect(Math.abs(v - fits[1].coef[j])).toBeLessThan(1e-7))
    expect(m.groups).toEqual([37, 27, 17, 7])
  })

  it('is zero at alpha max, and throws for bad input', () => {
    const { groups, alpha_max } = fx.cases[0]
    expect(groupLasso({ alpha: alpha_max * (1 + 1e-9), groups }).fit(dataset(x, y)).activeGroups).toBe(0)
    expect(groupLasso({ alpha: alpha_max * (1 - 1e-6), groups }).fit(dataset(x, y)).activeGroups).toBe(1)
    expect(() => groupLasso({ alpha: 0, groups })).toThrow(DomainError)
    expect(() => groupLasso({ alpha: 1, groups: [0, 1] }).fit(dataset(x, y))).toThrow(ShapeError)
    expect(() => groupLasso({ alpha: 1, groups: groups.map((g) => g + 0.5) }).fit(dataset(x, y))).toThrow(DomainError)
  })
})

describe('groupLassoPath', () => {
  it('fits the given penalties as groupLasso does, and starts its default grid at alpha max', () => {
    for (const c of fx.cases) {
      const path = groupLassoPath(x, y, { groups: c.groups, alphas: c.fits.map((f) => f.alpha) })
      expect(path.converged).toBe(true)
      const coef = toRows(path.coef)
      c.fits.forEach((f, t) => {
        for (let j = 0; j < 10; j++) expect(Math.abs(coef[j][t] - f.coef[j])).toBeLessThan(1e-7)
        expect(path.intercept[t]).toBeCloseTo(f.intercept, 7)
      })
      const grid = groupLassoPath(x, y, { groups: c.groups, count: 21, span: 3 })
      expect(grid.alphas[0]).toBeCloseTo(c.alpha_max, 10)
      expect(grid.activeGroups[0]).toBe(0)
      expect(grid.activeGroups.at(-1)).toBe(new Set(c.groups).size)
      expect(grid.groupNorms.shape).toEqual([new Set(c.groups).size, 21])
      // Each group's norm along the path agrees with its weights.
      const norms = toRows(grid.groupNorms)
      const w = toRows(grid.coef)
      grid.groups.forEach((label, gi) => {
        const cols = c.groups.flatMap((l, j) => (l === label ? [j] : []))
        for (let t = 0; t < 21; t++) expect(norms[gi][t]).toBeCloseTo(Math.hypot(...cols.map((j) => w[j][t])), 12)
      })
    }
  })

  it('throws for a bad count, and is registered', () => {
    expect(() => groupLassoPath(x, y, { count: 0 })).toThrow(DomainError)
    expect(() => groupLassoPath(x, y, { alphas: [-1] })).toThrow(DomainError)
    expect(linearFunctions.groupLassoPath).toBe(groupLassoPath)
  })
})
