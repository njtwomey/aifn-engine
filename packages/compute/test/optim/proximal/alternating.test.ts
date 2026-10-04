/**
 * Alternating projections and the projections LP-LLP uses: the group-sum projection is the orthogonal projection onto
 * its affine set (feasible, idempotent, residual orthogonal to the set's directions); rows land on the simplex;
 * alternating projections reach a point of box ∩ group sums; Dykstra's variant reaches the projection itself (checked
 * against the QP min ‖x − x₀‖² over the intersection).
 */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { quadprog } from 'aifn-compute/optim/programming'
import { alternatingProjections, projectBox, projectGroupSums, projectSimplexRows } from 'aifn-compute/optim/proximal'

const groups = [0, 0, 1, 1, 1, -1]
const targets = [1.2, 0.5]
const x0 = [0.9, 0.8, -0.3, 0.6, 0.4, 2]

describe('projectGroupSums', () => {
  it('meets the sums, leaves free coordinates alone and is the orthogonal projection', () => {
    const P = projectGroupSums(groups, targets)
    const y = toFlat(P(x0 as never))
    expect(y[0] + y[1]).toBeCloseTo(1.2, 12)
    expect(y[2] + y[3] + y[4]).toBeCloseTo(0.5, 12)
    expect(y[5]).toBe(2)
    toFlat(P(y as never)).forEach((v, i) => expect(v).toBeCloseTo(y[i], 14))
    // Directions inside the affine set: zero-sum changes within a group, or any change of a free coordinate.
    const directions = [
      [1, -1, 0, 0, 0, 0],
      [0, 0, 1, -1, 0, 0],
      [0, 0, 1, 1, -2, 0],
    ]
    for (const d of directions) expect(d.reduce((s, v, i) => s + v * (x0[i] - y[i]), 0)).toBeCloseTo(0, 12)
  })
})

describe('projectSimplexRows', () => {
  it('puts every row on the simplex', () => {
    const y = toFlat(projectSimplexRows(3)([2, -1, 0.5, 0.1, 0.1, 0.1] as never))
    expect(y[0] + y[1] + y[2]).toBeCloseTo(1, 12)
    expect(y[3] + y[4] + y[5]).toBeCloseTo(1, 12)
    expect(Math.min(...y)).toBeGreaterThanOrEqual(0)
    expect(Array.from(y.slice(3))).toEqual([1 / 3, 1 / 3, 1 / 3].map((v) => expect.closeTo(v, 12)))
  })
})

describe('alternatingProjections', () => {
  const box = projectBox(0, 1)
  const sums = projectGroupSums(groups, targets)
  it('reaches a point of the intersection', () => {
    const r = alternatingProjections([sums, box], x0, { tolerance: 1e-12, maxCycles: 5000 })
    const y = toFlat(r.x)
    expect(r.converged).toBe(true)
    for (const v of y) expect(v).toBeGreaterThanOrEqual(-1e-9)
    for (const v of y) expect(v).toBeLessThanOrEqual(1 + 1e-9)
    expect(y[0] + y[1]).toBeCloseTo(1.2, 8)
    expect(y[2] + y[3] + y[4]).toBeCloseTo(0.5, 8)
  })

  it('with Dykstra’s corrections, reaches the projection of the start (the QP solution)', () => {
    const r = alternatingProjections([sums, box], x0, { dykstra: true, tolerance: 1e-13, maxCycles: 20000 })
    const n = 6
    const Q = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
    const qp = quadprog({
      Q,
      c: x0.map((v) => -v),
      A: [...Q, ...Q.map((row) => row.map((v) => -v))],
      b: [...Array(n).fill(1), ...Array(n).fill(0)],
      E: [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
      ],
      e: targets,
    })
    const y = toFlat(r.x)
    toFlat(qp.x).forEach((v, i) => expect(y[i]).toBeCloseTo(v, 6))
  })
})
