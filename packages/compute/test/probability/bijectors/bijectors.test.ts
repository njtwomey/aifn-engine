import { describe, expect, it } from 'vitest'
import {
  affineBijector,
  chainBijectors,
  expBijector,
  logBijector,
  normalCdfBijector,
  powerBijector,
  sigmoidBijector,
  tanhBijector,
  orderedBijector,
  softplusBijector,
  supportInteriorPoint,
  UNIT_INTERVAL,
  POSITIVE,
  REALS,
} from 'aifn-compute/probability/bijectors'
import { jacobian } from 'aifn-compute/foundation/autodiff'
import { tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

const flat = (v: Value) => (typeof v === 'number' ? [v] : toFlat(v as Tensor))

describe('the ordered bijector', () => {
  it.each(['exp', 'softplus'] as const)('%s gaps: increasing output, inverse, and log|det J| by autodiff', (gap) => {
    const b = orderedBijector({ gap })
    expect(b.eventRank).toBe(1)
    const x = tensor([-1, 0.3, -2, 1.1])
    const y = flat(b.forward(x))
    for (let k = 1; k < y.length; k++) expect(y[k]).toBeGreaterThan(y[k - 1])
    flat(b.inverse(b.forward(x))).forEach((v, i) => expect(v).toBeCloseTo(flat(x)[i], 12))
    // The Jacobian is lower triangular: log|det J| = Σ log of its diagonal.
    const J = toRows(jacobian((v: Value) => b.forward(v))(x) as Tensor)
    const logDet = J.reduce((s, row, i) => s + Math.log(Math.abs(row[i])), 0)
    expect(flat(b.logAbsDetJacobian(x))[0]).toBeCloseTo(logDet, 12)
  })

  it('exp gaps: the documented example, and one value per row of a batch', () => {
    const b = orderedBijector()
    expect(flat(b.forward(tensor([-1, 0, 0])))).toEqual([-1, 0, 1])
    const batch = tensor([
      [0, 0, 0],
      [1, 1, 1],
    ])
    expect((b.logAbsDetJacobian(batch) as Tensor).shape).toEqual([2])
    expect(() => b.forward(3)).toThrow(/vector/)
  })
})

describe('scalar bijectors', () => {
  it('forward, inverse and log-Jacobian agree with autodiff', () => {
    const onReals = [
      expBijector,
      softplusBijector,
      affineBijector(2, -3),
      sigmoidBijector,
      tanhBijector,
      normalCdfBijector,
      chainBijectors(affineBijector(0, 0.5), sigmoidBijector),
    ]
    const onPositive = [logBijector, powerBijector(2.5), powerBijector(-0.5)]
    for (const [b, x] of [
      ...onReals.map((b) => [b, tensor([-0.7, 0.4, 1.9])] as const),
      ...onPositive.map((b) => [b, tensor([0.3, 1.1, 2.4])] as const),
    ]) {
      flat(b.inverse(b.forward(x))).forEach((v, i) => expect(v).toBeCloseTo(flat(x)[i], 12))
      const J = toRows(jacobian((v: Value) => b.forward(v))(x) as Tensor)
      flat(b.logAbsDetJacobian(x)).forEach((v, i) => expect(v).toBeCloseTo(Math.log(Math.abs(J[i][i])), 10))
    }
  })

  it('supportInteriorPoint lies inside each support', () => {
    expect(supportInteriorPoint(REALS)).toBe(0)
    expect(supportInteriorPoint(POSITIVE)).toBeGreaterThan(0)
    const u = supportInteriorPoint(UNIT_INTERVAL)
    expect(u > 0 && u < 1).toBe(true)
  })
})
