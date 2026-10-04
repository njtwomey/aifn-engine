import { describe, expect, it } from 'vitest'
import { jacobian } from 'aifn-compute/foundation/autodiff'
import {
  add,
  fromData,
  matmul,
  mul,
  sin,
  tanh,
  toFlat,
  toRows,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { det } from 'aifn-compute/numerics/linalg'
import { affineCouplingBijector } from 'aifn-compute/probability/bijectors'

// A fixed nonlinear conditioner of the kept coordinates (any function will do).
const conditioner = (xm: Value) => ({
  shift: mul(2, sin(add(xm, 0.3))),
  logScale: tanh(mul(-1.5, xm)),
})
// The kept coordinates must condition the others across positions, so mix them along the last axis.
const mixing = (xm: Value) => {
  const W = fromData(Float64Array.from([0.5, -1, 0.7, 1.2, 0.3, -0.4, -0.8, 0.9, 0.2]), [3, 3])
  return conditioner(matmul(xm, W))
}

const num = (v: Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

describe('affineCouplingBijector', () => {
  const b = affineCouplingBijector([1, 0, 1], mixing)
  const x = fromData(Float64Array.from([0.4, -1.3, 2.1]), [3])
  it('leaves the masked coordinates and inverts exactly', () => {
    const y = toFlat(b.forward(x) as Tensor)
    expect(y[0]).toBe(0.4)
    expect(y[2]).toBe(2.1)
    const back = toFlat(b.inverse(b.forward(x)) as Tensor)
    back.forEach((v, i) => expect(v).toBeCloseTo(toFlat(x)[i], 12))
  })
  it('has log |det J| equal to the autodiff Jacobian’s', () => {
    const J = jacobian((v: Value) => b.forward(v))(x) as Tensor
    const logDet = Math.log(Math.abs(det(J) as number))
    expect(num(b.logAbsDetJacobian(x))).toBeCloseTo(logDet, 10)
    expect(toRows(J)[0][1]).toBe(0)
  })
  it('works on a batch, one log-determinant per row, and reduces to additive coupling', () => {
    const X = fromData(Float64Array.from([0.4, -1.3, 2.1, -0.2, 0.5, 0.9]), [2, 3])
    const ld = toFlat(b.logAbsDetJacobian(X) as Tensor)
    expect(ld[0]).toBeCloseTo(num(b.logAbsDetJacobian(x)), 12)
    const additive = affineCouplingBijector([1, 0], (xm) => ({
      shift: matmul(xm, fromData(Float64Array.from([0, 3, 0, 0]), [2, 2])),
    }))
    const v = fromData(Float64Array.from([1, 2]), [2])
    expect(Array.from(toFlat(additive.forward(v) as Tensor))).toEqual([1, 5])
    expect(num(additive.logAbsDetJacobian(v))).toBe(0)
  })
})
