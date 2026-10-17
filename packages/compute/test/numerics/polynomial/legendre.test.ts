/**
 * aifn-compute/numerics/polynomial `legendre`: values at z = 0.5 (the HiPPO note's check), the end values
 * $P_n(\pm 1)$, orthogonality by Gauss–Legendre quadrature, and the derivative against the closed form.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { slice, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { legendre } from 'aifn-compute/numerics/polynomial'
import { gaussLegendre } from 'aifn-compute/numerics/quadrature'

const flat = (v: Value) => toFlat(v as Tensor)

describe('legendre', () => {
  it('gives P_0..P_3 at z = 0.5 as 1, 0.5, -0.125, -0.4375', () => {
    flat(legendre(4, 0.5)).forEach((v, i) => expect(v).toBeCloseTo([1, 0.5, -0.125, -0.4375][i], 14))
  })

  it('has P_n(1) = 1 and P_n(-1) = (-1)^n, with the shape [...shape(z), N]', () => {
    const P = legendre(6, tensor([-1, 1]))
    expect((P as Tensor).shape).toEqual([2, 6])
    expect(flat(P)).toEqual([1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1])
    expect((legendre(0, tensor([0.2, 0.3])) as Tensor).shape).toEqual([2, 0])
  })

  it('is orthogonal on [-1, 1] with norms 2/(2n + 1)', () => {
    const { nodes, weights } = gaussLegendre(10)
    const P = flat(legendre(5, nodes))
    const w = toFlat(weights)
    for (let m = 0; m < 5; m++)
      for (let n = 0; n < 5; n++) {
        let s = 0
        w.forEach((wi, i) => (s += wi * P[i * 5 + m] * P[i * 5 + n]))
        expect(s).toBeCloseTo(m === n ? 2 / (2 * n + 1) : 0, 12)
      }
  })

  it('differentiates in z: dP_3/dz = (15 z^2 - 3)/2', () => {
    const d = grad((z: Value) => sum(slice(legendre(4, z), 3)))(0.3) as number
    expect(d).toBeCloseTo((15 * 0.09 - 3) / 2, 12)
  })
})
