/** Implicit differentiation: implicitRoot, implicitFixedPoint, atConvergence (design K §4.3, docs/aifn-autodiff.md §11). */
import { describe, expect, it } from 'vitest'
import { grad, jacobian, jvp, vmap } from 'aifn-compute/foundation/autodiff'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { atConvergence, implicitFixedPoint, implicitRoot } from 'aifn-compute/numerics/implicit'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { LinAlgError } from 'aifn-compute/numerics/linalg'
import {
  add,
  cos,
  mul,
  sin,
  sub,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

// √p by Newton's method on raw numbers; the derivative comes from r(p, x) = x² − p = 0.
const newtonSqrt = (p: Value): Value => {
  const q = unwrap(p)
  const one = (v: number) => {
    let x = Math.max(v, 1)
    for (let k = 0; k < 60; k++) x = 0.5 * (x + v / x)
    return x
  }
  return typeof q === 'number' ? one(q) : tensor(toFlat(q).map(one), q.shape)
}
const residual = (p: Value, x: Value) => sub(mul(x, x), p)

describe('implicitRoot', () => {
  for (const solve of ['dense', 'iterative'] as const) {
    it(`√p: first and second derivatives, forward mode and vmap (${solve})`, () => {
      const root = implicitRoot(newtonSqrt, residual, { solve })
      expect(grad(root)(2)).toBeCloseTo(1 / (2 * Math.SQRT2), 12)
      expect(grad((p: Value) => grad(root)(p) as Value)(2)).toBeCloseTo(-1 / (4 * 2 ** 1.5), 9)
      expect(jvp(root, 4, 1).tangent).toBeCloseTo(0.25, 12)
      const ps = tensor([1, 4, 9])
      // Per-example gradients: the adjoint solve runs on the whole batch (the iterative one in lockstep, each example
      // stopping on its own values).
      toFlat(vmap(grad(root))(ps) as Tensor).forEach((g, k) => expect(g).toBeCloseTo(1 / (2 * (k + 1)), 11))
      toFlat(vmap((p: Value) => jvp(root, p, 1).tangent as Value)(ps) as Tensor).forEach((g, k) =>
        expect(g).toBeCloseTo(1 / (2 * (k + 1)), 11),
      )
      toFlat(vmap(grad((p: Value) => grad(root)(p) as Value))(ps) as Tensor).forEach((g, k) =>
        expect(g).toBeCloseTo(-1 / (4 * (k + 1) ** 3), 9),
      )
      toFlat(grad((q: Value) => sum(vmap(root)(q) as Value))(ps) as Tensor).forEach((g, k) =>
        expect(g).toBeCloseTo(1 / (2 * (k + 1)), 11),
      )
    })
  }
  it('a vector root: the Jacobian is diag(1/(2√p))', () => {
    const root = implicitRoot(newtonSqrt, residual)
    const J = toFlat(jacobian(root)(tensor([1, 4])) as Tensor)
    ;[0.5, 0, 0, 0.25].forEach((v, k) => expect(J[k]).toBeCloseTo(v, 12))
  })
  it('a solver whose answer does not satisfy the equation is refused', () => {
    const wrong = implicitRoot((p: Value) => mul(p, 0.5), residual)
    expect(() => grad(wrong)(2)).toThrow(NumericalError)
  })
})

describe('implicitFixedPoint', () => {
  // x = cos(p x): dx/dp = −x sin(px) / (1 + p sin(px)).
  const iterate = (p: Value, x0: Value): Value => {
    let x = unwrap(x0) as number
    const q = unwrap(p) as number
    for (let k = 0; k < 500; k++) x = Math.cos(q * x)
    return x
  }
  const F = (p: Value, x: Value) => cos(mul(p, x))
  for (const solve of ['dense', 'iterative'] as const) {
    it(`matches the closed form (${solve}), to second order by differences`, () => {
      const x = implicitFixedPoint(iterate, F, { solve })
      const p = 0.5
      const xs = iterate(p, 0) as number
      const want = (-xs * Math.sin(p * xs)) / (1 + p * Math.sin(p * xs))
      expect(grad((q: Value) => x(q, 0))(p)).toBeCloseTo(want, 10)
      const g = (q: number) => grad((r: Value) => x(r, 0))(q) as number
      const h = 1e-5
      expect(grad((q: Value) => grad((r: Value) => x(r, 0))(q) as Value)(p)).toBeCloseTo(
        (g(p + h) - g(p - h)) / (2 * h),
        6,
      )
    })
  }
  it('batches under vmap(grad) and vmap(jvp), with batched parameters and examples of different speeds', () => {
    // x = cos(p·x) + c, with c batched and held constant. The examples contract at different
    // rates (p from 0.1 to 0.9), so the iterative adjoint finishes them at different steps.
    const ps = tensor([0.1, 0.5, 0.9])
    const cs = tensor([0, 0.2, -0.1])
    // An example's value as a number (inside vmap an example of a vector is a rank-0 tensor).
    const num = (v: Value) => {
      const r = unwrap(v)
      return typeof r === 'number' ? r : toFlat(r)[0]
    }
    const solveOne = (p: number, c: number) => {
      let v = 0
      for (let k = 0; k < 2000; k++) v = Math.cos(p * v) + c
      return v
    }
    // dx/dp = −x sin(px) / (1 + p sin(px)) at the fixed point of each example.
    const want = [0, 1, 2].map((k) => {
      const [p, c] = [toFlat(ps)[k], toFlat(cs)[k]]
      const xs = solveOne(p, c)
      return (-xs * Math.sin(p * xs)) / (1 + p * Math.sin(p * xs))
    })
    for (const solve of ['dense', 'iterative'] as const) {
      // The parameters are the pair [p, c]: the solver sees each example's raw values, F the batched ones.
      const x = implicitFixedPoint(
        ([q, c]: Value[]) => solveOne(num(q), num(c)),
        ([q, c]: Value[], v: Value) => add(cos(mul(q, v)), c),
        { solve },
      )
      const perExample = (p: Value, c: Value) => x([p, c], 0)
      const g = vmap((p: Value, c: Value) => grad((q: Value) => perExample(q, c))(p) as Value)(ps, cs) as Tensor
      toFlat(g).forEach((v, k) => expect(v).toBeCloseTo(want[k], 9))
      const t = vmap((p: Value, c: Value) => jvp((q: Value) => perExample(q, c), p, 1).tangent as Value)(ps, cs)
      toFlat(t as Tensor).forEach((v, k) => expect(v).toBeCloseTo(want[k], 9))
    }
  })
  it('a divergent adjoint iteration reports not-converged', () => {
    // x = 3x − 2p has the fixed point x = p, but its iteration map is expanding (∂F/∂x = 3).
    const x = implicitFixedPoint(
      (p: Value) => p,
      (p: Value, v: Value) => sub(mul(3, v), mul(2, p)),
      {
        solve: 'iterative',
      },
    )
    expect(() => grad((q: Value) => sum(x(q, 0) as Value))(tensor([1, 2]))).toThrow(NumericalError)
    void sin
  })
})

describe('atConvergence', () => {
  type N = { t: number; x: number; converged: boolean }
  // Newton's method for x² = p on raw numbers.
  const newton = (p: Value): Algorithm<number, N> => ({
    name: 'newton',
    init: (x0) => ({ t: 0, x: x0, converged: false }),
    step: (s) => {
      const x = 0.5 * (s.x + (unwrap(p) as number) / s.x)
      return { t: s.t + 1, x, converged: Math.abs(x - s.x) < 1e-15 }
    },
  })
  it('differentiates the converged solution by the implicit function theorem', () => {
    const root = atConvergence(newton, (p: Value, x: Value) => sub(mul(x, x), p), { start: 1, select: (s) => s.x })
    expect(root(2)).toBeCloseTo(Math.SQRT2, 14)
    expect(grad(root)(2)).toBeCloseTo(1 / (2 * Math.SQRT2), 12)
  })
  it('an algorithm that does not converge is refused', () => {
    const stuck = (_p: Value): Algorithm<number, N> => ({
      name: 'stuck',
      init: (x0) => ({ t: 0, x: x0, converged: false }),
      step: (s) => ({ ...s, t: s.t + 1 }),
    })
    const r = atConvergence(stuck, (p: Value, x: Value) => sub(x, p), { start: 1, select: (s) => s.x, maxSteps: 5 })
    expect(() => r(2)).toThrow(NumericalError)
  })
})

describe('the dense adjoint solve is linalg.solve', () => {
  it('a singular adjoint system raises LinAlgError (singular), never a gradient', () => {
    // r(p, x) = x² − p at p = 0 has the root x = 0, where ∂r/∂x = 2x = 0.
    const root = implicitRoot(
      (_p: Value) => tensor([0]) as Value,
      (p: Value, x: Value) => sub(mul(x, x), p),
      {
        solve: 'dense',
      },
    )
    expect(() => grad((p: Value) => sum(root(p)))(tensor([0]))).toThrow(LinAlgError)
  })
  it('the dense and iterative adjoints agree on a 3-vector fixed point', () => {
    const F = (p: Value, x: Value) => mul(0.5, cos(add(x, p)))
    const iterate = (p: Value, x0: Value): Value => {
      let x = toFlat(x0 as Tensor)
      const q = toFlat(p as Tensor)
      for (let k = 0; k < 200; k++) x = x.map((v, i) => 0.5 * Math.cos(v + q[i]))
      return tensor(x)
    }
    const p = tensor([0.1, -0.4, 0.9])
    const x0 = tensor([0, 0, 0])
    const g = (solve: 'dense' | 'iterative') =>
      toFlat(grad((q: Value) => sum(implicitFixedPoint(iterate, F, { solve })(q, x0)))(p) as Tensor)
    const a = g('dense')
    g('iterative').forEach((v, i) => expect(v).toBeCloseTo(a[i], 9))
  })
})
