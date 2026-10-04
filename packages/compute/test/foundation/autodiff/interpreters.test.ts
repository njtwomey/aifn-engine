/**
 * The three interpreters and their nesting: forward (jvp), reverse (vjp, grad) and batch (vmap), with Jacobians,
 * Hessians and linearize built from them (design K §4).
 */
import { describe, expect, it } from 'vitest'
import { grad, hessian, hvp, jacobian, jvp, linearize, vjp, vmap } from 'aifn-compute/foundation/autodiff'
import {
  add,
  allclose,
  dot,
  exp,
  logsumexp,
  matmul,
  mul,
  sin,
  stack,
  sum,
  tanh,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

const A = tensor([
  [1, -0.5, 2],
  [0.3, 1.2, -1],
])
const f = (x: Value): Value => tanh(matmul(A, x))
const loss = (x: Value): Value => add(logsumexp(matmul(A, sin(x))), mul(0.1, sum(mul(x, x))))
const num = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
const x0 = tensor([0.3, -0.8, 0.5])
const same = (a: Value, b: Value, tol = 1e-12) =>
  expect(allclose(a as Tensor, b as Tensor, { rtol: tol, atol: tol })).toBe(true)

describe('forward and reverse agree', () => {
  it('⟨u, J v⟩ = ⟨Jᵀ u, v⟩', () => {
    const v = tensor([0.2, 1, -0.4])
    const u = tensor([1.5, -0.7])
    const Jv = jvp(f, x0, v).tangent as Tensor
    const JTu = vjp(f, x0).pullback(u) as Tensor
    expect(num(dot(u, Jv))).toBeCloseTo(num(dot(JTu, v)), 13)
  })
  it('jacobian by forward and reverse mode', () => {
    same(jacobian(f, { mode: 'forward' })(x0) as Value, jacobian(f, { mode: 'reverse' })(x0) as Value)
    expect((jacobian(f)(x0) as Tensor).shape).toEqual([2, 3])
  })
  it('hessian = jacobian of grad; hvp = H v', () => {
    const H = hessian(loss)(x0) as Tensor
    same(H, jacobian((x: Value) => grad(loss)(x) as Value)(x0) as Value, 1e-10)
    const v = tensor([1, 0, -1])
    same(hvp(loss, x0, v) as Value, matmul(H, v), 1e-10)
  })
  it('linearize gives the value and a linear map equal to jvp', () => {
    const lin = linearize(f, x0)
    const v = tensor([0.2, 1, -0.4])
    same(lin.value as Value, f(x0))
    same(lin.jvp(v) as Value, jvp(f, x0, v).tangent as Value)
  })
})

describe('vmap', () => {
  const X = tensor([
    [0.3, -0.8, 0.5],
    [1, 0.2, -0.1],
    [-0.4, 0.6, 0.9],
    [0, 0, 0],
  ])
  const rows = [0, 1, 2, 3].map((i) => tensor(toFlat(X).slice(3 * i, 3 * i + 3)))
  it('equals a loop over examples, for values and per-example gradients', () => {
    same(vmap(f)(X) as Value, stack(rows.map(f) as Tensor[]))
    same(vmap(grad(loss))(X) as Value, stack(rows.map((r) => grad(loss)(r) as Tensor)), 1e-12)
  })
  it('in and out axes, shared arguments and pytrees', () => {
    const g = (x: Value, w: Value) => mul(x, w)
    const w = tensor([1, 2, 3])
    same(vmap(g, { inAxes: [0, null] })(X, w) as Value, mul(X, w))
    const T = vmap((x: Value) => mul(x, 2), { inAxes: 1, outAxes: 1 })(X) as Tensor
    same(T, mul(X, 2))
    const out = vmap((p: { a: Value; b: Value }) => ({ s: sum(mul(p.a, p.b)) }))({ a: X, b: X }) as { s: Tensor }
    same(out.s, sum(mul(X, X), 1))
  })
  it('nests with itself and with the derivative transforms', () => {
    const XX = stack([X, mul(X, 2)]) as Tensor
    const inner = (x: Value) => sum(exp(x))
    same(vmap(vmap(inner))(XX) as Value, sum(exp(XX), 2))
    // grad of a sum over a vmap = the sum of per-example grads.
    const total = (Y: Value) => sum(vmap(loss)(Y) as Value)
    same(grad(total)(X) as Value, vmap(grad(loss))(X) as Value, 1e-12)
    // vmap(jvp) = jvp(vmap).
    const V = mul(X, -0.5)
    same(
      vmap((x: Value, v: Value) => jvp(f, x, v).tangent as Value)(X, V) as Value,
      jvp(vmap(f), X, V).tangent as Value,
      1e-12,
    )
  })
})

describe('nesting without perturbation confusion', () => {
  it('d/dx [x · d/dy (x · y)] = 2x (Siskind and Pearlmutter, 2005), in every combination of modes', () => {
    const inner = (x: Value) => grad((y: Value) => mul(x, y))(1) as Value
    expect(grad((x: Value) => mul(x, inner(x)))(3)).toBe(6)
    const innerF = (x: Value) => jvp((y: Value) => mul(x, y), 1, 1).tangent as Value
    expect(jvp((x: Value) => mul(x, innerF(x)), 3, 1).tangent).toBe(6)
    expect(grad((x: Value) => mul(x, innerF(x)))(3)).toBe(6)
  })
  it('third derivatives by forward over reverse over reverse', () => {
    const g = (x: Value) => sin(mul(x, x))
    const d3 = (x: number) => jvp((y: Value) => grad((z: Value) => grad(g)(z) as Value)(y) as Value, x, 1).tangent
    // d³/dx³ sin(x²) = −12x sin(x²) − 8x³ cos(x²).
    const x = 0.7
    expect(d3(x) as number).toBeCloseTo(-12 * x * Math.sin(x * x) - 8 * x ** 3 * Math.cos(x * x), 11)
  })
})
