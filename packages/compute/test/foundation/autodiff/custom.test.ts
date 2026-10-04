/** Custom derivative rules: customVjp, customJvp and checkpoint, under every interpreter (design K §4.3). */
import { describe, expect, it } from 'vitest'
import {
  checkpoint,
  customJvp,
  customVjp,
  grad,
  hessian,
  jvp,
  refuseTraced,
  traceGraph,
  vmap,
} from 'aifn-compute/foundation/autodiff'
import { AifnError, NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  allclose,
  complex,
  conj,
  div,
  dot,
  equalTo,
  exp,
  map,
  mul,
  norm,
  scalar,
  sin,
  sum,
  tanh,
  tensor,
  toFlat,
  unwrap,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

// ‖x‖ with the subgradient 0 at x = 0 instead of NaN, written with primitives so that it also runs under vmap.
const safeNorm = customJvp(
  (x: Value) => norm(x),
  ([x], [t]) => {
    const n = norm(x)
    const zero = equalTo(n, 0)
    return [n, where(zero, 0, div(dot(x, t), where(zero, 1, n)))]
  },
)

// The straight-through estimator: round in the forward pass, identity in the backward pass.
const round = (x: Value) => map(x, Math.round)
const ste = customVjp(
  (x: Value) => round(x),
  (x: Value) => ({ out: round(x), residuals: null }),
  (_r, g) => [g],
)

describe('customJvp', () => {
  it('the rule is used in forward and reverse mode, and to second order', () => {
    const zero = tensor([0, 0])
    expect(toFlat(grad(safeNorm)(zero) as Tensor)).toEqual([0, 0])
    const x = tensor([3, 4])
    toFlat(grad(safeNorm)(x) as Tensor).forEach((g, k) => expect(g).toBeCloseTo([0.6, 0.8][k], 14))
    expect(jvp(safeNorm, x, tensor([1, 0])).tangent).toBeCloseTo(0.6, 14)
    // ∇²‖x‖ = (I − x̂x̂ᵀ)/‖x‖.
    const H = toFlat(hessian(safeNorm)(x) as Tensor)
    ;[0.64, -0.48, -0.48, 0.36].forEach((v, k) => expect(H[k]).toBeCloseTo(v / 5, 12))
  })
  it('under vmap, per example', () => {
    const X = tensor([
      [3, 4],
      [0, 0],
    ])
    toFlat(vmap(grad(safeNorm))(X) as Tensor).forEach((g, k) => expect(g).toBeCloseTo([0.6, 0.8, 0, 0][k], 14))
  })
})

describe('customVjp', () => {
  it('the straight-through estimator: rounded values, identity gradient, in every mode', () => {
    const x = tensor([0.4, 1.6])
    expect(toFlat(ste(x) as Tensor)).toEqual([0, 2])
    expect(toFlat(grad((v: Value) => sum(mul(ste(v), tensor([2, 3]))))(x) as Tensor)).toEqual([2, 3])
    expect(toFlat(jvp(ste, x, tensor([1, -1])).tangent as Tensor)).toEqual([1, -1])
    expect(toFlat(vmap(grad((v: Value) => sum(ste(v))))(tensor([[0.2, 0.7]])) as Tensor)).toEqual([1, 1])
  })
  it('pytree arguments and residuals, differentiated twice', () => {
    // f(a, b) = a·sin(b) with its hand-written vjp, residuals (a, b).
    const f = customVjp(
      (a: Value, b: Value) => mul(a, sin(b)),
      (a: Value, b: Value) => ({ out: mul(a, sin(b)), residuals: [a, b] as const }),
      ([a, b], g) => [mul(g, sin(b)), mul(g, mul(a, Math.cos(unwrap(b) as number)))],
    )
    const [ga, gb] = grad(f, { argnums: [0, 1] })(2, 0.5) as number[]
    expect(ga).toBeCloseTo(Math.sin(0.5), 14)
    expect(gb).toBeCloseTo(2 * Math.cos(0.5), 14)
    expect(grad((a: Value) => grad(f)(a, 0.5) as Value)(2)).toBe(0)
  })
})

describe('checkpoint', () => {
  const block = (x: Value) => tanh(mul(exp(sin(x)), x))
  const deep = (x: Value) => sum(block(block(block(x))))
  const saved = (x: Value) => sum(checkpoint(block)(checkpoint(block)(checkpoint(block)(x))))
  const x = tensor([0.3, -1.2, 0.8])
  it('the same values and derivatives (first, second, batched) with fewer records', () => {
    const same = (a: Value, b: Value) =>
      expect(allclose(a as Tensor, b as Tensor, { rtol: 1e-12, atol: 1e-14 })).toBe(true)
    expect(unwrap(saved(x))).toBeCloseTo(unwrap(deep(x)) as number, 14)
    same(grad(saved)(x) as Value, grad(deep)(x) as Value)
    same(hessian(saved)(x) as Value, hessian(deep)(x) as Value)
    same(
      vmap(grad(saved))(tensor([toFlat(x), [0, 1, 2]])) as Value,
      vmap(grad(deep))(tensor([toFlat(x), [0, 1, 2]])) as Value,
    )
    expect(traceGraph(saved, x).nodes.length).toBeLessThan(traceGraph(deep, x).nodes.length)
  })
  it('refuses a closure over a value traced at the checkpointed level', () => {
    // The argument is traced at the same level as the closed-over w, so recomputing the block would lose w's trace.
    expect(() => grad((w: Value) => sum(checkpoint((v: Value) => mul(v, w))(mul(w, tensor([1, 2])))))(2)).toThrow(
      AifnError,
    )
    // Passing w as an argument is fine.
    expect(grad((w: Value) => sum(checkpoint((v: Value, u: Value) => mul(v, u))(mul(w, tensor([1, 2])), w)))(2)).toBe(
      12,
    )
  })
})

describe('refuseTraced', () => {
  it('raw-array code says it cannot be differentiated', () => {
    const raw = (x: Value) => {
      refuseTraced('raw', x)
      return x
    }
    expect(raw(1)).toBe(1)
    expect(() => grad(raw)(1)).toThrow(NotDifferentiableError)
  })
})

describe('customVjp in forward mode with complex values (review 2026-10-01)', () => {
  it('the transpose trick through bwd uses the ℝ² inner product, so J·t matches the plain jvp', () => {
    const w = complex(scalar(2), scalar(3))
    const f = (z: Value) => mul(z, w)
    const g = customVjp(
      f,
      (z: Value) => ({ out: f(z), residuals: null }),
      (_r: null, ct: Value) => [mul(ct, conj(w))],
    )
    const z = complex(tensor([1]), tensor([1]))
    const t = complex(tensor([0]), tensor([1]))
    // (2 + 3i)·i = −3 + 2i; the plain Σ c·t pairing gave 3 − 2i.
    expect(toFlat(jvp(g, z, t).tangent as Tensor)).toEqual(toFlat(jvp(f, z, t).tangent as Tensor))
    expect(toFlat(jvp(g, z, t).tangent as Tensor)).toEqual([-3, 2])
  })
})
