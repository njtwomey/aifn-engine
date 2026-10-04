/**
 * Differentiating through first-order methods with `unrolled`: learning-rate hypergradients and the gradient with
 * respect to the starting point, for objectives written with primitives (an `ObjectiveFn`, and an `Objective` through
 * `objectiveFn`), checked against closed forms and central differences of the same method run on raw values.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import type { ObjectiveFn } from 'aifn-compute/foundation/contracts'
import {
  mul,
  square,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { run, unrolled } from 'aifn-compute/foundation/trace'
import { adam, gradientDescent, momentum, rmsprop } from 'aifn-compute/optim/first-order'
import { objectiveFn } from 'aifn-compute/optim'

// f(x) = ½ Σ aᵢ xᵢ², with gradient a ⊙ x, written with primitives.
const a = tensor([1, 3, 0.5])
const quadratic: ObjectiveFn = (x) =>
  ({ value: mul(0.5, sum(mul(a, square(x)))), grad: mul(a, x) }) as unknown as ReturnType<ObjectiveFn>
const x0 = [1, -2, 0.5]
const n = 30
const valueOf = (v: unknown) => unwrap(v as Value) as number

function central(f: (v: number) => number, at: number, eps = 1e-6): number {
  return (f(at + eps) - f(at - eps)) / (2 * eps)
}

describe('learning-rate hypergradients', () => {
  it('gradient descent: d f(x_n)/dη in closed form', () => {
    // x_n = (1 − ηa)ⁿ x₀, so d f(x_n)/dη = Σ aᵢ x_{n,i} · n(1 − ηaᵢ)ⁿ⁻¹(−aᵢ) x₀ᵢ.
    const eta = 0.1
    const fAfter = (lr: Value) =>
      unrolled((s: Value) => gradientDescent(quadratic, { stepSize: s, tolerance: 0 }), { x0 }, n, { params: lr })
        .value as unknown as Value
    const g = valueOf(grad(fAfter)(eta))
    const A = toFlat(a)
    const expected = x0.reduce((s, xi, i) => {
      const xn = (1 - eta * A[i]) ** n * xi
      return s + A[i] * xn * n * (1 - eta * A[i]) ** (n - 1) * -A[i] * xi
    }, 0)
    expect(g).toBeCloseTo(expected, 12)
  })

  for (const [name, method] of [
    ['momentum', (lr: Value) => momentum(quadratic, { stepSize: lr, tolerance: 0 })],
    ['rmsprop', (lr: Value) => rmsprop(quadratic, { stepSize: lr, tolerance: 0 })],
    ['adam', (lr: Value) => adam(quadratic, { stepSize: lr, tolerance: 0 })],
  ] as const) {
    it(`${name}: matches central differences, with and without checkpoints`, () => {
      const eta = 0.05
      const raw = (lr: number) => run(method(lr), { x0 }, n).value
      const fd = central(raw, eta)
      for (const checkpointEvery of [undefined, 7]) {
        const g = valueOf(
          grad((lr: Value) => unrolled(method, { x0 }, n, { params: lr, checkpointEvery }).value as unknown as Value)(
            eta,
          ),
        )
        expect(g).toBeCloseTo(fd, 6)
      }
    })
  }
})

describe('gradient with respect to the start', () => {
  it('gradient descent on an Objective (nested reverse mode through objectiveFn)', () => {
    const f = objectiveFn({
      kind: 'objective',
      name: 'quadratic',
      dim: 3,
      value: (x: Tensor) => mul(0.5, sum(mul(a, square(x)))),
    })
    const alg = gradientDescent(f, { stepSize: 0.1, tolerance: 0 })
    const gx = toFlat(
      grad((start: Value) => unrolled(alg, { x0: start as unknown as Vector }, n).value as unknown as Value)(
        tensor(x0),
      ) as Tensor,
    )
    // f(x_n) = ½ Σ aᵢ (1 − ηaᵢ)²ⁿ x₀ᵢ², so ∂/∂x₀ᵢ = aᵢ(1 − ηaᵢ)²ⁿ x₀ᵢ.
    const A = toFlat(a)
    gx.forEach((g, i) => expect(g).toBeCloseTo(A[i] * (1 - 0.1 * A[i]) ** (2 * n) * x0[i], 12))
  })

  it('the raw path is unchanged: same iterates as before tracing', () => {
    const s = run(gradientDescent(quadratic, { stepSize: 0.1, tolerance: 0 }), { x0 }, n)
    const A = toFlat(a)
    toFlat(s.x).forEach((v, i) => expect(v).toBeCloseTo((1 - 0.1 * A[i]) ** n * x0[i], 14))
    expect(typeof s.value).toBe('number')
    expect(s.stepSize).toBe(0.1)
  })
})
