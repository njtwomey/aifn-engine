/** Differentiating through algorithms step by step: `unrolled` (optionally checkpointed). */
import { describe, expect, it } from 'vitest'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { mul, sub, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { unrolled, type Algorithm } from 'aifn-compute/foundation/trace'

type GD = { t: number; x: Value }
/** Gradient descent on ½ a x² with learning rate lr, written with primitives. */
const gd = (lr: Value): Algorithm<Value, GD> => ({
  name: 'gd',
  init: (x0) => ({ t: 0, x: x0 }),
  step: (s) => ({ t: s.t + 1, x: sub(s.x, mul(lr, mul(2, s.x))) }),
})
// After n steps x = x0 (1 − 2 lr)ⁿ.
const closed = (lr: number, x0: number, n: number) => x0 * (1 - 2 * lr) ** n

describe('unrolled', () => {
  for (const checkpointEvery of [undefined, 1, 4, 30]) {
    it(`d x_n / d lr and d x_n / d x0 (checkpointEvery ${checkpointEvery})`, () => {
      const n = 25
      const xn = (lr: Value, x0: Value) => unrolled(gd, x0, n, { params: lr, checkpointEvery }).x
      const dlr = grad((lr: Value) => xn(lr, 1.5))(0.1) as number
      expect(dlr).toBeCloseTo(1.5 * n * (1 - 0.2) ** (n - 1) * -2, 10)
      expect(grad((x0: Value) => xn(0.1, x0))(1.5)).toBeCloseTo(closed(0.1, 1, n), 12)
      // Second order: d²x_n/d lr² = x0 · 4n(n − 1)(1 − 2lr)^(n − 2).
      const d2 = grad((lr: Value) => grad((l: Value) => xn(l, 1.5))(lr) as Value)(0.1) as number
      expect(d2).toBeCloseTo(1.5 * 4 * n * (n - 1) * 0.8 ** (n - 2), 8)
    })
  }
  it('vmap over learning rates equals a loop', () => {
    const lrs = tensor([0.05, 0.1, 0.2])
    const g = vmap(grad((lr: Value) => unrolled(gd, 1, 10, { params: lr, checkpointEvery: 3 }).x))(lrs) as Tensor
    toFlat(g).forEach((v, k) => {
      const lr = toFlat(lrs)[k]
      expect(v).toBeCloseTo(10 * (1 - 2 * lr) ** 9 * -2, 11)
    })
  })
  it('an algorithm that computes on raw arrays is refused', () => {
    const raw: Algorithm<Value, GD> = {
      name: 'raw',
      init: (x0) => ({ t: 0, x: x0 }),
      step: (s) => ({ t: s.t + 1, x: 0.5 * (unwrap(s.x) as number) }),
    }
    expect(() => grad((x0: Value) => unrolled(raw, x0, 3).x)(1)).toThrow(NotDifferentiableError)
  })
})
