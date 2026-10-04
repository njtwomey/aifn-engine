import { describe, expect, it } from 'vitest'
import {
  AifnError,
  DomainError,
  DTypeError,
  NotDifferentiableError,
  NumericalError,
  ShapeError,
} from 'aifn-compute/foundation/errors'
import { matmul, sin, sum, tensor, type Value } from 'aifn-compute/foundation/tensor'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream, integers } from 'aifn-compute/foundation/random'

describe('error classes', () => {
  it('every error is an AifnError naming its operation, with its own name and data', () => {
    const cases: [AifnError, string][] = [
      [new ShapeError('matmul', 'm', [[2, 3], [4]]), 'ShapeError'],
      [new DTypeError('less', 'm', ['complex128']), 'DTypeError'],
      [new NotDifferentiableError('floor'), 'NotDifferentiableError'],
      [new NumericalError('cholesky', 'm', 'not-positive-definite'), 'NumericalError'],
      [new DomainError('logit', 'm'), 'DomainError'],
    ]
    for (const [e, name] of cases) {
      expect(e).toBeInstanceOf(Error)
      expect(e).toBeInstanceOf(AifnError)
      expect(e.name).toBe(name)
      expect(e.op).toBeTruthy()
    }
    const s = cases[0][0] as ShapeError
    expect(s.shapes).toEqual([[2, 3], [4]])
    expect((cases[1][0] as DTypeError).dtypes).toEqual(['complex128'])
    expect((cases[3][0] as NumericalError).kind).toBe('not-positive-definite')
    expect(new NotDifferentiableError('floor').message).toMatch(/floor has no derivative rule/)
  })

  it('shapes are copied, so the error does not alias the caller’s arrays', () => {
    const shape = [2, 3]
    const e = new ShapeError('op', 'm', [shape])
    shape[0] = 9
    expect(e.shapes[0]).toEqual([2, 3])
  })
})

describe('the library raises them', () => {
  it('shape, dtype, domain, numerical and differentiability failures', () => {
    expect(() => matmul(tensor([[1, 2]]), tensor([[1, 2]]))).toThrow(ShapeError)
    expect(() => integers(stream(1), 0)).toThrow(DomainError)
    // A failed Cholesky factor is reported (`failed`), and differentiating it is a NumericalError.
    const bad = tensor([
      [1, 2],
      [2, 1],
    ])
    expect(cholesky(bad, { jitter: 0 }).failed).toBe(true)
    try {
      grad((a: Value) => sum(cholesky(a, { jitter: 0 }).L as Value))(bad)
      expect.unreachable()
    } catch (e) {
      expect(e).toBeInstanceOf(NumericalError)
      expect((e as NumericalError).kind).toBe('not-positive-definite')
    }
    expect(() => grad((z: never) => sin(z))(tensor([1, 2]) as never)).toThrow(ShapeError)
  })
})
