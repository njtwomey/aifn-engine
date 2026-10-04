import { expect, it } from 'vitest'
import { grad, mul, ShapeError, stream, tensor, toArray, trace, type Value } from 'aifn-compute'
import { uniform } from 'aifn-compute/foundation/random'

it('the package root re-exports foundation’s common surface (D1)', () => {
  expect(toArray(tensor([1, 2]))).toEqual([1, 2])
  expect(grad((x: Value) => mul(x, x))(3)).toBe(6)
  expect(typeof uniform(stream(1))).toBe('number')
  expect(typeof trace).toBe('function')
  expect(new ShapeError('op', 'm')).toBeInstanceOf(Error)
})
