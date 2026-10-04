/**
 * The values a program sees: numbers and plain (nested) arrays of numbers. Prelude functions take numbers, arrays,
 * typed arrays or tensors, compute with aifn tensors, and hand back numbers or plain arrays, so a program can use
 * `x.map(…)`, `x[i]` and `x.length` as in any JavaScript.
 */
import { isTensor, shapeOf, tensor, toArray, type NestedArray, type Tensor } from 'aifn-compute/foundation/tensor'

/** What a program passes to the prelude: a number, a boolean, a (nested) array, a typed array or a tensor. */
export type Input = number | boolean | Tensor | ArrayLike<unknown>

/** A program-side value: a number or a nested array of numbers. */
export type Data = number | NestedArray

/** An input as an aifn value: numbers stay numbers (booleans become 0 or 1), arrays become float64 tensors. */
export function toValue(x: unknown, what = 'argument'): number | Tensor {
  if (typeof x === 'number') return x
  if (typeof x === 'boolean') return x ? 1 : 0
  if (isTensor(x)) return x
  if (ArrayBuffer.isView(x)) return tensor(Array.from(x as unknown as ArrayLike<number>))
  if (Array.isArray(x)) {
    try {
      return tensor(x as NestedArray)
    } catch (err) {
      throw new TypeError(`${what}: expected numbers or a rectangular array of numbers (${(err as Error).message})`)
    }
  }
  throw new TypeError(`${what}: expected a number or an array of numbers, got ${describe(x)}`)
}

/** An input as a tensor (a number becomes a scalar tensor). */
export function toTensor(x: unknown, what = 'argument'): Tensor {
  const v = toValue(x, what)
  return typeof v === 'number' ? tensor(v) : v
}

/** An aifn result as program data: numbers stay numbers, a scalar tensor becomes a number, others plain arrays. */
export function toData(v: unknown): Data {
  if (typeof v === 'number') return v
  if (isTensor(v)) {
    const t = v as Tensor
    return shapeOf(t).length === 0 ? (toArray(t) as number) : toArray(t)
  }
  throw new TypeError(`expected a number or tensor result, got ${describe(v)}`)
}

/** A short description of a value's type, for error messages. */
export function describe(x: unknown): string {
  if (x === null) return 'null'
  if (Array.isArray(x)) return 'an array'
  if (typeof x === 'function') return 'a function'
  if (typeof x === 'object') return 'an object'
  return typeof x
}

/** A length or shape argument as a shape: `n` is `[n]`, an array is itself. */
export function toShape(n: unknown, what: string): number[] {
  const shape = typeof n === 'number' ? [n] : Array.isArray(n) ? n : null
  if (shape === null || !shape.every((d) => typeof d === 'number' && Number.isInteger(d) && d >= 0))
    throw new TypeError(`${what}: expected a length or a shape (an array of non-negative integers)`)
  return shape as number[]
}
