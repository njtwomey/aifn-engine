/**
 * The values a program sees: numbers and plain (nested) arrays of numbers, and the conversions between them and aifn
 * tensors.
 *
 * Prelude functions take numbers, arrays, typed arrays or tensors (`toValue`, `toTensor`), compute with aifn tensors,
 * and hand back numbers or plain arrays (`toData`), so a program can use `x.map(…)`, `x[i]` and `x.length` as in any
 * JavaScript. A value of the wrong kind is a `TypeError` that names the argument, for the program's error panel.
 */
import { isTensor, shapeOf, tensor, toArray, type NestedArray, type Tensor } from 'aifn-compute/foundation/tensor'

/** What a program passes to the prelude: a number, a boolean, a (nested) array, a typed array or a tensor. */
export type Input = number | boolean | Tensor | ArrayLike<unknown>

/** A program-side value: a number or a nested array of numbers. */
export type Data = number | NestedArray

/**
 * An input as an aifn value: numbers stay numbers (booleans become 0 or 1), tensors pass through, and arrays and typed
 * arrays become float64 tensors. Throws a `TypeError` naming `what` for anything else, or for a ragged array.
 *
 * @param x The program's value.
 * @param what The caller's name for the argument, used in error messages (`'stats.mean: argument 1'`).
 * @returns A number, or a tensor of the array's shape.
 *
 * @example Numbers stay numbers; arrays become tensors
 * print('3 →', toValue(3))
 * print('true →', toValue(true))
 * print('[[1, 2], [3, 4]] →', toValue([[1, 2], [3, 4]]))
 *
 * @example A ragged array is an error that names the argument
 * try {
 *   toValue([[1, 2], [3]], 'linalg.det: argument 1')
 * } catch (e) {
 *   print(e.message)
 * }
 */
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

/**
 * An input as a tensor: as `toValue`, and a number (or boolean) becomes a scalar tensor.
 *
 * @param x The program's value.
 * @param what The caller's name for the argument, used in error messages.
 * @returns A tensor: of shape `[]` for a number, of the array's shape otherwise.
 *
 * @example A number becomes a scalar tensor
 * const t = toTensor(2.5)
 * print('shape', shapeOf(t), 'value', t)
 * print('vector shape', shapeOf(toTensor([1, 2, 3])))
 */
export function toTensor(x: unknown, what = 'argument'): Tensor {
  const v = toValue(x, what)
  return typeof v === 'number' ? tensor(v) : v
}

/**
 * An aifn result as program data: numbers stay numbers, a scalar tensor becomes a number, any other tensor a plain
 * nested array. Throws a `TypeError` for anything else.
 *
 * @param v The result of an aifn function: a number or a tensor.
 * @returns A number, or a nested array of numbers with the tensor's shape.
 *
 * @example Tensors back to plain JavaScript
 * print('scalar →', toData(tensor(7)))
 * const rows = toData(tensor([[1, 2], [3, 4]]))
 * print('matrix →', rows, 'is an array:', Array.isArray(rows))
 */
export function toData(v: unknown): Data {
  if (typeof v === 'number') return v
  if (isTensor(v)) {
    const t = v as Tensor
    return shapeOf(t).length === 0 ? (toArray(t) as number) : toArray(t)
  }
  throw new TypeError(`expected a number or tensor result, got ${describe(v)}`)
}

/**
 * A short description of a value's type, for error messages: `'an array'`, `'a function'`, `'an object'`, `'null'`,
 * or the `typeof` name.
 *
 * @param x Any value.
 * @returns The description.
 *
 * @example What an error message calls a value
 * print(describe([1, 2]), '|', describe(null), '|', describe('text'), '|', describe(() => 0))
 */
export function describe(x: unknown): string {
  if (x === null) return 'null'
  if (Array.isArray(x)) return 'an array'
  if (typeof x === 'function') return 'a function'
  if (typeof x === 'object') return 'an object'
  return typeof x
}

/**
 * A length or shape argument as a shape: `n` is `[n]`, an array is itself. Throws a `TypeError` naming `what` unless
 * every entry is a non-negative integer.
 *
 * @param n A length, or a shape as an array of lengths.
 * @param what The caller's name, used in error messages (`'array.zeros'`).
 * @returns The shape, as an array of non-negative integers (the array `n` itself when one was given).
 *
 * @example A length and a shape
 * print(toShape(4, 'array.zeros'), toShape([2, 3], 'array.zeros'))
 */
export function toShape(n: unknown, what: string): number[] {
  const shape = typeof n === 'number' ? [n] : Array.isArray(n) ? n : null
  if (shape === null || !shape.every((d) => typeof d === 'number' && Number.isInteger(d) && d >= 0))
    throw new TypeError(`${what}: expected a length or a shape (an array of non-negative integers)`)
  return shape as number[]
}
