/**
 * Dtypes: the one promotion table and the result rules every primitive declares (design K §3.2).
 *
 * | a \ b      | bool      | int32   | float32 | float64 | complex128 | JS number (weak)               |
 * | ---------- | --------- | ------- | ------- | ------- | ---------- | ------------------------------ |
 * | bool       | bool $^1$ | int32   | float32 | float64 | complex128 | int32 if integer, else float64 |
 * | int32      |           | int32   | float64 | float64 | complex128 | int32 if integer, else float64 |
 * | float32    |           |         | float32 | float64 | complex128 | float32                        |
 * | float64    |           |         |         | float64 | complex128 | float64                        |
 * | complex128 |           |         |         |         | complex128 | complex128                     |
 *
 * $^1$ Logical operations only: arithmetic on bool promotes to int32 (the `same` rule below). int32 with float32 gives
 * float64 because float32 cannot hold every int32 exactly, as in NumPy. A JS number is a *weak* operand: it takes the
 * tensor's dtype where it fits, so `x + 1` keeps float32 and `mask + 1` is int32. JS numbers are always real; a complex
 * scalar is a rank-0 complex128 tensor (`complex(re, im)`).
 */

import type { DType } from 'aifn-compute/foundation/contracts'

/** Position of each dtype in the promotion lattice (int32 with float32 is the one exception to "the larger wins"). */
const RANK: Readonly<Record<DType, number>> = { bool: 0, int32: 1, float32: 2, float64: 3, complex128: 4 }

/**
 * The dtype two tensor operands promote to (the table above): the larger in the order bool, int32, float32, float64,
 * complex128, except that int32 with float32 gives float64. Symmetric.
 *
 * @param a The dtype of one operand.
 * @param b The dtype of the other.
 * @returns The dtype both are converted to before the operation.
 *
 * @example Promotion of tensor dtypes
 * print('int32 with float64 ->', promoteTypes('int32', 'float64'))
 * print('int32 with float32 ->', promoteTypes('int32', 'float32'))
 * print('float32 with complex128 ->', promoteTypes('float32', 'complex128'))
 */
export function promoteTypes(a: DType, b: DType): DType {
  if (a === b) return a
  if ((a === 'int32' && b === 'float32') || (a === 'float32' && b === 'int32')) return 'float64'
  return RANK[a] >= RANK[b] ? a : b
}

/**
 * The dtype a JS number takes next to a tensor of dtype `other` (NumPy's weak scalar rule; the last column above).
 *
 * @param value The JS number; only whether it is an integer matters, and only next to bool or int32.
 * @param other The dtype of the tensor it is combined with.
 * @returns int32 for an integer next to bool or int32, float64 for a fraction there, and `other` otherwise.
 *
 * @example A number takes the tensor's dtype where it fits
 * print('1 next to float32 ->', weakType(1, 'float32'))
 * print('1 next to int32 ->', weakType(1, 'int32'))
 * print('0.5 next to int32 ->', weakType(0.5, 'int32'))
 */
export function weakType(value: number, other: DType): DType {
  if (other === 'bool' || other === 'int32') return Number.isInteger(value) ? 'int32' : 'float64'
  return other
}

/**
 * How a primitive's result dtype follows from its promoted input dtype:
 *
 * - `same`: the promoted dtype, with bool arithmetic giving int32 (neg, add, mul, reshape, …);
 * - `float`: integers and bool become float64, floats and complex stay (exp, div, mean, sum);
 * - `bool`: always bool (comparisons);
 * - `real`: complex becomes float64, bool int32, the rest stay (abs, realPart, imagPart);
 * - `realFloat`: complex and integers become float64, float32 stays (complexAbs, angle: moduli and phases);
 * - `index`: always int32 (argmax, argmin);
 * - `complex`: always complex128 (`complex(re, im)`, expj).
 */
export type ResultRule = 'same' | 'float' | 'bool' | 'real' | 'realFloat' | 'index' | 'complex'

/**
 * The result dtype of a rule applied to the promoted input dtype.
 *
 * @param rule How the primitive's result dtype follows from its inputs' (see `ResultRule`).
 * @param promoted The inputs' dtype after promotion (`promoteTypes`, `weakType`).
 * @returns The dtype of the primitive's result.
 *
 * @example Result dtypes of a few rules
 * print('float rule on int32 ->', resultType('float', 'int32'))
 * print('same rule on bool ->', resultType('same', 'bool'))
 * print('real rule on complex128 ->', resultType('real', 'complex128'))
 */
export function resultType(rule: ResultRule, promoted: DType): DType {
  switch (rule) {
    case 'same':
      return promoted === 'bool' ? 'int32' : promoted
    case 'float':
      return promoted === 'bool' || promoted === 'int32' ? 'float64' : promoted
    case 'bool':
      return 'bool'
    case 'real':
      return promoted === 'complex128' ? 'float64' : promoted === 'bool' ? 'int32' : promoted
    case 'realFloat':
      return promoted === 'float32' ? 'float32' : 'float64'
    case 'index':
      return 'int32'
    case 'complex':
      return 'complex128'
  }
}

/**
 * True for complex128.
 *
 * @param dtype The dtype to test.
 * @returns Whether it is complex128.
 *
 * @example Which dtypes are complex
 * print('complex128:', isComplexDType('complex128'))
 * print('float64:', isComplexDType('float64'))
 */
export const isComplexDType = (dtype: DType): boolean => dtype === 'complex128'

/**
 * True for the real floating dtypes (float32, float64).
 *
 * @param dtype The dtype to test.
 * @returns Whether it is float32 or float64 (complex128 is not counted).
 *
 * @example Which dtypes are real floating point
 * print('float32:', isFloatDType('float32'))
 * print('int32:', isFloatDType('int32'))
 * print('complex128:', isFloatDType('complex128'))
 */
export const isFloatDType = (dtype: DType): boolean => dtype === 'float64' || dtype === 'float32'

/**
 * The number of storage slots one element occupies: 2 for complex128 (re, im), else 1.
 *
 * @param dtype The dtype.
 * @returns 2 for complex128, else 1.
 *
 * @example Slots per element
 * print('float64:', elementWidth('float64'))
 * print('complex128:', elementWidth('complex128'))
 */
export const elementWidth = (dtype: DType): 1 | 2 => (dtype === 'complex128' ? 2 : 1)
