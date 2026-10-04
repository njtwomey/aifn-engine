/**
 * Dtypes: the one promotion table and the result rules every primitive declares (design K §3.2).
 *
 * | a \ b      | bool    | int32   | float32 | float64 | complex128 | JS number (weak)               |
 * | ---------- | ------- | ------- | ------- | ------- | ---------- | ------------------------------ |
 * | bool       | bool¹   | int32   | float32 | float64 | complex128 | int32 if integer, else float64 |
 * | int32      |         | int32   | float64 | float64 | complex128 | int32 if integer, else float64 |
 * | float32    |         |         | float32 | float64 | complex128 | float32                        |
 * | float64    |         |         |         | float64 | complex128 | float64                        |
 * | complex128 |         |         |         |         | complex128 | complex128                     |
 *
 * ¹ Logical operations only: arithmetic on bool promotes to int32 (the `same` rule below). int32 with float32 gives
 * float64 because float32 cannot hold every int32 exactly, as in NumPy. A JS number is a *weak* operand: it takes the
 * tensor's dtype where it fits, so `x + 1` keeps float32 and `mask + 1` is int32. JS numbers are always real; a complex
 * scalar is a rank-0 complex128 tensor (`complex(re, im)`).
 */

import type { DType } from 'aifn-compute/foundation/contracts'

/** Position of each dtype in the promotion lattice (int32 with float32 is the one exception to "the larger wins"). */
const RANK: Readonly<Record<DType, number>> = { bool: 0, int32: 1, float32: 2, float64: 3, complex128: 4 }

/** The dtype two tensor operands promote to (the table above). */
export function promoteTypes(a: DType, b: DType): DType {
  if (a === b) return a
  if ((a === 'int32' && b === 'float32') || (a === 'float32' && b === 'int32')) return 'float64'
  return RANK[a] >= RANK[b] ? a : b
}

/** The dtype a JS number takes next to a tensor of dtype `other` (NumPy's weak scalar rule; the last column above). */
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
 * - `complex`: always complex128 (complex(re, im), expj).
 */
export type ResultRule = 'same' | 'float' | 'bool' | 'real' | 'realFloat' | 'index' | 'complex'

/** The result dtype of a rule applied to the promoted input dtype. */
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

/** True for complex128. */
export const isComplexDType = (dtype: DType): boolean => dtype === 'complex128'

/** True for the real floating dtypes (float32, float64). */
export const isFloatDType = (dtype: DType): boolean => dtype === 'float64' || dtype === 'float32'

/** The number of storage slots one element occupies: 2 for complex128 (re, im), else 1. */
export const elementWidth = (dtype: DType): 1 | 2 => (dtype === 'complex128' ? 2 : 1)
