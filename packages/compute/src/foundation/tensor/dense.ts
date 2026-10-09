/**
 * Dense kernels for inner loops: small vector and matrix arithmetic on row-major `Float64Array`s, and the conversions
 * between the public surface (`VectorLike`, `MatrixLike`, `Tensor`) and working arrays. Exported from
 * `aifn-compute/foundation/tensor` as the `dense` namespace (`dense.dot`, `dense.matVec`, …), so these names never
 * collide with the differentiable primitives (`dot`, `add`, `norm`, …). They are not primitives: they do not
 * broadcast, do not record on a tape and do not check lengths. Public functions take and return tensors; these are for
 * the loops inside them.
 *
 * A matrix is a row-major array: entry $(i, j)$ of an $m \times n$ matrix is at index $i n + j$, and its dimensions are
 * passed alongside. Every arithmetic helper returns a new array; inputs are never mutated. `vec`, `mat` and `data`
 * share storage instead of copying.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DTypeError, ShapeError } from 'aifn-compute/foundation/errors'
import { float64Data, fromData, isTensor, type Tensor } from './core'
import { toFlat } from './create'
import type { Value } from './trace'

// The input aliases are defined once, in `aifn-compute/foundation/contracts`.
export type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/** A float64 working array (one over a plain `ArrayBuffer`, as `new Float64Array(n)` makes). */
export type F64 = Float64Array<ArrayBuffer>

/**
 * A copy of a vector argument as a Float64Array (reads strided tensors correctly). Accepts any `Value` so that
 * functions written with primitives type-check, but at run time it must be a rank-1 (or scalar) tensor or an array:
 * a plain number, a tensor of higher rank or a complex one is an error.
 *
 * @param v The vector: a real tensor of rank 0 or 1, or an array-like of numbers.
 * @param where The caller's name, for error messages.
 * @returns A new array of its elements.
 */
export function toF64(v: VectorLike | Value, where: string): F64 {
  if (isTensor(v)) {
    realOnly(v, where)
    if (v.shape.length > 1)
      throw new ShapeError(where, `${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return Float64Array.from(toFlat(v))
  }
  if (typeof v === 'number' || typeof (v as ArrayLike<number>).length !== 'number')
    throw new ShapeError(where, `${where}: expected a vector (a rank-1 tensor or an array of numbers)`)
  return Float64Array.from(v as ArrayLike<number>)
}

/**
 * A copy of a matrix argument ($m \times n$) as a row-major Float64Array, checking its shape when `m`, `n` are given.
 * A tensor that is not a real matrix, ragged rows and a shape other than the one asked for are errors.
 *
 * @param a The matrix: a real rank-2 tensor, or an array of rows of equal length.
 * @param where The caller's name, for error messages.
 * @param m The number of rows required; not checked when left out.
 * @param n The number of columns required; not checked when left out.
 * @returns A new row-major array `data` of the entries, with the matrix's actual rows `m` and columns `n`.
 */
export function toMatrixF64(a: MatrixLike, where: string, m?: number, n?: number): { data: F64; m: number; n: number } {
  let rows: number
  let cols: number
  let data: F64
  if (isTensor(a)) {
    realOnly(a, where)
    if (a.shape.length !== 2)
      throw new ShapeError(where, `${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    ;[rows, cols] = a.shape
    data = Float64Array.from(toFlat(a))
  } else {
    rows = a.length
    cols = rows > 0 ? a[0].length : 0
    data = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++) {
      const row = a[i]
      if (row.length !== cols) throw new ShapeError(where, `${where}: ragged matrix rows`)
      for (let j = 0; j < cols; j++) data[i * cols + j] = row[j]
    }
  }
  if ((m !== undefined && rows !== m) || (n !== undefined && cols !== n))
    throw new ShapeError(where, `${where}: expected a ${m ?? '?'}×${n ?? '?'} matrix, got ${rows}×${cols}`)
  return { data, m: rows, n: cols }
}

/**
 * The dense kernels are real: a complex tensor is a `DTypeError` rather than silently read as interleaved pairs.
 *
 * @param t The tensor to check.
 * @param where The caller's name, for the error message.
 */
function realOnly(t: Tensor, where: string): void {
  if (t.dtype === 'complex128')
    throw new DTypeError(where, `${where}: expected real values, got complex128 (take realPart, imagPart or abs)`, [
      t.dtype,
    ])
}

/**
 * Wraps a working array as a rank-1 tensor (no copy: the array must not be written afterwards).
 *
 * @param a The array, handed over to the tensor.
 * @returns A vector of its `a.length` elements.
 */
export const vec = (a: F64): Tensor => fromData(a, [a.length])

/**
 * Wraps a row-major working array as an $m \times n$ tensor (no copy: the array must not be written afterwards).
 *
 * @param a The row-major array of $m n$ values, handed over to the tensor.
 * @param m The number of rows.
 * @param n The number of columns.
 * @returns The matrix over `a`.
 */
export const mat = (a: F64, m: number, n: number): Tensor => fromData(a, [m, n])

/**
 * The elements of a tensor in row-major order: a zero-copy view of its storage (`readonlyData`) when it is contiguous
 * float64, a copy otherwise. Callers must not write to the result. A complex tensor is a `DTypeError`.
 *
 * @param t A real tensor of any shape.
 * @returns Its elements as float64, flattened.
 */
export function data(t: Tensor): F64 {
  realOnly(t, 'dense.data')
  return float64Data(t) as F64
}

/**
 * $\sum_i a_i b_i$ over the length of $\avec$.
 *
 * @param a The first vector; its length sets the number of terms.
 * @param b The second vector, at least as long as `a`.
 * @returns The inner product.
 */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/**
 * The Euclidean norm, scaled so that huge components give a large finite value rather than overflowing: computed as
 * $s \lVert \avec / s \rVert$ with $s$ the largest absolute component.
 *
 * @param a The vector.
 * @returns $\lVert \avec \rVert$; 0 for an empty or zero vector, and infinite or NaN when a component is.
 */
export const norm = (a: ArrayLike<number>): number => {
  let scale = 0
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(a[i]))
  if (scale === 0 || !Number.isFinite(scale)) return scale
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] / scale) ** 2
  return scale * Math.sqrt(s)
}

/**
 * $\yvec + \alpha \xvec$.
 *
 * @param alpha The scalar $\alpha$.
 * @param x The vector $\xvec$, at least as long as `y`.
 * @param y The vector $\yvec$; its length sets the result's. Not modified.
 * @returns A new array.
 */
export function axpy(alpha: number, x: ArrayLike<number>, y: ArrayLike<number>): F64 {
  const out = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) out[i] = y[i] + alpha * x[i]
  return out
}

/**
 * $\alpha \xvec$.
 *
 * @param alpha The scalar $\alpha$.
 * @param x The vector $\xvec$.
 * @returns A new array.
 */
export function scale(alpha: number, x: ArrayLike<number>): F64 {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = alpha * x[i]
  return out
}

/**
 * $\avec - \bvec$.
 *
 * @param a The vector $\avec$; its length sets the result's.
 * @param b The vector $\bvec$, at least as long as `a`.
 * @returns A new array.
 */
export function sub(a: ArrayLike<number>, b: ArrayLike<number>): F64 {
  const out = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i]
  return out
}

/**
 * $\avec + \bvec$.
 *
 * @param a The vector $\avec$; its length sets the result's.
 * @param b The vector $\bvec$, at least as long as `a`.
 * @returns A new array.
 */
export function add(a: ArrayLike<number>, b: ArrayLike<number>): F64 {
  const out = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
  return out
}

/**
 * $\Amat\xvec$ for a row-major $m \times n$ matrix.
 *
 * @param a The matrix $\Amat$ as a row-major array of $m n$ values.
 * @param x The vector $\xvec$ of $n$ values.
 * @param m The number of rows of $\Amat$.
 * @param n The number of columns of $\Amat$.
 * @returns A new array of $m$ values.
 */
export function matVec(a: ArrayLike<number>, x: ArrayLike<number>, m: number, n: number): F64 {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += a[i * n + j] * x[j]
    out[i] = s
  }
  return out
}

/**
 * $\Amat^\top\xvec$ for a row-major $m \times n$ matrix, without forming $\Amat^\top$.
 *
 * @param a The matrix $\Amat$ as a row-major array of $m n$ values.
 * @param x The vector $\xvec$ of $m$ values.
 * @param m The number of rows of $\Amat$.
 * @param n The number of columns of $\Amat$.
 * @returns A new array of $n$ values.
 */
export function matTVec(a: ArrayLike<number>, x: ArrayLike<number>, m: number, n: number): F64 {
  const out = new Float64Array(n)
  for (let i = 0; i < m; i++) {
    const xi = x[i]
    for (let j = 0; j < n; j++) out[j] += a[i * n + j] * xi
  }
  return out
}

/**
 * $\Amat\Bmat$ for row-major $\Amat$ ($m \times k$) and $\Bmat$ ($k \times n$).
 *
 * @param a The matrix $\Amat$ as a row-major array of $m k$ values.
 * @param b The matrix $\Bmat$ as a row-major array of $k n$ values.
 * @param m The number of rows of $\Amat$.
 * @param k The number of columns of $\Amat$ and rows of $\Bmat$.
 * @param n The number of columns of $\Bmat$.
 * @returns The product as a new row-major array of $m n$ values.
 */
export function matMul(a: ArrayLike<number>, b: ArrayLike<number>, m: number, k: number, n: number): F64 {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < k; l++) {
      // No skip of zero entries: 0·∞ and 0·NaN must give NaN, as the matmul primitive does.
      const ail = a[i * k + l]
      for (let j = 0; j < n; j++) out[i * n + j] += ail * b[l * n + j]
    }
  return out
}

/**
 * $\Amat^\top$ ($n \times m$) of a row-major $m \times n$ matrix.
 *
 * @param a The matrix $\Amat$ as a row-major array of $m n$ values.
 * @param m The number of rows of $\Amat$.
 * @param n The number of columns of $\Amat$.
 * @returns The transpose as a new row-major array.
 */
export function transpose(a: ArrayLike<number>, m: number, n: number): F64 {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = a[i * n + j]
  return out
}

/**
 * $\Amat\Bmat\Amat^\top$ ($r \times r$) for a row-major $\Amat$ ($r \times c$) and $\Bmat$ ($c \times c$): a
 * covariance carried through a linear map.
 *
 * @param a The map $\Amat$ as a row-major array of $r c$ values.
 * @param b The matrix $\Bmat$ (a covariance) as a row-major array of $c^2$ values.
 * @param r The number of rows of $\Amat$.
 * @param c The number of columns of $\Amat$, and the size of $\Bmat$.
 * @returns The product as a new row-major array of $r^2$ values (symmetric up to rounding when $\Bmat$ is).
 */
export function sandwich(a: ArrayLike<number>, b: ArrayLike<number>, r: number, c: number): F64 {
  const ab = matMul(a, b, r, c, c)
  const out = new Float64Array(r * r)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++) {
      let s = 0
      for (let l = 0; l < c; l++) s += ab[i * c + l] * a[j * c + l]
      out[i * r + j] = s
    }
  return out
}

/**
 * $(\Amat + \Amat^\top)/2$ of a row-major $n \times n$ matrix: removes the asymmetry rounding leaves in a covariance.
 *
 * @param a The matrix $\Amat$ as a row-major array of $n^2$ values.
 * @param n The number of rows (and columns) of $\Amat$.
 * @returns The symmetric part as a new row-major array.
 */
export function symmetrise(a: ArrayLike<number>, n: number): F64 {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i])
  return out
}

/**
 * $\Amat^\top\Amat$ ($n \times n$) for a row-major $m \times n$ matrix.
 *
 * @param a The matrix $\Amat$ as a row-major array of $m n$ values.
 * @param m The number of rows of $\Amat$.
 * @param n The number of columns of $\Amat$.
 * @returns The Gram matrix as a new row-major array of $n^2$ values.
 */
export function gram(a: ArrayLike<number>, m: number, n: number): F64 {
  const out = new Float64Array(n * n)
  for (let k = 0; k < m; k++)
    for (let i = 0; i < n; i++) {
      const aki = a[k * n + i]
      for (let j = 0; j < n; j++) out[i * n + j] += aki * a[k * n + j]
    }
  return out
}

/**
 * The $n \times n$ identity as a row-major array.
 *
 * @param n The number of rows (and columns).
 * @returns A new array of $n^2$ values.
 */
export function identity(n: number): F64 {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = 1
  return out
}

/**
 * Are all elements finite?
 *
 * @param a The values to check.
 * @returns False when any is infinite or NaN (true for an empty array).
 */
export const allFinite = (a: ArrayLike<number>): boolean => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false
  return true
}

/**
 * The largest absolute element ($\infty$-norm of the flattened array).
 *
 * @param a The values, of any layout.
 * @returns $\max_i \lvert a_i \rvert$; 0 for an empty array, and NaN when an element is NaN.
 */
export function maxAbs(a: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]))
  return m
}

/**
 * The matrix 1-norm (largest absolute column sum) of a row-major $n \times n$ array.
 *
 * @param a The matrix as a row-major array of $n^2$ values.
 * @param n The number of rows (and columns).
 * @returns $\max_j \sum_i \lvert A_{ij} \rvert$.
 */
export function norm1(a: ArrayLike<number>, n: number): number {
  let best = 0
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += Math.abs(a[i * n + j])
    best = Math.max(best, s)
  }
  return best
}
