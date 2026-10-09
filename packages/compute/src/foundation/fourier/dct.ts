/**
 * The orthonormal discrete cosine transform (DCT-II, Ahmed, Natarajan and Rao, 1974) and its inverse, the DCT-III.
 *
 * Both are products with the orthogonal $N \times N$ DCT matrix $\Cmat$ along the last axis, $\xvec \mapsto \Cmat\xvec$
 * and its inverse $\Cmat^\top$, so they are differentiable and batch through `matmul`. They cost $O(N^2)$
 * per row, which suits short feature vectors (cepstra, image blocks); there is no fast DCT here.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, matmul, transpose, type Tensor, type Traced, type Value } from 'aifn-compute/foundation/tensor'
import { shapeOfValue } from 'aifn-compute/foundation/tensor'

/**
 * The $n \times n$ orthonormal DCT-II matrix, $C_{ki} = \sqrt{(2 - \delta_{k0})/n} \cos(\pi k (2i + 1) / (2n))$:
 * row $k$ is the $k$-th cosine basis vector. Orthogonal, so its transpose (the DCT-III) is its inverse. Throws
 * `DomainError` unless $n$ is a positive integer.
 *
 * @param n The transform length: the number of rows and columns.
 * @returns $\Cmat$ as a float64 tensor of shape $[n, n]$, row-major.
 *
 * @example The matrix is orthogonal
 * const C = dctMatrix(4)
 * print('C =', C)
 * print('C Cᵀ =', matmul(C, transpose(C)))
 */
export function dctMatrix(n: number): Tensor {
  if (!Number.isInteger(n) || n < 1)
    throw new DomainError('dctMatrix', `dctMatrix: n must be a positive integer, got ${n}`)
  const out = new Float64Array(n * n)
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++)
      out[k * n + i] = Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n)) * Math.sqrt((k === 0 ? 1 : 2) / n)
  return fromData(out, [n, n])
}

type Out<X> = X extends Traced ? Traced : Tensor

/**
 * The length of the last axis of `x`, the transform length of `dct` and `idct`. Throws `ShapeError` (naming `dct`, also
 * when `idct` calls it) for a scalar.
 *
 * @param x The input of the transform, of rank at least 1.
 * @returns The size of its last axis.
 */
const lastLength = (x: Value): number => {
  const shape = shapeOfValue(x)
  if (shape.length === 0) throw new ShapeError('dct', 'dct: needs a tensor of rank ≥ 1')
  return shape[shape.length - 1]
}

/**
 * The orthonormal DCT-II along the last axis, as `scipy.fft.dct(x, norm='ortho')`:
 * $X_k = \sqrt{(2 - \delta_{k0})/N} \sum_{n=0}^{N-1} x_n \cos(\pi k (2n + 1) / (2N))$. Computed as the product
 * with the transposed `dctMatrix`, $O(N^2)$ per row, so it is differentiable and every leading axis is a batch.
 *
 * @param x The signal: a tensor of rank at least 1 transformed along its last axis (length $N$), or an array of
 *   numbers (read as a float64 vector). A traced value makes the result differentiable.
 * @returns The coefficients $X_0, \dots, X_{N-1}$, with the shape of `x`.
 *
 * @example A constant has only the first coefficient
 * print('dct([1, 1, 1, 1]) =', dct([1, 1, 1, 1]))
 *
 * @example An impulse spreads over every coefficient, and idct brings it back
 * const X = dct([1, 0, 0, 0])
 * print('dct(impulse) =', X)
 * print('idct(dct(impulse)) =', idct(X))
 *
 * @example Every row of a matrix is transformed
 * print('rows =', dct(tensor([[1, 1], [1, -1]])))
 */
export function dct<X extends Value | ArrayLike<number>>(x: X): Out<X> {
  const v = typeof x === 'object' && x !== null && 'length' in x ? fromData(Float64Array.from(x)) : (x as Value)
  return matmul(v, transpose(dctMatrix(lastLength(v)))) as Out<X>
}

/**
 * The inverse of `dct`, the orthonormal DCT-III, along the last axis: $x_n = \sum_k C_{kn} X_k$, the product with
 * `dctMatrix`. Differentiable, and every leading axis is a batch.
 *
 * @param x The DCT-II coefficients: a tensor of rank at least 1, transformed along its last axis.
 * @returns The signal whose `dct` is `x`, with the shape of `x`.
 *
 * @example The first coefficient alone gives a constant
 * print('idct([2, 0, 0, 0]) =', idct(tensor([2, 0, 0, 0])))
 */
export function idct<X extends Value>(x: X): Out<X> {
  return matmul(x, dctMatrix(lastLength(x))) as Out<X>
}
