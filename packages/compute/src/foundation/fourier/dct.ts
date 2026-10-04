/**
 * The discrete cosine transform (DCT-II, orthonormal) and its inverse, part of `aifn-compute/foundation/fourier`: products
 * with the orthogonal DCT matrix along the last axis, so they are differentiable and batch through `matmul`.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, matmul, transpose, type Tensor, type Traced, type Value } from 'aifn-compute/foundation/tensor'
import { shapeOfValue } from 'aifn-compute/foundation/tensor'

/**
 * The n × n orthonormal DCT-II matrix: Cₖᵢ = √((2 − δₖ₀)/n) cos(πk(2i + 1)/(2n)). Orthogonal, so its transpose (the
 * DCT-III) is its inverse.
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

const lastLength = (x: Value): number => {
  const shape = shapeOfValue(x)
  if (shape.length === 0) throw new ShapeError('dct', 'dct: needs a tensor of rank ≥ 1')
  return shape[shape.length - 1]
}

/**
 * The orthonormal DCT-II along the last axis, as `scipy.fft.dct(x, norm='ortho')`:
 * X[k] = √((2 − δ_k0)/N) Σ_n x[n] cos(πk(2n + 1)/(2N)). x · Cᵀ, O(N²) per row, suited to short feature vectors.
 */
export function dct<X extends Value | ArrayLike<number>>(x: X): Out<X> {
  const v = typeof x === 'object' && x !== null && 'length' in x ? fromData(Float64Array.from(x)) : (x as Value)
  return matmul(v, transpose(dctMatrix(lastLength(v)))) as Out<X>
}

/** The inverse of `dct` (the orthonormal DCT-III) along the last axis: x · C. */
export function idct<X extends Value>(x: X): Out<X> {
  return matmul(x, dctMatrix(lastLength(x))) as Out<X>
}
