/**
 * Triangular solves by forward and back substitution (Golub and Van Loan, 2013, "Matrix Computations", 4th ed.,
 * Algorithms 3.1.1–3.1.2), as a primitive with its derivative rules: for X = T⁻¹B the cotangents are
 * B̄ = T⁻ᵀX̄ and T̄ = −B̄Xᵀ restricted to T's triangle, and the tangent is Ẋ = T⁻¹(Ḃ − ṪX) (Giles, 2008, §2.2.3).
 */

import {
  defineOp,
  matmul,
  mul,
  neg,
  type Op,
  type Raw,
  reshape,
  shapeOfValue,
  sub,
  type Tensor,
  type TensorResult,
  transpose,
  avalOf,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, denseSquare, LinAlgError, matrix, wellConditioned } from './dense'
import { column, float64Aval, foldColumns, kernelBatch, lowerMask, upperMask } from './rules'

/** Options of `solveTriangular`. */
export type TriangularOptions = {
  /** T is lower triangular (default true); otherwise upper. Only that triangle of T is read. */
  lower?: boolean
  /** Solve Tᵀ X = B instead of T X = B (default false). */
  transpose?: boolean
  /** Take T's diagonal as ones without reading it (default false). */
  unitDiagonal?: boolean
}

type Params = Required<TriangularOptions>

/** Solve in place on a dense n×r right-hand side. Throws on a zero diagonal. */
export function substitute(t: Float64Array, n: number, b: Float64Array, r: number, p: Params, where: string): void {
  // Tᵀ of a lower matrix is upper: solving with Tᵀ walks the other way and reads T by columns.
  const forward = p.lower !== p.transpose
  const at = p.transpose ? (i: number, j: number) => t[j * n + i] : (i: number, j: number) => t[i * n + j]
  for (let step = 0; step < n; step++) {
    const i = forward ? step : n - 1 - step
    const d = p.unitDiagonal ? 1 : at(i, i)
    if (d === 0)
      throw new LinAlgError(`${where}: the triangular matrix is singular (zero at diagonal ${i})`, 'singular')
    for (let c = 0; c < r; c++) {
      let s = b[i * r + c]
      if (forward) for (let j = 0; j < i; j++) s -= at(i, j) * b[j * r + c]
      else for (let j = i + 1; j < n; j++) s -= at(i, j) * b[j * r + c]
      b[i * r + c] = s / d
    }
  }
}

/** T's triangle as read (with its diagonal unless the diagonal is taken as ones), as a constant mask. */
const readMask = (n: number, p: Params) => (p.lower ? lowerMask(n, !p.unitDiagonal) : upperMask(n, !p.unitDiagonal))

// op(T) X = B with op(T) = T or Tᵀ. Reverse: B̄ = op(T)⁻ᵀX̄ and T̄ = −B̄Xᵀ (or −XB̄ᵀ for Tᵀ), restricted to the triangle
// read (Giles, 2008, §2.2.3). Forward: Ẋ = op(T)⁻¹(Ḃ − op(Ṫ)X) (Giles, 2008, §2.2.3). Linear in B, with transpose
// B̄ = op(T)⁻ᵀX̄. Batch: an unbatched T solves every example at once, the batch folded into B's columns; a batched T
// goes through the batched kernel (one impl call looping over the contiguous examples).
const solveTriangularOp: Op<Params> = defineOp<Params>(
  'numerics/linalg/solveTriangular',
  ([t, b], p) => {
    const T = denseSquare(t, 'solveTriangular')
    const isVector = typeof b !== 'number' && b.shape.length === 1
    const B = isVector ? dense(reshape(b as Tensor, [-1, 1]), 'solveTriangular') : dense(b as Raw, 'solveTriangular')
    if (B.m !== T.n)
      throw new ShapeError('solveTriangular', `solveTriangular: T is ${T.n}×${T.n} but B has ${B.m} rows`)
    substitute(T.a, T.n, B.a, B.n, p, 'solveTriangular')
    return isVector ? reshape(matrix(B.a, B.m, B.n), [-1]) : matrix(B.a, B.m, B.n)
  },
  (g, [t, b], x, p, needed) => {
    const n = shapeOfValue(t)[0]
    const gb = solveTriangularOp([t, g], { ...p, transpose: !p.transpose })
    if (!needed[0]) return [null, avalOf(b).number ? null : gb]
    const outer = p.transpose ? matmul(column(x), transpose(column(gb))) : matmul(column(gb), transpose(column(x)))
    const gt = mul(neg(outer), readMask(n, p))
    return [gt, avalOf(b).number ? null : gb]
  },
  {
    arity: 2,
    jvp: ([dt, db], [t], x, p) => {
      if (dt === null && db === null) return null
      const n = shapeOfValue(t)[0]
      let rhs: Value | null = db
      if (dt !== null) {
        const read = mul(dt, readMask(n, p))
        const term = matmul(p.transpose ? transpose(read) : read, column(x))
        const shaped = shapeOfValue(x).length === 1 ? reshape(term, [-1]) : term
        rhs = rhs === null ? neg(shaped) : sub(rhs, shaped)
      }
      return solveTriangularOp([t, rhs as Value], p)
    },
    transpose: (ct, [t], which, p) => {
      if (which !== 1) throw new AifnError('solveTriangular', 'solveTriangular: linear only in B')
      return solveTriangularOp([t, ct], { ...p, transpose: !p.transpose })
    },
    batch: ([t, b], [at, ab], p, size) =>
      at === null && ab !== null
        ? foldColumns((rhs) => solveTriangularOp([t, rhs], p), b, ab, size)
        : kernelBatch<Params>('numerics/linalg/solveTriangular')([t, b], [at, ab], p, size),
    shape: ([, b]) => float64Aval(b.shape),
    doc: { summary: 'Solve a triangular system by substitution.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        {
          inputs: [wellConditioned(draw, 3), draw([3])],
          params: { lower: true, transpose: false, unitDiagonal: false },
        },
        {
          inputs: [wellConditioned(draw, 3), draw([3, 2])],
          params: { lower: false, transpose: true, unitDiagonal: false },
        },
      ],
    },
  },
)

/**
 * Solve T X = B (or Tᵀ X = B) for triangular T (n×n) and B (n or n×r), by substitution. Only T's triangle is read.
 * Throws `LinAlgError` ('singular') when a diagonal entry is exactly zero.
 */
export function solveTriangular<T extends Value, B extends Value>(
  t: T,
  b: B,
  { lower = true, transpose = false, unitDiagonal = false }: TriangularOptions = {},
): TensorResult<T | B> {
  return solveTriangularOp([t, b], { lower, transpose, unitDiagonal }) as TensorResult<T | B>
}
