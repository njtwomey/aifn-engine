/**
 * Triangular solves by forward and back substitution (Golub and Van Loan, 2013, "Matrix Computations", 4th ed.,
 * Algorithms 3.1.1–3.1.2), as a primitive with its derivative rules: for $\Xmat = \Tmat^{-1}\Bmat$ the cotangents are
 * $\bar{\Bmat} = \Tmat^{-\top}\bar{\Xmat}$ and $\bar{\Tmat} = -\bar{\Bmat}\Xmat^\top$ restricted to $\Tmat$'s
 * triangle, and the tangent is $\dot{\Xmat} = \Tmat^{-1}(\dot{\Bmat} - \dot{\Tmat}\Xmat)$ (Giles, 2008, §2.2.3).
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
  /** $\Tmat$ is lower triangular (default true); otherwise upper. Only that triangle of $\Tmat$ is read. */
  lower?: boolean
  /** Solve $\Tmat^\top\Xmat = \Bmat$ instead of $\Tmat\Xmat = \Bmat$ (default false). */
  transpose?: boolean
  /** Take $\Tmat$'s diagonal as ones without reading it (default false). */
  unitDiagonal?: boolean
}

type Params = Required<TriangularOptions>

/**
 * Solve in place on a dense $n \times r$ right-hand side. Throws on a zero diagonal.
 *
 * @param t The triangular matrix $\Tmat$ as a row-major array of $n^2$ values. Only the triangle named by `p.lower`
 *   is read (and not its diagonal when `p.unitDiagonal`); it is not modified.
 * @param n The number of rows (and columns) of $\Tmat$.
 * @param b The right-hand side $\Bmat$ as a row-major array of $n \cdot r$ values. Overwritten with the solution
 *   $\Xmat$.
 * @param r The number of columns of the right-hand side (1 for a vector).
 * @param p Which system to solve: `lower` says which triangle of $\Tmat$ holds the matrix, `transpose` solves with
 *   $\Tmat^\top$ instead of $\Tmat$, and `unitDiagonal` takes the diagonal as ones.
 * @param where The caller's name, used in error messages.
 */
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

/**
 * $\Tmat$'s triangle as read (with its diagonal unless the diagonal is taken as ones), as a constant mask.
 *
 * @param n The number of rows (and columns) of $\Tmat$.
 * @param p The parameters of the solve: `lower` chooses the lower or upper triangle, and `unitDiagonal` leaves the
 *   diagonal out of the mask (`transpose` is not used).
 * @returns An $n \times n$ matrix with ones at the entries of $\Tmat$ that the solve reads and zeros elsewhere.
 */
const readMask = (n: number, p: Params) => (p.lower ? lowerMask(n, !p.unitDiagonal) : upperMask(n, !p.unitDiagonal))

/**
 * The triangular solve primitive, $\operatorname{op}(\Tmat) \Xmat = \Bmat$ where $\operatorname{op}(\Tmat)$ is $\Tmat$
 * or $\Tmat^\top$, with its derivative rules (Giles, 2008, §2.2.3). Reverse: $\bar{\Bmat} =
 * \operatorname{op}(\Tmat)^{-\top} \bar{\Xmat}$ and $\bar{\Tmat} = -\bar{\Bmat} \Xmat^\top$ (or $-\Xmat
 * \bar{\Bmat}^\top$ for $\Tmat^\top$), restricted to the triangle read. Forward: $\dot{\Xmat} =
 * \operatorname{op}(\Tmat)^{-1} (\dot{\Bmat} - \operatorname{op}(\dot{\Tmat}) \Xmat)$. Linear in $\Bmat$, with
 * transpose $\bar{\Bmat} = \operatorname{op}(\Tmat)^{-\top} \bar{\Xmat}$. Batch: an unbatched $\Tmat$ solves every
 * example at once, the batch folded into $\Bmat$'s columns; a batched $\Tmat$ goes through the batched kernel (one impl
 * call looping over the contiguous examples).
 */
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
 * Solve $\Tmat\Xmat = \Bmat$ (or $\Tmat^\top\Xmat = \Bmat$) for triangular $\Tmat$ ($n \times n$) and $\Bmat$ ($n$ or
 * $n \times r$), by substitution. Only $\Tmat$'s triangle is read. Throws `LinAlgError` ('singular') when a diagonal
 * entry is exactly zero.
 *
 * @param t The triangular matrix $\Tmat$ ($n \times n$). Only the triangle chosen by `lower` is read, so the other
 *   one may hold anything. Not modified; a traced value makes the solution differentiable in it.
 * @param b The right-hand side $\Bmat$: a vector of $n$ values, or an $n \times r$ matrix whose columns are solved
 *   together. Not modified.
 * @param options Which triangular system to solve (default: lower triangular, not transposed, diagonal as stored).
 * @param options.lower When true (the default) $\Tmat$ is lower triangular and its lower triangle is read; when
 *   false, upper.
 * @param options.transpose When true, solve $\Tmat^\top\Xmat = \Bmat$ instead of $\Tmat\Xmat = \Bmat$ (default
 *   false).
 * @param options.unitDiagonal When true, take $\Tmat$'s diagonal as ones without reading it (default false).
 * @returns The solution $\Xmat$, with the shape of `b`.
 *
 * @example Forward substitution with a lower-triangular matrix
 * const L = tensor([[2, 0], [1, 3]])
 * print(solveTriangular(L, tensor([2, 7])))
 *
 * @example An upper-triangular system
 * const U = tensor([[2, 1], [0, 3]])
 * print(solveTriangular(U, tensor([5, 6]), { lower: false }))
 */
export function solveTriangular<T extends Value, B extends Value>(
  t: T,
  b: B,
  { lower = true, transpose = false, unitDiagonal = false }: TriangularOptions = {},
): TensorResult<T | B> {
  return solveTriangularOp([t, b], { lower, transpose, unitDiagonal }) as TensorResult<T | B>
}
