/**
 * LU factorisation with partial pivoting, $\Pmat\Amat = \Lmat\Umat$ (Golub and Van Loan, 2013, "Matrix Computations",
 * 4th ed., Algorithm 3.4.1), and the solves, inverse and determinants built on it. A matrix is factored once:
 * `luFactor` keeps the packed factor, `luSolve` is a primitive that only substitutes with it (linear in $\Bmat$), and
 * `solve`, `inverse`, `det` and `logDet` factor once and hand the factor to their derivative rules, which never
 * refactor. The rules are Giles (2008), "Collected matrix derivative results for forward and reverse mode algorithmic
 * differentiation", §2.2: for $\Xmat = \Amat^{-1}\Bmat$, $\bar{\Bmat} = \Amat^{-\top}\bar{\Xmat}$ (a transposed solve
 * with the same factor) and $\bar{\Amat} = -\bar{\Bmat}\Xmat^\top$; for $d = \det\Amat$,
 * $\bar{\Amat} = \bar{d} \cdot d \cdot \Amat^{-\top}$; for $\log\lvert\det\Amat\rvert$, $\bar{\Amat} = \Amat^{-\top}$.
 */

import {
  add,
  eye,
  fromData,
  isTraced,
  matmul,
  mul,
  neg,
  type NumberResult,
  type Op,
  reshape,
  shapeOfValue,
  type Tensor,
  type TensorResult,
  sub,
  transpose,
  unwrap,
  definePrimitive,
  zeros,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Index } from 'aifn-compute/foundation/contracts'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, denseSquare, EPS, LinAlgError, matrix, maxAbs, wellConditioned, type Dense } from './dense'
import { matrixTrace } from './products'
import {
  column,
  concrete,
  concreteExamples,
  exampleOf,
  float64Aval,
  foldColumns,
  kernelBatch,
  lowerMask,
  pack,
  packRaw,
  scaleOf,
  unpack,
  upperMask,
} from './rules'
import { solveTriangular } from './triangular'

/**
 * A matrix factored once, $\Pmat\Amat = \Lmat\Umat$, for repeated solves: `luSolve(f, b)` substitutes with it, and
 * derivatives of the solves with respect to $\Amat$ reuse it (no refactorisation).
 */
export type LuFactor<A extends Value = Tensor> = {
  /** The matrix that was factored (kept, possibly traced, so solves are differentiable in it). */
  readonly matrix: A
  /**
   * The packed factor ($n \times n$): $\Lmat$ strictly below the diagonal (its unit diagonal implied), $\Umat$ on and
   * above.
   */
  readonly packed: Tensor
  /** The row permutation: row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$. */
  readonly perm: Int32Array
  /** $\det\Pmat = \pm 1$, the parity of the row swaps. */
  readonly sign: 1 | -1
  /**
   * True when a pivot is at most $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$ in magnitude: $\Amat$ is singular to
   * working precision, and solves with it would be dominated by rounding. `luSolve` then throws.
   */
  readonly singular: boolean
}

/**
 * The result of `lu`: the factor of `luFactor` with $\Lmat$, $\Umat$ and $\Pmat$ unpacked as matrices ($\Lmat$ and
 * $\Umat$ traced for traced $\Amat$).
 */
export type LU<F extends Value = Tensor> = LuFactor<F> & {
  /** Unit lower-triangular factor ($n \times n$). */
  readonly L: F
  /** Upper-triangular factor ($n \times n$). */
  readonly U: F
  /** Permutation matrix with $\Pmat\Amat = \Lmat\Umat$. */
  readonly P: Tensor
}

/**
 * Factor a dense square matrix in place: $\Lmat$ below the diagonal (unit diagonal implied), $\Umat$ on and above.
 * Shared with `solveDense` (the one LU elimination in this module).
 *
 * @param options The matrix to factor, as a dense working copy.
 * @param options.n The number of rows (and columns) $n$ of the matrix.
 * @param options.a The matrix as a row-major array of $n^2$ values. Overwritten with the packed factor: the
 *   multipliers of $\Lmat$ strictly below the diagonal, $\Umat$ on and above, with the rows swapped as the pivoting
 *   chose.
 * @returns `perm`, the row permutation (row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$); `sign`, $\det\Pmat$
 *   ($+1$ for an even number of row swaps, $-1$ for an odd one); and `singular`, true when a pivot is at most
 *   $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$ in magnitude.
 */
export function factor({ n, a }: Dense): { perm: Int32Array; sign: 1 | -1; singular: boolean } {
  const perm = Int32Array.from({ length: n }, (_, i) => i)
  const tolerance = n * EPS * maxAbs(a)
  let sign: 1 | -1 = 1
  let singular = false
  for (let k = 0; k < n; k++) {
    // Partial pivoting: the largest |a_ik| on or below the diagonal; the first one wins ties.
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i * n + k]) > Math.abs(a[p * n + k])) p = i
    if (p !== k) {
      for (let j = 0; j < n; j++) [a[k * n + j], a[p * n + j]] = [a[p * n + j], a[k * n + j]]
      ;[perm[k], perm[p]] = [perm[p], perm[k]]
      sign = -sign as 1 | -1
    }
    const pivot = a[k * n + k]
    if (Math.abs(pivot) <= tolerance) singular = true
    // An exactly zero column needs no elimination (and dividing by the zero pivot would make NaN).
    if (pivot === 0) continue
    for (let i = k + 1; i < n; i++) {
      const l = (a[i * n + k] /= pivot)
      if (l !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= l * a[k * n + j]
    }
  }
  return { perm, sign, singular }
}

/**
 * The raw factor of a square matrix (traced input is read through its value).
 *
 * @param a The square matrix $\Amat$ to factor ($n \times n$); it is copied, not modified. Not square throws
 *   `ShapeError`, and a non-finite entry `LinAlgError`.
 * @param where The caller's name, used in error messages.
 * @returns The packed factor `packed` (row-major, $n^2$ values: $\Lmat$ strictly below the diagonal, $\Umat$ on and
 *   above), the size `n`, the row permutation `perm`, its `sign` ($\det\Pmat$) and whether a pivot was `singular`.
 */
function factorOf(
  a: Value,
  where: string,
): { packed: Float64Array; n: number; perm: Int32Array; sign: 1 | -1; singular: boolean } {
  const d = denseSquare(a, where)
  return { ...factor(d), packed: d.a, n: d.n }
}

/**
 * Factor a square matrix once, $\Pmat\Amat = \Lmat\Umat$ with partial pivoting, for `luSolve`. Accepts traced input:
 * the factor keeps the matrix, so solves with it are differentiable in $\Amat$ and $\Bmat$. Never throws for singular
 * input (`singular` reports it); non-finite entries throw `LinAlgError`.
 *
 * @param a The square matrix $\Amat$ to factor ($n \times n$). It is not modified; a traced value is kept in the
 *   result so that later solves are differentiable in it.
 * @returns The factor: the matrix itself, the packed $\Lmat$ and $\Umat$, the row permutation `perm`, its `sign`
 *   and the `singular` flag.
 *
 * @example Factor once, solve for several right-hand sides
 * const f = luFactor(tensor([[4, 1], [1, 3]]))
 * print('singular =', f.singular)
 * print('x1 =', luSolve(f, tensor([1, 2])))
 * print('x2 =', luSolve(f, tensor([0, 1])))
 */
export function luFactor<A extends Value>(a: A): LuFactor<A> {
  const { packed, n, perm, sign, singular } = factorOf(a, 'luFactor')
  return { matrix: a, packed: matrix(packed, n, n), perm, sign, singular }
}

/**
 * $\Lmat$ (unit lower) and $\Umat$ (upper), $n \times n$ each, unpacked from a packed factor.
 *
 * @param packed The packed factor as a row-major array of $n^2$ values: $\Lmat$ strictly below the diagonal, $\Umat$
 *   on and above. Read only.
 * @param n The number of rows (and columns) of the factored matrix.
 * @returns `L` and `U` as new row-major arrays of $n^2$ values, `L` with ones on its diagonal and zeros above, `U`
 *   with zeros below its diagonal.
 */
function unpackLU(packed: ArrayLike<number>, n: number): { L: Float64Array; U: Float64Array } {
  const L = new Float64Array(n * n)
  const U = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j < i) L[i * n + j] = packed[i * n + j]
      else U[i * n + j] = packed[i * n + j]
    }
    L[i * n + i] = 1
  }
  return { L, U }
}

/**
 * The permutation matrix $\Pmat$ of a row permutation (row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$).
 *
 * @param perm The row permutation: $n$ row indices of $\Amat$, in the order the rows take in $\Pmat\Amat$.
 * @returns $\Pmat$ as an $n \times n$ tensor, with a one at column `perm[i]` of row $i$ and zeros elsewhere.
 */
function permutationMatrix(perm: ArrayLike<number>): Tensor {
  const n = perm.length
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) P[i * n + perm[i]] = 1
  return matrix(P, n, n)
}

/** Parameters of the `lu` primitive: the factor of its input (computed when absent). */
type LuParams = {
  readonly factor?: { readonly packed: ArrayLike<number>; readonly perm: Int32Array; readonly singular: boolean }
}

/**
 * The derivative of $\Lmat$ and $\Umat$ does not exist at a singular $\Amat$ ($\Umat^{-1}$ is needed): report it,
 * example by example inside `vmap`. The test is the factorisation's: a pivot of $\Umat$ at most
 * $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$.
 *
 * @param a The matrix $\Amat$ that was factored ($n \times n$, or a batch of them inside `vmap`); only its largest
 *   absolute entry is used, to scale the tolerance.
 * @param U The upper-triangular factor $\Umat$ of `a` ($n \times n$, batched as `a` is), whose diagonal holds the
 *   pivots tested. When `a` or `U` has no concrete value nothing is checked.
 * @param n The number of rows (and columns) of $\Amat$.
 */
function refuseSingular(a: Value, U: Value, n: number): void {
  const us = concreteExamples(U)
  const as = concreteExamples(a)
  if (us === null || as === null) return
  us.forEach((u, b) => {
    const tolerance = n * EPS * scaleOf(exampleOf(as, b) ?? [])
    for (let i = 0; i < n; i++) {
      if (Math.abs(u[i * n + i]) <= tolerance) {
        const which = us.length > 1 ? ` (batch example ${b})` : ''
        throw new LinAlgError(
          `lu: the matrix${which} is singular to working precision, so its factor cannot be differentiated`,
          'singular',
        )
      }
    }
  })
}

/**
 * $\Lmat$, $\Umat$ and $\Pmat$ ($n \times n$ each) of a packed output of the `lu` primitive (or its tangent or
 * cotangent).
 *
 * @param v The packed output: a vector of $3n^2$ values holding $\Lmat$, $\Umat$ and $\Pmat$ in that order, each
 *   row-major.
 * @param n The number of rows (and columns) of each matrix.
 * @returns The three $n \times n$ matrices, in the order $\Lmat$, $\Umat$, $\Pmat$.
 */
const parts = (v: Value, n: number): Value[] =>
  unpack(v, [
    [n, n],
    [n, n],
    [n, n],
  ])

/**
 * The LU primitive, with its derivative rules for $\Pmat\Amat = \Lmat\Umat$ with $\Pmat$ held fixed (it is locally
 * constant). With $\Xmat = \Lmat^{-1} \Pmat \dot{\Amat} \Umat^{-1}$, the tangents are $\dot{\Lmat} = \Lmat
 * \operatorname{tril}(\Xmat, -1)$ and $\dot{\Umat} = \operatorname{triu}(\Xmat) \Umat$, and the adjoint is $\bar{\Amat}
 * = \Pmat^\top \Lmat^{-\top} (\operatorname{tril}(\Lmat^\top \bar{\Lmat}, -1) + \operatorname{triu}(\bar{\Umat}
 * \Umat^\top)) \Umat^{-\top}$ (Giles, 2008, §3.1; de Hoog, Anderssen and Lukas, 2011, "Differentiation of matrix
 * functionals using triangular factorization"). The output is $\Lmat$, $\Umat$ and $\Pmat$ packed into one vector (in
 * that order); $\Pmat$ is part of the output, rather than a parameter, so that the rules read each example's own
 * permutation inside `vmap`. $\Pmat$'s tangent is zero and its cotangent ignored.
 */
const luOp: Op<LuParams> = definePrimitive<LuParams>({
  id: 'numerics/linalg/lu',
  arity: 1,
  impl: ([a], p) => {
    const f = p.factor ?? factorOf(a, 'lu')
    const n = f.perm.length
    const { L, U } = unpackLU(f.packed, n)
    return packRaw([L, U, permutationMatrix(f.perm).data as Float64Array])
  },
  vjp: (g, [a], out) => {
    const n = shapeOfValue(a)[0]
    const [L, U, P] = parts(out, n)
    refuseSingular(a, U, n)
    const [gL, gU] = parts(g, n)
    const Z = add(mul(matmul(transpose(L), gL), lowerMask(n, false)), mul(matmul(gU, transpose(U)), upperMask(n)))
    const W = transpose(solveTriangular(U, transpose(Z), { lower: false }))
    const LW = solveTriangular(L, W, { transpose: true, unitDiagonal: true })
    return [matmul(transpose(P), LW)]
  },
  jvp: ([t], [a], out) => {
    if (t === null) return null
    const n = shapeOfValue(a)[0]
    const [L, U, P] = parts(out, n)
    refuseSingular(a, U, n)
    const left = solveTriangular(L, matmul(P, t), { unitDiagonal: true })
    const X = transpose(solveTriangular(U, transpose(left), { lower: false, transpose: true }))
    return pack([matmul(L, mul(X, lowerMask(n, false))), matmul(mul(X, upperMask(n)), U), zeros([n, n])])
  },
  batch: kernelBatch('numerics/linalg/lu'),
  shape: ([a]) => float64Aval([3 * a.shape[0] * a.shape[0]]),
  doc: {
    note: 'lower-upper-decomposition',
    summary: 'The unpacked LU factors L, U and the permutation P of PA = LU with partial pivoting.',
  },
  test: { secondOrder: true, cases: (draw) => [{ inputs: [wellConditioned(draw, 3)], params: {} }] },
})

/**
 * LU factorisation with partial pivoting of a square matrix, unpacked: $\Pmat\Amat = \Lmat\Umat$ with $\Lmat$ unit
 * lower-triangular, $\Umat$ upper triangular and $\Pmat$ the permutation matrix (for display and teaching; `luFactor`
 * is the solver's form). Never throws for singular input: `singular` reports it. Non-finite entries throw
 * `LinAlgError`. Differentiable in $\Amat$ ($\Pmat$ is held fixed, as it is locally constant); differentiating at a
 * singular $\Amat$ throws `LinAlgError` ('singular'). The wrapper is not available inside `vmap` (its permutation and
 * flags are per example); the primitive batches.
 *
 * @param a The square matrix $\Amat$ to factor ($n \times n$); not modified. A traced value makes `L` and `U`
 *   differentiable in it.
 * @returns The factor of `luFactor` (so it can be passed to `luSolve`) together with `L`, `U` and `P` as
 *   $n \times n$ matrices.
 *
 * @example The factors of PA = LU
 * const A = tensor([[1, 2], [3, 4]])
 * const { L, U, P } = lu(A)
 * print('L =', L)
 * print('U =', U)
 * print('P =', P)
 * print('P A =', matmul(P, A))
 * print('L U =', matmul(L, U))
 */
export function lu<A extends Value>(a: A): LU<TensorResult<A>> {
  const f = luFactor(a)
  const n = f.perm.length
  const P = permutationMatrix(f.perm)
  if (!isTraced(a)) {
    const { L, U } = unpackLU(f.packed.data as ArrayLike<number>, n)
    return { ...f, L: matrix(L, n, n), U: matrix(U, n, n), P } as unknown as LU<TensorResult<A>>
  }
  const out = luOp([a], { factor: { packed: f.packed.data as ArrayLike<number>, perm: f.perm, singular: f.singular } })
  const [L, U] = parts(out, n)
  return { ...f, L, U, P } as unknown as LU<TensorResult<A>>
}

/**
 * Solve with a packed factor, in place on an $n \times r$ right-hand side whose rows are already permuted: $\Lmat$
 * then $\Umat$.
 *
 * @param lu The packed factor as a row-major array of $n^2$ values: $\Lmat$ strictly below the diagonal (unit
 *   diagonal implied), $\Umat$ on and above. Read only; a zero on the diagonal is not checked for.
 * @param n The number of rows (and columns) of the factored matrix.
 * @param b The right-hand side $\Pmat\Bmat$ as a row-major array of $n \cdot r$ values (row $i$ is row `perm[i]` of
 *   $\Bmat$). Overwritten with the solution $\Xmat$.
 * @param r The number of columns of the right-hand side (1 for a vector).
 */
export function substitute(lu: ArrayLike<number>, n: number, b: Float64Array, r: number): void {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const l = lu[i * n + j]
      if (l !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= l * b[j * r + c]
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      const u = lu[i * n + j]
      if (u !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= u * b[j * r + c]
    }
    const d = lu[i * n + i]
    for (let c = 0; c < r; c++) b[i * r + c] /= d
  }
}

/**
 * Solve $\Amat^\top\Xmat = \Bmat$ with the factor of $\Amat$, in place on an $n \times r$ right-hand side:
 * $\Pmat\Amat = \Lmat\Umat$ gives $\Amat^\top = \Umat^\top\Lmat^\top\Pmat$, so solve $\Umat^\top\Ymat = \Bmat$
 * (forward), $\Lmat^\top\Zmat = \Ymat$ (backward, unit diagonal), then $\Xmat = \Pmat^\top\Zmat$.
 *
 * @param lu The packed factor of $\Amat$ as a row-major array of $n^2$ values: $\Lmat$ strictly below the diagonal
 *   (unit diagonal implied), $\Umat$ on and above. Read only.
 * @param n The number of rows (and columns) of $\Amat$.
 * @param perm The row permutation of the factor: row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$.
 * @param b The right-hand side $\Bmat$ as a row-major array of $n \cdot r$ values, its rows in their original order.
 *   Overwritten with the solution $\Xmat$.
 * @param r The number of columns of the right-hand side (1 for a vector).
 */
function substituteTransposed(lu: ArrayLike<number>, n: number, perm: ArrayLike<Index>, b: Float64Array, r: number) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const u = lu[j * n + i]
      if (u !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= u * b[j * r + c]
    }
    const d = lu[i * n + i]
    for (let c = 0; c < r; c++) b[i * r + c] /= d
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      const l = lu[j * n + i]
      if (l !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= l * b[j * r + c]
    }
  }
  const out = new Float64Array(n * r)
  for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) out[perm[i] * r + c] = b[i * r + c]
  b.set(out)
}

/**
 * A vector or matrix right-hand side as a dense $n \times r$ array, and whether it was a vector.
 *
 * @param b The right-hand side: a vector of $n$ values (taken as one column) or an $n \times r$ matrix. It is copied,
 *   not modified; a non-finite entry throws `LinAlgError`.
 * @param where The caller's name, used in error messages.
 * @returns `B`, a row-major working copy with its dimensions ($n \times 1$ for a vector), and `isVector`, true when
 *   `b` had rank 1 (so the solution is returned as a vector).
 */
function rightHandSide(b: Value, where: string): { B: Dense; isVector: boolean } {
  const shape = shapeOfValue(b)
  if (shape.length === 1) return { B: dense(reshape(unwrap(b), [-1, 1]), where), isVector: true }
  return { B: dense(b, where), isVector: false }
}

/**
 * Parameters of the `luSolve` primitive: the factor of its first input (computed from it when absent, as in the
 * generated tests) and whether to solve with $\Amat^\top$.
 */
type SolveParams = {
  readonly factor?: { readonly packed: ArrayLike<number>; readonly perm: Int32Array; readonly singular: boolean }
  readonly transpose: boolean
}

/**
 * The LU solve primitive, with its derivative rules (Giles, 2008, §2.2.3) for $\operatorname{op}(\Amat) \Xmat = \Bmat$,
 * where $\operatorname{op}(\Amat)$ is $\Amat$ or $\Amat^\top$, reusing the one factor. Reverse: $\bar{\Bmat} =
 * \operatorname{op}(\Amat)^{-\top} \bar{\Xmat}$ (the other orientation) and $\bar{\Amat} = -\bar{\Bmat} \Xmat^\top$
 * ($-\Xmat \bar{\Bmat}^\top$ for $\Amat^\top$). Forward: $\dot{\Xmat} = \operatorname{op}(\Amat)^{-1} (\dot{\Bmat} -
 * \operatorname{op}(\dot{\Amat}) \Xmat)$. Linear in $\Bmat$, with transpose $\bar{\Bmat} =
 * \operatorname{op}(\Amat)^{-\top} \bar{\Xmat}$. Batch: an unbatched $\Amat$ solves every example at once with its
 * factor, the batch folded into $\Bmat$'s columns; a batched $\Amat$ goes through the batched kernel, each example
 * factoring itself (the cached factor belongs to no example).
 */
const luSolveOp: Op<SolveParams> = definePrimitive<SolveParams>({
  id: 'numerics/linalg/luSolve',
  arity: 2,
  impl: ([a, b], p) => {
    const f = p.factor ?? factorOf(a, 'luSolve')
    if (f.singular) throw new LinAlgError('luSolve: the matrix is singular to working precision', 'singular')
    const n = f.perm.length
    const { B, isVector } = rightHandSide(b, 'luSolve')
    if (B.m !== n) throw new ShapeError('luSolve', `luSolve: A is ${n}×${n} but B has ${B.m} rows`)
    const r = B.n
    const x = new Float64Array(n * r)
    if (p.transpose) {
      x.set(B.a)
      substituteTransposed(f.packed, n, f.perm, x, r)
    } else {
      for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) x[i * r + c] = B.a[f.perm[i] * r + c]
      substitute(f.packed, n, x, r)
    }
    return isVector ? fromData(x, [n]) : matrix(x, n, r)
  },
  vjp: (g, [a], x, p, needed) => {
    const gb = luSolveOp([a, g], { ...p, transpose: !p.transpose })
    if (!needed[0]) return [null, gb]
    const ga = p.transpose ? matmul(column(x), transpose(column(gb))) : matmul(column(gb), transpose(column(x)))
    return [neg(ga), gb]
  },
  jvp: ([da, db], [a], x, p) => {
    if (da === null && db === null) return null
    let rhs: Value | null = db
    if (da !== null) {
      const term = matmul(p.transpose ? transpose(da) : da, column(x))
      const shaped = shapeOfValue(x).length === 1 ? reshape(term, [-1]) : term
      rhs = rhs === null ? neg(shaped) : sub(rhs, shaped)
    }
    return luSolveOp([a, rhs as Value], p)
  },
  transpose: (ct, [a], which, p) => {
    if (which !== 1) throw new AifnError('luSolve', 'luSolve: linear only in B')
    return luSolveOp([a, ct], { ...p, transpose: !p.transpose })
  },
  batch: ([a, b], [axisA, axisB], p, size) =>
    axisA === null && axisB !== null
      ? foldColumns((rhs) => luSolveOp([a, rhs], p), b, axisB, size)
      : kernelBatch<SolveParams>('numerics/linalg/luSolve')([a, b], [axisA, axisB], p, size),
  shape: ([, b]) => float64Aval(b.shape),
  doc: {
    note: 'lower-upper-decomposition',
    summary: 'Solve A X = B (or Aᵀ X = B) by substitution with the LU factor of A.',
  },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [wellConditioned(draw, 3), draw([3, 2])], params: { transpose: false } },
      { inputs: [wellConditioned(draw, 3), draw([3])], params: { transpose: true } },
    ],
  },
})

/**
 * The primitive's parameters for a factor.
 *
 * @param f The factor from `luFactor` (or `lu`) whose packed data, permutation and `singular` flag the primitive
 *   will reuse instead of factoring again.
 * @param transpose Whether the solve is with $\Amat^\top$ instead of $\Amat$ (default false).
 * @returns The parameters of the `luSolve` primitive: the factor's raw data and the `transpose` flag.
 */
const paramsOf = (f: LuFactor<Value>, transpose = false): SolveParams => ({
  factor: { packed: f.packed.data, perm: f.perm, singular: f.singular },
  transpose,
})

/**
 * Solve a factored system: $\Xmat$ with $\Amat\Xmat = \Bmat$ (or $\Amat^\top\Xmat = \Bmat$ with `transpose`), given
 * `luFactor(A)` (or `lu(A)`) and $\Bmat$ ($n$ or $n \times r$), by substitution only. Linear in $\Bmat$ and
 * differentiable in both $\Amat$ (through the factor's matrix) and $\Bmat$; the derivative reuses the factor. Throws
 * `LinAlgError` ('singular') when the factor is `singular`.
 *
 * @param f The factor of $\Amat$ ($n \times n$), as `luFactor` or `lu` returns it. Not $\Amat$ itself.
 * @param b The right-hand side $\Bmat$: a vector of $n$ values, or an $n \times r$ matrix whose columns are solved
 *   together. Not modified.
 * @param options Which system to solve with the factor.
 * @param options.transpose When true, solve $\Amat^\top\Xmat = \Bmat$ instead of $\Amat\Xmat = \Bmat$ (default
 *   false).
 * @returns The solution $\Xmat$, with the shape of `b`.
 *
 * @example Solve with a factor, and with its transpose
 * const f = luFactor(tensor([[1, 2], [3, 4]]))
 * print('A x = b:', luSolve(f, tensor([5, 6])))
 * print('Aᵀ x = b:', luSolve(f, tensor([5, 6]), { transpose: true }))
 */
export function luSolve<A extends Value, B extends Value>(
  f: LuFactor<A>,
  b: B,
  { transpose = false }: { transpose?: boolean } = {},
): TensorResult<A | B> {
  return luSolveOp([f.matrix, b], paramsOf(f, transpose)) as TensorResult<A | B>
}

/**
 * Solve $\Amat\Xmat = \Bmat$ for square $\Amat$ ($n \times n$) and $\Bmat$ ($n$ or $n \times r$): one LU
 * factorisation with partial pivoting, then `luSolve`, so the derivative reuses the factor. Throws `LinAlgError`
 * ('singular') when $\Amat$ is singular to working precision; call `luFactor` first to test without throwing.
 *
 * @param a The square coefficient matrix $\Amat$ ($n \times n$). Not modified; a traced value makes the solution
 *   differentiable in it.
 * @param b The right-hand side $\Bmat$: a vector of $n$ values, or an $n \times r$ matrix whose columns are solved
 *   together. Not modified.
 * @returns The solution $\Xmat$, with the shape of `b`.
 *
 * @example Solve a linear system
 * const A = tensor([[4, 1], [1, 3]])
 * const b = tensor([1, 2])
 * print(solve(A, b))
 *
 * @example Differentiate through the solve
 * // d/db of sum(A⁻¹ b) is the column sums of A⁻¹.
 * const A = tensor([[4, 1], [1, 3]])
 * print(grad((b) => sum(solve(A, b)))(tensor([1, 2])))
 */
export function solve<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  return luSolveOp([a, b], factorIfConcrete(a)) as TensorResult<A | B>
}

/**
 * The solve parameters with $\Amat$'s factor when $\Amat$ has a concrete value; inside `vmap` the impl factors per
 * example.
 *
 * @param a The square matrix $\Amat$ of the solve ($n \times n$); factored here when it has a concrete value.
 * @param transpose Whether the solve is with $\Amat^\top$ instead of $\Amat$ (default false).
 * @returns The parameters of the `luSolve` primitive: the `transpose` flag, with the factor of `a` when it could be
 *   computed here.
 */
function factorIfConcrete(a: Value, transpose = false): SolveParams {
  return concrete(a) === null ? { transpose } : paramsOf(luFactor(a), transpose)
}

/**
 * The inverse of a square matrix: `luSolve` of its factor against $\Imat$ (so differentiable, with
 * $\bar{\Amat} = -\Amat^{-\top} \overline{\Amat^{-1}} \Amat^{-\top}$ through the solve's rule). Throws `LinAlgError`
 * ('singular') when $\Amat$ is singular to working precision. Prefer `solve` or `choleskySolve` to multiplying by an
 * inverse.
 *
 * @param a The square matrix $\Amat$ to invert ($n \times n$). Not modified; a traced value makes the inverse
 *   differentiable in it.
 * @returns $\Amat^{-1}$, an $n \times n$ matrix.
 *
 * @example Invert a matrix
 * const A = tensor([[4, 1], [1, 3]])
 * const Ainv = inverse(A)
 * print('A⁻¹ =', Ainv)
 * print('A A⁻¹ =', matmul(A, Ainv))
 */
export function inverse<A extends Value>(a: A): TensorResult<A> {
  return luSolveOp([a, eye(shapeOfValue(a)[0])], factorIfConcrete(a)) as TensorResult<A>
}

/**
 * Sign, $\log\lvert\det\Amat\rvert$ and $\det\Amat$ from a raw factor.
 *
 * @param f The raw factor of $\Amat$: `packed`, a row-major array of $n^2$ values of which only the diagonal (the
 *   pivots of $\Umat$) is read; `n`, the number of rows; and `sign`, $\det\Pmat$ of its row permutation.
 * @returns `sign`, the sign of $\det\Amat$ ($-1$, $0$ or $1$); `logAbs`, $\log\lvert\det\Amat\rvert$ ($-\infty$ when
 *   a pivot is zero); and `det`, the determinant as the product of the pivots and the permutation's sign.
 */
function determinant(f: { packed: ArrayLike<number>; n: number; sign: 1 | -1 }): {
  sign: number
  logAbs: number
  det: number
} {
  let s: number = f.sign
  let logAbs = 0
  let det: number = f.sign
  for (let i = 0; i < f.n; i++) {
    const u = f.packed[i * f.n + i]
    s *= Math.sign(u)
    logAbs += Math.log(Math.abs(u))
    det *= u
  }
  // Report an exactly zero determinant as +0, not −0.
  return { sign: s === 0 ? 0 : s, logAbs, det: det === 0 ? 0 : det }
}

/** Parameters of `det` and `logDet`: the factor of the input (computed when absent, as in the tests and `vmap`). */
type DetParams = { readonly factor?: SolveParams['factor'] }

/**
 * $\Amat^{-1}\Ymat$ (or $\Amat^{-\top}\Ymat$) with the one factor, for the rules of det and logDet. At a singular
 * $\Amat$ the derivative of $\log\lvert\det\Amat\rvert$ does not exist, and that of $\det\Amat$ is the adjugate, which
 * is not implemented: both are reported as `LinAlgError` ('singular') rather than returned as garbage.
 *
 * @param a The square matrix $\Amat$ ($n \times n$) whose determinant is being differentiated.
 * @param y The right-hand side $\Ymat$ ($n \times n$): the identity for the reverse rule, the tangent of $\Amat$ for
 *   the forward one.
 * @param p The parameters of the determinant primitive, carrying the factor of `a` when the caller already found it
 *   (the solve then reuses it; otherwise the solve factors `a` itself).
 * @param transpose When true the solve is with $\Amat^\top$, giving $\Amat^{-\top}\Ymat$; when false,
 *   $\Amat^{-1}\Ymat$.
 * @param where The caller's name (`det` or `logDet`), used in error messages.
 * @returns The solution, an $n \times n$ matrix.
 */
function solveForRule(a: Value, y: Value, p: DetParams, transpose: boolean, where: string): Value {
  if (p.factor?.singular) {
    throw new LinAlgError(
      `${where}: the matrix is singular to working precision; the derivative there (for det, the adjugate) is not ` +
        'implemented',
      'singular',
    )
  }
  return luSolveOp([a, y], { factor: p.factor, transpose })
}

/**
 * The packed LU factor of `a` for a determinant: the one already found by the caller when the parameters carry it,
 * else a fresh factorisation.
 *
 * @param a The square matrix $\Amat$ ($n \times n$); read (and factored) only when `p` carries no factor.
 * @param p The parameters of the determinant primitive, with the factor of `a` when the caller already found it.
 * @param where The caller's name (`det` or `logDet`), used in error messages.
 * @returns The raw factor: `packed` (row-major, $n^2$ values), the size `n` and `sign`, $\det\Pmat$ of its row
 *   permutation.
 */
const factorFor = (a: Value, p: DetParams, where: string) => {
  if (p.factor) return { packed: p.factor.packed, n: p.factor.perm.length, sign: signOf(p.factor.perm) }
  return factorOf(a, where)
}

/**
 * $\det\Pmat$ of a row permutation: $\pm 1$ by the parity of its cycles.
 *
 * @param perm The row permutation: $n$ row indices, each of $0, \dots, n - 1$ appearing once. Read only.
 * @returns $+1$ for an even permutation, $-1$ for an odd one (each cycle of even length flips the sign).
 */
function signOf(perm: Int32Array): 1 | -1 {
  const seen = new Uint8Array(perm.length)
  let sign: 1 | -1 = 1
  for (let i = 0; i < perm.length; i++) {
    if (seen[i]) continue
    let length = 0
    for (let j = i; !seen[j]; j = perm[j]) {
      seen[j] = 1
      length++
    }
    if (length % 2 === 0) sign = -sign as 1 | -1
  }
  return sign
}

/**
 * The parameters of det and logDet for an input: its factor when it has a concrete value.
 *
 * @param a The square matrix $\Amat$ ($n \times n$) whose determinant is wanted; factored here when it has a concrete
 *   value.
 * @returns The parameters of the determinant primitives: the factor of `a`, or nothing inside `vmap` (where the
 *   primitive factors each example itself).
 */
function detParams(a: Value): DetParams {
  return concrete(a) === null ? {} : { factor: paramsOf(luFactor(a)).factor }
}

/**
 * The determinant primitive, with its derivative rules (Giles, 2008, §2.2.2; Magnus and Neudecker, 2019, §8.3): for $d
 * = \det\Amat$, the adjoint is $\bar{\Amat} = \bar{d} \, d \, \Amat^{-\top}$ and the tangent $\dot{d} = d
 * \trace(\Amat^{-1} \dot{\Amat})$. Both reuse the one factor through `luSolve` ($\bar{\Amat}$ through a transposed
 * solve against $\Imat$).
 */
const detOp: Op<DetParams> = definePrimitive<DetParams>({
  id: 'numerics/linalg/det',
  arity: 1,
  impl: ([a], p) => determinant(factorFor(a, p, 'det')).det,
  vjp: (g, [a], d, p) => [mul(mul(g, d), solveForRule(a, eye(shapeOfValue(a)[0]), p, true, 'det'))],
  jvp: ([t], [a], d, p) => (t === null ? null : mul(d, matrixTrace(solveForRule(a, t, p, false, 'det')))),
  batch: kernelBatch('numerics/linalg/det'),
  shape: () => float64Aval([], true),
  doc: { note: 'determinant', summary: 'The determinant of a square matrix.' },
  test: { secondOrder: true, cases: (draw) => [{ inputs: [wellConditioned(draw, 3)], params: {} }] },
})

/**
 * The determinant of a square matrix (0 for a singular one), from its LU factorisation. Its derivatives
 * ($d \cdot \Amat^{-\top}$, and $d \cdot \operatorname{tr}(\Amat^{-1}\dot{\Amat})$ forward) reuse the factor; at a
 * singular $\Amat$ differentiating throws `LinAlgError` ('singular').
 *
 * @param a The square matrix $\Amat$ ($n \times n$). Not modified; a traced value makes the determinant
 *   differentiable in it.
 * @returns $\det\Amat$ as a number (or a traced scalar when `a` is traced).
 *
 * @example The determinant
 * print(det(tensor([[4, 1], [1, 3]])))
 * print(det(tensor([[1, 2], [2, 4]])))
 */
export function det<A extends Value>(a: A): NumberResult<A> {
  return detOp([a], detParams(a)) as NumberResult<A>
}

/**
 * The log-determinant primitive, with its derivative rules (as for `detOp`): for $\log \lvert \det\Amat \rvert$, the
 * adjoint is $\bar{\Amat} = \Amat^{-\top}$ and the tangent is $\trace(\Amat^{-1} \dot{\Amat})$. Both reuse the one
 * factor through `luSolve`.
 */
const logDetOp: Op<DetParams> = definePrimitive<DetParams>({
  id: 'numerics/linalg/logDet',
  arity: 1,
  impl: ([a], p) => determinant(factorFor(a, p, 'logDet')).logAbs,
  vjp: (g, [a], _y, p) => [mul(g, solveForRule(a, eye(shapeOfValue(a)[0]), p, true, 'logDet'))],
  jvp: ([t], [a], _y, p) => (t === null ? null : matrixTrace(solveForRule(a, t, p, false, 'logDet'))),
  batch: kernelBatch('numerics/linalg/logDet'),
  shape: () => float64Aval([], true),
  doc: { note: 'determinant', summary: 'log |det A|.' },
  test: { secondOrder: true, cases: (draw) => [{ inputs: [wellConditioned(draw, 3)], params: {} }] },
})

/**
 * $\log\lvert\det\Amat\rvert$ for a square matrix, from its LU factorisation, without the overflow of forming
 * $\det\Amat$ ($-\infty$ for a singular matrix, where differentiating throws `LinAlgError`). The sign is that of
 * `det(A)`; for a positive-definite matrix, `choleskyLogDet` is cheaper.
 *
 * @param a The square matrix $\Amat$ ($n \times n$). Not modified; a traced value makes the result differentiable in
 *   it.
 * @returns $\log\lvert\det\Amat\rvert$ as a number (or a traced scalar when `a` is traced).
 *
 * @example The log-determinant of a matrix with positive determinant
 * const A = tensor([[4, 1], [1, 3]])
 * print('log det A =', logDet(A))
 * print('log(det A) =', Math.log(det(A)))
 */
export function logDet<A extends Value>(a: A): NumberResult<A> {
  return logDetOp([a], detParams(a)) as NumberResult<A>
}

/**
 * The sign of $\det\Amat$ ($-1$, $0$ or $1$), from its LU factorisation.
 *
 * @param a The square matrix $\Amat$ ($n \times n$), as a concrete (untraced) tensor. Not modified.
 * @returns $-1$, $0$ or $1$: the sign of $\det\Amat$, 0 when a pivot is exactly zero.
 *
 * @example The sign of the determinant
 * print(signDet(tensor([[4, 1], [1, 3]])))
 * print(signDet(tensor([[0, 1], [1, 0]])))
 */
export function signDet(a: Tensor): number {
  return determinant(factorOf(a, 'signDet')).sign
}
