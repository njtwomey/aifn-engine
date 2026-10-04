/**
 * LU factorisation with partial pivoting, PA = LU (Golub and Van Loan, 2013, "Matrix Computations", 4th ed.,
 * Algorithm 3.4.1), and the solves, inverse and determinants built on it. A matrix is factored once: `luFactor` keeps
 * the packed factor, `luSolve` is a primitive that only substitutes with it (linear in B), and `solve`, `inverse`,
 * `det` and `logDet` factor once and hand the factor to their derivative rules, which never refactor. The rules are
 * Giles (2008), "Collected matrix derivative results for forward and reverse mode algorithmic differentiation", §2.2:
 * for X = A⁻¹B, B̄ = A⁻ᵀX̄ (a transposed solve with the same factor) and Ā = −B̄Xᵀ; for d = det A, Ā = d̄·d·A⁻ᵀ; for
 * log |det A|, Ā = A⁻ᵀ.
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
 * A matrix factored once, PA = LU, for repeated solves: `luSolve(f, b)` substitutes with it, and derivatives of the
 * solves with respect to A reuse it (no refactorisation).
 */
export type LuFactor<A extends Value = Tensor> = {
  /** The matrix that was factored (kept, possibly traced, so solves are differentiable in it). */
  readonly matrix: A
  /** The packed factor (n×n): L strictly below the diagonal (its unit diagonal implied), U on and above. */
  readonly packed: Tensor
  /** The row permutation: row i of PA is row `perm[i]` of A. */
  readonly perm: Int32Array
  /** det P = ±1, the parity of the row swaps. */
  readonly sign: 1 | -1
  /**
   * True when a pivot is at most n·ε·max|A| in magnitude: A is singular to working precision, and solves with it
   * would be dominated by rounding. `luSolve` then throws.
   */
  readonly singular: boolean
}

/** The result of `lu`: the factor of `luFactor` with L, U and P unpacked as matrices (L and U traced for traced A). */
export type LU<F extends Value = Tensor> = LuFactor<F> & {
  /** Unit lower-triangular factor (n×n). */
  readonly L: F
  /** Upper-triangular factor (n×n). */
  readonly U: F
  /** Permutation matrix with PA = LU. */
  readonly P: Tensor
}

/**
 * Factor a dense square matrix in place: L below the diagonal (unit diagonal implied), U on and above. Shared with
 * `solveDense` (the one LU elimination in this module).
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

/** The raw factor of a square matrix (traced input is read through its value). */
function factorOf(
  a: Value,
  where: string,
): { packed: Float64Array; n: number; perm: Int32Array; sign: 1 | -1; singular: boolean } {
  const d = denseSquare(a, where)
  return { ...factor(d), packed: d.a, n: d.n }
}

/**
 * Factor a square matrix once, PA = LU with partial pivoting, for `luSolve`. Accepts traced input: the factor keeps
 * the matrix, so solves with it are differentiable in A and B. Never throws for singular input (`singular` reports
 * it); non-finite entries throw `LinAlgError`.
 *
 * @example const f = luFactor(A); const x = luSolve(f, b); const y = luSolve(f, c) // one factorisation
 */
export function luFactor<A extends Value>(a: A): LuFactor<A> {
  const { packed, n, perm, sign, singular } = factorOf(a, 'luFactor')
  return { matrix: a, packed: matrix(packed, n, n), perm, sign, singular }
}

/** L (unit lower) and U (upper), n×n each, unpacked from a packed factor. */
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

/** The permutation matrix P of a row permutation (row i of PA is row perm[i] of A). */
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
 * The derivative of L and U does not exist at a singular A (U⁻¹ is needed): report it, example by example inside
 * `vmap`. The test is the factorisation's: a pivot of U at most n·ε·max|A|.
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

/** L, U and P (n×n each) of a packed output of the `lu` primitive (or its tangent or cotangent). */
const parts = (v: Value, n: number): Value[] =>
  unpack(v, [
    [n, n],
    [n, n],
    [n, n],
  ])

// Rules for PA = LU with P held fixed (it is locally constant): with X = L⁻¹ P Ȧ U⁻¹, the tangents are
// L̇ = L·tril(X, −1) and U̇ = triu(X)·U, and the adjoint is Ā = Pᵀ L⁻ᵀ (tril(LᵀL̄, −1) + triu(ŪUᵀ)) U⁻ᵀ (Giles, 2008,
// §3.1; de Hoog, Anderssen and Lukas, 2011, "Differentiation of matrix functionals using triangular factorization").
// The output is L, U and P packed into one vector (in that order); P is part of the output, rather than a parameter,
// so that the rules read each example's own permutation inside `vmap`. P's tangent is zero and its cotangent ignored.
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
 * LU factorisation with partial pivoting of a square matrix, unpacked: PA = LU with L unit lower-triangular, U upper
 * triangular and P the permutation matrix (for display and teaching; `luFactor` is the solver's form). Never throws
 * for singular input: `singular` reports it. Non-finite entries throw `LinAlgError`. Differentiable in A (P is held
 * fixed, as it is locally constant); differentiating at a singular A throws `LinAlgError` ('singular'). The wrapper is
 * not available inside `vmap` (its permutation and flags are per example); the primitive batches.
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

/** Solve with a packed factor, in place on an n×r right-hand side whose rows are already permuted: L then U. */
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
 * Solve Aᵀ X = B with the factor of A, in place on an n×r right-hand side: PA = LU gives Aᵀ = Uᵀ Lᵀ P, so solve
 * Uᵀ y = B (forward), Lᵀ z = y (backward, unit diagonal), then X = Pᵀ z.
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

/** A vector or matrix right-hand side as a dense n×r array, and whether it was a vector. */
function rightHandSide(b: Value, where: string): { B: Dense; isVector: boolean } {
  const shape = shapeOfValue(b)
  if (shape.length === 1) return { B: dense(reshape(unwrap(b), [-1, 1]), where), isVector: true }
  return { B: dense(b, where), isVector: false }
}

/**
 * Parameters of the `luSolve` primitive: the factor of its first input (computed from it when absent, as in the
 * generated tests) and whether to solve with Aᵀ.
 */
type SolveParams = {
  readonly factor?: { readonly packed: ArrayLike<number>; readonly perm: Int32Array; readonly singular: boolean }
  readonly transpose: boolean
}

// Rules (Giles, 2008, §2.2.3) for op(A) X = B, op(A) = A or Aᵀ, reusing the one factor. Reverse: B̄ = op(A)⁻ᵀX̄ (the
// other orientation) and Ā = −B̄Xᵀ (−XB̄ᵀ for Aᵀ). Forward: Ẋ = op(A)⁻¹(Ḃ − op(Ȧ)X). Linear in B, with transpose
// B̄ = op(A)⁻ᵀX̄. Batch: an unbatched A solves every example at once with its factor, the batch folded into B's
// columns; a batched A goes through the batched kernel, each example factoring itself (the cached factor belongs to no
// example).
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

/** The primitive's parameters for a factor. */
const paramsOf = (f: LuFactor<Value>, transpose = false): SolveParams => ({
  factor: { packed: f.packed.data, perm: f.perm, singular: f.singular },
  transpose,
})

/**
 * Solve a factored system: X with A X = B (or Aᵀ X = B with `transpose`), given `luFactor(A)` (or `lu(A)`) and B (n
 * or n×r), by substitution only. Linear in B and differentiable in both A (through the factor's matrix) and B; the
 * derivative reuses the factor. Throws `LinAlgError` ('singular') when the factor is `singular`.
 */
export function luSolve<A extends Value, B extends Value>(
  f: LuFactor<A>,
  b: B,
  { transpose = false }: { transpose?: boolean } = {},
): TensorResult<A | B> {
  return luSolveOp([f.matrix, b], paramsOf(f, transpose)) as TensorResult<A | B>
}

/**
 * Solve A X = B for square A (n×n) and B (n or n×r): one LU factorisation with partial pivoting, then `luSolve`, so
 * the derivative reuses the factor. Throws `LinAlgError` ('singular') when A is singular to working precision; call
 * `luFactor` first to test without throwing.
 */
export function solve<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  return luSolveOp([a, b], factorIfConcrete(a)) as TensorResult<A | B>
}

/** The solve parameters with A's factor when A has a concrete value; inside `vmap` the impl factors per example. */
function factorIfConcrete(a: Value, transpose = false): SolveParams {
  return concrete(a) === null ? { transpose } : paramsOf(luFactor(a), transpose)
}

/**
 * The inverse of a square matrix: `luSolve` of its factor against I (so differentiable, with Ā = −A⁻ᵀ Ā⁻¹ A⁻ᵀ through
 * the solve's rule). Throws `LinAlgError` ('singular') when A is singular to working precision. Prefer `solve` or
 * `choleskySolve` to multiplying by an inverse.
 */
export function inverse<A extends Value>(a: A): TensorResult<A> {
  return luSolveOp([a, eye(shapeOfValue(a)[0])], factorIfConcrete(a)) as TensorResult<A>
}

/** Sign, log |det| and det from a raw factor. */
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
 * A⁻¹Y (or A⁻ᵀY) with the one factor, for the rules of det and logDet. At a singular A the derivative of log |det A|
 * does not exist, and that of det A is the adjugate, which is not implemented: both are reported as `LinAlgError`
 * ('singular') rather than returned as garbage.
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

const factorFor = (a: Value, p: DetParams, where: string) => {
  if (p.factor) return { packed: p.factor.packed, n: p.factor.perm.length, sign: signOf(p.factor.perm) }
  return factorOf(a, where)
}

/** det P of a row permutation: ±1 by the parity of its cycles. */
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

/** The parameters of det and logDet for an input: its factor when it has a concrete value. */
function detParams(a: Value): DetParams {
  return concrete(a) === null ? {} : { factor: paramsOf(luFactor(a)).factor }
}

// Rules (Giles, 2008, §2.2.2; Magnus and Neudecker, 2019, §8.3): for d = det A, Ā = d̄·d·A⁻ᵀ and ḋ = d·tr(A⁻¹Ȧ); for
// log |det A|, Ā = A⁻ᵀ and the tangent is tr(A⁻¹Ȧ). Both reuse the one factor through luSolve (Ā through a transposed
// solve against I).
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
 * The determinant of a square matrix (0 for a singular one), from its LU factorisation. Its derivatives (d·A⁻ᵀ, and
 * d·tr(A⁻¹Ȧ) forward) reuse the factor; at a singular A differentiating throws `LinAlgError` ('singular').
 */
export function det<A extends Value>(a: A): NumberResult<A> {
  return detOp([a], detParams(a)) as NumberResult<A>
}

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
 * log |det A| for a square matrix, from its LU factorisation, without the overflow of forming det A (−∞ for a
 * singular matrix, where differentiating throws `LinAlgError`). The sign is that of `det(A)`; for a positive-definite
 * matrix, `choleskyLogDet` is cheaper.
 */
export function logDet<A extends Value>(a: A): NumberResult<A> {
  return logDetOp([a], detParams(a)) as NumberResult<A>
}

/** The sign of det A (−1, 0 or 1), from its LU factorisation. */
export function signDet(a: Tensor): number {
  return determinant(factorOf(a, 'signDet')).sign
}
