/**
 * Thin singular value decomposition by one-sided Jacobi (Hestenes, 1958; Demmel and Veselić, 1992, "Jacobi's method
 * is more accurate than QR", SIAM J. Matrix Anal. Appl. 13(4)): plane rotations orthogonalise the columns of A in
 * place; their norms are then the singular values. Small singular values come out with high relative accuracy.
 * `pinv`, `lstsq` and `conditionNumber` are built on it.
 */

import {
  add,
  definePrimitive,
  diag,
  diagonal,
  div,
  greater,
  isTraced,
  matmul,
  mul,
  type Op,
  reshape,
  shapeOfValue,
  square,
  sub,
  type Tensor,
  type TensorResult,
  transpose,
  type Value,
  where,
} from 'aifn-compute/foundation/tensor'
import { NotDifferentiableError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, EPS, matrix, vector } from './dense'
import {
  concrete,
  concreteExamples,
  exampleOf,
  float64Aval,
  fMatrix,
  kernelBatch,
  negligible,
  pack,
  packRaw,
  scaleOf,
  unpack,
} from './rules'

/** The result of `svd` (U, S and V traced for traced input). */
export type SVD<T = Tensor> = {
  /** Left singular vectors as columns, m×k with k = min(m, n). */
  U: T
  /** Singular values in descending order, length k. */
  S: T
  /**
   * Right singular vectors as columns, n×k (so A = U diag(S) Vᵀ; NumPy returns Vᵀ as `vh`). Each pair (uⱼ, vⱼ) is
   * signed so that the largest-magnitude component of vⱼ (the first of equals) is positive.
   */
  V: T
  /** Number of Jacobi sweeps used (NaN inside `vmap`). */
  sweeps: number
  /** False when `maxSweeps` ran out before every pair of columns was orthogonal to working precision. */
  converged: boolean
}

type Jacobi = {
  U: Float64Array
  S: Float64Array
  V: Float64Array
  m: number
  n: number
  sweeps: number
  converged: boolean
}

/** The smallest normal double: a squared column norm below it has lost its precision to underflow. */
const SUBNORMAL_SQUARE = 2.2250738585072014e-308

/** One-sided Jacobi on an m×n matrix with m ≥ n (row-major in `a`, overwritten). */
function jacobi(a: Float64Array, m: number, n: number, maxSweeps: number): Jacobi {
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  let sweeps = 0
  let converged = false
  for (; sweeps < maxSweeps; sweeps++) {
    let rotated = false
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          alpha += x * x
          beta += y * y
          gamma += x * y
        }
        // A column whose squared norm is subnormal (norm below about 1.5e-154, as a rank-deficient input's null column
        // becomes) cannot have its orthogonality measured, and rotating it never converges: treat it as done. √α·√β
        // rather than √(αβ), which underflows sooner.
        if (gamma === 0 || Math.min(alpha, beta) < SUBNORMAL_SQUARE) continue
        if (Math.abs(gamma) <= EPS * Math.sqrt(alpha) * Math.sqrt(beta)) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          a[i * n + p] = c * x - s * y
          a[i * n + q] = s * x + c * y
        }
        for (let i = 0; i < n; i++) {
          const x = V[i * n + p]
          const y = V[i * n + q]
          V[i * n + p] = c * x - s * y
          V[i * n + q] = s * x + c * y
        }
      }
    }
    if (!rotated) {
      converged = true
      break
    }
  }
  const S = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    // Scaled by the column's largest entry, so a singular value near 1e-300 does not underflow to 0 when squared.
    let big = 0
    for (let i = 0; i < m; i++) big = Math.max(big, Math.abs(a[i * n + j]))
    let s = 0
    if (big > 0) for (let i = 0; i < m; i++) s += (a[i * n + j] / big) ** 2
    S[j] = big * Math.sqrt(s)
  }
  return { U: a, S, V, m, n, sweeps, converged }
}

/**
 * Normalise U's columns and complete those of zero singular values to an orthonormal set, sort by descending
 * singular value, and fix signs. Returns U (m×n), S (n), V (n×n).
 */
function finish({ U, S, V, m, n }: Jacobi): { U: Float64Array; S: Float64Array; V: Float64Array } {
  const order = Array.from({ length: n }, (_, j) => j).sort((i, j) => S[j] - S[i])
  const u = new Float64Array(m * n)
  const v = new Float64Array(n * n)
  const s = new Float64Array(n)
  const smax = S[order[0]] ?? 0
  order.forEach((src, col) => {
    s[col] = S[src]
    for (let i = 0; i < n; i++) v[i * n + col] = V[i * n + src]
    if (S[src] > 0 && S[src] > smax * EPS * EPS) for (let i = 0; i < m; i++) u[i * n + col] = U[i * n + src] / S[src]
    else fillOrthogonal(u, m, n, col)
    let big = 0
    for (let i = 0; i < n; i++) if (Math.abs(v[i * n + col]) > Math.abs(v[big * n + col])) big = i
    if (v[big * n + col] < 0) {
      for (let i = 0; i < n; i++) v[i * n + col] = -v[i * n + col]
      for (let i = 0; i < m; i++) u[i * n + col] = -u[i * n + col]
    }
  })
  return { U: u, S: s, V: v }
}

/** Set column `col` of u (m×n) to a unit vector orthogonal to columns 0…col−1, by Gram–Schmidt on basis vectors. */
function fillOrthogonal(u: Float64Array, m: number, n: number, col: number): void {
  let best = new Float64Array(m)
  let bestNorm = -1
  for (let e = 0; e < m; e++) {
    const w = new Float64Array(m)
    w[e] = 1
    for (let pass = 0; pass < 2; pass++) {
      for (let c = 0; c < col; c++) {
        let d = 0
        for (let i = 0; i < m; i++) d += u[i * n + c] * w[i]
        for (let i = 0; i < m; i++) w[i] -= d * u[i * n + c]
      }
    }
    const norm = Math.hypot(...w)
    if (norm > bestNorm) {
      best = w
      bestNorm = norm
    }
    if (norm > 0.5) break
  }
  for (let i = 0; i < m; i++) u[i * n + col] = best[i] / bestNorm
}

/**
 * The SVD of a tall (m ≥ n, row-major in `data`, overwritten) matrix, with each pair of singular vectors signed so that
 * the largest-magnitude component of vⱼ (`signOn` 'V') or of uⱼ ('U') is positive.
 */
function tall(data: Float64Array, m: number, n: number, maxSweeps: number, signOn: 'U' | 'V'): SVD {
  const raw = jacobi(data, m, n, maxSweeps)
  const { U, S, V } = finish(raw)
  if (signOn === 'U') {
    for (let col = 0; col < n; col++) {
      let big = 0
      for (let i = 0; i < m; i++) if (Math.abs(U[i * n + col]) > Math.abs(U[big * n + col])) big = i
      if (U[big * n + col] < 0) {
        for (let i = 0; i < m; i++) U[i * n + col] = -U[i * n + col]
        for (let i = 0; i < n; i++) V[i * n + col] = -V[i * n + col]
      }
    }
  }
  return { U: matrix(U, m, n), S: vector(S), V: matrix(V, n, n), sweeps: raw.sweeps, converged: raw.converged }
}

/** Parameters of the `svd` primitive (tall input): the sweep limit, the sign convention, and the SVD already found. */
type Params = { readonly maxSweeps: number; readonly signOn: 'U' | 'V'; readonly found?: SVD }

function refuseUnconverged(p: Params): void {
  if (p.found && !p.found.converged) {
    throw new NumericalError(
      'svd',
      'svd: the Jacobi iteration did not converge, so the decomposition cannot be differentiated',
      'not-converged',
    )
  }
}

/**
 * Refuse the projection term of a tall SVD at a zero singular value whose left vector's cotangent is used
 * (`used(j, b)`, every one by default), example by example inside `vmap`.
 */
function refuseZeroSingular(
  examples: readonly ArrayLike<number>[] | null,
  m: number,
  n: number,
  used: (j: number, b: number) => boolean = () => true,
): void {
  if (examples === null || m === n) return
  examples.forEach((s, b) => {
    const smax = scaleOf(s)
    for (let j = 0; j < n; j++) {
      if (s[j] <= n * EPS * smax && used(j, b)) {
        const which = examples.length > 1 ? ` of batch example ${b}` : ''
        throw new NumericalError(
          'svd',
          `svd: singular value ${j}${which} is zero, so its left singular vector has no derivative`,
          'degenerate',
        )
      }
    }
  })
}

// Rules for the thin SVD A = U diag(s) Vᵀ of a tall m×n matrix (Townsend, 2016, "Differentiating the singular value
// decomposition"; Seeger et al., 2017, arXiv:1710.08717), with Fᵢⱼ = 1/(sⱼ² − sᵢ²), S = diag(s), and P = UᵀȦV:
//   ṡ = diag(P),  U̇ = U(F∘(PS + SPᵀ)) + (I − UUᵀ)ȦVS⁻¹,  V̇ = V(F∘(SP + PᵀS))  (V is square);
//   Ā = U[diag(s̄) + (F∘(UᵀŪ − ŪᵀU))S + S(F∘(VᵀV̄ − V̄ᵀV))]Vᵀ + (I − UUᵀ)ŪS⁻¹Vᵀ.
// Repeated singular values are handled as eigh's repeated eigenvalues. A wide matrix is decomposed through its
// transpose. The output is U (m×n), s (n) and V (n×n) packed into one vector.
const svdOp: Op<Params> = definePrimitive<Params>({
  id: 'numerics/linalg/svd',
  arity: 1,
  impl: ([a], p) => {
    const f = p.found ?? decomposeTall(a, p)
    if (!f.converged) refuseUnconverged({ ...p, found: f })
    return packRaw([f.U.data as Float64Array, f.S.data as Float64Array, f.V.data as Float64Array])
  },
  vjp: (g, [a], out, p) => {
    refuseUnconverged(p)
    const [m, n] = shapeOfValue(a)
    const shapes = [[m, n], [n], [n, n]]
    const [U, s, V] = unpack(out, shapes)
    const [gU, gs, gV] = unpack(g, shapes)
    const S = diag(s)
    const UtgU = matmul(transpose(U), gU)
    const VtgV = matmul(transpose(V), gV)
    const J = sub(UtgU, transpose(UtgU))
    const K = sub(VtgV, transpose(VtgV))
    const js = concreteExamples(J)
    const ks = concreteExamples(K)
    const gss = concreteExamples(gs)
    const ss = concreteExamples(s)
    const invariant =
      js && ks && gss && ss
        ? (i: number, j: number, b: number) => {
            const [jc, kc, gc, sc] = [exampleOf(js, b), exampleOf(ks, b), exampleOf(gss, b), exampleOf(ss, b)]
            if (!jc || !kc || !gc || !sc) return false
            const scale = Math.max(scaleOf(jc), scaleOf(kc), scaleOf(gc))
            return (
              negligible((jc[i * n + j] + kc[i * n + j]) * sc[j], scale * scaleOf(sc)) &&
              negligible(gc[i] - gc[j], scale)
            )
          }
        : null
    const F = fMatrix(square(s), 'svd', invariant)
    const inner = add(diag(gs), add(matmul(mul(F, J), S), matmul(S, mul(F, K))))
    let ga: Value = matmul(matmul(U, inner), transpose(V))
    if (m > n) {
      const gUs = concreteExamples(gU)
      refuseZeroSingular(ss, m, n, (j, b) => {
        const gUc = gUs && exampleOf(gUs, b)
        return !gUc || Array.from({ length: m }, (_, i) => gUc[i * n + j]).some((v) => v !== 0)
      })
      const projected = sub(gU, matmul(U, UtgU))
      // A zero singular value passes the check above only when its column of Ū is zero, so its projected column is
      // exactly zero: divide it by 1 rather than 0 (0 · ∞ would be NaN).
      const nonzero = greater(s, 0)
      const inverse = where(nonzero, div(1, where(nonzero, s, 1)), 0)
      ga = add(ga, matmul(mul(projected, reshape(inverse, [1, n])), transpose(V)))
    }
    return [ga]
  },
  jvp: ([t], [a], out, p) => {
    if (t === null) return null
    refuseUnconverged(p)
    const [m, n] = shapeOfValue(a)
    const [U, s, V] = unpack(out, [[m, n], [n], [n, n]])
    const S = diag(s)
    const P = matmul(matmul(transpose(U), t), V)
    const ps = concreteExamples(P)
    const invariant = ps
      ? (i: number, j: number, b: number) => {
          const pc = exampleOf(ps, b)
          return pc !== undefined && negligible(pc[i * n + j] + pc[j * n + i], scaleOf(pc))
        }
      : null
    const F = fMatrix(square(s), 'svd', invariant)
    let dU: Value = matmul(U, mul(F, add(matmul(P, S), matmul(S, transpose(P)))))
    const dV = matmul(V, mul(F, add(matmul(S, P), matmul(transpose(P), S))))
    if (m > n) {
      refuseZeroSingular(concreteExamples(s), m, n)
      const AV = matmul(t, V)
      const projected = sub(AV, matmul(U, matmul(transpose(U), AV)))
      dU = add(dU, mul(projected, reshape(div(1, s), [1, n])))
    }
    return pack([dU, diagonal(P), dV])
  },
  batch: kernelBatch('numerics/linalg/svd'),
  shape: ([a]) => {
    const [m, n] = a.shape
    return float64Aval([m * n + n + n * n])
  },
  doc: { note: 'singular-value-decomposition', summary: 'The thin singular value decomposition of a tall matrix.' },
  test: {
    rtol: 1e-4,
    cases: (draw) => [{ inputs: [draw([4, 3])], params: { maxSweeps: 60, signOn: 'V' } }],
  },
})

/** The tall SVD of a raw input, per the primitive's parameters. */
function decomposeTall(a: Value, p: Params): SVD {
  const { m, n, a: data } = dense(a, 'svd')
  return tall(data, m, n, p.maxSweeps, p.signOn)
}

/**
 * Thin singular value decomposition A = U diag(S) Vᵀ of an m×n matrix: U is m×k, S has length k and V is n×k, with
 * k = min(m, n) and S descending. Left singular vectors of zero singular values are completed to an orthonormal set.
 * Differentiable in both modes (Townsend, 2016): repeated singular values are handled as `eigh`'s repeated eigenvalues
 * (`NumericalError` 'degenerate' unless the function is invariant), and the vectors of a zero singular value of a
 * non-square matrix have no derivative. Inside `vmap`, `sweeps` is NaN and an example that does not converge throws.
 */
export function svd<X extends Value>(a: X, { maxSweeps = 60 }: { maxSweeps?: number } = {}): SVD<TensorResult<X>> {
  const [m, n] = shapeOfValue(a)
  if (!isTraced(a)) {
    const { a: data } = dense(a, 'svd')
    if (m >= n) return tall(data, m, n, maxSweeps, 'V') as SVD<TensorResult<X>>
    // A wide matrix: decompose Aᵀ = V diag(S) Uᵀ, signed on Aᵀ's left vectors (our V), and swap the factors.
    const t = new Float64Array(n * m)
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) t[j * m + i] = data[i * n + j]
    const r = tall(t, n, m, maxSweeps, 'U')
    return { U: r.V, S: r.S, V: r.U, sweeps: r.sweeps, converged: r.converged } as SVD<TensorResult<X>>
  }
  const wide = m < n
  const input: Value = wide ? transpose(a) : a
  const [rows, cols] = wide ? [n, m] : [m, n]
  const params: Params = { maxSweeps, signOn: wide ? 'U' : 'V' }
  const found = concrete(input) === null ? undefined : decomposeTall(input, params)
  const [U, S, V] = unpack(svdOp([input], { ...params, found }), [[rows, cols], [cols], [cols, cols]])
  const r = { S, sweeps: found?.sweeps ?? NaN, converged: found?.converged ?? true }
  return (wide ? { ...r, U: V, V: U } : { ...r, U, V }) as SVD<TensorResult<X>>
}

/** The default relative cutoff for treating a singular value as zero: max(m, n)·ε, as NumPy's `matrix_rank`. */
function defaultRtol(m: number, n: number): number {
  return Math.max(m, n) * EPS
}

/** The dense routines below read the SVD's values; under a transformation they say so instead of failing inside. */
function concreteOnly(a: unknown, where: string): void {
  if (isTraced(a as Value))
    throw new NotDifferentiableError(
      where,
      `${where}: not differentiable (it thresholds singular values); differentiate through \`svd\` or \`solve\``,
    )
}

/**
 * Moore–Penrose pseudo-inverse (n×m) of an m×n matrix from its SVD: singular values at most `rtol`·σ_max are
 * treated as zero (default max(m, n)·ε).
 */
export function pinv(a: Tensor, { rtol }: { rtol?: number } = {}): Tensor {
  concreteOnly(a, 'pinv')
  const [m, n] = a.shape
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const out = new Float64Array(n * m)
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    for (let i = 0; i < n; i++) {
      const vi = V.data[i * k + c] / s
      for (let j = 0; j < m; j++) out[i * m + j] += vi * U.data[j * k + c]
    }
  }
  return matrix(out, n, m)
}

/** The result of `lstsq`. */
export type LeastSquares = {
  /** The minimum-norm least-squares solution (n, or n×r for a matrix right-hand side). */
  x: Tensor
  /** Squared residual norm ‖Ax − b‖² (per column for a matrix right-hand side). */
  residuals: Tensor
  /** Numerical rank: the number of singular values above the cutoff. */
  rank: number
  /** Singular values of A, descending. */
  singularValues: Tensor
}

/**
 * Least squares: the x minimising ‖Ax − b‖, and among those the one of least norm, from the SVD of A (m×n) with
 * singular values at most `rtol`·σ_max treated as zero (default max(m, n)·ε). Rank deficiency is reported by `rank`.
 */
export function lstsq(a: Tensor, b: Tensor, { rtol }: { rtol?: number } = {}): LeastSquares {
  concreteOnly(a, 'lstsq')
  concreteOnly(b, 'lstsq')
  const [m, n] = a.shape
  const isVector = b.shape.length === 1
  const rhs = dense(isVector ? reshape(b, [-1, 1]) : b, 'lstsq')
  if (rhs.m !== m) throw new ShapeError('lstsq', `lstsq: A has ${m} rows but b has ${rhs.m}`)
  const r = rhs.n
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const x = new Float64Array(n * r)
  let rank = 0
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    rank++
    for (let col = 0; col < r; col++) {
      let d = 0
      for (let j = 0; j < m; j++) d += U.data[j * k + c] * rhs.a[j * r + col]
      d /= s
      for (let i = 0; i < n; i++) x[i * r + col] += V.data[i * k + c] * d
    }
  }
  const residuals = new Float64Array(r)
  const A = dense(a, 'lstsq')
  for (let col = 0; col < r; col++) {
    for (let j = 0; j < m; j++) {
      let e = -rhs.a[j * r + col]
      for (let i = 0; i < n; i++) e += A.a[j * n + i] * x[i * r + col]
      residuals[col] += e * e
    }
  }
  return {
    x: isVector ? vector(x) : matrix(x, n, r),
    residuals: vector(residuals),
    rank,
    singularValues: S,
  }
}

/** The 2-norm condition number σ_max / σ_min (∞ when σ_min = 0) of an m×n matrix, over its min(m, n) singular values. */
export function conditionNumber(a: Tensor): number {
  concreteOnly(a, 'conditionNumber')
  const { S } = svd(a)
  const k = S.shape[0]
  if (k === 0) return 0
  const smin = S.data[k - 1]
  return smin === 0 ? Infinity : S.data[0] / smin
}
