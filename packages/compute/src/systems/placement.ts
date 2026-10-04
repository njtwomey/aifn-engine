/**
 * Multi-input pole placement by the robust eigenstructure assignment of Kautsky, Nichols & Van Dooren (1985, "Robust
 * pole assignment in linear state feedback", Int. J. Control 41(5)), method 0, as scipy's
 * `place_poles(method='KNV0')`, extended to complex-conjugate poles.
 */

import { eig, eigh, factorDense, lstsq, qr, solveFactored, svd } from 'aifn-compute/numerics/linalg'
import { complexVector, type ComplexLike } from 'aifn-compute/numerics/polynomial'
import { dense, fromData, imagPart, realPart, toFlat, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

/** Options for `placePoles`. */
export type PlacePolesOptions = {
  /** Stop when |det X| changes by a relative amount below this between sweeps. Default 1e-3 (scipy's). */
  rtol?: Scalar
  /** Most sweeps over the eigenvector columns. Default 30. */
  maxIterations?: Size
}

/** The result of `placePoles`. */
export type PolePlacementResult = {
  /** The gain K (m × n): u = −Kx gives eig(A − BK) = the requested poles. */
  K: Matrix
  /** The requested poles in the solver's order (real ones ascending, then each conjugate pair, negative part first). */
  requested: Tensor
  /** The eigenvalues of A − BK actually obtained, complex128, sorted as `eig`. */
  closedLoop: Tensor
  /**
   * The closed-loop eigenvector matrix X (n × n, complex128, unit columns, column j for `requested[j]`). The method
   * maximises |det X|, i.e. makes X as well conditioned as it can, so the placed poles are insensitive to perturbation.
   */
  X: Tensor
  /** The condition number of X (κ₂); the bound on how far a perturbation of A − BK moves the poles. */
  conditioning: Scalar
  /** Sweeps taken (0 when B has full row rank or rank 1: nothing to optimise). */
  iterations: Size
  /** The last relative change of |det X| (NaN when there was nothing to optimise). */
  change: Scalar
  /** True when the change fell below `rtol` (or there was nothing to optimise). */
  converged: boolean
}

type Pole = { re: number; im: number }

/** Real poles ascending, then conjugate pairs (negative imaginary part first, by real part), as scipy. */
function orderPoles(poles: Pole[]): Pole[] {
  const size = Math.max(1, ...poles.map((p) => Math.hypot(p.re, p.im)))
  const isReal = (p: Pole) => Math.abs(p.im) <= 1e-12 * size
  const real = poles.filter(isReal).map((p) => ({ re: p.re, im: 0 }))
  real.sort((a, b) => a.re - b.re)
  const lower = poles.filter((p) => !isReal(p) && p.im < 0).sort((a, b) => a.re - b.re || a.im - b.im)
  const upper = poles.filter((p) => !isReal(p) && p.im > 0)
  const out = [...real]
  for (const p of lower) {
    const k = upper.findIndex((q) => Math.abs(q.re - p.re) <= 1e-9 * size && Math.abs(q.im + p.im) <= 1e-9 * size)
    if (k < 0) throw new DomainError('placePoles', 'placePoles: complex poles must come with their conjugates')
    upper.splice(k, 1)
    out.push(p, { re: p.re, im: -p.im })
  }
  if (upper.length) throw new DomainError('placePoles', 'placePoles: complex poles must come with their conjugates')
  return out
}

/** A full QR's Q (m × m, row-major). */
function fullQ(a: F64, m: number, n: number): F64 {
  return dense.data(qr(fromData(a, [m, n]), { mode: 'complete' }).Q)
}

/** Columns [from, to) of a row-major r × c matrix. */
function columns(a: F64, r: number, c: number, from: number, to: number): F64 {
  const w = to - from
  const out = new Float64Array(r * w)
  for (let i = 0; i < r; i++) for (let j = 0; j < w; j++) out[i * w + j] = a[i * c + from + j]
  return out
}

/** log |det X| of a row-major square matrix (−∞ when singular). */
function logAbsDet(x: F64, n: number): number {
  return factorDense(x, n).logAbsDet
}

/**
 * State-feedback gain K with eig(A − BK) equal to the requested poles, for a controllable pair (A, B) with m ≥ 1
 * inputs. With several inputs the gain is not unique: the freedom is in the closed-loop eigenvectors, and the
 * Kautsky–Nichols–Van Dooren method 0 chooses them to make the eigenvector matrix X as well conditioned as possible.
 *
 * B = [U₀ U₁][Z; 0] (QR, rank r). Each eigenvector x_j must lie in S_j = ker U₁ᵀ(A − λ_j I) (an r-dimensional space for a
 * controllable pair), because then A − BK = XΛX⁻¹ is solvable for K: K = −Z⁻¹U₀ᵀ(XΛX⁻¹ − A). Method 0 sweeps over the
 * columns, replacing x_j by the unit vector of S_j closest to orthogonal to the other columns: the projection onto
 * S_j of the normal to their span. Each such replacement can only increase |det X| (for unit columns, |det X| ≤ 1 with
 * equality for orthonormal X), and the sweeps stop when |det X| changes by less than `rtol` relative.
 *
 * Real poles follow scipy's KNV0 step for step. A complex pair λ, λ̄ (scipy's KNV0 rejects them) is held as the real
 * columns [Re x, Im x], with S_j realified to {(xᵣ, xᵢ) : U₁ᵀ[(A − aI)xᵣ + b xᵢ] = 0, U₁ᵀ[(A − aI)xᵢ − b xᵣ] = 0} for
 * λ = a + ib; the update picks the unit (xᵣ, xᵢ) in it that maximises |det X| with the other n − 2 columns fixed (an
 * extreme eigenvector of a 2r × 2r symmetric form), the pair analogue of the real update. When B has full row rank, K is found by least squares and X = I; with one input K is
 * unique (as Ackermann's formula) and no sweep is needed.
 */
export function placePoles(
  plant: { A: MatrixLike; B: MatrixLike },
  poles: ComplexLike,
  { rtol = 1e-3, maxIterations = 30 }: PlacePolesOptions = {},
): PolePlacementResult {
  const { data: A, m: n, n: nA } = dense.toMatrixF64(plant.A, 'placePoles A')
  if (n !== nA) throw new DomainError('placePoles', 'placePoles: A must be square')
  const { data: B, n: m } = dense.toMatrixF64(plant.B, 'placePoles B', n)
  if (!(rtol <= 1)) throw new DomainError('placePoles', 'placePoles: rtol cannot exceed 1')
  if (!(maxIterations >= 1)) throw new DomainError('placePoles', 'placePoles: maxIterations must be at least 1')
  const pv = complexVector(poles, 'placePoles')
  const re = toFlat(realPart(pv))
  const im = toFlat(imagPart(pv))
  const P = orderPoles(Array.from(re, (r, i) => ({ re: r, im: im[i] })))
  if (P.length !== n) throw new DomainError('placePoles', `placePoles: need ${n} poles, got ${P.length}`)
  const sB = toFlat(svd(fromData(B, [n, m])).S)
  const rank = sB.filter((s) => s > Math.max(n, m) * Number.EPSILON * (sB[0] ?? 0)).length
  for (const p of P) {
    const same = P.filter((q) => q.re === p.re && q.im === p.im).length
    if (same > rank)
      throw new DomainError('placePoles', 'placePoles: a requested pole is repeated more than rank(B) times')
  }

  // B = U [Z; 0]: U₀ = U[:, :r], U₁ = U[:, r:], Z = R[:r, :].
  const Qb = qr(fromData(B, [n, m]), { mode: 'complete' })
  const U = dense.data(Qb.Q)
  const z = dense.data(Qb.R).slice(0, rank * m) // Z = R[:r, :] (r × m)
  const U0 = columns(U, n, n, 0, rank)
  const U1 = columns(U, n, n, rank, n)
  const nr = n - rank

  let K: F64
  let X = dense.identity(n)
  let iterations = 0
  let change = NaN
  let converged = true
  // The real block form Λ: [[a, b], [−b, a]] for a pair a ± ib (A[xᵣ xᵢ] = [xᵣ xᵢ]Λ for λ = a + ib).
  const Lambda = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const p = P[j]
    if (p.im === 0) {
      Lambda[j * n + j] = p.re
      continue
    }
    Lambda[j * n + j] = p.re
    Lambda[(j + 1) * n + j + 1] = p.re
    Lambda[j * n + j + 1] = p.im
    Lambda[(j + 1) * n + j] = -p.im
    j++
  }

  if (rank === n) {
    // B has full row rank: solve B K' = Λ' − A in the least-squares sense (scipy's diag_poles block form).
    const D = new Float64Array(n * n)
    for (let j = 0; j < n; j++) {
      const p = P[j]
      D[j * n + j] = p.re
      if (p.im !== 0) {
        D[j * n + j + 1] = -p.im
        D[(j + 1) * n + j + 1] = p.re
        D[(j + 1) * n + j] = p.im
        j++
      }
    }
    K = dense.data(lstsq(fromData(B, [n, m]), fromData(dense.sub(D, A), [n, n])).x)
  } else {
    // The kernel bases S_j (n × r real, or 2n × 2r realified for a pair) and the starting columns of X.
    const kernels: { basis: F64; rows: number; cols: number; pair: boolean }[] = []
    for (let j = 0; j < n; j++) {
      const p = P[j]
      if (p.im === 0) {
        // (U₁ᵀ(A − pI))ᵀ: n × (n − r); the last r columns of its full Q span the kernel.
        const M = dense.matMul(dense.transpose(U1, n, nr), dense.sub(A, dense.scale(p.re, dense.identity(n))), nr, n, n)
        const Q = fullQ(dense.transpose(M, nr, n), n, nr)
        const basis = columns(Q, n, n, nr, n)
        kernels.push({ basis, rows: n, cols: rank, pair: false })
        let x = new Float64Array(n)
        for (let i = 0; i < n; i++) for (let k = 0; k < rank; k++) x[i] += basis[i * rank + k]
        x = dense.scale(1 / dense.norm(x), x)
        for (let i = 0; i < n; i++) X[i * n + j] = x[i]
        continue
      }
      const a = p.re
      const b = p.im
      const Mc = dense.matMul(dense.transpose(U1, n, nr), dense.sub(A, dense.scale(a, dense.identity(n))), nr, n, n)
      const U1t = dense.transpose(U1, n, nr)
      const G = new Float64Array(2 * nr * 2 * n)
      for (let i = 0; i < nr; i++)
        for (let k = 0; k < n; k++) {
          G[i * 2 * n + k] = Mc[i * n + k]
          G[i * 2 * n + n + k] = b * U1t[i * n + k]
          G[(nr + i) * 2 * n + k] = -b * U1t[i * n + k]
          G[(nr + i) * 2 * n + n + k] = Mc[i * n + k]
        }
      const Q = fullQ(dense.transpose(G, 2 * nr, 2 * n), 2 * n, 2 * nr)
      const basis = columns(Q, 2 * n, 2 * n, 2 * nr, 2 * n)
      kernels.push(
        { basis, rows: 2 * n, cols: 2 * rank, pair: true },
        { basis, rows: 2 * n, cols: 2 * rank, pair: true },
      )
      const v = new Float64Array(2 * n)
      for (let i = 0; i < 2 * n; i++) for (let k = 0; k < 2 * rank; k++) v[i] += basis[i * 2 * rank + k]
      const s = 1 / dense.norm(v)
      for (let i = 0; i < n; i++) {
        X[i * n + j] = v[i] * s
        X[i * n + j + 1] = v[n + i] * s
      }
      j++
    }

    if (rank > 1) {
      const floor = Math.sqrt(Number.EPSILON)
      converged = false
      while (iterations < maxIterations && !converged) {
        const before = Math.exp(logAbsDet(X, n))
        for (let j = 0; j < n; j++) {
          const ker = kernels[j]
          if (!ker.pair) {
            // The normal to the other columns' span: the last column of the full Q of X without column j.
            const others = new Float64Array(n * (n - 1))
            for (let i = 0; i < n; i++)
              for (let k = 0, c = 0; k < n; k++) if (k !== j) others[i * (n - 1) + c++] = X[i * n + k]
            const Q = fullQ(others, n, n - 1)
            const q = Float64Array.from({ length: n }, (_, i) => Q[i * n + n - 1])
            // y = S Sᵀ q, the projection onto S_j.
            const c = dense.matTVec(ker.basis, q, n, rank)
            const y = dense.matVec(ker.basis, c, n, rank)
            // scipy keeps the column when y ≈ 0 (np.allclose(y, 0): atol 1e-8).
            if (y.some((v) => Math.abs(v) > 1e-8)) {
              const s = 1 / dense.norm(y)
              for (let i = 0; i < n; i++) X[i * n + j] = y[i] * s
            }
            continue
          }
          // A conjugate pair (columns j, j + 1) holds x = xᵣ + i xᵢ with ‖xᵣ‖² + ‖xᵢ‖² = 1. With the other n − 2 columns
          // fixed, |det X| is proportional to |det [Cᵀxᵣ, Cᵀxᵢ]|, C an orthonormal basis of their complement. For
          // x = S v that determinant is the quadratic form vᵀHv, so the best unit v is H's eigenvector of largest |λ|.
          const others = new Float64Array(n * (n - 2))
          for (let i = 0; i < n; i++)
            for (let k = 0, c = 0; k < n; k++) if (k !== j && k !== j + 1) others[i * (n - 2) + c++] = X[i * n + k]
          const Q = n > 2 ? fullQ(others, n, n - 2) : dense.identity(n)
          const C = columns(Q, n, n, n - 2, n)
          const w = ker.cols
          const Ct = dense.transpose(C, n, 2)
          const R = dense.matMul(Ct, ker.basis.subarray(0, n * w), 2, n, w) // rows: c₁ᵣ, c₂ᵣ (as functions of v)
          const I = dense.matMul(Ct, ker.basis.subarray(n * w, 2 * n * w), 2, n, w) // rows: c₁ᵢ, c₂ᵢ
          // det = c₁ᵣc₂ᵢ − c₁ᵢc₂ᵣ = vᵀ(r₁ᵀi₂ − i₁ᵀr₂)v, symmetrised.
          const H = new Float64Array(w * w)
          for (let a = 0; a < w; a++)
            for (let b = 0; b < w; b++) {
              const ab = R[a] * I[w + b] - I[a] * R[w + b]
              const ba = R[b] * I[w + a] - I[b] * R[w + a]
              H[a * w + b] = 0.5 * (ab + ba)
            }
          const top = eigh(fromData(H, [w, w]))
          const lam = dense.data(top.values)
          const col = Math.abs(lam[0]) >= Math.abs(lam[w - 1]) ? 0 : w - 1
          const coef = Float64Array.from({ length: w }, (_, k) => dense.data(top.vectors)[k * w + col])
          const v = dense.matVec(ker.basis, coef, 2 * n, w)
          const s = 1 / dense.norm(v)
          for (let i = 0; i < n; i++) {
            X[i * n + j] = v[i] * s
            X[i * n + j + 1] = v[n + i] * s
          }
          j++
        }
        const after = Math.max(floor, Math.exp(logAbsDet(X, n)))
        change = Math.abs((after - before) / after)
        if (change < rtol && after > floor) converged = true
        iterations++
      }
    }

    // M = X Λ X⁻¹ (real: X holds [Re x, Im x] for each pair), then Z K' = U₀ᵀ(M − A) and K = −K'.
    const XL = dense.matMul(X, Lambda, n, n, n)
    const fx = factorDense(dense.transpose(X, n, n), n)
    const Mt = solveFactored(fx, dense.transpose(XL, n, n))
    if (Mt === null)
      throw new DomainError(
        'placePoles',
        'placePoles: the eigenvector matrix is singular; the poles cannot be placed (check controllability)',
      )
    const M = dense.transpose(Mt, n, n)
    const rhs = dense.matMul(dense.transpose(U0, n, rank), dense.sub(M, A), rank, n, n)
    const fz = factorDense(z, rank)
    const Kp = rank === m ? solveFactored(fz, rhs) : null
    if (Kp === null)
      throw new DomainError('placePoles', 'placePoles: B is rank deficient in its inputs; the gain is not determined')
    K = Kp
  }
  K = dense.scale(-1, K)
  const Acl = dense.sub(A, dense.matMul(B, K, n, m, n))
  const closedLoop = eig(fromData(Acl, [n, n]), { vectors: false }).values
  // The complex eigenvectors: x = xᵣ − i xᵢ for the pole with negative imaginary part and its conjugate, as scipy.
  const Xc = new Float64Array(2 * n * n)
  for (let j = 0; j < n; j++) {
    if (P[j].im === 0) {
      for (let i = 0; i < n; i++) Xc[2 * (i * n + j)] = X[i * n + j]
      continue
    }
    for (let i = 0; i < n; i++) {
      const r = X[i * n + j]
      const g = X[i * n + j + 1]
      Xc[2 * (i * n + j)] = r
      Xc[2 * (i * n + j) + 1] = -g
      Xc[2 * (i * n + j + 1)] = r
      Xc[2 * (i * n + j + 1) + 1] = g
    }
    j++
  }
  // κ₂ of the complex X, from its real form [[Xᵣ, −Xᵢ], [Xᵢ, Xᵣ]] (2n × 2n, the same singular values, each twice).
  const R2 = new Float64Array(4 * n * n)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const xr = Xc[2 * (i * n + k)]
      const xi = Xc[2 * (i * n + k) + 1]
      R2[i * 2 * n + k] = xr
      R2[i * 2 * n + n + k] = -xi
      R2[(n + i) * 2 * n + k] = xi
      R2[(n + i) * 2 * n + n + k] = xr
    }
  const sX = toFlat(svd(fromData(R2, [2 * n, 2 * n])).S)
  const requested = fromData(Float64Array.from(P.flatMap((p) => [p.re, p.im])), [n], 'complex128')
  return {
    K: fromData(K, [m, n]),
    requested,
    closedLoop,
    X: fromData(Xc, [n, n], 'complex128'),
    conditioning: sX[0] / sX[2 * n - 1],
    iterations,
    change,
    converged,
  }
}
