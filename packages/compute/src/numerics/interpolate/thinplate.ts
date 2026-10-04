/**
 * Thin-plate splines: the exact (radial-basis) thin-plate spline through or near scattered data in d = 1, 2 or 3
 * dimensions (Duchon, 1977; Wahba, 1990, "Spline Models for Observational Data", §2.4), and the low-rank thin-plate
 * regression spline basis of Wood (2003), "Thin plate regression splines", JRSS B 65(1), used by GAMs.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Size } from 'aifn-compute/foundation/contracts'
import { eigh, solve } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

type F64 = Float64Array
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/** Points as rows [n, d]; a vector [n] is one-dimensional. */
function points(x: Tensor): { v: F64; n: number; d: number } {
  if (x.shape.length === 1) return { v: f64(x), n: x.shape[0], d: 1 }
  if (x.shape.length === 2) return { v: f64(x), n: x.shape[0], d: x.shape[1] }
  throw new DomainError('thinplate', 'thin-plate: points must be [n] or [n, d]')
}

/**
 * The second-order thin-plate radial function η(r): r² log r in two dimensions, r³ in one and r in three (the
 * penalty ∫‖∇²f‖² needs 2m > d, so m = 2 covers d ≤ 3). `scaled` multiplies by Duchon's constants (1/(8π), 1/12 and
 * −1/(8π)), which makes the penalty equal βᵀEβ exactly (Wood, 2003, eq. 1).
 */
function eta(r: number, d: number, scaled: boolean): number {
  if (d === 2) return r === 0 ? 0 : (r * r * Math.log(r)) / (scaled ? 8 * Math.PI : 1)
  if (d === 1) return (r * r * r) / (scaled ? 12 : 1)
  if (d === 3) return scaled ? -r / (8 * Math.PI) : r
  throw new ShapeError('thinplate', `thin-plate: dimension ${d} is not supported (1, 2 or 3)`)
}

function radial(a: F64, n: number, b: F64, m: number, d: number, scaled: boolean): F64 {
  const E = new Float64Array(n * m)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0
      for (let k = 0; k < d; k++) s += (a[i * d + k] - b[j * d + k]) ** 2
      E[i * m + j] = eta(Math.sqrt(s), d, scaled)
    }
  return E
}

/** The polynomial null space: 1, x₁, …, x_d per row, [n, d + 1]. */
function affine(a: F64, n: number, d: number): F64 {
  const T = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    T[i * (d + 1)] = 1
    for (let k = 0; k < d; k++) T[i * (d + 1) + k + 1] = a[i * d + k]
  }
  return T
}

/** A fitted thin-plate spline. */
export type ThinPlateSpline = {
  readonly kind: 'thin-plate-spline'
  /** Radial weights wᵢ [n] (they satisfy Tᵀw = 0). */
  readonly weights: Tensor
  /** Coefficients of 1, x₁, …, x_d. */
  readonly polynomial: Tensor
  readonly smoothing: number
  /** f at new points [m, d] (or [m]). */
  evaluate(x: Tensor): Tensor
}

/**
 * The thin-plate spline f(x) = Σᵢ wᵢ η(‖x − xᵢ‖) + c₀ + cᵀx through (xᵢ, yᵢ), or, with `smoothing` λ > 0, the
 * smoother that solves (E + λI)w + Tc = y, Tᵀw = 0 (Wahba, 1990, §2.4). The radial function is unscaled, as
 * scipy's `RBFInterpolator(kernel='thin_plate_spline', degree=1, smoothing=λ)` in two dimensions.
 */
export function thinPlateSpline(x: Tensor, y: Tensor, { smoothing = 0 }: { smoothing?: number } = {}): ThinPlateSpline {
  const { v, n, d } = points(x)
  const Y = f64(y)
  if (Y.length !== n) throw new ShapeError('thinPlateSpline', `thinPlateSpline: ${n} points but ${Y.length} values`)
  const M = d + 1
  const E = radial(v, n, v, n, d, false)
  const T = affine(v, n, d)
  const size = n + M
  const A = new Float64Array(size * size)
  const r = new Float64Array(size)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i * size + j] = E[i * n + j] + (i === j ? smoothing : 0)
    for (let k = 0; k < M; k++) {
      A[i * size + n + k] = T[i * M + k]
      A[(n + k) * size + i] = T[i * M + k]
    }
    r[i] = Y[i]
  }
  const sol = f64(solve(fromData(A, [size, size]), fromData(r, [size])) as Tensor)
  const w = sol.slice(0, n)
  const c = sol.slice(n)
  return {
    kind: 'thin-plate-spline',
    weights: fromData(w, [n]),
    polynomial: fromData(c, [M]),
    smoothing,
    evaluate: (xs) => {
      const q = points(xs)
      if (q.d !== d) throw new ShapeError('thinPlateSpline', `thinPlateSpline: fitted in ${d} dimensions, given ${q.d}`)
      const Es = radial(q.v, q.n, v, n, d, false)
      const Ts = affine(q.v, q.n, d)
      const out = Float64Array.from({ length: q.n }, (_, i) => {
        let s = 0
        for (let j = 0; j < n; j++) s += Es[i * n + j] * w[j]
        for (let k = 0; k < M; k++) s += Ts[i * M + k] * c[k]
        return s
      })
      return fromData(out, [q.n])
    },
  }
}

/** A low-rank thin-plate regression spline basis. */
export type ThinPlateRegressionBasis = {
  readonly kind: 'thin-plate-regression-basis'
  /** The design at the data [n, k]: k − M wiggly columns, then the M = d + 1 polynomial columns. */
  readonly design: Tensor
  /** The penalty [k, k] (zero on the polynomial columns), so that βᵀSβ = ∫‖∇²f‖². */
  readonly penalty: Tensor
  /** Dimension M of the unpenalised polynomial space. */
  readonly nullSpace: number
  /** The eigenvalues of E kept, largest magnitude first. */
  readonly eigenvalues: Tensor
  /** The basis at new points [m, d] → [m, k]. */
  evaluate(x: Tensor): Tensor
}

/**
 * Wood's (2003) thin-plate regression spline basis of rank k: the k eigenvectors of E = [η(‖xᵢ − xⱼ‖)] with the
 * largest-magnitude eigenvalues, constrained to be orthogonal to the polynomials (Tᵀδ = 0) by an orthonormal null-space
 * basis Z, plus the polynomials. The truncation is optimal in the sense of perturbing the full thin-plate spline least.
 * The eigendecomposition is dense (O(n³)), so use at most a few hundred points.
 */
export function thinPlateRegressionBasis(x: Tensor, rank: Size): ThinPlateRegressionBasis {
  const { v, n, d } = points(x)
  const M = d + 1
  if (!(rank > M && rank <= n))
    throw new ShapeError('thinPlateRegressionBasis', `thinPlateRegressionBasis: rank must be in (${M}, ${n}]`)
  const E = radial(v, n, v, n, d, true)
  const { values, vectors } = eigh(fromData(E, [n, n]))
  const lam = f64(values)
  const V = f64(vectors)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => Math.abs(lam[b]) - Math.abs(lam[a]))
  const keep = order.slice(0, rank)
  const k = rank
  const U = new Float64Array(n * k)
  for (let i = 0; i < n; i++) keep.forEach((c, j) => (U[i * k + j] = V[i * n + c]))
  const D = Float64Array.from(keep, (c) => lam[c])
  const T = affine(v, n, d)
  // C = U_kᵀT [k, M]; Z spans the null space of Cᵀ (the eigenvectors of CCᵀ with zero eigenvalues).
  const C = new Float64Array(k * M)
  for (let a = 0; a < k; a++)
    for (let b = 0; b < M; b++) for (let i = 0; i < n; i++) C[a * M + b] += U[i * k + a] * T[i * M + b]
  const CCt = new Float64Array(k * k)
  for (let a = 0; a < k; a++)
    for (let b = 0; b < k; b++) for (let c = 0; c < M; c++) CCt[a * k + b] += C[a * M + c] * C[b * M + c]
  const ec = eigh(fromData(CCt, [k, k]))
  const Vc = f64(ec.vectors)
  const q = k - M
  const Z = new Float64Array(k * q)
  for (let a = 0; a < k; a++) for (let j = 0; j < q; j++) Z[a * q + j] = Vc[a * k + M + j]
  // UZ = U_k Z [n, q] maps the constrained δ to data space; the wiggly design is E U_k Z = U_k D_k Z.
  const UZ = new Float64Array(n * q)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < q; j++) for (let a = 0; a < k; a++) UZ[i * q + j] += U[i * k + a] * Z[a * q + j]
  const design = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < q; j++) {
      let s = 0
      for (let a = 0; a < k; a++) s += U[i * k + a] * D[a] * Z[a * q + j]
      design[i * k + j] = s
    }
    for (let b = 0; b < M; b++) design[i * k + q + b] = T[i * M + b]
  }
  const S = new Float64Array(k * k)
  for (let a = 0; a < q; a++)
    for (let b = 0; b < q; b++) for (let c = 0; c < k; c++) S[a * k + b] += Z[c * q + a] * D[c] * Z[c * q + b]
  return {
    kind: 'thin-plate-regression-basis',
    design: fromData(design, [n, k]),
    penalty: fromData(S, [k, k]),
    nullSpace: M,
    eigenvalues: fromData(D, [k]),
    evaluate: (xs) => {
      const p = points(xs)
      if (p.d !== d)
        throw new ShapeError(
          'thinPlateRegressionBasis',
          `thinPlateRegressionBasis: built in ${d} dimensions, given ${p.d}`,
        )
      const Es = radial(p.v, p.n, v, n, d, true)
      const Ts = affine(p.v, p.n, d)
      const out = new Float64Array(p.n * k)
      for (let i = 0; i < p.n; i++) {
        for (let j = 0; j < q; j++) {
          let s = 0
          for (let a = 0; a < n; a++) s += Es[i * n + a] * UZ[a * q + j]
          out[i * k + j] = s
        }
        for (let b = 0; b < M; b++) out[i * k + q + b] = Ts[i * M + b]
      }
      return fromData(out, [p.n, k])
    },
  }
}
