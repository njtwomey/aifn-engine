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

/**
 * Extract a flat Float64Array view from a tensor.
 *
 * @param t Input tensor.
 * @returns Flattened 64-bit float array.
 */
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/**
 * Parse points tensor as flattened Float64Array, returning point count $n$ and dimension $d$.
 *
 * @param x 1D vector $[n]$ or 2D matrix $[n, d]$.
 * @returns Object with data array `v`, point count `n`, and spatial dimension `d`.
 */
function points(x: Tensor): { v: F64; n: number; d: number } {
  if (x.shape.length === 1) return { v: f64(x), n: x.shape[0], d: 1 }
  if (x.shape.length === 2) return { v: f64(x), n: x.shape[0], d: x.shape[1] }
  throw new DomainError('thinplate', 'thin-plate: points must be [n] or [n, d]')
}

/**
 * Second-order thin-plate radial basis function $\eta(r)$.
 *
 * In $d = 2$, $\eta(r) = r^2 \log r$; in $d = 1$, $\eta(r) = r^3$; in $d = 3$, $\eta(r) = r$.
 * When `scaled` is true, multiplies by Duchon's normalization constants so that the bending
 * penalty exactly equals $\boldsymbol{\beta}^\top \mathbf{E} \boldsymbol{\beta}$ (Wood, 2003, eq. 1).
 *
 * @param r Euclidean distance between two points ($r \ge 0$).
 * @param d Spatial dimensionality ($d \in \{1, 2, 3\}$).
 * @param scaled Whether to apply Duchon's penalty normalization scaling.
 * @returns Radial basis function value $\eta(r)$.
 */
function eta(r: number, d: number, scaled: boolean): number {
  if (d === 2) return r === 0 ? 0 : (r * r * Math.log(r)) / (scaled ? 8 * Math.PI : 1)
  if (d === 1) return (r * r * r) / (scaled ? 12 : 1)
  if (d === 3) return scaled ? -r / (8 * Math.PI) : r
  throw new ShapeError('thinplate', `thin-plate: dimension ${d} is not supported (1, 2 or 3)`)
}

/**
 * Compute the pairwise thin-plate radial kernel matrix $E_{ij} = \eta(\|a_i - b_j\|)$.
 *
 * @param a First coordinate set as flattened Float64Array.
 * @param n Number of points in first set.
 * @param b Second coordinate set as flattened Float64Array.
 * @param m Number of points in second set.
 * @param d Spatial dimensionality ($d \in \{1, 2, 3\}$).
 * @param scaled Whether to scale radial evaluations by Duchon's constants.
 * @returns Flattened pairwise kernel matrix of shape $[n, m]$.
 */
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

/**
 * Polynomial null-space matrix: row $i$ contains $[1, x_{i,1}, \dots, x_{i,d}]$ of shape $[n, d + 1]$.
 *
 * @param a Coordinate data as flattened Float64Array.
 * @param n Number of data points.
 * @param d Spatial dimensionality.
 * @returns Flattened design matrix of shape $[n, d + 1]$.
 */
function affine(a: F64, n: number, d: number): F64 {
  const T = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    T[i * (d + 1)] = 1
    for (let k = 0; k < d; k++) T[i * (d + 1) + k + 1] = a[i * d + k]
  }
  return T
}

/** A fitted thin-plate spline model. */
export type ThinPlateSpline = {
  /** Discriminator kind tag. */
  readonly kind: 'thin-plate-spline'
  /** Radial weights $w_i$, shape $[n]$, satisfying $\mathbf{T}^\top \mathbf{w} = \mathbf{0}$. */
  readonly weights: Tensor
  /** Affine polynomial coefficients for $[1, x_1, \dots, x_d]$, shape $[d + 1]$. */
  readonly polynomial: Tensor
  /** Smoothing penalty parameter $\lambda \ge 0$. */
  readonly smoothing: number
  /**
   * Evaluate the fitted thin-plate spline at query points.
   *
   * @param x Query coordinate tensor of shape $[m, d]$ or $[m]$.
   * @returns Predicted values tensor of shape $[m]$.
   */
  evaluate(x: Tensor): Tensor
}

/**
 * Fit a thin-plate spline $f(\xvec) = \sum_i w_i \eta(\|\xvec - \xvec_i\|) + c_0 + \mathbf{c}^\top \xvec$ through data.
 *
 * For smoothing parameter $\lambda = 0$, exact interpolation is performed. When $\lambda > 0$, solves the regularised
 * system $(\mathbf{E} + \lambda \mathbf{I})\mathbf{w} + \mathbf{T}\mathbf{c} = \mathbf{y}$ subject to $\mathbf{T}^\top \mathbf{w} = \mathbf{0}$
 * (Wahba, 1990, §2.4). Matches SciPy's `RBFInterpolator(kernel='thin_plate_spline', degree=1, smoothing=λ)`.
 *
 * @param x Observed point coordinates tensor of shape $[n, d]$ (or $[n]$ for 1D).
 * @param y Observed values vector of length $n$.
 * @param options Fitting options.
 * @param options.smoothing Ridge smoothing parameter $\lambda \ge 0$ (default 0).
 * @returns A fitted `ThinPlateSpline` object supporting out-of-sample evaluation.
 *
 * @example Fit 2D thin-plate spline
 * const pts = tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
 * const vals = tensor([0, 1, 1, 2])
 * const tps = thinPlateSpline(pts, vals)
 * const pred = tps.evaluate(tensor([[0.5, 0.5]]))
 * print('fitted value =', pred)
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

/** A low-rank thin-plate regression spline basis (Wood, 2003). */
export type ThinPlateRegressionBasis = {
  /** Discriminator kind tag. */
  readonly kind: 'thin-plate-regression-basis'
  /** The design matrix at the data points, shape $[n, k]$: $k - M$ wiggly columns followed by $M = d + 1$ polynomial columns. */
  readonly design: Tensor
  /** The penalty matrix $\mathbf{S}$ of shape $[k, k]$ (zero on polynomial columns) such that $\boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} = \int \|\nabla^2 f\|^2$. */
  readonly penalty: Tensor
  /** Dimension $M = d + 1$ of the unpenalised polynomial null space. */
  readonly nullSpace: number
  /** Retained eigenvalues of kernel matrix $\mathbf{E}$, sorted by descending magnitude. */
  readonly eigenvalues: Tensor
  /**
   * Evaluate the basis functions at new query coordinates.
   *
   * @param x Query coordinate tensor of shape $[m, d]$ (or $[m]$ for 1D).
   * @returns Basis evaluation matrix of shape $[m, k]$.
   */
  evaluate(x: Tensor): Tensor
}

/**
 * Construct Wood's (2003) low-rank thin-plate regression spline basis of rank $k$.
 *
 * Computes the $k$ eigenvectors of kernel matrix $E_{ij} = \eta(\|\xvec_i - \xvec_j\|)$ with largest eigenvalue
 * magnitudes, constrained to be orthogonal to polynomials ($\mathbf{T}^\top \boldsymbol{\delta} = \mathbf{0}$) via an
 * orthonormal null-space basis $\mathbf{Z}$, plus the polynomial basis. Truncation optimally perturbs the full spline.
 *
 * @param x Observed point coordinates tensor of shape $[n, d]$ (or $[n]$ for 1D).
 * @param rank Basis rank $k$ satisfying $d + 1 < k \le n$.
 * @returns A `ThinPlateRegressionBasis` object with design matrix, penalty matrix, and evaluator.
 *
 * @example Build thin-plate regression basis
 * const pts = tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
 * const basis = thinPlateRegressionBasis(pts, 4)
 * print('design shape =', basis.design.shape)
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
