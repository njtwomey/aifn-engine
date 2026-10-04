/**
 * One-class kernel methods: the one-class support vector machine (Schölkopf et al., 2001), which separates the data
 * from the origin in feature space with maximum margin, and support vector data description (Tax and Duin, 2004),
 * which encloses the data in the smallest ball. Both are quadratic programmes in the dual, solved by
 * `aifn-compute/optim/programming`'s interior-point method, with a Gaussian kernel from `aifn-compute/learning/kernels`; with that
 * kernel (k(x, x) = 1) the two give the same boundary.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { quadprog } from 'aifn-compute/optim/programming'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `oneClassSvm` and `supportVectorDataDescription`. */
export type OneClassOptions = {
  /** ν ∈ (0, 1]: an upper bound on the share of training points outside the boundary (default 0.1). */
  nu?: number
  /** The Gaussian kernel exp(−γ‖x − y‖²), with γ as scikit-learn's (default 1/(d · variance of X), its 'scale'). */
  gamma?: number
}

/** A fitted one-class model: support points, their dual weights, the kernel's γ and the offset. */
export type OneClassModel = {
  kind: 'one-class-svm' | 'svdd'
  support: Tensor
  /** Dual weights α of the support points (they sum to 1). */
  alpha: Float64Array
  gamma: number
  /** OCSVM: the offset ρ of f(x) = Σ αᵢ k(xᵢ, x) − ρ. SVDD: the squared radius R². */
  offset: number
  /** SVDD: αᵀKα, the squared norm of the centre. */
  centreNorm: number
  /** The share of training points with α at its upper bound (outside the boundary). */
  boundShare: number
}

const kernelOf = (gamma: number) => rbf({ lengthscale: 1 / Math.sqrt(2 * gamma) })

function setup(x: MatrixLike, options: OneClassOptions, where: string) {
  const m = dense.toMatrixF64(x, where)
  const X = fromData(Float64Array.from(m.data), [m.m, m.n])
  const { nu = 0.1 } = options
  if (!(nu > 0 && nu <= 1)) throw new DomainError(where, `${where}: ν must lie in (0, 1], got ${nu}`)
  let gamma = options.gamma
  if (gamma === undefined) {
    let s = 0
    let s2 = 0
    for (const v of m.data) {
      s += v
      s2 += v * v
    }
    const variance = s2 / m.data.length - (s / m.data.length) ** 2
    gamma = 1 / (m.n * (variance > 0 ? variance : 1))
  }
  const K = Float64Array.from(toFlat(gram(kernelOf(gamma), X)))
  return { X, n: m.m, nu, gamma, K, C: 1 / (nu * m.m) }
}

/** Solve min ½αᵀQα + cᵀα subject to 0 ≤ α ≤ C and Σα = 1 by the interior-point method. */
function boxSimplexQp(Q: Float64Array, c: Float64Array, n: Size, C: number): Float64Array {
  const A = new Float64Array(2 * n * n)
  const b = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    A[i * n + i] = 1
    b[i] = C
    A[(n + i) * n + i] = -1
  }
  const r = quadprog(
    {
      Q: fromData(Q, [n, n]),
      c: fromData(c, [n]),
      A: fromData(A, [2 * n, n]),
      b: fromData(b, [2 * n]),
      E: fromData(new Float64Array(n).fill(1), [1, n]),
      e: fromData(Float64Array.of(1), [1]),
    },
    { method: 'interior-point', tolerance: 1e-10 },
  )
  const alpha = Float64Array.from(toFlat(r.x), (v) => Math.min(C, Math.max(0, v)))
  return polish(Q, c, alpha, n, C) ?? alpha
}

/**
 * The interior point stops strictly inside the box, so its weights at the bounds are only near them. Starting from the
 * bound set it found, solve the KKT system of the free weights exactly, Q_FF α_F − μ1 = −Q_FB α_B − c_F with
 * 1ᵀα_F = 1 − 1ᵀα_B, moving a free weight that leaves the box to its bound and freeing a bound weight whose
 * multiplier has the wrong sign, until the set settles (a few rounds of an active-set method from a good start).
 * Returns null if it does not settle (the interior point's weights are then kept).
 */
function polish(Q: Float64Array, c: Float64Array, alpha: Float64Array, n: Size, C: number): Float64Array | null {
  const tol = 1e-4 * C
  // 0 or C for a weight held at a bound, NaN for a free one.
  const fixed = Float64Array.from(alpha, (a) => (a < tol ? 0 : a > C - tol ? C : NaN))
  for (let round = 0; round < 50; round++) {
    const free: number[] = []
    fixed.forEach((a, i) => Number.isNaN(a) && free.push(i))
    if (free.length === 0) return null
    const f = free.length
    const M = new Float64Array((f + 1) * (f + 1))
    const rhs = new Float64Array(f + 1)
    let budget = 1
    for (let i = 0; i < n; i++) if (!Number.isNaN(fixed[i])) budget -= fixed[i]
    free.forEach((i, a) => {
      free.forEach((j, b) => (M[a * (f + 1) + b] = Q[i * n + j]))
      M[a * (f + 1) + f] = -1
      M[f * (f + 1) + a] = 1
      let r = -c[i]
      for (let j = 0; j < n; j++) if (fixed[j] === C) r -= Q[i * n + j] * C
      rhs[a] = r
    })
    rhs[f] = budget
    const sol = solveDense(M, rhs, f + 1)
    if (!sol.x) return null
    const out = Float64Array.from(fixed, (a) => (Number.isNaN(a) ? 0 : a))
    free.forEach((i, a) => (out[i] = sol.x![a]))
    const mu = sol.x[f]
    let changed = false
    // A free weight outside the box goes to the bound it crossed.
    free.forEach((i) => {
      if (out[i] < 0) {
        fixed[i] = 0
        changed = true
      } else if (out[i] > C) {
        fixed[i] = C
        changed = true
      }
    })
    if (changed) continue
    // A bound weight whose gradient says it should move inward is freed.
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(fixed[i])) continue
      let g = c[i]
      for (let j = 0; j < n; j++) g += Q[i * n + j] * out[j]
      if ((fixed[i] === 0 && g - mu < -1e-12) || (fixed[i] === C && g - mu > 1e-12)) {
        fixed[i] = NaN
        changed = true
      }
    }
    if (!changed) return out
  }
  return null
}

/** Indices with 0 < α < C (free support vectors), allowing a small tolerance. */
function freeSupport(alpha: Float64Array, C: number): number[] {
  const tol = 1e-9 * C
  const out: number[] = []
  alpha.forEach((a, i) => a > tol && a < C - tol && out.push(i))
  return out
}

function keepSupport(X: Tensor, alpha: Float64Array, C: number) {
  const idx: number[] = []
  alpha.forEach((a, i) => a > 1e-9 * C && idx.push(i))
  const d = X.shape[1]
  const rows = new Float64Array(idx.length * d)
  idx.forEach((i, r) => rows.set(X.data.subarray(i * d, (i + 1) * d) as Float64Array, r * d))
  const bound = Array.from(alpha).filter((a) => a >= C * (1 - 1e-9)).length / alpha.length
  return { support: fromData(rows, [idx.length, d]), alpha: Float64Array.from(idx, (i) => alpha[i]), bound }
}

/**
 * The one-class SVM (Schölkopf, Platt, Shawe-Taylor, Smola and Williamson, 2001) in its dual: minimise ½αᵀKα subject
 * to 0 ≤ αᵢ ≤ 1/(νn) and Σαᵢ = 1. The decision function f(x) = Σ αᵢ k(xᵢ, x) − ρ is positive inside the boundary;
 * ρ = (Kα)ᵢ at the free support vectors. ν bounds the share of training points outside from above and the share of
 * support vectors from below. scikit-learn's `OneClassSVM` scales α and ρ by νn.
 */
export function oneClassSvm(x: MatrixLike, options: OneClassOptions = {}): OneClassModel {
  const { X, n, gamma, K, C } = setup(x, options, 'oneClassSvm')
  const alpha = boxSimplexQp(K, new Float64Array(n), n, C)
  const Ka = dense.matVec(K, alpha, n, n)
  const free = freeSupport(alpha, C)
  const pool = free.length ? free : Array.from({ length: n }, (_, i) => i).filter((i) => alpha[i] > 0)
  const offset = pool.reduce((s, i) => s + Ka[i], 0) / pool.length
  const kept = keepSupport(X, alpha, C)
  return {
    kind: 'one-class-svm',
    support: kept.support,
    alpha: kept.alpha,
    gamma,
    offset,
    centreNorm: 0,
    boundShare: kept.bound,
  }
}

/**
 * Support vector data description (Tax and Duin, 2004): the smallest ball (centre a, radius R) in feature space holding
 * the data, with slack for a share of at most ν outside. Dual: minimise αᵀKα − Σ αᵢ k(xᵢ, xᵢ) subject to
 * 0 ≤ αᵢ ≤ 1/(νn) and Σαᵢ = 1; then a = Σ αᵢ φ(xᵢ) and R² is the squared distance of a free support vector from a.
 */
export function supportVectorDataDescription(x: MatrixLike, options: OneClassOptions = {}): OneClassModel {
  const { X, n, gamma, K, C } = setup(x, options, 'supportVectorDataDescription')
  const Q = K.map((v) => 2 * v)
  const c = Float64Array.from({ length: n }, (_, i) => -K[i * n + i])
  const alpha = boxSimplexQp(Q, c, n, C)
  const Ka = dense.matVec(K, alpha, n, n)
  const centreNorm = alpha.reduce((s, a, i) => s + a * Ka[i], 0)
  const free = freeSupport(alpha, C)
  const pool = free.length ? free : Array.from({ length: n }, (_, i) => i).filter((i) => alpha[i] > 0)
  const offset = pool.reduce((s, i) => s + K[i * n + i] - 2 * Ka[i] + centreNorm, 0) / pool.length
  const kept = keepSupport(X, alpha, C)
  return { kind: 'svdd', support: kept.support, alpha: kept.alpha, gamma, offset, centreNorm, boundShare: kept.bound }
}

/**
 * The anomaly score of each row of x under a one-class model, positive outside the boundary and negative inside: −f(x)
 * for the one-class SVM, and ‖φ(x) − a‖² − R² for SVDD.
 */
export function oneClassScore(model: OneClassModel, x: MatrixLike): Float64Array {
  const m = dense.toMatrixF64(x, 'oneClassScore')
  const Q = fromData(Float64Array.from(m.data), [m.m, m.n])
  const Kq = toFlat(gram(kernelOf(model.gamma), Q, model.support))
  const s = model.alpha.length
  const out = new Float64Array(m.m)
  for (let q = 0; q < m.m; q++) {
    let f = 0
    for (let j = 0; j < s; j++) f += model.alpha[j] * Kq[q * s + j]
    out[q] = model.kind === 'one-class-svm' ? model.offset - f : 1 - 2 * f + model.centreNorm - model.offset
  }
  return out
}
