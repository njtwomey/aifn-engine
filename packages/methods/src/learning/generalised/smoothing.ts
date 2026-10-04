/**
 * Smoothing-parameter selection for penalised generalised models (Wood, 2017, "Generalized Additive Models: An
 * Introduction with R", 2nd ed.): the penalty S_λ = Σ λₖ Sₖ, a penalised IRLS fit at fixed λ with the quantities
 * inference needs (H = XᵀWX + S_λ, its inverse and log-determinant, the effective degrees of freedom tr(H⁻¹XᵀWX),
 * §6.1.2), and the criteria minimised over log λ: GCV and UBRE (§6.2.3–6.2.4) and REML/LAML (§6.2.5–6.2.6). Shared by
 * the generalised models; `gam` searches log λ with Nelder–Mead over these.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import { cholesky, choleskySolve, eigh } from 'aifn-compute/numerics/linalg'
import type { Family, Link } from 'aifn-compute/probability/likelihoods'
import { irls, type IrlsState } from './irls'
import { NumericalError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array

/** A penalised design: the model matrix X [n, P] (row-major), the penalties Sₖ (each P × P) and the null-space size. */
export type PenalisedDesign = {
  X: F64
  n: number
  P: number
  penalties: readonly { S: F64 }[]
  /** Dimension of the unpenalised space M_p = P − rank(Σ Sₖ). */
  nullSpace: number
}

/** The data a penalised fit is made on. */
export type PenalisedData = { y: Tensor; weights?: Tensor; offset?: Tensor }

/** S_λ = Σ λₖ Sₖ (+ an extra penalty), P × P. */
export function penaltyMatrix(A: PenalisedDesign, lambdas: readonly number[], extra?: F64): F64 {
  const S = new Float64Array(A.P * A.P)
  A.penalties.forEach(({ S: Sk }, k) => {
    const l = lambdas[k]
    if (l === 0) return
    for (let i = 0; i < S.length; i++) S[i] += l * Sk[i]
  })
  if (extra) for (let i = 0; i < S.length; i++) S[i] += extra[i]
  return S
}

/** The rank of Σ Sₖ gives the unpenalised dimension M_p = P − rank. */
export function nullSpaceDimension(penalties: readonly { S: F64 }[], P: number): number {
  const sumS = new Float64Array(P * P)
  for (const { S } of penalties) for (let k = 0; k < P * P; k++) sumS[k] += S[k]
  const ev = dense.data(eigh(fromData(sumS, [P, P])).values)
  const top = Math.max(...Array.from(ev, Math.abs), 0)
  return P - ev.filter((v) => v > 1e-9 * top).length
}

/** log of the product of the positive eigenvalues of S (its rank fixed by the structure). */
function logPseudoDeterminant(S: F64, P: number, rank: number): number {
  if (rank === 0) return 0
  const ev = dense.data(eigh(fromData(S, [P, P])).values)
  let s = 0
  for (let i = 0; i < rank; i++) s += Math.log(ev[i])
  return s
}

/**
 * The quantities inference needs at coefficients β with working weights W: H = XᵀWX + S, its inverse and
 * log-determinant, the effective degrees of freedom tr(H⁻¹XᵀWX) and their diagonal, and the penalty βᵀSβ (Wood, 2017,
 * §6.1.2). At a P-IRLS optimum these are the fit's; at any other β (a step of another fitter) they describe that β.
 */
export type PenalisedInference = {
  /** H = XᵀWX + S (+ shape penalty) and its inverse. */
  H: F64
  Hinv: F64
  /** XᵀWX at β. */
  XtWX: F64
  edf: number
  /** diag(H⁻¹XᵀWX), [P]: per-coefficient EDF. */
  edfDiag: F64
  penalty: number
  logDetH: number
  /** Jitter added to H to factor it. */
  jitter: number
}

/** H, H⁻¹, the EDF and the penalty at β with working weights W [n] (see `PenalisedInference`). */
export function penalisedInference(
  A: PenalisedDesign,
  W: ArrayLike<number>,
  S: F64,
  beta: ArrayLike<number>,
): PenalisedInference {
  const { P, n, X } = A
  const XtWX = new Float64Array(P * P)
  for (let i = 0; i < n; i++) {
    const wi = W[i]
    if (wi === 0) continue
    for (let a = 0; a < P; a++) {
      const xa = X[i * P + a] * wi
      if (xa === 0) continue
      for (let b = 0; b <= a; b++) XtWX[a * P + b] += xa * X[i * P + b]
    }
  }
  for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) XtWX[a * P + b] = XtWX[b * P + a]
  const H = Float64Array.from(XtWX, (v, k) => v + S[k])
  // Overlapping null spaces (e.g. a linear term inside a tensor smooth's) make H singular: jitter is added and
  // reported.
  const c = cholesky(fromData(H, [P, P]))
  let logDetH = 0
  const L = dense.data(c.L)
  for (let i = 0; i < P; i++) logDetH += 2 * Math.log(L[i * P + i])
  const eye = new Float64Array(P * P)
  for (let i = 0; i < P; i++) eye[i * P + i] = 1
  const Hinv = Float64Array.from(dense.data(choleskySolve(c.L, fromData(eye, [P, P])) as Tensor))
  const edfDiag = new Float64Array(P)
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) edfDiag[a] += Hinv[a * P + b] * XtWX[b * P + a]
  let penalty = 0
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) penalty += beta[a] * S[a * P + b] * beta[b]
  return { H, Hinv, XtWX, edf: edfDiag.reduce((a, b) => a + b, 0), edfDiag, penalty, logDetH, jitter: c.jitter }
}

/** One penalised fit at fixed λ, with everything the scores and inference need. */
export type PenalisedFit = PenalisedInference & {
  training: Trace<IrlsState>
  final: IrlsState
  beta: F64
}

/**
 * Penalised IRLS at the penalty S (`irls` with tolerance 1e-12 on the penalised deviance, so that β is accurate to
 * about 1e-7 even where Fisher scoring converges only linearly), warm-started from `start` when given.
 */
export function penalisedFit(
  A: PenalisedDesign,
  data: PenalisedData,
  family: Family,
  link: Link,
  S: F64,
  maxSteps: number,
  start?: F64,
): PenalisedFit {
  const problem = {
    design: fromData(A.X, [A.n, A.P]),
    y: data.y,
    family,
    link,
    weights: data.weights,
    offset: data.offset,
    penalty: fromData(S, [A.P, A.P]),
    tolerance: 1e-12,
  }
  const training = trace(irls(problem), start ? { coefficients: fromData(start, [start.length]) } : {}, maxSteps, {
    record: { deviance: (s) => s.deviance, penalisedDeviance: (s) => s.penalisedDeviance },
  })
  const final = training.final
  if (!final.coefficients)
    throw new NumericalError('penalisedFit', 'penalisedFit: P-IRLS took no step', 'not-converged')
  const beta = Float64Array.from(dense.data(final.coefficients))
  return { training, final, beta, ...penalisedInference(A, dense.data(final.workingWeights), S, beta) }
}

/**
 * The smoothing-parameter criterion to minimise at a penalised fit: GCV n·D/(n − γ·edf)² when φ is estimated, UBRE
 * D/n − φ + 2γφ·edf/n when it is known; REML (Laplace-approximate restricted likelihood, φ profiled out when
 * estimated) −2 log L_R up to a constant.
 */
export function smoothingCriterion(
  method: 'reml' | 'gcv',
  A: PenalisedDesign,
  family: Family,
  fit: PenalisedFit,
  S: F64,
  gamma: number,
): number {
  const n = A.n
  const D = fit.final.deviance
  if (method === 'gcv') {
    if (family.dispersion === null) return (n * D) / (n - gamma * fit.edf) ** 2
    const phi = family.dispersion
    return D / n - phi + (2 * gamma * phi * fit.edf) / n
  }
  const rank = A.P - A.nullSpace
  const logS = logPseudoDeterminant(S, A.P, rank)
  const Dp = D + fit.penalty
  if (family.dispersion === null) {
    // φ profiled out: φ̂ = D_p/(n − M_p).
    const m = n - A.nullSpace
    return m * (1 + Math.log((2 * Math.PI * Dp) / m)) + fit.logDetH - logS
  }
  return Dp / family.dispersion + fit.logDetH - logS
}
