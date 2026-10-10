/**
 * Smoothing-parameter selection for penalised generalised models (Wood, 2017, "Generalized Additive Models: An
 * Introduction with R", 2nd ed.): the penalty $\Smat_\lambda = \sum_k \lambda_k \Smat_k$, a penalised IRLS fit at
 * fixed $\lambdavec$ with the quantities inference needs ($\Hmat = \Xmat^\top\Wmat\Xmat + \Smat_\lambda$, its
 * inverse and log-determinant, the effective degrees of freedom $\trace(\Hmat^{-1}\Xmat^\top\Wmat\Xmat)$, §6.1.2),
 * and the criteria minimised over $\log\lambdavec$: GCV and UBRE (§6.2.3–6.2.4) and REML/LAML (§6.2.5–6.2.6). Shared
 * by the generalised models; `gam` searches $\log\lambdavec$ with Nelder–Mead over these. Matrices are row-major
 * `Float64Array`s.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import { cholesky, choleskySolve, eigh } from 'aifn-compute/numerics/linalg'
import type { Family, Link } from 'aifn-compute/probability/likelihoods'
import { irls, type IrlsState } from './irls'
import { NumericalError } from 'aifn-compute/foundation/errors'

/** A row-major array of numbers. */
type F64 = Float64Array

/**
 * A penalised design: the model matrix $\Xmat$ ($n \times P$, row-major), the penalties $\Smat_k$ (each
 * $P \times P$) and the null-space size.
 */
export type PenalisedDesign = {
  /** The model matrix $\Xmat$, row-major, $n P$ values. */
  X: F64
  /** Number of observations $n$. */
  n: number
  /** Number of coefficients $P$. */
  P: number
  /** The penalties $\Smat_k$, each $P \times P$ row-major, before their smoothing parameters. */
  penalties: readonly { S: F64 }[]
  /** Dimension of the unpenalised space $M_p = P - \rank(\sum_k \Smat_k)$. */
  nullSpace: number
}

/**
 * The data a penalised fit is made on: responses `y` ($n$), and optional prior weights `weights` and offset `offset`
 * ($n$ each), as `irls` takes them.
 */
export type PenalisedData = { y: Tensor; weights?: Tensor; offset?: Tensor }

/**
 * The total penalty $\Smat_\lambda = \sum_k \lambda_k \Smat_k$ (plus an extra penalty), $P \times P$.
 *
 * @param A The design whose penalties are combined.
 * @param lambdas The smoothing parameters $\lambda_k$, one per penalty; a zero skips its penalty.
 * @param extra A further $P \times P$ penalty added as it is (e.g. a shape-constraint penalty), or none.
 * @returns $\Smat_\lambda$, row-major, $P^2$ values.
 *
 * @example Two one-coefficient penalties weighted by their smoothing parameters
 * const A = {
 *   X: Float64Array.of(1, 0, 0, 1), n: 2, P: 2, nullSpace: 0,
 *   penalties: [{ S: Float64Array.of(1, 0, 0, 0) }, { S: Float64Array.of(0, 0, 0, 1) }],
 * }
 * print('S =', penaltyMatrix(A, [10, 0.1]))
 * print('S + extra =', penaltyMatrix(A, [10, 0.1], Float64Array.of(0, 1, 1, 0)))
 */
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

/**
 * The dimension of the unpenalised space, $M_p = P - \rank(\sum_k \Smat_k)$, the rank counting the eigenvalues
 * above $10^{-9}$ times the largest in absolute value.
 *
 * @param penalties The penalties $\Smat_k$, each $P \times P$ row-major.
 * @param P The number of coefficients.
 * @returns $M_p$.
 *
 * @example A second-difference penalty leaves constants and lines unpenalised
 * // Second differences of four coefficients: two rows, D, and the penalty S = D'D.
 * const D = [[1, -2, 1, 0], [0, 1, -2, 1]]
 * const S = Float64Array.from({ length: 16 }, (_, k) => D.reduce((a, r) => a + r[Math.floor(k / 4)] * r[k % 4], 0))
 * print('null-space dimension of S =', nullSpaceDimension([{ S }], 4))
 * print('of a ridge =', nullSpaceDimension([{ S: Float64Array.from({ length: 16 }, (_, k) => +(k % 5 === 0)) }], 4))
 */
export function nullSpaceDimension(penalties: readonly { S: F64 }[], P: number): number {
  const sumS = new Float64Array(P * P)
  for (const { S } of penalties) for (let k = 0; k < P * P; k++) sumS[k] += S[k]
  const ev = dense.data(eigh(fromData(sumS, [P, P])).values)
  const top = Math.max(...Array.from(ev, Math.abs), 0)
  return P - ev.filter((v) => v > 1e-9 * top).length
}

/**
 * $\log\lvert\Smat\rvert_+$, the log of the product of the positive eigenvalues of $\Smat$, with its rank given
 * (fixed by the structure) rather than detected: the logs of the `rank` largest eigenvalues are summed.
 *
 * @param S The penalty, $P \times P$ row-major.
 * @param P The number of coefficients.
 * @param rank The rank of $\Smat$, $P - M_p$.
 * @returns The log pseudo-determinant (0 for rank 0).
 */
function logPseudoDeterminant(S: F64, P: number, rank: number): number {
  if (rank === 0) return 0
  const ev = dense.data(eigh(fromData(S, [P, P])).values)
  let s = 0
  for (let i = 0; i < rank; i++) s += Math.log(ev[i])
  return s
}

/**
 * The quantities inference needs at coefficients $\betavec$ with working weights $\Wmat$:
 * $\Hmat = \Xmat^\top\Wmat\Xmat + \Smat$, its inverse and log-determinant, the effective degrees of freedom
 * $\trace(\Hmat^{-1}\Xmat^\top\Wmat\Xmat)$ and their diagonal, and the penalty $\betavec^\top\Smat\betavec$
 * (Wood, 2017, §6.1.2). At a P-IRLS optimum these are the fit's; at any other $\betavec$ (a step of another fitter)
 * they describe that $\betavec$. Matrices are $P \times P$, row-major.
 */
export type PenalisedInference = {
  /** $\Hmat = \Xmat^\top\Wmat\Xmat + \Smat$ ($\Smat$ including any shape penalty), without the jitter. */
  H: F64
  /** $\Hmat^{-1}$ (of $\Hmat$ plus the jitter). */
  Hinv: F64
  /** $\Xmat^\top\Wmat\Xmat$ at $\betavec$. */
  XtWX: F64
  /** The effective degrees of freedom, the sum of `edfDiag`. */
  edf: number
  /** $\diag(\Hmat^{-1}\Xmat^\top\Wmat\Xmat)$, $P$: per-coefficient EDF. */
  edfDiag: F64
  /** The penalty $\betavec^\top\Smat\betavec$. */
  penalty: number
  /** $\log\det\Hmat$ (of $\Hmat$ plus the jitter), from its Cholesky factor. */
  logDetH: number
  /** Jitter added to the diagonal of $\Hmat$ to factor it (0 when it factored as given). */
  jitter: number
}

/**
 * $\Hmat$, $\Hmat^{-1}$, the EDF and the penalty at $\betavec$ with working weights $\Wmat$ (see
 * `PenalisedInference`). Overlapping null spaces make $\Hmat$ singular; `cholesky`'s jitter is then added and
 * reported.
 *
 * @param A The design.
 * @param W The working weights $W_{ii}$, $n$.
 * @param S The total penalty, $P \times P$ row-major.
 * @param beta The coefficients $\betavec$, $P$.
 * @returns The inference quantities.
 */
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

/** One penalised fit at fixed $\lambdavec$, with everything the scores and inference need. */
export type PenalisedFit = PenalisedInference & {
  /** The P-IRLS trace (series `deviance` and `penalisedDeviance`). */
  training: Trace<IrlsState>
  /** Its last state. */
  final: IrlsState
  /** The fitted coefficients $\hat\betavec$, $P$. */
  beta: F64
}

/**
 * Penalised IRLS at the penalty $\Smat$ (`irls` with tolerance 1e-12 on the penalised deviance, so that $\betavec$ is
 * accurate to about 1e-7 even where Fisher scoring converges only linearly), warm-started from `start` when given,
 * with the inference quantities at the result. Throws `NumericalError` when P-IRLS takes no step.
 *
 * @param A The design.
 * @param data The responses, and optional prior weights and offset.
 * @param family The response family.
 * @param link The link.
 * @param S The total penalty $\Smat_\lambda$, $P \times P$ row-major (from `penaltyMatrix`).
 * @param maxSteps Most P-IRLS steps.
 * @param start Starting coefficients, $P$ (default: the family's initial mean).
 * @returns The fit.
 *
 * @example A ridge on the slope shrinks it and lowers the effective degrees of freedom
 * import { gaussianFamily, link } from 'aifn-compute/probability/likelihoods'
 * // Columns x and 1; only the slope is penalised.
 * const X = Float64Array.of(0, 1, 1, 1, 2, 1, 3, 1)
 * const A = { X, n: 4, P: 2, penalties: [{ S: Float64Array.of(1, 0, 0, 0) }], nullSpace: 1 }
 * const y = tensor([1, 3, 2, 5])
 * for (const lambda of [0, 1, 10]) {
 *   const fit = penalisedFit(A, { y }, gaussianFamily(), link('identity'), penaltyMatrix(A, [lambda]), 10)
 *   print('lambda =', lambda, ' beta =', fit.beta, ' edf =', fit.edf)
 * }
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
 * The smoothing-parameter criterion to minimise at a penalised fit, with the deviance $D$ and the penalised deviance
 * $D_p = D + \betavec^\top\Smat\betavec$. `gcv`: GCV $n D / (n - \gamma\,\text{edf})^2$ when $\phi$ is estimated,
 * UBRE $D/n - \phi + 2\gamma\phi\,\text{edf}/n$ when it is known. `reml` (Laplace-approximate restricted
 * likelihood) is $-2\log L_R$ up to a constant: $D_p/\phi + \log\lvert\Hmat\rvert - \log\lvert\Smat\rvert_+$
 * with $\phi$ known, and with $\phi$ profiled out at $\hat\phi = D_p/(n - M_p)$,
 * $(n - M_p)(1 + \log(2\pi D_p/(n - M_p))) + \log\lvert\Hmat\rvert - \log\lvert\Smat\rvert_+$.
 *
 * @param method `reml`, or `gcv` (GCV or UBRE by whether the family's dispersion is known).
 * @param A The design (its size and null-space dimension).
 * @param family The family: its dispersion, known or null.
 * @param fit The penalised fit at $\Smat$.
 * @param S The total penalty $\Smat_\lambda$ the fit was made at.
 * @param gamma The GCV/UBRE inflation $\gamma$ of the effective degrees of freedom (1 for none; 1.4 smooths more).
 *   Unused by `reml`.
 * @returns The criterion.
 *
 * @example GCV and REML over a grid of smoothing parameters for a Whittaker smoother
 * import { gaussianFamily, link } from 'aifn-compute/probability/likelihoods'
 * const n = 30
 * const xs = Array.from({ length: n }, (_, i) => i / (n - 1))
 * const y = add(tensor(xs.map((x) => Math.sin(2 * Math.PI * x))), normals(stream(1), n, 0, 0.3))
 * // One coefficient per point (X = I) and a second-difference penalty, whose null space is the lines.
 * const X = Float64Array.from({ length: n * n }, (_, k) => +(k % (n + 1) === 0))
 * const S = new Float64Array(n * n)
 * for (let i = 0; i + 2 < n; i++) {
 *   const d = [[i, 1], [i + 1, -2], [i + 2, 1]]
 *   for (const [a, va] of d) for (const [b, vb] of d) S[a * n + b] += va * vb
 * }
 * const A = { X, n, P: n, penalties: [{ S }], nullSpace: 2 }
 * for (const lambda of [0.01, 1, 100, 10000]) {
 *   const Sl = penaltyMatrix(A, [lambda])
 *   const fit = penalisedFit(A, { y }, gaussianFamily(), link('identity'), Sl, 10)
 *   const gcv = smoothingCriterion('gcv', A, gaussianFamily(), fit, Sl, 1)
 *   const reml = smoothingCriterion('reml', A, gaussianFamily(), fit, Sl, 1)
 *   print('lambda =', lambda, ' edf =', fit.edf, ' GCV =', gcv, ' REML =', reml)
 * }
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
