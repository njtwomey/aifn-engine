/**
 * Generalised additive models g(E[y]) = α + Σⱼ fⱼ(x) fitted by penalised IRLS (`aifn-methods/learning/glm`'s `irls`
 * with the penalty S_λ = Σ λₖ Sₖ), with smoothing parameters chosen by GCV/UBRE or REML (Laplace-approximate restricted
 * likelihood), effective degrees of freedom, Bayesian posterior bands and draws, and shape constraints.
 *
 * References: Wood (2017), "Generalized Additive Models: An Introduction with R", 2nd ed.: penalised IRLS §6.1.1,
 * GCV and UBRE §6.2.3–6.2.4, REML and LAML §6.2.5–6.2.6, EDF §6.1.2, posterior covariance V_β = (XᵀWX + S_λ)⁻¹φ §6.10.
 * Shape constraints follow pyGAM (Servén and Brummitt, 2018): a large penalty on the coefficient differences that
 * violate the constraint, added until none do.
 */

import {
  withExpectation,
  withSampling,
  type Decides,
  type Distribution,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Samples,
  type Trained,
} from 'aifn-compute/learning/estimators'
import type { Family, Link } from 'aifn-compute/probability/likelihoods'
import type { Trace } from 'aifn-compute/foundation/trace'
import type { IrlsState } from '../irls'
import { residuals, type ResidualKind } from '../residuals'
import { penalisedInference } from '../smoothing'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { termBasis, termGridRows, times, type BuiltTerm, type TermSpec } from './terms'
import { gamDesignAt, gamProblem, type GamData, type GamProblem, type GamSpec, type SmoothingMethod } from './problem'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

export type { GamData, SmoothingMethod } from './problem'

type F64 = Float64Array
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))
const vec = (a: F64) => fromData(a, [a.length])

/** Hyperparameters of `gam`: a problem specification (`GamSpec`) with a family object or name. */
export type GamParams = GamSpec & { terms: readonly TermSpec[] }

/** Partial effect of one term on a grid, with pointwise standard errors. */
export type PartialEffect = { fit: Tensor; se: Tensor }

/**
 * A term's basis on a grid with the coefficients: the raw basis and the constrained columns (see `termBasis`), the
 * coefficients on each, the raw columns weighted by their coefficients, and their sum, the partial effect.
 */
export type GamTermBasis = ReturnType<typeof termBasis> & {
  /** βⱼ [size], the coefficients of the constrained columns. */
  coefficients: Tensor
  /** Zβⱼ [rawSize], the same function's coefficients on the raw basis. */
  rawCoefficients: Tensor
  /** Each raw basis column times its coefficient, [m, rawSize]. */
  weighted: Tensor
  /** Σ of the weighted columns: the partial effect fⱼ on the grid, [m]. */
  sum: Tensor
}

/** A fitted GAM. */
export interface GamModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Distribution>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<IrlsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gam'
  readonly family: Family
  readonly link: Link
  readonly terms: readonly BuiltTerm[]
  /** Term labels, e.g. "s(x0)", "te(x1, x2)". */
  readonly labels: string[]
  /** All coefficients [P]: the intercept, then each term's block. */
  readonly coefficients: Tensor
  readonly intercept: number
  /** One smoothing parameter per penalty, in term order. */
  readonly lambdas: number[]
  /** Total effective degrees of freedom tr(H⁻¹XᵀWX), including the intercept. */
  readonly edf: number
  /** EDF of each term. */
  readonly termEdf: number[]
  /** φ: fixed by the family or estimated as Σwᵢ(yᵢ − μᵢ)²/V(μᵢ) / (n − edf). */
  readonly dispersion: number
  readonly deviance: number
  /** 1 − D/D₀ against the intercept-only model's deviance D₀. */
  readonly devianceExplained: number
  /** λ-weighted penalty βⱼᵀS_λβⱼ of each term. */
  readonly termPenalties: number[]
  /** D + βᵀS_λβ (with any shape-constraint penalty). */
  readonly penalisedDeviance: number
  /** AIC = −2 log L(β̂, φ̂) + 2(edf + 1 when φ is estimated) (Wood, 2017, §6.11.2). */
  readonly aic: number
  /** The smoothing-parameter criterion at the chosen λ (REML: −2 × restricted log-likelihood up to a constant). */
  readonly smoothingScore: { method: SmoothingMethod; value: number; evaluations: number }
  /** Bayesian covariance V_β = H⁻¹φ [P, P]. */
  readonly covariance: Tensor
  readonly fitted: Tensor
  readonly linearPredictor: Tensor
  readonly converged: boolean
  /** Jitter added to XᵀWX + S_λ to factor it (non-zero when unpenalised directions overlap). */
  readonly jitter: number
  /** Shape constraints: active difference rows and how many differences still violate (with tolerance 1e-6). */
  readonly shape: { active: number; violations: number }
  /** The problem the model solves (design, penalties, λ, the reference optimum). */
  readonly problem: GamProblem
  /**
   * The partial effect fⱼ on a grid: [m] for a one-dimensional term (a `by` variable set to 1 or its level), [m, 2] for
   * a tensor.
   */
  partial(term: number, grid: Tensor): PartialEffect
  /** Term j's basis on a grid with its coefficients (raw and constrained), the weighted columns and their sum. */
  basis(term: number, grid: Tensor): GamTermBasis
  /** Partial residuals of term j at the training data: f̂ⱼ(xᵢ) + the working residual (yᵢ − μᵢ)/μ′(ηᵢ), [n]. */
  partialResiduals(term: number): Tensor
  /** Training residuals of a kind (default deviance). */
  residuals(kind?: ResidualKind): Tensor
  /** `count` posterior draws of fⱼ on a grid from β ~ N(β̂, V_β), [count, m]. */
  partialDraws(s: Stream, term: number, grid: Tensor, count: number): Tensor
}

/**
 * The GAM of a problem at coefficients β: at the problem's optimum this is the fitted model; at any other β (a step of
 * a fitter) it is the model those coefficients define, with EDF, covariance and bands from H = XᵀWX + S_λ at that β.
 */
export function gamModel(problem: GamProblem, coefficients: ArrayLike<number>, training?: Trace<IrlsState>): GamModel {
  const A = problem.design
  const { family, link: lk, data } = problem
  const n = A.n
  const beta = Float64Array.from(coefficients)
  if (beta.length !== A.P) throw new ShapeError('gamModel', `gamModel: ${beta.length} coefficients for ${A.P} columns`)
  const ev = problem.evaluate(beta)
  const { W } = problem.working(beta)
  const inf = penalisedInference(A, W, problem.penalty, beta)
  const mu = ev.mu
  const y = problem.y
  const w = problem.w
  const V = f64(family.variance(vec(mu)) as Tensor)
  let pearson = 0
  for (let i = 0; i < n; i++) pearson += (w[i] * (y[i] - mu[i]) ** 2) / V[i]
  const dispersion = family.dispersion ?? pearson / (n - inf.edf)
  const cov = Float64Array.from(inf.Hinv, (v) => v * dispersion)
  const termEdf = A.terms.map((t, j) => {
    let s = 0
    for (let c = 0; c < t.size; c++) s += inf.edfDiag[A.offsets[j] + c]
    return s
  })
  const logLik = family.logLikelihood(data.y, vec(mu), dispersion, data.weights ?? vec(w))
  const aic = -2 * logLik + 2 * (inf.edf + (family.dispersion === null ? 1 : 0))

  const forward = (x: Tensor) => {
    const Xd = gamDesignAt(A, x)
    const m = x.shape[0]
    const eta = new Float64Array(m)
    for (let i = 0; i < m; i++) for (let a = 0; a < A.P; a++) eta[i] += Xd[i * A.P + a] * beta[a]
    return vec(eta)
  }
  // A term's block of the design on a grid of its own feature(s).
  const termGrid = (j: number, grid: Tensor): { B: F64; m: number } => {
    const t = A.terms[j]
    if (!t) throw new DomainError('gam', `gam: no term ${j}`)
    const { rows, m } = termGridRows(t, grid, A.d)
    return { B: times(t.raw(rows, m, A.d), m, t.rawSize, t.Z, t.size), m }
  }
  const blockCov = (j: number) => {
    const t = A.terms[j]
    const o = A.offsets[j]
    const C = new Float64Array(t.size * t.size)
    for (let a = 0; a < t.size; a++) for (let b = 0; b < t.size; b++) C[a * t.size + b] = cov[(o + a) * A.P + o + b]
    return C
  }
  const partial = (j: number, grid: Tensor): PartialEffect => {
    const t = A.terms[j]
    const { B, m } = termGrid(j, grid)
    const C = blockCov(j)
    const fitv = new Float64Array(m)
    const se = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      let f = 0
      let v = 0
      for (let a = 0; a < t.size; a++) {
        f += B[i * t.size + a] * beta[A.offsets[j] + a]
        for (let b = 0; b < t.size; b++) v += B[i * t.size + a] * C[a * t.size + b] * B[i * t.size + b]
      }
      fitv[i] = f
      se[i] = Math.sqrt(Math.max(v, 0))
    }
    return { fit: vec(fitv), se: vec(se) }
  }
  const basis = (j: number, grid: Tensor): GamTermBasis => {
    const t = A.terms[j]
    if (!t) throw new DomainError('gam', `gam: no term ${j}`)
    const b = termBasis(t, grid, A.d)
    const bj = beta.slice(A.offsets[j], A.offsets[j] + t.size)
    const rawBeta = times(t.Z, t.rawSize, t.size, bj, 1)
    const raw = f64(b.raw)
    const m = grid.shape[0]
    const weighted = new Float64Array(m * t.rawSize)
    const total = new Float64Array(m)
    for (let i = 0; i < m; i++)
      for (let c = 0; c < t.rawSize; c++) {
        const v = raw[i * t.rawSize + c] * rawBeta[c]
        weighted[i * t.rawSize + c] = v
        total[i] += v
      }
    return {
      ...b,
      coefficients: vec(bj),
      rawCoefficients: vec(rawBeta),
      weighted: fromData(weighted, [m, t.rawSize]),
      sum: vec(total),
    }
  }
  const partialDraws = (s: Stream, j: number, grid: Tensor, count: number) => {
    const t = A.terms[j]
    const { B, m } = termGrid(j, grid)
    const { L } = cholesky(fromData(blockCov(j), [t.size, t.size]))
    const Lf = f64(L)
    const z = f64(normals(s, [count, t.size]))
    const out = new Float64Array(count * m)
    for (let r = 0; r < count; r++) {
      const b = new Float64Array(t.size)
      for (let a = 0; a < t.size; a++) {
        let v = beta[A.offsets[j] + a]
        for (let c = 0; c <= a; c++) v += Lf[a * t.size + c] * z[r * t.size + c]
        b[a] = v
      }
      for (let i = 0; i < m; i++) for (let a = 0; a < t.size; a++) out[r * m + i] += B[i * t.size + a] * b[a]
    }
    return fromData(out, [count, m])
  }
  const eta = vec(ev.eta)
  const muT = vec(mu)
  const partialResiduals = (j: number) => {
    const t = A.terms[j]
    if (!t) throw new DomainError('gam', `gam: no term ${j}`)
    const dmu = f64(lk.derivative(eta) as Tensor)
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let f = 0
      for (let c = 0; c < t.size; c++) f += A.X[i * A.P + A.offsets[j] + c] * beta[A.offsets[j] + c]
      out[i] = f + (y[i] - mu[i]) / dmu[i]
    }
    return vec(out)
  }
  // Violated shape differences at β (tolerance 1e-6).
  let violations = 0
  A.terms.forEach((t, j) => {
    if (!t.shape) return
    const R = times(t.shape.matrix, t.shape.rows, t.rawSize, t.Z, t.size)
    for (let r = 0; r < t.shape.rows; r++) {
      let v = 0
      for (let c = 0; c < t.size; c++) v += R[r * t.size + c] * beta[A.offsets[j] + c]
      if (v < -1e-6) violations++
    }
  })
  const meanAt = (x: Tensor) => lk.inverse(forward(x)) as Tensor
  const base = {
    kind: 'model' as const,
    name: 'gam' as const,
    family,
    link: lk,
    terms: A.terms,
    labels: A.terms.map((t) => t.label),
    coefficients: vec(beta),
    intercept: beta[0],
    lambdas: problem.lambdas,
    edf: inf.edf,
    termEdf,
    dispersion,
    deviance: ev.deviance,
    devianceExplained: 1 - ev.deviance / problem.nullDeviance,
    termPenalties: ev.termPenalties,
    penalisedDeviance: ev.penalisedDeviance,
    aic,
    smoothingScore: problem.smoothing,
    covariance: fromData(cov, [A.P, A.P]),
    fitted: muT,
    linearPredictor: eta,

    jitter: inf.jitter,
    shape: { active: problem.active.reduce((s, a) => s + a.length, 0), violations },
    problem,

    partial,
    basis,
    partialResiduals,
    residuals: (kind?: ResidualKind) =>
      residuals({ y: data.y, mu: muT, eta, weights: data.weights, family, link: lk }, kind),
    partialDraws,
    forward,
    decide: meanAt,
    predictive: (x: Tensor) => family.predictive(meanAt(x), dispersion),
  }
  const model = withSampling(withExpectation(base))
  // The reference optimum is read only when asked for, so a fitter's step costs no P-IRLS refit (the mixins copy
  // properties, so these getters are added after them).
  Object.defineProperties(model, {
    training: { enumerable: true, get: () => training ?? problem.optimum.training },
    converged: { enumerable: true, get: () => (training ? training.final.converged : problem.optimum.final.converged) },
  })
  return model as unknown as GamModel
}

/**
 * The linear predictor η(x) = x̃ᵀβ̂ at new inputs x [m, d] with its pointwise standard error √(x̃ᵀV_βx̃), the intercept's
 * uncertainty included (Wood, 2017, §6.10); g⁻¹ of η ± 2 se is the usual band on the response scale. [m] each.
 */
export function gamLinkBand(model: GamModel, x: Tensor): PartialEffect {
  const A = model.problem.design
  const Xd = gamDesignAt(A, x)
  const m = x.shape[0]
  const P = A.P
  const beta = f64(model.coefficients)
  const V = f64(model.covariance)
  const fit = new Float64Array(m)
  const se = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let f = 0
    let v = 0
    for (let a = 0; a < P; a++) {
      const xa = Xd[i * P + a]
      if (xa === 0) continue
      f += xa * beta[a]
      for (let b = 0; b < P; b++) v += xa * V[a * P + b] * Xd[i * P + b]
    }
    fit[i] = f
    se[i] = Math.sqrt(Math.max(v, 0))
  }
  return { fit: vec(fit), se: vec(se) }
}

/**
 * A generalised additive model: `gam({ terms: [s(0), s(1, { k: 20 }), linearTerm(2)], family: poissonFamily() })`.
 * Fitting builds the penalised problem (`gamProblem`: design, λ by REML/GCV or fixed, shape constraints) and returns
 * the model at its P-IRLS optimum, kept in `training`. Capabilities: `forward` (η), `decide` and `expect` (μ),
 * `predictive` (the family at μ with the fitted dispersion), `sample`.
 */
export function gam(params: GamParams): Estimator<GamData, GamModel> {
  const fam = typeof params.family === 'string' ? params.family : (params.family?.name ?? 'gaussian')
  return {
    name: `gam-${fam}`,
    params,
    fit(data) {
      const problem = gamProblem(params, data)
      return gamModel(problem, problem.optimum.beta)
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gam',
    module: 'learning/generalised/gam',
    name: 'Generalised additive model',
    summary:
      'A sum of penalised smooth terms with smoothing parameters chosen by REML or GCV; the terms are a required argument.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      method: oneOf(['reml', 'gcv', 'fixed']),
      gamma: real(1, 2, { default: 1 }),
      maxSteps: int(1, 500, { default: 50 }),
    }),
    notes: ['generalised-additive-model', 'smoothing-and-penalised-splines'],
    cite: ['hastie1990'],
  },
  gam,
)
