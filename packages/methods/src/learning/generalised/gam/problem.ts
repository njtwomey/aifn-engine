/**
 * The penalised-likelihood problem of a GAM, separated from how it is solved. `gamProblem(spec, data)` builds the
 * model matrix X = [1, B₁, …, B_J] and the penalties from the terms, chooses the smoothing parameters (REML, GCV/UBRE
 * or fixed; Wood, 2017, "Generalized Additive Models", 2nd ed., §6.2) and the active shape-constraint rows, and solves
 * it once by penalised IRLS for the reference optimum. Every fitter of `fitters.ts` then minimises the same objective
 *
 *   J(β) = (D(β) + βᵀS_λβ) / (2n),  D(β) = Σᵢ wᵢ d(yᵢ, g⁻¹(xᵢᵀβ + oᵢ)),
 *
 * the penalised deviance scaled per observation (for a family with known dispersion φ, D/2φ is the negative
 * log-likelihood up to a constant, so J is the penalised negative log-likelihood per observation). `objective` writes
 * J (or its minibatch estimate) with primitives, so autodiff gives its gradient; `gradient` is the closed form
 * ∇J = (Sβ − Xᵀr)/n with rᵢ = wᵢ(yᵢ − μᵢ)μ′(ηᵢ)/V(μᵢ).
 */

import {
  family as familyByName,
  checkLink,
  likelihood,
  type Family,
  type FamilyName,
  type Link,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'
import { add, div, fromData, matmul, mul, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { eigh } from 'aifn-compute/numerics/linalg'
import { minimize } from 'aifn-compute/optim/minimize'
import { deviance as devianceOf } from '../irls'
import {
  nullSpaceDimension,
  penalisedFit,
  penaltyMatrix,
  smoothingCriterion,
  type PenalisedDesign,
  type PenalisedFit,
} from '../smoothing'
import { buildTerms, times, type BuiltTerm, type TermSpec } from './terms'
import { ShapeError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array
const lazy = <T>(f: () => T) => {
  let done = false
  let v: T
  return () => {
    if (!done) {
      v = f()
      done = true
    }
    return v
  }
}
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/** Data for a GAM: features x [n, d] (factor columns as integer codes), responses y [n], optional weights and offset. */
export type GamData = { x: Tensor; y: Tensor; weights?: Tensor; offset?: Tensor }

/** How smoothing parameters are chosen. */
export type SmoothingMethod = 'reml' | 'gcv' | 'fixed'

/** What defines a GAM's penalised problem: the terms, the likelihood and how λ is chosen. */
export type GamSpec = {
  terms: readonly TermSpec[]
  /** A family, or its name (default Gaussian). */
  family?: Family | FamilyName
  /** Default the family's default link; a link the family does not take is rejected (`checkLink`). */
  link?: LinkName | Link
  /** `reml` (default), `gcv` (GCV when φ is estimated, UBRE when it is known) or `fixed`. */
  method?: SmoothingMethod
  /** With `fixed`: λ for every penalty whose term does not fix its own (default 1). */
  lambda?: number
  /** One λ per penalty, in term order: skips the search (a problem reports its choice in `lambdas`). */
  lambdas?: readonly number[]
  /** GCV/UBRE inflation γ ≥ 1 of the EDF, against overfitting (default 1). */
  gamma?: number
  /** Most P-IRLS steps per fit (default 50). */
  maxSteps?: number
  /** Most Nelder–Mead steps of the smoothing-parameter search (default 200). */
  maxSearchSteps?: number
  /** Weight of the shape-constraint penalty relative to the mean diagonal of XᵀWX (default 1e6). */
  constraintWeight?: number
  /** Active shape-constraint rows per term (a problem reports them in `active`): skips the active-set refits. */
  active?: readonly (readonly number[])[]
}

/** The model matrix, penalties and fixed data of a GAM (a `PenalisedDesign` with its terms). */
export type GamDesign = PenalisedDesign & {
  terms: BuiltTerm[]
  /** Column offset of each term's block (column 0 is the intercept). */
  offsets: number[]
  P: number
  n: number
  d: number
  /** The model matrix [n, P], row-major. */
  X: F64
  /** Each penalty embedded in P × P, with the term it belongs to. */
  penalties: { S: F64; term: number }[]
  /** Fixed λ per penalty (NaN: selected). */
  fixed: number[]
  /** Dimension of the unpenalised space (intercept and null spaces). */
  nullSpace: number
}

/** Build the terms on x [n, d] and assemble the model matrix and penalties. */
export function gamDesign(specs: readonly TermSpec[], x: Tensor): GamDesign {
  if (x.shape.length !== 2) throw new ShapeError('gam', 'gam: x must be [n, d]')
  const [n, d] = x.shape
  const Xin = f64(x)
  const terms = buildTerms(specs, Xin, n, d)
  const offsets: number[] = []
  let P = 1
  for (const t of terms) {
    offsets.push(P)
    P += t.size
  }
  const X = new Float64Array(n * P)
  for (let i = 0; i < n; i++) X[i * P] = 1
  terms.forEach((t, j) => {
    const B = times(t.raw(Xin, n, d), n, t.rawSize, t.Z, t.size)
    for (let i = 0; i < n; i++) for (let c = 0; c < t.size; c++) X[i * P + offsets[j] + c] = B[i * t.size + c]
  })
  const penalties: { S: F64; term: number }[] = []
  const fixed: number[] = []
  terms.forEach((t, j) => {
    t.penalties.forEach((S, k) => {
      const E = new Float64Array(P * P)
      for (let a = 0; a < t.size; a++)
        for (let b = 0; b < t.size; b++) E[(offsets[j] + a) * P + offsets[j] + b] = S[a * t.size + b]
      penalties.push({ S: E, term: j })
      fixed.push(t.fixedLambda[k])
    })
  })
  // The null-space size (an eigendecomposition) is needed only by the REML criterion: computed on first use.
  let nullSpace: number | undefined
  return {
    terms,
    offsets,
    P,
    n,
    d,
    X,
    penalties,
    fixed,
    get nullSpace() {
      return (nullSpace ??= nullSpaceDimension(penalties, P))
    },
  }
}

/** Rows of the model matrix at new inputs x [m, d], row-major [m, P]. */
export function gamDesignAt(A: GamDesign, x: Tensor): F64 {
  const [m, d] = x.shape
  if (d !== A.d) throw new ShapeError('gam', `gam: fitted on ${A.d} features, given ${d}`)
  const Xin = f64(x)
  const out = new Float64Array(m * A.P)
  for (let i = 0; i < m; i++) out[i * A.P] = 1
  A.terms.forEach((t, j) => {
    const B = times(t.raw(Xin, m, d), m, t.rawSize, t.Z, t.size)
    for (let i = 0; i < m; i++) for (let c = 0; c < t.size; c++) out[i * A.P + A.offsets[j] + c] = B[i * t.size + c]
  })
  return out
}

/** The objective and fit quantities at one β. */
export type GamEvaluation = {
  /** η = Xβ + o and μ = g⁻¹(η), [n]. */
  eta: F64
  mu: F64
  /** Whether every μ lies in the family's mean space (else the deviance is NaN). */
  valid: boolean
  deviance: number
  /** βᵀS_λβ, and its part from each term's penalties (λ-weighted, without the shape penalty). */
  penalty: number
  termPenalties: number[]
  /** D + βᵀS_λβ. */
  penalisedDeviance: number
  /** J(β) = (D + βᵀS_λβ)/(2n). */
  objective: number
}

/** A GAM's penalised problem with its smoothing parameters chosen (see the module comment). */
export type GamProblem = {
  readonly kind: 'gam-problem'
  readonly spec: GamSpec
  readonly family: Family
  readonly link: Link
  readonly design: GamDesign
  readonly data: GamData
  /** y, prior weights and offset, [n]. */
  readonly y: F64
  readonly w: F64
  readonly o: F64
  /** One λ per penalty, in term order. */
  readonly lambdas: number[]
  /** S_λ = Σ λₖSₖ plus the shape-constraint penalty, [P, P] row-major. */
  readonly penalty: F64
  /** Active shape-constraint rows per term. */
  readonly active: number[][]
  /** The smoothing criterion at the chosen λ (NaN when fixed) and how many fits the search took. */
  readonly smoothing: { method: SmoothingMethod; value: number; evaluations: number }
  /** The P-IRLS solution at the chosen penalty: the reference optimum every fitter should reach. */
  readonly optimum: PenalisedFit
  /** The common start β₀: the intercept at the weighted mean of g(μ₀) for the family's starting mean μ₀, 0 elsewhere. */
  readonly start: F64
  /** The deviance of the intercept-only model (μ = the weighted mean of y). */
  readonly nullDeviance: number
  /** The largest eigenvalue L of ∇²J = H/n at the optimum: a fixed gradient step is stable near it below 2/L. */
  readonly curvature: number
  /** J(β) with primitives (differentiable in β); with `rows`, the unbiased minibatch estimate on those rows. */
  objective(beta: Value, rows?: ArrayLike<number>): Value
  /** ∇J(β) in closed form, [P]. */
  gradient(beta: ArrayLike<number>): F64
  /** The fit quantities at β. */
  evaluate(beta: ArrayLike<number>): GamEvaluation
  /** μ′(η), V(μ), the working response z = η − o + (y − μ)/μ′(η) and weights W = wμ′(η)²/V(μ) at β, [n] each. */
  working(beta: ArrayLike<number>): { z: F64; W: F64 }
}

/** The family by name or as given, and its link checked against it. */
export function resolveLikelihood(
  spec: Pick<GamSpec, 'family' | 'link'>,
  where = 'gam',
): { family: Family; link: Link } {
  const family = typeof spec.family === 'string' ? familyByName(spec.family) : (spec.family ?? familyByName('gaussian'))
  return { family, link: checkLink(family, spec.link, where) }
}

/**
 * The penalised problem of a GAM on data: the design, λ chosen by `spec.method` (Nelder–Mead on log λ over the REML or
 * GCV/UBRE criterion, each evaluation a warm-started P-IRLS fit), the active shape-constraint rows (a heavy penalty on
 * violated coefficient differences, added until none is violated; pyGAM's scheme, Servén and Brummitt, 2018) and the
 * P-IRLS optimum.
 */
export function gamProblem(spec: GamSpec, data: GamData): GamProblem {
  const { family, link } = resolveLikelihood(spec)
  const lik = likelihood(family, link.name)
  const { method = 'reml', gamma = 1, maxSteps = 50, maxSearchSteps = 200, constraintWeight = 1e6 } = spec
  const A = gamDesign(spec.terms, data.x)
  const { n, P } = A
  const y = f64(data.y)
  if (y.length !== n) throw new ShapeError('gam', `gam: ${n} rows of x but ${y.length} responses`)
  const w = data.weights ? f64(data.weights) : new Float64Array(n).fill(1)
  const o = data.offset ? f64(data.offset) : new Float64Array(n)

  // Smoothing parameters.
  const free = A.fixed.map((v, k) => (Number.isNaN(v) ? k : -1)).filter((k) => k >= 0)
  const common = spec.lambda ?? 1
  const lambdasFrom = (logs: ArrayLike<number>) =>
    A.fixed.map((v, k) => (Number.isNaN(v) ? (method === 'fixed' ? common : Math.exp(logs[free.indexOf(k)])) : v))
  let lambdas = spec.lambdas ? [...spec.lambdas] : lambdasFrom(new Float64Array(free.length))
  if (spec.lambdas && spec.lambdas.length !== A.penalties.length)
    throw new ShapeError('gam', `gam: ${spec.lambdas.length} λ given for ${A.penalties.length} penalties`)
  let evaluations = 0
  if (!spec.lambdas && method !== 'fixed' && free.length > 0) {
    let warm: F64 | undefined
    const objective = (logs: Tensor) => {
      evaluations++
      const l = toFlat(logs) as number[]
      // Keep log λ in [−15, 15]; beyond it the criterion is flat and the search wanders.
      const excess = l.reduce((s, v) => s + Math.max(0, Math.abs(v) - 15) ** 2, 0)
      const S = penaltyMatrix(A, lambdasFrom(l.map((v) => Math.max(-15, Math.min(15, v)))))
      try {
        const fit = penalisedFit(A, data, family, link, S, maxSteps, warm)
        warm = fit.beta
        const v = smoothingCriterion(method, A, family, fit, S, gamma)
        return Number.isFinite(v) ? v + excess : Infinity
      } catch {
        return Infinity
      }
    }
    const x0 = new Float64Array(free.length)
    const simplex = [Array.from(x0), ...free.map((_, i) => Array.from(x0, (v, j) => v + (i === j ? 2 : 0)))]
    const best = minimize(objective, x0, {
      method: 'nelder-mead',
      maxSteps: maxSearchSteps,
      initialSimplex: simplex,
      xTolerance: 1e-4,
      fTolerance: 1e-8,
    })
    lambdas = lambdasFrom(Array.from(toFlat(best.x)).map((v) => Math.max(-15, Math.min(15, v))))
  }
  const S = penaltyMatrix(A, lambdas)
  // The optimum, the criterion and the curvature are computed on first use, so a problem rebuilt from known choices
  // (to evaluate a fitter's steps) costs only its design.
  let fitted: PenalisedFit | undefined
  const fitAtS = () => (fitted ??= penalisedFit(A, data, family, link, S, maxSteps))

  // Shape constraints: penalise violated differences heavily, adding rows until none are violated.
  const shapeRows = A.terms.map((t) => (t.shape ? times(t.shape.matrix, t.shape.rows, t.rawSize, t.Z, t.size) : null))
  const active = A.terms.map((_, j) => new Set<number>(spec.active?.[j] ?? []))
  const violations = (beta: F64, tol: number) => {
    const out: [number, number][] = []
    A.terms.forEach((t, j) => {
      const R = shapeRows[j]
      if (!R || !t.shape) return
      for (let r = 0; r < t.shape.rows; r++) {
        let v = 0
        for (let c = 0; c < t.size; c++) v += R[r * t.size + c] * beta[A.offsets[j] + c]
        if (v < -tol) out.push([j, r])
      }
    })
    return out
  }
  let shapePenalty: F64 | null = null
  let penalty = S
  const needsShape = shapeRows.some((r) => r !== null) && (!spec.active || active.some((a) => a.size > 0))
  // The criterion is that of the unconstrained fit at λ, as the search scored it.
  let scoreBeforeShape: number | undefined
  if (needsShape) {
    let fit = fitAtS()
    if (method !== 'fixed') scoreBeforeShape = smoothingCriterion(method, A, family, fit, S, gamma)
    let meanDiag = 0
    for (let a = 0; a < P; a++) meanDiag += fit.XtWX[a * P + a] / P
    const kappa = constraintWeight * meanDiag
    const extraOf = () => {
      const extra = new Float64Array(P * P)
      A.terms.forEach((t, j) => {
        const R = shapeRows[j]
        if (!R) return
        for (const r of active[j])
          for (let a = 0; a < t.size; a++)
            for (let b = 0; b < t.size; b++)
              extra[(A.offsets[j] + a) * P + A.offsets[j] + b] += kappa * R[r * t.size + a] * R[r * t.size + b]
      })
      return extra
    }
    if (spec.active && active.some((a) => a.size > 0)) {
      shapePenalty = extraOf()
      penalty = penaltyMatrix(A, lambdas, shapePenalty)
      fit = penalisedFit(A, data, family, link, penalty, maxSteps, fit.beta)
    }
    for (let refits = 0; !spec.active && refits < 50; refits++) {
      const fresh = violations(fit.beta, 1e-10).filter(([j, r]) => !active[j].has(r))
      if (fresh.length === 0) break
      fresh.forEach(([j, r]) => active[j].add(r))
      shapePenalty = extraOf()
      penalty = penaltyMatrix(A, lambdas, shapePenalty)
      fit = penalisedFit(A, data, family, link, penalty, maxSteps, fit.beta)
    }
    fitted = fit
  }
  const scoreAt = lazy(() =>
    method === 'fixed' ? NaN : (scoreBeforeShape ?? smoothingCriterion(method, A, family, fitAtS(), S, gamma)),
  )

  // The common start: a constant η at the weighted mean of g(μ₀).
  const eta0 = f64(link.link(family.initialMean(data.y, fromData(w, [n]))) as Tensor)
  let alpha = 0
  let sw = 0
  for (let i = 0; i < n; i++) {
    alpha += w[i] * (eta0[i] - o[i])
    sw += w[i]
  }
  const start = new Float64Array(P)
  start[0] = alpha / sw
  let ybar = 0
  for (let i = 0; i < n; i++) ybar += (w[i] * y[i]) / sw
  const nullDeviance = devianceOf(family, data.y, fromData(new Float64Array(n).fill(ybar), [n]), data.weights)
  const curvatureAt = lazy(() => Math.max(...toFlat(eigh(fromData(fitAtS().H, [P, P])).values)) / n)

  // J(β) with primitives. The data enter as constants; only β is differentiated.
  const Xt = fromData(A.X, [n, P])
  const St = fromData(penalty, [P, P])
  const full = { X: Xt, y: data.y, w: fromData(w, [n]), o: fromData(o, [n]), scale: 1 }
  const batchOf = (rows: ArrayLike<number>) => {
    const b = rows.length
    const Xb = new Float64Array(b * P)
    const yb = new Float64Array(b)
    const wb = new Float64Array(b)
    const ob = new Float64Array(b)
    for (let r = 0; r < b; r++) {
      const i = rows[r]
      Xb.set(A.X.subarray(i * P, (i + 1) * P), r * P)
      yb[r] = y[i]
      wb[r] = w[i]
      ob[r] = o[i]
    }
    const v = (a: F64) => fromData(a, [b])
    return { X: fromData(Xb, [b, P]), y: v(yb), w: v(wb), o: v(ob), scale: n / b }
  }
  const objective = (beta: Value, rows?: ArrayLike<number>): Value => {
    const B = rows ? batchOf(rows) : full
    const eta = add(matmul(B.X, beta), B.o)
    const dev = sum(mul(B.w, lik.unitDeviance(B.y, eta)))
    const pen = sum(mul(beta, matmul(St, beta)))
    return div(add(mul(B.scale, dev), pen), 2 * n)
  }

  const linear = (beta: ArrayLike<number>) => {
    const eta = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let v = o[i]
      for (let a = 0; a < P; a++) v += A.X[i * P + a] * beta[a]
      eta[i] = v
    }
    return eta
  }
  const quadratic = (M: F64, beta: ArrayLike<number>, from = 0, to = P) => {
    let s = 0
    for (let a = from; a < to; a++) for (let b = from; b < to; b++) s += beta[a] * M[a * P + b] * beta[b]
    return s
  }
  const evaluate = (beta: ArrayLike<number>): GamEvaluation => {
    const eta = linear(beta)
    const mu = f64(link.inverse(fromData(eta, [n])) as Tensor)
    const valid = link.total ? eta.every(Number.isFinite) : family.validMean(fromData(mu, [n]))
    const d = f64(lik.unitDeviance(data.y, fromData(eta, [n])) as Tensor)
    let dev = 0
    for (let i = 0; i < n; i++) dev += w[i] * d[i]
    if (!valid) dev = NaN
    const pen = quadratic(penalty, beta)
    const termPenalties = A.terms.map((t, j) => quadratic(S, beta, A.offsets[j], A.offsets[j] + t.size))
    return {
      eta,
      mu,
      valid,
      deviance: dev,
      penalty: pen,
      termPenalties,
      penalisedDeviance: dev + pen,
      objective: (dev + pen) / (2 * n),
    }
  }
  const working = (beta: ArrayLike<number>) => {
    const eta = linear(beta)
    const etaT = fromData(eta, [n])
    const mu = f64(link.inverse(etaT) as Tensor)
    const dmu = f64(link.derivative(etaT) as Tensor)
    const V = f64(family.variance(fromData(mu, [n])) as Tensor)
    // As in `irls`: an observation whose μ has rounded to the edge of the mean space carries no weight.
    const W = Float64Array.from(dmu, (g, i) => {
      const v = (w[i] * g * g) / V[i]
      return Number.isFinite(v) ? v : 0
    })
    return { z: Float64Array.from(eta, (e, i) => (W[i] > 0 ? e - o[i] + (y[i] - mu[i]) / dmu[i] : e - o[i])), W }
  }
  const gradient = (beta: ArrayLike<number>) => {
    const eta = linear(beta)
    const etaT = fromData(eta, [n])
    const mu = f64(link.inverse(etaT) as Tensor)
    const dmu = f64(link.derivative(etaT) as Tensor)
    const V = f64(family.variance(fromData(mu, [n])) as Tensor)
    const g = new Float64Array(P)
    for (let i = 0; i < n; i++) {
      const r = (w[i] * (y[i] - mu[i]) * dmu[i]) / V[i]
      for (let a = 0; a < P; a++) g[a] -= A.X[i * P + a] * r
    }
    for (let a = 0; a < P; a++) {
      let s = 0
      for (let b = 0; b < P; b++) s += penalty[a * P + b] * beta[b]
      g[a] = (g[a] + s) / n
    }
    return g
  }

  return {
    kind: 'gam-problem',
    spec,
    family,
    link,
    design: A,
    data,
    y,
    w,
    o,
    lambdas,
    penalty,
    active: active.map((a) => [...a].sort((p, q) => p - q)),
    smoothing: {
      method,
      evaluations,
      get value() {
        return scoreAt()
      },
    },
    get optimum() {
      return fitAtS()
    },
    start,
    nullDeviance,
    get curvature() {
      return curvatureAt()
    },
    objective,
    gradient,
    evaluate,
    working,
  }
}

/** The smoothing criterion and the EDF along a path of one common log λ (see `smoothingPath`). */
export type SmoothingPath = { logLambdas: number[]; criterion: number[]; edf: number[] }

/**
 * The smoothing criterion (REML, or GCV/UBRE) and the total EDF of a problem's design and data against one common
 * log λ for every selectable penalty (fixed ones kept), at each value of `logLambdas`: the curve whose minimum a common
 * λ would take, and how flexible the fit is there. Each point is a P-IRLS fit warm-started from the last.
 */
export function smoothingPath(
  problem: GamProblem,
  logLambdas: readonly number[],
  method: 'reml' | 'gcv' = 'reml',
): SmoothingPath {
  const A = problem.design
  const maxSteps = problem.spec.maxSteps ?? 50
  let warm: F64 | undefined
  const criterion: number[] = []
  const edf: number[] = []
  for (const l of logLambdas) {
    const S = penaltyMatrix(
      A,
      A.fixed.map((v) => (Number.isNaN(v) ? Math.exp(l) : v)),
    )
    try {
      const fit = penalisedFit(A, problem.data, problem.family, problem.link, S, maxSteps, warm)
      warm = fit.beta
      criterion.push(smoothingCriterion(method, A, problem.family, fit, S, problem.spec.gamma ?? 1))
      edf.push(fit.edf)
    } catch {
      criterion.push(NaN)
      edf.push(NaN)
    }
  }
  return { logLambdas: [...logLambdas], criterion, edf }
}

/** The smoothing criterion alone along a common log λ (`smoothingPath`'s `criterion`). */
export function smoothingProfile(
  problem: GamProblem,
  logLambdas: readonly number[],
  method: 'reml' | 'gcv' = 'reml',
): number[] {
  return smoothingPath(problem, logLambdas, method).criterion
}
