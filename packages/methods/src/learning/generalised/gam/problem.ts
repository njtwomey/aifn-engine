/**
 * The penalised-likelihood problem of a GAM, separated from how it is solved. `gamProblem(spec, data)` builds the
 * model matrix $\Xmat = [\ones, \Bmat_1, \dots, \Bmat_J]$ and the penalties from the terms, chooses the smoothing
 * parameters (REML, GCV/UBRE or fixed; Wood, 2017, "Generalized Additive Models", 2nd ed., §6.2) and the active
 * shape-constraint rows, and solves it once by penalised IRLS for the reference optimum. Every fitter of `fitters.ts`
 * then minimises the same objective
 *
 * $$
 * J(\betavec) = \frac{D(\betavec) + \betavec^\top\Smat_\lambda\betavec}{2n}, \qquad
 * D(\betavec) = \sum_i w_i \, d\big(y_i, g^{-1}(\xvec_i^\top\betavec + o_i)\big),
 * $$
 *
 * the penalised deviance scaled per observation, with $\Smat_\lambda = \sum_k \lambda_k\Smat_k$ plus any
 * shape-constraint penalty (for a family with known dispersion $\phi$, $D/2\phi$ is the negative log-likelihood up to
 * a constant, so $J$ is the penalised negative log-likelihood per observation). `objective` writes $J$ (or its
 * minibatch estimate) with primitives, so autodiff gives its gradient; `gradient` is the closed form
 * $\nabla J = (\Smat_\lambda\betavec - \Xmat^\top\rvec)/n$ with
 * $r_i = w_i (y_i - \mu_i) \mu'(\eta_i) / V(\mu_i)$.
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
/**
 * A memoised thunk: `f` runs on the first call only, and every call returns its value.
 *
 * @param f The computation to defer.
 * @returns A function that returns `f()`, computed once.
 */
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
/**
 * A tensor's values, flattened in row-major order, as a fresh `Float64Array`.
 *
 * @param t The tensor to copy; it is not modified.
 * @returns Its entries in row-major order.
 */
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/**
 * Data for a GAM: features `x` ($n \times d$; a factor's column holds its level codes), responses `y` ($n$), and
 * optional prior `weights` ($n$, default 1) and `offset` ($n$, added to the linear predictor; default 0).
 */
export type GamData = { x: Tensor; y: Tensor; weights?: Tensor; offset?: Tensor }

/** How smoothing parameters are chosen. */
export type SmoothingMethod = 'reml' | 'gcv' | 'fixed'

/** What defines a GAM's penalised problem: the terms, the likelihood and how $\lambda$ is chosen. */
export type GamSpec = {
  /** The terms of the predictor, after the intercept (`s`, `te`, `linearTerm`, ...). */
  terms: readonly TermSpec[]
  /** A family, or its name (default Gaussian). */
  family?: Family | FamilyName
  /** Default the family's default link; a link the family does not take is rejected (`checkLink`). */
  link?: LinkName | Link
  /** `reml` (default), `gcv` (GCV when $\phi$ is estimated, UBRE when it is known) or `fixed`. */
  method?: SmoothingMethod
  /** With `fixed`: $\lambda$ for every penalty whose term does not fix its own (default 1). */
  lambda?: number
  /**
   * One $\lambda$ per penalty, in term order, overriding the terms' own: skips the search (a problem reports its
   * choice in `lambdas`). A count other than the number of penalties throws `ShapeError`.
   */
  lambdas?: readonly number[]
  /** GCV/UBRE inflation $\gamma \ge 1$ of the EDF, against overfitting (default 1). */
  gamma?: number
  /** Most P-IRLS steps per fit (default 50). */
  maxSteps?: number
  /** Most Nelder–Mead steps of the smoothing-parameter search (default 200). */
  maxSearchSteps?: number
  /**
   * Weight of the shape-constraint penalty relative to the mean diagonal of $\Xmat^\top\Wmat\Xmat$ at the
   * unconstrained fit (default 1e6).
   */
  constraintWeight?: number
  /** Active shape-constraint rows per term (a problem reports them in `active`): skips the active-set refits. */
  active?: readonly (readonly number[])[]
}

/** The model matrix, penalties and fixed data of a GAM (a `PenalisedDesign` with its terms). */
export type GamDesign = Omit<PenalisedDesign, 'penalties'> & {
  /** The built terms, in model order. */
  terms: BuiltTerm[]
  /** Column offset of each term's block (column 0 is the intercept). */
  offsets: number[]
  /** Columns of the model matrix: 1 plus every term's `size`. */
  P: number
  /** Training rows. */
  n: number
  /** Features. */
  d: number
  /** The model matrix ($n \times P$), row-major. */
  X: F64
  /** Each penalty embedded in $P \times P$ (row-major, unweighted by $\lambda$), with the term it belongs to. */
  penalties: { S: F64; term: number }[]
  /** Fixed $\lambda$ per penalty (NaN: selected). */
  fixed: number[]
  /** Dimension of the unpenalised space (intercept and null spaces). */
  nullSpace: number
}

/**
 * Build the terms on the training features and assemble the model matrix $\Xmat = [\ones, \Bmat_1\Zmat_1, \dots]$
 * and the penalties, each embedded at its term's block. Throws `ShapeError` when `x` is not a matrix.
 *
 * @param specs The term specifications, in model order.
 * @param x The training features ($n \times d$).
 * @returns The design: built terms, block offsets, $\Xmat$, embedded penalties and fixed $\lambda$.
 *
 * @example A smooth and a factor: an intercept, then one block per term
 * const x = tensor([[0.1, 0], [0.4, 1], [0.6, 0], [0.9, 1], [0.3, 2], [0.7, 2]])
 * const A = gamDesign([s(0, { k: 5 }), factorTerm(1)], x)
 * print('labels =', A.terms.map((t) => t.label))
 * print('P =', A.P, 'offsets =', A.offsets, 'penalties =', A.penalties.length)
 * print('column sums =', sum(fromData(A.X, [6, A.P]), 0))
 */
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

/**
 * Rows of the model matrix at new inputs, with the training data's constraint maps and centring. Throws `ShapeError`
 * when `x` has a different number of features from the training data.
 *
 * @param A The design built on the training data.
 * @param x The new inputs ($m \times d$).
 * @returns The model-matrix rows ($m \times P$), row-major.
 */
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

/** The objective and fit quantities at one $\betavec$. */
export type GamEvaluation = {
  /** The linear predictor $\etavec = \Xmat\betavec + \ovec$ ($n$). */
  eta: F64
  /** The mean $\muvec = g^{-1}(\etavec)$ ($n$). */
  mu: F64
  /** Whether every $\mu_i$ lies in the family's mean space (else the deviance is NaN). */
  valid: boolean
  /** The deviance $D(\betavec) = \sum_i w_i d(y_i, \mu_i)$. */
  deviance: number
  /** $\betavec^\top\Smat_\lambda\betavec$, the shape-constraint penalty included. */
  penalty: number
  /** Each term's part of the $\lambda$-weighted penalty, without the shape-constraint penalty. */
  termPenalties: number[]
  /** $D + \betavec^\top\Smat_\lambda\betavec$. */
  penalisedDeviance: number
  /** $J(\betavec) = (D + \betavec^\top\Smat_\lambda\betavec)/(2n)$. */
  objective: number
}

/** A GAM's penalised problem with its smoothing parameters chosen (see the module comment). */
export type GamProblem = {
  /** Tags a GAM problem. */
  readonly kind: 'gam-problem'
  /** The specification it was built from. */
  readonly spec: GamSpec
  /** The response family. */
  readonly family: Family
  /** The link, checked against the family. */
  readonly link: Link
  /** The model matrix and penalties. */
  readonly design: GamDesign
  /** The data it was built on. */
  readonly data: GamData
  /** The responses $\yvec$ ($n$). */
  readonly y: F64
  /** The prior weights ($n$; 1 when the data have none). */
  readonly w: F64
  /** The offset ($n$; 0 when the data have none). */
  readonly o: F64
  /** One $\lambda$ per penalty, in term order. */
  readonly lambdas: number[]
  /** $\Smat_\lambda = \sum_k \lambda_k\Smat_k$ plus the shape-constraint penalty ($P \times P$, row-major). */
  readonly penalty: F64
  /** Active shape-constraint rows per term. */
  readonly active: number[][]
  /** The smoothing criterion at the chosen $\lambda$ (NaN when fixed) and how many fits the search took. */
  readonly smoothing: { method: SmoothingMethod; value: number; evaluations: number }
  /** The P-IRLS solution at the chosen penalty: the reference optimum every fitter should reach. */
  readonly optimum: PenalisedFit
  /**
   * The common start $\betavec_0$: the intercept at the weighted mean of $g(\mu_0) - o$ for the family's starting
   * mean $\mu_0$, 0 elsewhere.
   */
  readonly start: F64
  /** The deviance of the intercept-only model ($\mu$ the weighted mean of $\yvec$). */
  readonly nullDeviance: number
  /**
   * The largest eigenvalue $L$ of $\nabla^2 J = \Hmat/n$ at the optimum: a fixed gradient step is stable near it
   * below $2/L$.
   */
  readonly curvature: number
  /**
   * $J(\betavec)$ with primitives (differentiable in $\betavec$); with `rows`, the unbiased minibatch estimate on
   * those rows.
   */
  objective(beta: Value, rows?: ArrayLike<number>): Value
  /** $\nabla J(\betavec)$ in closed form ($P$). */
  gradient(beta: ArrayLike<number>): F64
  /** The fit quantities at $\betavec$. */
  evaluate(beta: ArrayLike<number>): GamEvaluation
  /**
   * The working response $z = \eta - o + (y - \mu)/\mu'(\eta)$ and weights $W = w\mu'(\eta)^2/V(\mu)$ at
   * $\betavec$ ($n$ each). A non-finite weight is set to 0, and $z = \eta - o$ there.
   */
  working(beta: ArrayLike<number>): { z: F64; W: F64 }
}

/**
 * The family by name or as given (Gaussian when left out), and its link checked against it (the family's default
 * when left out). A link the family does not take throws `DomainError`.
 *
 * @param spec The `family` and `link` of a specification.
 * @param where The caller's name for error messages.
 * @returns The family and the link.
 *
 * @example A Poisson GAM takes the log link unless told otherwise
 * const { family, link } = resolveLikelihood({ family: 'poisson' })
 * print(family.name, 'with the', link.name, 'link')
 * print('binomial with', resolveLikelihood({ family: 'binomial', link: 'probit' }).link.name)
 *
 * @example A link the family does not take is rejected
 * try {
 *   resolveLikelihood({ family: 'poisson', link: 'logit' })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function resolveLikelihood(
  spec: Pick<GamSpec, 'family' | 'link'>,
  where = 'gam',
): { family: Family; link: Link } {
  const family = typeof spec.family === 'string' ? familyByName(spec.family) : (spec.family ?? familyByName('gaussian'))
  return { family, link: checkLink(family, spec.link, where) }
}

/**
 * The penalised problem of a GAM on data: the design, $\lambda$ chosen by `spec.method` (Nelder–Mead on
 * $\log\lambda$ from 0, kept in $[-15, 15]$, over the REML or GCV/UBRE criterion, each evaluation a warm-started
 * P-IRLS fit), the active shape-constraint rows (a heavy penalty on violated coefficient differences, added until none
 * is violated; pyGAM's scheme, Servén and Brummitt, 2018) and the P-IRLS optimum. The optimum, the criterion and the
 * curvature are computed on first use. Throws `ShapeError` when `y` and `x` differ in length, and `DomainError` for a
 * link the family does not take.
 *
 * @param spec The terms, family, link and smoothing-parameter choice (see `GamSpec`).
 * @param data The features, responses, and optional weights and offset.
 * @returns The problem, with its chosen $\lambda$, objective, gradient and reference optimum.
 *
 * @example REML and GCV choose similar smoothness for a noisy sine
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const reml = gamProblem({ terms: [s(0)] }, { x, y })
 * const gcv = gamProblem({ terms: [s(0)], method: 'gcv' }, { x, y })
 * print('REML: lambda =', reml.lambdas, 'edf =', reml.optimum.edf)
 * print('GCV: lambda =', gcv.lambdas, 'edf =', gcv.optimum.edf)
 *
 * @example The objective at the common start and at the optimum
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * print('J(start) =', problem.evaluate(problem.start).objective)
 * print('J(optimum) =', problem.evaluate(problem.optimum.beta).objective)
 * print('gradient norm there =', Math.hypot(...problem.gradient(problem.optimum.beta)))
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

/**
 * The smoothing criterion and the EDF along a path of one common $\log\lambda$ (see `smoothingPath`): `logLambdas`
 * as given, and at each the `criterion` and the total `edf` (NaN where the fit failed).
 */
export type SmoothingPath = { logLambdas: number[]; criterion: number[]; edf: number[] }

/**
 * The smoothing criterion (REML, or GCV/UBRE) and the total EDF of a problem's design and data against one common
 * $\log\lambda$ for every selectable penalty (those its terms fix are kept), at each value of `logLambdas`: the curve
 * whose minimum a common $\lambda$ would take, and how flexible the fit is there. Each point is a P-IRLS fit
 * warm-started from the last, without shape constraints; a fit that fails gives NaN.
 *
 * @param problem The problem whose design, data, family and link are used (its own $\lambda$ and method are not).
 * @param logLambdas The values of $\log\lambda$, in the order they are visited.
 * @param method The criterion: `'reml'`, or `'gcv'` (GCV, or UBRE when the family's dispersion is known).
 * @returns The path: the criterion and EDF at each $\log\lambda$.
 *
 * @example REML is lowest near the chosen lambda, and the EDF falls as lambda grows
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * const path = smoothingPath(problem, [-6, -3, 0, 3, 6])
 * print('REML =', path.criterion)
 * print('edf =', path.edf)
 * print('chosen log lambda =', Math.log(problem.lambdas[0]))
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

/**
 * The smoothing criterion alone along a common $\log\lambda$ (`smoothingPath`'s `criterion`).
 *
 * @param problem The problem whose design, data, family and link are used.
 * @param logLambdas The values of $\log\lambda$.
 * @param method The criterion: `'reml'` or `'gcv'`.
 * @returns The criterion at each $\log\lambda$ (NaN where the fit failed).
 *
 * @example The GCV score against log lambda
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * print('GCV =', smoothingProfile(problem, [-6, -3, 0, 3, 6], 'gcv'))
 */
export function smoothingProfile(
  problem: GamProblem,
  logLambdas: readonly number[],
  method: 'reml' | 'gcv' = 'reml',
): number[] {
  return smoothingPath(problem, logLambdas, method).criterion
}
