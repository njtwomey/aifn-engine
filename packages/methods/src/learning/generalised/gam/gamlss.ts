/**
 * Generalised additive models for location, scale and shape (GAMLSS; Rigby and Stasinopoulos, 2005, JRSS C 54(3)):
 * $y \sim \mathcal{D}(\theta_1, \dots, \theta_K)$ with $g_k(\thetavec_k) = \etavec_k = \Xmat_k\betavec_k$ for
 * every parameter, each predictor an intercept plus GAM terms (`s`, `linearTerm`, ... on the GAM bases and penalties
 * of `gamDesign`), fitted by maximising the penalised log-likelihood
 * $\ell - \tfrac12 \sum_k \lambda_k \betavec_k^\top\Smat_k\betavec_k$.
 *
 * The RS algorithm (`gamlssRs`, one step = one cycle) updates the parameters in turn, each by penalised IRLS with the
 * others held fixed: with $u = \partial\ell/\partial\eta_k = (\partial\ell/\partial\theta_k)(d\theta_k/d\eta_k)$
 * and $w = -\expect[\partial^2\ell/\partial\theta_k^2](d\theta_k/d\eta_k)^2$, it solves
 * $(\Xmat_k^\top\Wmat\Xmat_k + \Smat_\lambda)\betavec = \Xmat_k^\top\Wmat\zvec$ for the working response
 * $z = \eta_k + u/w$, halving a step that raises the penalised global deviance, until that deviance settles; then it
 * moves to the next parameter. The cycle ends with the global deviance $\text{GD} = -2\ell$. Smoothing parameters are
 * fixed or chosen by local maximum likelihood: after each parameter's inner fit,
 * $\lambda \leftarrow (r - \lambda \trace(\Hmat^{-1}\Smat))/(\betavec^\top\Smat\betavec)$ ($r = \rank\Smat$),
 * the Fellner–Schall fixed point of the working model's marginal likelihood (Wood and Fasiolo, 2017), which Rigby and
 * Stasinopoulos's local ML estimates. The new $\lambda$ is used from the next cycle on.
 *
 * Outputs: parameter curves, centile curves, normalised quantile residuals and a worm plot (`gamlssModel`), and the
 * generalised AIC $\text{GAIC}(k) = \text{GD} + k \cdot \text{df}$ with df the summed effective degrees of freedom.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import {
  distributionalFamily,
  distributionalLinks,
  quantileResidual,
  wormPlot,
  type DistributionalFamily,
  type DistributionalFamilyName,
  type DistributionalParameter,
  type Link,
  type LinkName,
  type WormPlot,
} from 'aifn-compute/probability/likelihoods'
import { nullSpaceDimension, penalisedInference, penaltyMatrix } from '../smoothing'
import { gamDesign, gamDesignAt, type GamDesign } from './problem'
import type { TermSpec } from './terms'

type F64 = Float64Array

/** One parameter's predictor: GAM `terms` (default none: a constant) and its `link` (default the family's). */
export type GamlssParameterSpec = { terms?: readonly TermSpec[]; link?: LinkName }

/** How the smoothing parameters are chosen. */
export type GamlssSmoothing = 'fixed' | 'local-ml'

/** What defines a GAMLSS problem. */
export type GamlssSpec = {
  /** The distribution: a distributional family or its name (`'normal'`, `'student-t'`, `'gamma'`, ...). */
  family: DistributionalFamily | DistributionalFamilyName
  /** One predictor per parameter name; a parameter left out is a constant. */
  parameters?: Partial<Record<DistributionalParameter, GamlssParameterSpec>>
  /** `local-ml` (default) or `fixed` (each term's own `lambda`, default 1). */
  smoothing?: GamlssSmoothing
  /** Most inner IRLS iterations per parameter per cycle (default 20). */
  maxInner?: number
  /** A cycle has converged when the global deviance changes by less than this (default 1e-3, gamlss's c.crit). */
  tolerance?: number
}

/** The data: features `x` ($n \times d$) and responses `y` ($n$). */
export type GamlssData = { x: Tensor; y: Tensor }

/** A GAMLSS problem: the family, one link and design per parameter, and the data. */
export type GamlssProblem = {
  /** Tags a GAMLSS problem. */
  readonly kind: 'gamlss-problem'
  /** The specification it was built from. */
  readonly spec: GamlssSpec
  /** The distributional family. */
  readonly family: DistributionalFamily
  /** One link per parameter, in the family's parameter order. */
  readonly links: readonly Link[]
  /** One design per parameter, in the family's parameter order. */
  readonly designs: readonly GamDesign[]
  /** The responses ($n$). */
  readonly y: F64
  /** The number of observations. */
  readonly n: number
  /** The rank of each penalty of each parameter's design. */
  readonly ranks: readonly (readonly number[])[]
  /** Starting $\lambda$ per parameter and penalty: the term's own `lambda`, or 1. */
  readonly lambdas: readonly (readonly number[])[]
}

/** One RS state: after `t` cycles. */
export type GamlssState = Status & {
  /** Cycles done. */
  t: number
  /** $\betavec_k$ per parameter. */
  coefficients: F64[]
  /** $\etavec_k$ per parameter ($n$ each). */
  eta: F64[]
  /** $\thetavec_k = g_k^{-1}(\etavec_k)$ per parameter ($n$ each). */
  theta: F64[]
  /**
   * $\lambda$ per parameter and penalty: with `fixed` smoothing, those of this cycle's fits; with `local-ml`, the
   * updated values the next cycle will use (`penalisedDeviance` is at the $\lambda$ this cycle used).
   */
  lambdas: number[][]
  /** Global deviance $-2\ell$. */
  deviance: number
  /** The global deviance plus the penalties $\sum \lambda \betavec^\top\Smat\betavec$. */
  penalisedDeviance: number
  /**
   * Effective degrees of freedom per parameter, $\trace(\Hmat_k^{-1}\Xmat_k^\top\Wmat\Xmat_k)$ (at cycle 0, the
   * number of columns of each design).
   */
  edf: number[]
  /** Inner IRLS iterations per parameter in this cycle. */
  inner: number[]
  /** Whether the global deviance changed by less than the tolerance in this cycle. */
  converged: boolean
  /** Whether the global deviance is no longer finite. */
  diverged: boolean
}

/**
 * The distributional family by name, or as given.
 *
 * @param f A family or its name.
 * @returns The family.
 */
const asFamily = (f: GamlssSpec['family']) => (typeof f === 'string' ? distributionalFamily(f) : f)
/**
 * Treat a value as a number: a link applied to a number returns one, though it is typed for tensors too.
 *
 * @param v The value.
 * @returns The same value, typed as a number.
 */
const num = (v: unknown) => v as number

/**
 * Build the designs, links and penalties of a GAMLSS: one GAM design per parameter of the family (an intercept alone
 * for a parameter left out). Throws `ShapeError` when `x` and `y` differ in length, and `DomainError` for a parameter
 * the family does not have.
 *
 * @param spec The family, each parameter's terms and link, and the smoothing and convergence settings.
 * @param data The features and responses.
 * @returns The problem.
 *
 * @example A normal with a linear mean and a linear log standard deviation
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [60, 1] })
 * const y = add(reshape(x, [60]), mul(add(0.2, mul(0.8, reshape(x, [60]))), normals(r, 60)))
 * const spec = { family: 'normal', parameters: { mu: { terms: [linearTerm(0)] }, sigma: { terms: [linearTerm(0)] } } }
 * const problem = gamlssProblem(spec, { x, y })
 * print('parameters =', problem.family.parameters.map((p) => p.name))
 * print('links =', problem.links.map((l) => l.name), 'columns =', problem.designs.map((A) => A.P))
 */
export function gamlssProblem(spec: GamlssSpec, data: GamlssData): GamlssProblem {
  const family = asFamily(spec.family)
  const y = Float64Array.from(toFlat(data.y))
  const n = y.length
  if (data.x.shape[0] !== n) throw new ShapeError('gamlss', `gamlss: ${data.x.shape[0]} rows of x, ${n} responses`)
  const parameters = spec.parameters ?? {}
  for (const name of Object.keys(parameters))
    if (!family.parameters.some((p) => p.name === name))
      throw new DomainError('gamlss', `gamlss: ${family.abbreviation} has no parameter ${name}`)
  const links = distributionalLinks(
    family,
    Object.fromEntries(Object.entries(parameters).flatMap(([k, v]) => (v?.link ? [[k, v.link]] : []))),
  )
  const designs = family.parameters.map((p) => gamDesign(parameters[p.name]?.terms ?? [], data.x))
  const ranks = designs.map((A) => A.penalties.map((pen) => A.P - nullSpaceDimension([pen], A.P)))
  const lambdas = designs.map((A) => A.fixed.map((l) => (Number.isNaN(l) ? 1 : l)))
  return { kind: 'gamlss-problem', spec, family, links, designs, y, n, ranks, lambdas }
}

/**
 * The linear predictor $\Xmat\betavec$ of a design.
 *
 * @param A The design, whose model matrix $\Xmat$ ($n \times P$) is read.
 * @param beta $\betavec$, $P$ values.
 * @returns $\Xmat\betavec$, $n$ values.
 */
function matVec(A: GamDesign, beta: F64): F64 {
  const out = new Float64Array(A.n)
  for (let i = 0; i < A.n; i++) {
    let s = 0
    for (let a = 0; a < A.P; a++) s += A.X[i * A.P + a] * beta[a]
    out[i] = s
  }
  return out
}

/**
 * Solve $(\Xmat^\top\Wmat\Xmat + \Smat)\betavec = \Xmat^\top\Wmat\zvec$ by Cholesky (with jitter if needed).
 *
 * @param A The design, whose model matrix $\Xmat$ ($n \times P$) is read.
 * @param W The working weights, the diagonal of $\Wmat$ ($n$ values; zeros drop their rows).
 * @param z The working response $\zvec$ ($n$ values).
 * @param S The penalty $\Smat$, $P \times P$ row-major, already weighted by $\lambda$; not modified.
 * @returns $\betavec$, $P$ values.
 */
function penalisedSolve(A: GamDesign, W: ArrayLike<number>, z: ArrayLike<number>, S: F64): F64 {
  const { P, n, X } = A
  const M = Float64Array.from(S)
  const b = new Float64Array(P)
  for (let i = 0; i < n; i++) {
    const wi = W[i]
    if (wi === 0) continue
    for (let a = 0; a < P; a++) {
      const xa = X[i * P + a] * wi
      if (xa === 0) continue
      b[a] += xa * z[i]
      for (let c = 0; c < P; c++) M[a * P + c] += xa * X[i * P + c]
    }
  }
  const { L } = cholesky(fromData(M, [P, P]))
  return Float64Array.from(toFlat(choleskySolve(L, fromData(b, [P])) as Tensor))
}

/**
 * The quadratic form $\betavec^\top\Smat\betavec$.
 *
 * @param beta $\betavec$, $P$ values.
 * @param S $\Smat$, $P \times P$ row-major.
 * @param P The number of coefficients.
 * @returns $\betavec^\top\Smat\betavec$.
 */
const quad = (beta: F64, S: F64, P: number) => {
  let s = 0
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) s += beta[a] * S[a * P + b] * beta[b]
  return s
}

/**
 * $\thetavec_k = g_k^{-1}(\etavec_k)$ elementwise.
 *
 * @param lk The parameter's link $g_k$.
 * @param eta $\etavec_k$, $n$ values.
 * @returns $\thetavec_k$, $n$ values.
 */
const inverse = (lk: Link, eta: F64) => Float64Array.from(eta, (e) => num(lk.inverse(e)))

/**
 * The global deviance $-2\ell$ at per-parameter $\thetavec$ arrays, or NaN when some $\thetavec$ leaves the
 * parameter space or the log-likelihood is not finite.
 *
 * @param problem The problem, whose family and responses are read.
 * @param theta One array of $n$ values per parameter, in the family's order.
 * @returns $-2\ell$, or NaN.
 */
function globalDeviance(problem: GamlssProblem, theta: readonly F64[]): number {
  const { family, y, n } = problem
  const th = new Array<number>(theta.length)
  let s = 0
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < theta.length; k++) th[k] = theta[k][i]
    if (!family.valid(th)) return NaN
    s += family.logPdf(y[i], th)
  }
  return Number.isFinite(s) ? -2 * s : NaN
}

/**
 * The working response and weights of parameter $k$ (see the module comment). An observation whose weight is not
 * finite and positive gets weight 0 and $z = \eta$.
 *
 * @param problem The problem, whose family, responses and links are read.
 * @param k The parameter's index, in the family's order.
 * @param eta $\etavec_k$, $n$ values.
 * @param theta Every parameter's $\thetavec$, $n$ values each.
 * @returns `z` and `W`, $n$ values each.
 */
function working(problem: GamlssProblem, k: number, eta: F64, theta: readonly F64[]) {
  const { family, y, n } = problem
  const lk = problem.links[k]
  const z = new Float64Array(n)
  const W = new Float64Array(n)
  const th = new Array<number>(theta.length)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < theta.length; j++) th[j] = theta[j][i]
    const d = num(lk.derivative(eta[i]))
    const w = -family.expectedSecond(k, th) * d * d
    W[i] = Number.isFinite(w) && w > 0 ? w : 0
    z[i] = W[i] > 0 ? eta[i] + (family.score(k, y[i], th) * d) / W[i] : eta[i]
  }
  return { z, W }
}

/**
 * The RS algorithm as a traceable algorithm: `init` projects each parameter's starting values (the family's
 * `initial`, on the link scale) onto its predictor by a penalised least-squares fit; each `step` is one cycle over the
 * parameters (see the module comment). It is converged when the global deviance changes by less than the
 * specification's `tolerance` over a cycle.
 *
 * @param problem The problem.
 * @returns The algorithm.
 *
 * @example The global deviance falls over a few cycles
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [60, 1] })
 * const y = add(reshape(x, [60]), mul(add(0.2, mul(0.8, reshape(x, [60]))), normals(r, 60)))
 * const spec = { family: 'normal', parameters: { mu: { terms: [linearTerm(0)] }, sigma: { terms: [linearTerm(0)] } } }
 * const problem = gamlssProblem(spec, { x, y })
 * for (const st of trace(gamlssRs(problem), undefined, 20).steps) print('cycle', st.t, 'GD =', st.deviance)
 */
export function gamlssRs(problem: GamlssProblem): Algorithm<undefined, GamlssState> {
  const { designs, links, family } = problem
  const K = designs.length
  const maxInner = problem.spec.maxInner ?? 20
  const tol = problem.spec.tolerance ?? 1e-3
  const localMl = (problem.spec.smoothing ?? 'local-ml') === 'local-ml'
  const penalties = (lambdas: number[][]) => designs.map((A, k) => penaltyMatrix(A, lambdas[k]))
  const totalPenalty = (betas: F64[], S: F64[]) => betas.reduce((s, b, k) => s + quad(b, S[k], designs[k].P), 0)

  return {
    name: 'gamlss-rs',
    init: () => {
      const lambdas = problem.lambdas.map((l) => [...l])
      const S = penalties(lambdas)
      const start = family.initial(problem.y)
      const coefficients = designs.map((A, k) => {
        const eta0 = Float64Array.from(start[k], (v) => num(links[k].link(v)))
        return penalisedSolve(A, new Float64Array(A.n).fill(1), eta0, S[k])
      })
      const eta = coefficients.map((b, k) => matVec(designs[k], b))
      const theta = eta.map((e, k) => inverse(links[k], e))
      const deviance = globalDeviance(problem, theta)
      return {
        t: 0,
        coefficients,
        eta,
        theta,
        lambdas,
        deviance,
        penalisedDeviance: deviance + totalPenalty(coefficients, S),
        edf: designs.map((A) => A.P),
        inner: designs.map(() => 0),
        converged: false,
        diverged: false,
      }
    },
    step: (state) => {
      const coefficients: F64[] = state.coefficients.map((b) => Float64Array.from(b))
      const eta: F64[] = state.eta.map((e) => Float64Array.from(e))
      const theta = state.theta.map((e) => Float64Array.from(e))
      const lambdas = state.lambdas.map((l) => [...l])
      const inner = new Array<number>(K).fill(0)
      const edf = [...state.edf]
      const next = [...lambdas.map((l) => [...l])]
      for (let k = 0; k < K; k++) {
        const A = designs[k]
        const S = penalties(lambdas)
        let pdev = globalDeviance(problem, theta) + totalPenalty(coefficients, S)
        let W = new Float64Array(A.n)
        for (let it = 0; it < maxInner; it++) {
          const work = working(problem, k, eta[k], theta)
          W = work.W
          const old = coefficients[k]
          let beta = penalisedSolve(A, work.W, work.z, S[k])
          let accepted = false
          for (let h = 0; h <= 20; h++) {
            const e = matVec(A, beta)
            const trial = theta.map((t, j) => (j === k ? inverse(links[k], e) : t))
            const betas = coefficients.map((b, j) => (j === k ? beta : b))
            const p = globalDeviance(problem, trial) + totalPenalty(betas, S)
            if (Number.isFinite(p) && (!Number.isFinite(pdev) || p <= pdev + 1e-10 * Math.abs(pdev))) {
              const change = Math.abs(pdev - p)
              coefficients[k] = beta
              eta[k] = e
              theta[k] = trial[k]
              accepted = true
              inner[k] = it + 1
              const settled = Number.isFinite(pdev) && change < 1e-8 * (Math.abs(p) + 1)
              pdev = p
              if (settled) it = maxInner
              break
            }
            beta = Float64Array.from(beta, (b, j) => (b + old[j]) / 2)
          }
          if (!accepted) break
        }
        // The working weights at the fitted β give its EDF and the local-ML update of λ (for the next cycle).
        W = working(problem, k, eta[k], theta).W
        const inf = penalisedInference(A, W, S[k], coefficients[k])
        edf[k] = inf.edf
        if (localMl)
          A.penalties.forEach(({ S: Sj }, j) => {
            if (!Number.isNaN(A.fixed[j])) return
            let trHS = 0
            for (let a = 0; a < A.P; a++) for (let b = 0; b < A.P; b++) trHS += inf.Hinv[a * A.P + b] * Sj[b * A.P + a]
            const bSb = quad(coefficients[k], Sj, A.P)
            const l = lambdas[k][j]
            const proposed = (problem.ranks[k][j] - l * trHS) / Math.max(bSb, 1e-300)
            next[k][j] = Math.min(1e10, Math.max(1e-8, proposed))
          })
      }
      const S = penalties(lambdas)
      const deviance = globalDeviance(problem, theta)
      return {
        t: state.t + 1,
        coefficients,
        eta,
        theta,
        lambdas: localMl ? next : lambdas,
        deviance,
        penalisedDeviance: deviance + totalPenalty(coefficients, S),
        edf,
        inner,
        converged: Math.abs(deviance - state.deviance) < tol,
        diverged: !Number.isFinite(deviance),
      }
    },
  }
}

/**
 * Trace the RS algorithm for at most `maxCycles` cycles, keeping every state and recording the deviance, the
 * penalised deviance and the total EDF (`df`).
 *
 * @param problem The problem.
 * @param maxCycles The most cycles; the run stops earlier when it converges.
 * @returns The trace.
 *
 * @example A smooth for the standard deviation: local ML shrinks it to its linear null space
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [60, 1] })
 * const y = add(reshape(x, [60]), mul(add(0.2, mul(0.8, reshape(x, [60]))), normals(r, 60)))
 * const problem = gamlssProblem({ family: 'normal', parameters: { sigma: { terms: [s(0, { k: 6 })] } } }, { x, y })
 * const { final } = gamlssTrace(problem)
 * print('cycles =', final.t, 'GD =', final.deviance, 'converged =', final.converged)
 * print('edf per parameter =', final.edf)
 */
export function gamlssTrace(problem: GamlssProblem, maxCycles = 50): Trace<GamlssState> {
  return trace(gamlssRs(problem), undefined, maxCycles, {
    record: {
      deviance: (s) => s.deviance,
      penalisedDeviance: (s) => s.penalisedDeviance,
      df: (s) => s.edf.reduce((a, b) => a + b, 0),
    },
  })
}

/** A fitted GAMLSS at one RS state. */
export type GamlssModel = {
  /** Tags a fitted GAMLSS. */
  readonly kind: 'gamlss-model'
  /** The distributional family. */
  readonly family: DistributionalFamily
  /** The problem it solves. */
  readonly problem: GamlssProblem
  /** The RS state it is the model of. */
  readonly state: GamlssState
  /** Global deviance $-2\ell$. */
  readonly deviance: number
  /** EDF per parameter. */
  readonly edf: readonly number[]
  /** The total EDF. */
  readonly df: number
  /** $\text{GD} + k \cdot \text{df}$ ($k = 2$, the default: AIC; $k = \log n$: SBC). */
  gaic(k?: number): number
  /** $\thetavec_k$ at new inputs ($m \times d$), one array of $m$ values per parameter. */
  parameters(x: Tensor): F64[]
  /** The $p$-centile curve at the inputs for each level $p \in (0, 1)$: one array of $m$ values per level. */
  centiles(x: Tensor, levels: readonly number[]): F64[]
  /** Normalised quantile residuals $\Phi^{-1}(F(y_i \mid \hat{\thetavec}_i))$ ($n$). */
  residuals(): F64
  /** The worm plot of the residuals. */
  worm(): WormPlot
}

/**
 * The model at an RS state: parameter and centile curves, quantile residuals, the worm plot and GAIC.
 *
 * @param problem The problem.
 * @param state The RS state (default the final state of `gamlssTrace(problem)`, at most 50 cycles).
 * @returns The model.
 *
 * @example The spread grows with x, and modelling it lowers the AIC
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [60, 1] })
 * const y = add(reshape(x, [60]), mul(add(0.2, mul(0.8, reshape(x, [60]))), normals(r, 60)))
 * const spec = { family: 'normal', parameters: { mu: { terms: [linearTerm(0)] }, sigma: { terms: [linearTerm(0)] } } }
 * const model = gamlssModel(gamlssProblem(spec, { x, y }))
 * const grid = tensor([[0], [0.5], [1]])
 * const [mu, sigma] = model.parameters(grid)
 * print('mu =', mu, 'sigma =', sigma)
 * print('10% and 90% centiles =', model.centiles(grid, [0.1, 0.9]))
 * const constant = gamlssModel(gamlssProblem({ ...spec, parameters: { mu: spec.parameters.mu } }, { x, y }))
 * print('AIC =', model.gaic(), 'with a constant sigma:', constant.gaic())
 */
export function gamlssModel(problem: GamlssProblem, state?: GamlssState): GamlssModel {
  const s = state ?? gamlssTrace(problem).final
  const { family, links, designs } = problem
  const df = s.edf.reduce((a, b) => a + b, 0)
  const parameters = (x: Tensor) =>
    designs.map((A, k) => {
      const rows = gamDesignAt(A, x)
      const m = rows.length / A.P
      const out = new Float64Array(m)
      for (let i = 0; i < m; i++) {
        let e = 0
        for (let a = 0; a < A.P; a++) e += rows[i * A.P + a] * s.coefficients[k][a]
        out[i] = num(links[k].inverse(e))
      }
      return out
    })
  let cached: F64 | null = null
  const residuals = () =>
    (cached ??= Float64Array.from(problem.y, (yi, i) =>
      quantileResidual(
        family,
        yi,
        s.theta.map((t) => t[i]),
      ),
    ))
  return {
    kind: 'gamlss-model',
    family,
    problem,
    state: s,
    deviance: s.deviance,
    edf: s.edf,
    df,
    gaic: (k = 2) => s.deviance + k * df,
    parameters,
    centiles: (x, levels) => {
      const th = parameters(x)
      const m = th[0].length
      return levels.map((p) => {
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++)
          out[i] = family.quantile(
            p,
            th.map((t) => t[i]),
          )
        return out
      })
    },
    residuals,
    worm: () => wormPlot(residuals()),
  }
}
