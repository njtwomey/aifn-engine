/**
 * Expectation propagation for a vector parameter θ ∈ ℝᵈ with a Gaussian prior N(μ₀, Σ₀) and n factors that each depend
 * on one linear projection, fᵢ(θ) = gᵢ(aᵢᵀθ): Gaussian process classification (aᵢ = eᵢ, θ the latent function at the
 * training inputs), Bayesian probit regression (aᵢ = xᵢ), and paired comparisons (aᵢ = e_winner − e_loser). Each site
 * is a scalar Gaussian in aᵢᵀθ, t̃ᵢ(θ) ∝ exp(−½ τ̃ᵢ (aᵢᵀθ)² + ν̃ᵢ aᵢᵀθ), so
 *
 *   q(θ) = N(μ, Σ),  Σ = (Σ₀⁻¹ + Aᵀ T̃ A)⁻¹,  μ = Σ(Σ₀⁻¹μ₀ + Aᵀν̃),  T̃ = diag(τ̃).
 *
 * A site update needs only the marginal of aᵢᵀθ under q, N(aᵢᵀμ, aᵢᵀΣaᵢ): the cavity, the tilted moments (the same
 * scalar `TiltedFn` as `expectationPropagation`, e.g. `probitTilted`) and the new site are the scalar algebra, and the
 * posterior moves by a rank-one update, Σ ← Σ − c (Σaᵢ)(Σaᵢ)ᵀ with c = Δτ̃/(1 + Δτ̃ aᵢᵀΣaᵢ) (Rasmussen & Williams,
 * 2006, Algorithm 3.5; Minka, 2001). At the end of every sweep Σ and μ are recomputed from the sites, which removes
 * the rounding the rank-one updates accumulate, by the push-through form
 *
 *   Σ = Σ₀ − Σ₀Aᵀ M⁻¹ T̃ A Σ₀,  M = I + T̃ A Σ₀ Aᵀ,
 *
 * which needs neither Σ₀⁻¹ (a GP Gram matrix is often numerically singular) nor τ̃ ≥ 0. With α = 1 the run also
 * reports EP's log evidence (Minka 2001, eq. 3.30; R&W eq. 3.65), using |Σ|/|Σ₀| = 1/|M|.
 */

import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { solveDense } from 'aifn-compute/numerics/linalg'
import type { TiltedFn } from './ep'
import type { GaussianMoments } from './gaussian'
import type { Tilted } from './tilted'

type F64 = dense.F64

/** The problem and options of {@link multivariateExpectationPropagation}. */
export interface MvEpOptions {
  /** The Gaussian prior N(μ₀, Σ₀) of θ, d-dimensional. */
  prior: { mean: Vector; covariance: Matrix }
  /** Row i is the projection aᵢ that factor i depends on, [n, d]. Default: the identity (factor i depends on θᵢ). */
  projections?: Matrix
  /** Tilted moments of factor i (raised to `power`) against the cavity of aᵢᵀθ. */
  tilted: TiltedFn
  /** Weight of the old site in each update, in [0, 1). Default 0. */
  damping?: number
  /** α for power EP. Default 1. */
  power?: number
  /** The order sites are visited within a sweep (default 0 … n − 1). */
  order?: readonly number[]
  /** A sweep in which no site parameter moves more than this has converged. Default 1e-8. */
  tolerance?: number
}

/** The state of multivariate EP after `t` site updates. */
export interface MvEpState extends Status {
  /** Site precisions τ̃ and shifts ν̃, [n]. */
  sitePrecision: Tensor
  siteShift: Tensor
  /** log Z of each site's last tilted distribution and that update's cavity (natural parameters of aᵢᵀθ), [n]. */
  siteLogZ: Tensor
  cavityPrecision: Tensor
  cavityShift: Tensor
  /** q(θ) = N(mean, covariance). */
  mean: Vector
  covariance: Matrix
  /** The site updated last (−1 at the start), the cavity of aᵢᵀθ and the tilted moments that set it. */
  site: number
  cavity: GaussianMoments
  tiltedMoments: Tilted
  /** False when the last cavity had non-positive precision or improper tilted moments (the update was skipped). */
  ok: boolean
  sweep: number
  position: number
  /** Largest change of a site parameter in the last update, in this sweep, and in the last full sweep. */
  change: number
  sweepChange: number
  lastSweepChange: number
  /** Updates skipped in this sweep and in total. */
  skipped: number
  totalSkipped: number
  /** EP's log evidence as of the start or the last sweep's end (α = 1 and every site updated; NaN otherwise). */
  logEvidence: number
  converged: boolean
}

/** The model as row-major arrays. */
type Problem = {
  d: number
  n: number
  mu0: F64
  S0: F64
  /** Projections [n, d], or null for the identity. */
  A: F64 | null
  /** Σ₀⁻¹μ₀ (null when μ₀ = 0), for the evidence. */
  priorShift: F64 | null
  /** ½ μ₀ᵀΣ₀⁻¹μ₀. */
  priorQuad: number
}

/** log ∫ exp(−½ τ x² + ν x) dx = ν²/(2τ) + ½ log(2π/τ). */
const logNormaliser = (tau: number, nu: number) => (nu * nu) / (2 * tau) + 0.5 * Math.log((2 * Math.PI) / tau)

/** Σa and aᵀΣa, aᵀμ for projection i. */
function marginal(p: Problem, Sigma: F64, mu: F64, i: number): { Sa: F64; v: number; m: number } {
  const { d, A } = p
  if (A === null) {
    const Sa = Sigma.slice(i * d, (i + 1) * d) as F64
    return { Sa, v: Sigma[i * d + i], m: mu[i] }
  }
  const a = A.subarray(i * d, (i + 1) * d)
  const Sa = dense.matVec(Sigma, a, d, d)
  return { Sa, v: dense.dot(a, Sa), m: dense.dot(a, mu) }
}

/**
 * Σ, μ and log|M| from the sites by the push-through form (module notes). With G = AΣ₀ (n×d) and M = I + T̃ G Aᵀ:
 * Σ = Σ₀ − Gᵀ M⁻¹ T̃ G and μ = μ₀ − Gᵀ M⁻¹ T̃ A μ₀ + Σ Aᵀ ν̃. Null when M is singular.
 */
function refresh(p: Problem, tau: F64, nu: F64): { Sigma: F64; mu: F64; logDetM: number } | null {
  const { d, n, S0, mu0, A } = p
  const G = A === null ? S0 : dense.matMul(A, S0, n, d, d)
  const C = A === null ? S0 : dense.matMul(G, dense.transpose(A, n, d), n, d, n)
  const M = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * n + j] = (i === j ? 1 : 0) + tau[i] * C[i * n + j]
  // Right-hand sides [T̃G | T̃Aμ₀] (n × (d + 1)), solved with one LU of M.
  const Amu0 = A === null ? mu0 : dense.matVec(A, mu0, n, d)
  const rhs = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) rhs[i * (d + 1) + j] = tau[i] * G[i * d + j]
    rhs[i * (d + 1) + d] = tau[i] * Amu0[i]
  }
  const sol = solveDense(M, rhs, n)
  if (sol.x === null) return null
  const X = sol.x
  const Sigma = Float64Array.from(S0) as F64
  const mu = Float64Array.from(mu0) as F64
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < d; c++) {
      let s = 0
      for (let k = 0; k < n; k++) s += G[k * d + r] * X[k * (d + 1) + c]
      Sigma[r * d + c] -= s
    }
    let s = 0
    for (let k = 0; k < n; k++) s += G[k * d + r] * X[k * (d + 1) + d]
    mu[r] -= s
  }
  for (let r = 0; r < d; r++) for (let c = 0; c < d; c++) Sigma[r * d + c] = 0.5 * (Sigma[r * d + c] + Sigma[c * d + r])
  const Atnu = A === null ? nu : dense.matTVec(A, nu, n, d)
  const add = dense.matVec(Sigma, Atnu, d, d)
  for (let r = 0; r < d; r++) mu[r] += add[r]
  return { Sigma, mu, logDetM: sol.logAbsDet }
}

/**
 * EP's log evidence log ∫ p₀(θ) Πᵢ fᵢ(θ) dθ for α = 1: Σᵢ [log Zᵢ + A(cavityᵢ) − A(cavityᵢ · siteᵢ)] + A(q) − A(p₀),
 * A the Gaussian log-normaliser, with A(q) − A(p₀) = ½ μᵀ(Σ₀⁻¹μ₀ + Aᵀν̃) − ½ μ₀ᵀΣ₀⁻¹μ₀ − ½ log|M|. NaN until every site
 * has been updated with a proper cavity.
 */
function logEvidenceOf(p: Problem, mu: F64, nu: F64, tau: F64, logZ: F64, ct: F64, cn: F64, logDetM: number): number {
  const { d, n, A } = p
  const Atnu = A === null ? nu : dense.matTVec(A, nu, n, d)
  let quad = dense.dot(mu, Atnu)
  if (p.priorShift) quad += dense.dot(mu, p.priorShift)
  let total = 0.5 * quad - p.priorQuad - 0.5 * logDetM
  for (let i = 0; i < n; i++) {
    if (!(ct[i] > 0)) return NaN
    total += logZ[i] + logNormaliser(ct[i], cn[i]) - logNormaliser(ct[i] + tau[i], cn[i] + nu[i])
  }
  return total
}

/**
 * Multivariate EP with rank-one sites as a traceable `Algorithm` (module notes): one site update per step, Σ and μ
 * recomputed at the end of every sweep, `converged` once a full sweep moves no site parameter by more than `tolerance`
 * and skips none. Improper cavities are skipped and counted, never hidden. `init` takes optional starting sites (a warm
 * start); by default every site is 1 (τ̃ = ν̃ = 0), so the first sweep is ADF.
 */
export function multivariateExpectationPropagation(
  o: MvEpOptions,
): Algorithm<{ sitePrecision?: Tensor; siteShift?: Tensor } | void, MvEpState> {
  const where = 'multivariateExpectationPropagation'
  const mu0 = Float64Array.from(dense.data(o.prior.mean)) as F64
  const d = mu0.length
  const S0 = Float64Array.from(dense.data(o.prior.covariance)) as F64
  if (o.prior.covariance.shape.length !== 2 || S0.length !== d * d)
    throw new ShapeError(where, `${where}: the prior covariance must be ${d}×${d}`)
  let A: F64 | null = null
  let n = d
  if (o.projections) {
    const [rows, cols] = o.projections.shape
    if (cols !== d) throw new ShapeError(where, `${where}: projections have ${cols} columns for dimension ${d}`)
    A = Float64Array.from(dense.data(o.projections)) as F64
    n = rows
  }
  let priorShift: F64 | null = null
  let priorQuad = 0
  if (mu0.some((v) => v !== 0)) {
    const w = solveDense(S0, mu0, d).x
    if (w === null) throw new ShapeError(where, `${where}: a non-zero prior mean needs an invertible prior covariance`)
    priorShift = w as F64
    priorQuad = 0.5 * dense.dot(mu0, w)
  }
  const p: Problem = { d, n, mu0, S0, A, priorShift, priorQuad }
  const damping = o.damping ?? 0
  if (!(damping >= 0 && damping < 1)) throw new DomainError(where, `${where}: damping must be in [0, 1)`)
  const power = o.power ?? 1
  const order = [...(o.order ?? Array.from({ length: n }, (_, i) => i))]
  const tolerance = o.tolerance ?? 1e-8
  const vecN = (v: Float64Array) => fromData(v, [n])
  const evidence = (r: { mu: F64; logDetM: number }, tau: F64, nu: F64, logZ: F64, ct: F64, cn: F64) =>
    power === 1 ? logEvidenceOf(p, r.mu, nu, tau, logZ, ct, cn, r.logDetM) : NaN

  return {
    name: 'multivariate-expectation-propagation',
    init: (start) => {
      const tau = (
        start?.sitePrecision ? Float64Array.from(dense.data(start.sitePrecision)) : new Float64Array(n)
      ) as F64
      const nu = (start?.siteShift ? Float64Array.from(dense.data(start.siteShift)) : new Float64Array(n)) as F64
      const r = refresh(p, tau, nu)
      const zeros = () => new Float64Array(n) as F64
      return {
        t: 0,
        sitePrecision: vecN(tau),
        siteShift: vecN(nu),
        siteLogZ: vecN(zeros()),
        cavityPrecision: vecN(zeros()),
        cavityShift: vecN(zeros()),
        mean: fromData(r ? r.mu : mu0, [d]) as Vector,
        covariance: fromData(r ? r.Sigma : S0, [d, d]) as Matrix,
        site: -1,
        cavity: { mean: NaN, variance: NaN },
        tiltedMoments: { logZ: NaN, mean: NaN, variance: NaN },
        ok: r !== null,
        sweep: 0,
        position: 0,
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        skipped: 0,
        totalSkipped: 0,
        logEvidence: NaN,
        converged: false,
        diverged: r === null,
      }
    },
    step: (s) => {
      const i = order[s.position]
      const tau = Float64Array.from(dense.data(s.sitePrecision)) as F64
      const nu = Float64Array.from(dense.data(s.siteShift)) as F64
      const logZ = Float64Array.from(dense.data(s.siteLogZ)) as F64
      const cavT = Float64Array.from(dense.data(s.cavityPrecision)) as F64
      const cavN = Float64Array.from(dense.data(s.cavityShift)) as F64
      let Sigma = Float64Array.from(dense.data(s.covariance)) as F64
      let mu = Float64Array.from(dense.data(s.mean)) as F64
      const { Sa, v, m } = marginal(p, Sigma, mu, i)
      const a = power
      const ct = 1 / v - a * tau[i]
      const cn = m / v - a * nu[i]
      let ok = ct > 0
      let change = 0
      let cavity: GaussianMoments = { mean: NaN, variance: NaN }
      let t: Tilted = { logZ: NaN, mean: NaN, variance: NaN }
      if (ok) {
        cavity = { mean: cn / ct, variance: 1 / ct }
        t = o.tilted(i, cavity, a)
        ok = t.variance > 0 && Number.isFinite(t.mean)
      }
      if (ok) {
        const newT = (1 / t.variance - ct) / a
        const newN = (t.mean / t.variance - cn) / a
        const dT = (1 - damping) * newT + damping * tau[i]
        const dN = (1 - damping) * newN + damping * nu[i]
        const deltaT = dT - tau[i]
        const deltaN = dN - nu[i]
        change = Math.max(Math.abs(deltaT), Math.abs(deltaN))
        // Rank-one update: Σ ← Σ − c (Σa)(Σa)ᵀ, μ ← μ − c (Σa) m + Δν (1 − c v) Σa.
        const c = deltaT / (1 + deltaT * v)
        const k = deltaN * (1 - c * v) - c * m
        Sigma = Sigma.map((x, idx) => x - c * Sa[Math.floor(idx / d)] * Sa[idx % d]) as F64
        mu = mu.map((x, r) => x + k * Sa[r]) as F64
        tau[i] = dT
        nu[i] = dN
        logZ[i] = t.logZ
        cavT[i] = ct
        cavN[i] = cn
      }
      let position = s.position + 1
      let { sweep, sweepChange, lastSweepChange, skipped, logEvidence } = s
      let converged: boolean = s.converged
      let diverged = false
      sweepChange = Math.max(sweepChange, change)
      if (!ok) skipped++
      if (position >= order.length) {
        const r = refresh(p, tau, nu)
        if (r === null) diverged = true
        else {
          Sigma = r.Sigma
          mu = r.mu
          logEvidence = evidence(r, tau, nu, logZ, cavT, cavN)
        }
        converged = sweepChange < tolerance && skipped === 0
        lastSweepChange = sweepChange
        position = 0
        sweep++
        sweepChange = 0
        skipped = 0
      }
      return {
        ...s,
        t: s.t + 1,
        sitePrecision: vecN(tau),
        siteShift: vecN(nu),
        siteLogZ: vecN(logZ),
        cavityPrecision: vecN(cavT),
        cavityShift: vecN(cavN),
        mean: fromData(mu, [d]) as Vector,
        covariance: fromData(Sigma, [d, d]) as Matrix,
        site: i,
        cavity,
        tiltedMoments: t,
        ok,
        sweep,
        position,
        change,
        sweepChange,
        lastSweepChange,
        skipped,
        totalSkipped: s.totalSkipped + (ok ? 0 : 1),
        logEvidence,
        converged,
        diverged,
      }
    },
  }
}
