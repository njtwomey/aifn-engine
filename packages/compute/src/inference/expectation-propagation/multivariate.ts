/**
 * Expectation propagation for a vector parameter $\thetavec \in \reals^d$ with a Gaussian prior
 * $\Gauss(\muvec_0, \Sigmamat_0)$ and $n$ factors that each depend on one linear projection,
 * $f_i(\thetavec) = g_i(\avec_i^\top \thetavec)$: Gaussian process classification ($\avec_i = \evec_i$, $\thetavec$
 * the latent function at the training inputs), Bayesian probit regression ($\avec_i = \xvec_i$), and paired
 * comparisons ($\avec_i = \evec_\text{winner} - \evec_\text{loser}$). Each site is a scalar Gaussian in
 * $u_i = \avec_i^\top \thetavec$,
 * $\tilde{t}_i(\thetavec) \propto \exp(-\tfrac{1}{2} \tilde{\tau}_i u_i^2 + \tilde{\nu}_i u_i)$, so
 *
 * $q(\thetavec) = \Gauss(\muvec, \Sigmamat)$, $\Sigmamat = (\Sigmamat_0^{-1} + \Amat^\top \tilde{\Tmat} \Amat)^{-1}$,
 * $\muvec = \Sigmamat(\Sigmamat_0^{-1}\muvec_0 + \Amat^\top \tilde{\nuvec})$, $\tilde{\Tmat} = \diag(\tilde{\tauvec})$.
 *
 * A site update needs only the marginal of $\avec_i^\top \thetavec$ under $q$,
 * $\Gauss(\avec_i^\top \muvec, \avec_i^\top \Sigmamat \avec_i)$: the cavity, the tilted moments (the same scalar
 * `TiltedFn` as `expectationPropagation`, e.g. `probitTilted`) and the new site are the scalar algebra, and the
 * posterior moves by a rank-one update,
 * $\Sigmamat \leftarrow \Sigmamat - c (\Sigmamat \avec_i)(\Sigmamat \avec_i)^\top$ with
 * $c = \Delta\tilde{\tau} / (1 + \Delta\tilde{\tau}\, \avec_i^\top \Sigmamat \avec_i)$ (Rasmussen & Williams, 2006,
 * Algorithm 3.5; Minka, 2001). At the end of every sweep $\Sigmamat$ and $\muvec$ are recomputed from the sites,
 * which removes the rounding the rank-one updates accumulate, by the push-through form
 *
 * $\Sigmamat = \Sigmamat_0 - \Sigmamat_0 \Amat^\top \Mmat^{-1} \tilde{\Tmat} \Amat \Sigmamat_0$,
 * $\Mmat = \Imat + \tilde{\Tmat} \Amat \Sigmamat_0 \Amat^\top$,
 *
 * which needs neither $\Sigmamat_0^{-1}$ (a GP Gram matrix is often numerically singular) nor $\tilde{\tau} \ge 0$.
 * With $\alpha = 1$ the run also reports EP's log evidence (Minka 2001, eq. 3.30; R&W eq. 3.65), using
 * $\lvert \Sigmamat \rvert / \lvert \Sigmamat_0 \rvert = 1 / \lvert \Mmat \rvert$.
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
  /**
   * The Gaussian prior $\Gauss(\muvec_0, \Sigmamat_0)$ of $\thetavec$, $d$-dimensional. A non-zero mean needs an
   * invertible covariance (`ShapeError` otherwise).
   */
  prior: { mean: Vector; covariance: Matrix }
  /**
   * Row $i$ is the projection $\avec_i$ that factor $i$ depends on, $n \times d$. Default: the identity (factor $i$
   * depends on $\theta_i$, and $n = d$).
   */
  projections?: Matrix
  /** Tilted moments of factor $i$ (raised to `power`) against the cavity of $\avec_i^\top \thetavec$. */
  tilted: TiltedFn
  /** Weight of the old site in each update, in $[0, 1)$. Default 0; outside $[0, 1)$ throws `DomainError`. */
  damping?: number
  /** $\alpha$ for power EP. Default 1. */
  power?: number
  /** The order sites are visited within a sweep (default $0, \dots, n - 1$). */
  order?: readonly number[]
  /** A sweep in which no site parameter moves more than this has converged. Default 1e-8. */
  tolerance?: number
}

/** The state of multivariate EP after `t` site updates. */
export interface MvEpState extends Status {
  /** Site precisions $\tilde{\tau}_i$ (length $n$). */
  sitePrecision: Tensor
  /** Site shifts $\tilde{\nu}_i$ (length $n$). */
  siteShift: Tensor
  /** $\log Z_i$ of each site's last tilted distribution, for the evidence (length $n$). */
  siteLogZ: Tensor
  /** The cavity precision of $\avec_i^\top \thetavec$ at each site's last update (length $n$; 0 before it). */
  cavityPrecision: Tensor
  /** The cavity shift of $\avec_i^\top \thetavec$ at each site's last update (length $n$). */
  cavityShift: Tensor
  /** The mean $\muvec$ of $q(\thetavec)$ (length $d$). */
  mean: Vector
  /** The covariance $\Sigmamat$ of $q(\thetavec)$ ($d \times d$). */
  covariance: Matrix
  /** The site visited last ($-1$ at the start). */
  site: number
  /** The cavity of $\avec_i^\top \thetavec$ at the last update (NaN when it was improper). */
  cavity: GaussianMoments
  /** The tilted moments of the last update (NaN when the cavity was improper). */
  tiltedMoments: Tilted
  /** False when the last cavity had non-positive precision or improper tilted moments (the update was skipped). */
  ok: boolean
  /** Completed sweeps. */
  sweep: number
  /** Position in `order` of the next site to update (0 at the start of a sweep). */
  position: number
  /** Largest change of a site parameter in the last update. */
  change: number
  /** Largest change of a site parameter in this sweep so far. */
  sweepChange: number
  /** Largest change of a site parameter in the last full sweep ($\infty$ before the first). */
  lastSweepChange: number
  /** Updates skipped in this sweep. */
  skipped: number
  /** Updates skipped in total. */
  totalSkipped: number
  /**
   * EP's log evidence as of the last sweep's end ($\alpha = 1$ and every site updated; NaN otherwise, and before the
   * first sweep ends).
   */
  logEvidence: number
  /** Whether the last full sweep moved no site parameter by more than `tolerance` and skipped none. */
  converged: boolean
}

/** The model as row-major arrays. */
type Problem = {
  /** The dimension $d$ of $\thetavec$. */
  d: number
  /** The number of factors $n$. */
  n: number
  /** The prior mean $\muvec_0$ (length $d$). */
  mu0: F64
  /** The prior covariance $\Sigmamat_0$ ($d \times d$). */
  S0: F64
  /** Projections $\Amat$ ($n \times d$), or null for the identity. */
  A: F64 | null
  /** $\Sigmamat_0^{-1}\muvec_0$ (null when $\muvec_0 = \zeros$), for the evidence. */
  priorShift: F64 | null
  /** $\tfrac{1}{2} \muvec_0^\top \Sigmamat_0^{-1} \muvec_0$. */
  priorQuad: number
}

/**
 * The Gaussian log-normaliser
 * $\log \int \exp(-\tfrac{1}{2} \tau x^2 + \nu x)\,dx = \nu^2/(2\tau) + \tfrac{1}{2} \log(2\pi/\tau)$.
 *
 * @param tau The precision $\tau$ (positive).
 * @param nu The shift $\nu$.
 * @returns The log-normaliser.
 */
const logNormaliser = (tau: number, nu: number) => (nu * nu) / (2 * tau) + 0.5 * Math.log((2 * Math.PI) / tau)

/**
 * $\Sigmamat \avec_i$, $\avec_i^\top \Sigmamat \avec_i$ and $\avec_i^\top \muvec$ for projection $i$: the marginal of
 * $\avec_i^\top \thetavec$ under $q$, and the direction of the rank-one update.
 *
 * @param p The problem, for $d$ and the projections.
 * @param Sigma The current covariance $\Sigmamat$ (row-major, $d \times d$); not modified.
 * @param mu The current mean $\muvec$ (length $d$).
 * @param i The factor's index.
 * @returns `Sa` ($\Sigmamat \avec_i$, a copy), `v` (the variance) and `m` (the mean).
 */
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
 * $\Sigmamat$, $\muvec$ and $\log \lvert \Mmat \rvert$ from the sites by the push-through form (the file comment).
 * With $\Gmat = \Amat\Sigmamat_0$ ($n \times d$) and $\Mmat = \Imat + \tilde{\Tmat} \Gmat \Amat^\top$:
 * $\Sigmamat = \Sigmamat_0 - \Gmat^\top \Mmat^{-1} \tilde{\Tmat} \Gmat$ and
 * $\muvec = \muvec_0 - \Gmat^\top \Mmat^{-1} \tilde{\Tmat} \Amat \muvec_0 + \Sigmamat \Amat^\top \tilde{\nuvec}$.
 * Null when $\Mmat$ is singular.
 *
 * @param p The problem.
 * @param tau The site precisions $\tilde{\tau}$ (length $n$).
 * @param nu The site shifts $\tilde{\nu}$ (length $n$).
 * @returns The covariance (symmetrised), the mean and $\log \lvert \det \Mmat \rvert$, or null.
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
 * EP's log evidence $\log \int p_0(\thetavec) \prod_i f_i(\thetavec)\,d\thetavec$ for $\alpha = 1$:
 * $\sum_i [\log Z_i + A(q^{\setminus i}) - A(q^{\setminus i} \tilde{t}_i)] + A(q) - A(p_0)$, $A$ the Gaussian
 * log-normaliser, with
 * $A(q) - A(p_0) = \tfrac{1}{2} (\muvec^\top \hat{\nuvec} - \muvec_0^\top \hat{\nuvec}_0 - \log \lvert \Mmat \rvert)$,
 * where $\hat{\nuvec}_0 = \Sigmamat_0^{-1}\muvec_0$ and $\hat{\nuvec} = \hat{\nuvec}_0 + \Amat^\top \tilde{\nuvec}$.
 * NaN until every site has been updated with a proper cavity.
 *
 * @param p The problem, for the projections and the prior terms.
 * @param mu The posterior mean $\muvec$ (from `refresh`).
 * @param nu The site shifts $\tilde{\nu}$.
 * @param tau The site precisions $\tilde{\tau}$.
 * @param logZ Each site's last $\log Z_i$.
 * @param ct Each site's last cavity precision.
 * @param cn Each site's last cavity shift.
 * @param logDetM $\log \lvert \det \Mmat \rvert$ (from `refresh`).
 * @returns The log evidence, or NaN.
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
 * Multivariate EP with rank-one sites as a traceable `Algorithm` (see the file comment): one site update per step,
 * $\Sigmamat$ and $\muvec$ recomputed at the end of every sweep, `converged` once a full sweep moves no site parameter
 * by more than `tolerance` and skips none. Improper cavities are skipped and counted, never hidden; a singular
 * $\Mmat$ when recomputing stops the run as `diverged`. `init` takes optional starting sites (a warm start); by
 * default every site is 1 ($\tilde{\tau} = \tilde{\nu} = 0$), so the first sweep is ADF. Throws `ShapeError` for
 * mismatched shapes (or a non-zero prior mean with a singular covariance) and `DomainError` for a `damping` outside
 * $[0, 1)$.
 *
 * @param o The prior, the projections, the tilted-moment function, and the damping, power, order and tolerance.
 * @returns The algorithm, to run with `run(alg, start, steps)`, `start` being `undefined` or the warm-start sites
 *   `{ sitePrecision, siteShift }` (each of length $n$); a sweep is $n$ steps.
 *
 * @example A paired comparison: one site is exact
 * // Skills θ ~ N(0, I) and player 0 beat player 1: one probit site on θ₀ − θ₁, Φ(θ₀ − θ₁) with no noise. With a
 * // single site EP is the exact moment match, and the evidence is p(θ₀ > θ₁).
 * const tilted = (i, c) => probitTilted(c.mean, c.variance, 1, { noiseVariance: 1e-12 })
 * const alg = multivariateExpectationPropagation({
 *   prior: { mean: tensor([0, 0]), covariance: tensor([[1, 0], [0, 1]]) },
 *   projections: tensor([[1, -1]]),
 *   tilted,
 * })
 * const s = run(alg, undefined, 10)
 * print('mean =', s.mean)
 * print('covariance =', s.covariance)
 * print('evidence =', Math.exp(s.logEvidence))
 *
 * @example GP-style classification of three points
 * // A correlated prior over the latent function at three inputs, with labels +1, +1, -1 through a probit.
 * const y = [1, 1, -1]
 * const K = tensor([[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 1]])
 * const alg = multivariateExpectationPropagation({
 *   prior: { mean: tensor([0, 0, 0]), covariance: K },
 *   tilted: (i, c) => probitTilted(c.mean, c.variance, y[i]),
 * })
 * const s = run(alg, undefined, 300)
 * print('latent means =', s.mean)
 * print('sweeps =', s.sweep, 'converged =', s.converged)
 * print('log evidence =', s.logEvidence)
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
