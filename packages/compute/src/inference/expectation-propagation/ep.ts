/**
 * Expectation propagation and assumed density filtering for a scalar parameter θ with a Gaussian prior and n
 * factors (Minka 2001, "Expectation propagation for approximate Bayesian inference", UAI; Minka 2004, "Power EP",
 * MSR-TR-2004-149).
 *
 * EP keeps one Gaussian site t̃ᵢ per factor, with q(θ) ∝ p₀(θ) Πᵢ t̃ᵢ(θ). Updating site i: the cavity is
 * q^{\i} = q / t̃ᵢ^α; the tilted distribution is q^{\i} fᵢ^α; its moments give q_new; the site becomes
 * (q_new / q^{\i})^{1/α}, damped towards its old value. α = 1 is EP; α ≠ 1 is power EP. Sites start at 1, so the
 * first sweep of EP is ADF.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import type { GaussianMoments, NaturalGaussian } from './gaussian'
import type { Tilted } from './tilted'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Tilted moments of factor i (raised to `power`) against a cavity. */
export type TiltedFn = (i: number, cavity: GaussianMoments, power: number) => Tilted

/** The problem and options of {@link expectationPropagation}. */
export interface EpOptions {
  /** The Gaussian prior p₀(θ). */
  prior: GaussianMoments
  /** The number of factors. */
  factors: number
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

/** The state of EP after `t` site updates. Plain data: the tilted-moment function is closed over by the factory. */
export interface EpState extends Status {
  /** The prior in natural parameters, and α (both read by `epLogEvidence`). */
  prior: NaturalGaussian
  power: number
  /** Site precisions and shifts (length n). */
  sitePrecision: Tensor
  siteShift: Tensor
  /** log Z of each site's last tilted distribution, and that update's cavity (natural parameters), for the evidence. */
  siteLogZ: Tensor
  cavityPrecision: Tensor
  cavityShift: Tensor
  /** The approximate posterior q. */
  posterior: GaussianMoments & NaturalGaussian
  /** The site updated last (−1 at the start), its cavity and tilted moments. */
  site: number
  cavity: GaussianMoments
  tiltedMoments: Tilted
  /** False when the last cavity had non-positive precision and the update was skipped. */
  ok: boolean
  sweep: number
  position: number
  /** Largest change of a site parameter in the last update, in this sweep, and in the last full sweep. */
  change: number
  sweepChange: number
  lastSweepChange: number
  /** Updates skipped (improper cavities) in this sweep and in total. */
  skipped: number
  totalSkipped: number
  converged: boolean
}

function posteriorOf(prior: NaturalGaussian, tau: Float64Array, nu: Float64Array) {
  let precision = prior.precision
  let shift = prior.shift
  for (let i = 0; i < tau.length; i++) {
    precision += tau[i]
    shift += nu[i]
  }
  return { precision, shift, mean: shift / precision, variance: 1 / precision }
}

/**
 * EP (and power EP) as a traceable algorithm: one site update per step. The run is `converged` once a full sweep
 * moves no site parameter by more than `tolerance`. Improper cavities are skipped and counted, never hidden.
 */
export function expectationPropagation(o: EpOptions): Algorithm<void, EpState> {
  const n = o.factors
  const damping = o.damping ?? 0
  if (!(damping >= 0 && damping < 1))
    throw new DomainError('expectationPropagation', 'expectationPropagation: damping must be in [0, 1)')
  const power = o.power ?? 1
  const order = [...(o.order ?? Array.from({ length: n }, (_, i) => i))]
  const tolerance = o.tolerance ?? 1e-8
  return {
    name: 'expectation-propagation',
    init: () => {
      const prior = { precision: 1 / o.prior.variance, shift: o.prior.mean / o.prior.variance }
      const zeros = () => fromData(new Float64Array(n), [n])
      const tau = new Float64Array(n)
      return {
        t: 0,
        prior,
        power,
        sitePrecision: zeros(),
        siteShift: zeros(),
        siteLogZ: zeros(),
        cavityPrecision: zeros(),
        cavityShift: zeros(),
        posterior: posteriorOf(prior, tau, tau),
        site: -1,
        cavity: { mean: NaN, variance: NaN },
        tiltedMoments: { logZ: NaN, mean: NaN, variance: NaN },
        ok: true,
        sweep: 0,
        position: 0,
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        skipped: 0,
        totalSkipped: 0,
        converged: false,
      }
    },
    step: (s) => {
      const i = order[s.position]
      const tau = Float64Array.from(s.sitePrecision.data)
      const nu = Float64Array.from(s.siteShift.data)
      const logZs = Float64Array.from(s.siteLogZ.data)
      const cavT = Float64Array.from(s.cavityPrecision.data)
      const cavN = Float64Array.from(s.cavityShift.data)
      const a = s.power
      const q = s.posterior
      const ct = q.precision - a * tau[i]
      const cn = q.shift - a * nu[i]
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
        change = Math.max(Math.abs(dT - tau[i]), Math.abs(dN - nu[i]))
        tau[i] = dT
        nu[i] = dN
        logZs[i] = t.logZ
        cavT[i] = ct
        cavN[i] = cn
      }
      let position = s.position + 1
      let { sweep, sweepChange, lastSweepChange, skipped } = s
      let converged: boolean = s.converged
      sweepChange = Math.max(sweepChange, change)
      if (!ok) skipped++
      if (position >= order.length) {
        converged = sweepChange < tolerance && skipped === 0
        lastSweepChange = sweepChange
        position = 0
        sweep++
        sweepChange = 0
        skipped = 0
      }
      const n = tau.length
      return {
        ...s,
        t: s.t + 1,
        sitePrecision: fromData(tau, [n]),
        siteShift: fromData(nu, [n]),
        siteLogZ: fromData(logZs, [n]),
        cavityPrecision: fromData(cavT, [n]),
        cavityShift: fromData(cavN, [n]),
        posterior: posteriorOf(s.prior, tau, nu),
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
        converged,
      }
    },
  }
}

/** log ∫ exp(−½ τ θ² + ν θ) dθ = ν²/(2τ) + ½ log(2π/τ). */
const logNormaliser = (tau: number, nu: number) => (nu * nu) / (2 * tau) + 0.5 * Math.log((2 * Math.PI) / tau)

/**
 * EP's estimate of the log evidence log ∫ p₀(θ) Πᵢ fᵢ(θ) dθ (Minka 2001, thesis eq. 3.30; Rasmussen & Williams 2006,
 * eq. 3.65): Σᵢ [log Zᵢ + A(cavityᵢ) − A(cavityᵢ · siteᵢ)] + A(q) − A(p₀), with A the Gaussian
 * log-normaliser and each site's scale fixed at its last update. Exact when every factor is Gaussian. Defined for α = 1 only (NaN otherwise, or before every site has been updated).
 */
export function epLogEvidence(s: EpState): number {
  if (s.power !== 1) return NaN
  const q = s.posterior
  const Aq = logNormaliser(q.precision, q.shift)
  let total = Aq - logNormaliser(s.prior.precision, s.prior.shift)
  for (let i = 0; i < s.sitePrecision.data.length; i++) {
    if (!(s.cavityPrecision.data[i] > 0)) return NaN
    const ct = s.cavityPrecision.data[i]
    const cn = s.cavityShift.data[i]
    // The site's scale C_i makes ∫ cavity_i · site_i = Z_i: log C_i = log Z_i + A(cavity_i) − A(cavity_i · site_i).
    total +=
      s.siteLogZ.data[i] + logNormaliser(ct, cn) - logNormaliser(ct + s.sitePrecision.data[i], cn + s.siteShift.data[i])
  }
  return total
}

/** The problem of {@link assumedDensityFiltering}. */
export interface AdfOptions {
  prior: GaussianMoments
  factors: number
  tilted: TiltedFn
  order?: readonly number[]
}

/** The state of ADF after absorbing `absorbed` factors. */
export interface AdfState extends Status {
  absorbed: number
  /** The current Gaussian approximation. */
  posterior: GaussianMoments
  /** Σ log Zᵢ so far: ADF's estimate of the log evidence. */
  logEvidence: number
  /** The factor absorbed last (−1 at the start). */
  factor: number
  done: boolean
}

/**
 * Assumed density filtering (Maybeck 1982; Opper 1998; Minka 2001, §3.1): one pass over the factors, each absorbed
 * into the running Gaussian by moment matching, q ← proj(q fᵢ). The result depends on the order.
 */
export function assumedDensityFiltering(o: AdfOptions): Algorithm<void, AdfState> {
  const order = [...(o.order ?? Array.from({ length: o.factors }, (_, i) => i))]
  return {
    name: 'assumed-density-filtering',
    init: () => ({
      t: 0,
      absorbed: 0,
      posterior: { ...o.prior },
      logEvidence: 0,
      factor: -1,
      done: o.factors === 0,
    }),
    step: (s) => {
      const i = order[s.absorbed]
      const t = o.tilted(i, s.posterior, 1)
      const absorbed = s.absorbed + 1
      return {
        ...s,
        t: s.t + 1,
        absorbed,
        posterior: { mean: t.mean, variance: t.variance },
        logEvidence: s.logEvidence + t.logZ,
        factor: i,
        done: absorbed >= order.length,
      }
    },
    done: (s) => s.done,
  }
}
