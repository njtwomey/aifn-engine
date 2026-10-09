/**
 * Expectation propagation and assumed density filtering for a scalar parameter $\theta$ with a Gaussian prior and $n$
 * factors (Minka 2001, "Expectation propagation for approximate Bayesian inference", UAI; Minka 2004, "Power EP",
 * MSR-TR-2004-149).
 *
 * EP keeps one Gaussian site $\tilde{t}_i$ per factor, with
 * $q(\theta) \propto p_0(\theta) \prod_i \tilde{t}_i(\theta)$. Updating site $i$: the cavity is
 * $q^{\setminus i} = q / \tilde{t}_i^\alpha$; the tilted distribution is $q^{\setminus i} f_i^\alpha$; its moments
 * give $q_\text{new}$; the site becomes
 * $(q_\text{new} / q^{\setminus i})^{1/\alpha}$, damped towards its old value. $\alpha = 1$ is EP; $\alpha \ne 1$ is
 * power EP. Sites start at 1, so the first sweep of EP is ADF.
 *
 * The factors enter only through a `TiltedFn`, which returns the tilted moments of factor $i$ against a cavity: one of
 * the closed forms of `tilted.ts` (`probitTilted`, `stepTilted`, `intervalTilted`) or `tiltedByQuadrature`.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import type { GaussianMoments, NaturalGaussian } from './gaussian'
import type { Tilted } from './tilted'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Tilted moments of factor $i$ (raised to `power`) against a cavity: called with the factor's index, the cavity's mean
 * and variance, and the power $\alpha$, it returns $\log Z$ and the mean and variance of the cavity times $f_i^\alpha$.
 */
export type TiltedFn = (i: number, cavity: GaussianMoments, power: number) => Tilted

/** The problem and options of {@link expectationPropagation}. */
export interface EpOptions {
  /** The Gaussian prior $p_0(\theta)$. */
  prior: GaussianMoments
  /** The number of factors. */
  factors: number
  /** The tilted moments of each factor against a cavity. */
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

/** The state of EP after `t` site updates. Plain data: the tilted-moment function is closed over by the factory. */
export interface EpState extends Status {
  /** The prior in natural parameters (read by `epLogEvidence`). */
  prior: NaturalGaussian
  /** $\alpha$ (read by `epLogEvidence`). */
  power: number
  /** Site precisions $\tilde{\tau}_i$ (length $n$). */
  sitePrecision: Tensor
  /** Site shifts $\tilde{\nu}_i$ (length $n$). */
  siteShift: Tensor
  /** $\log Z_i$ of each site's last tilted distribution, for the evidence (length $n$). */
  siteLogZ: Tensor
  /** The precision of each site's cavity at its last update, for the evidence (length $n$; 0 before it). */
  cavityPrecision: Tensor
  /** The shift of each site's cavity at its last update, for the evidence (length $n$). */
  cavityShift: Tensor
  /** The approximate posterior $q$, in moments and natural parameters. */
  posterior: GaussianMoments & NaturalGaussian
  /** The site visited last ($-1$ at the start). */
  site: number
  /** The cavity of the last update (NaN when the cavity was improper). */
  cavity: GaussianMoments
  /** The tilted moments of the last update (NaN when the cavity was improper). */
  tiltedMoments: Tilted
  /**
   * False when the last update was skipped: its cavity had non-positive precision, or the tilted variance was not
   * positive or the tilted mean not finite.
   */
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
  /** Whether the last full sweep moved no site parameter by more than `tolerance` and skipped none. */
  converged: boolean
}

/**
 * The posterior $q$ of the prior times every site: precisions add, and shifts add.
 *
 * @param prior The prior in natural parameters.
 * @param tau The site precisions $\tilde{\tau}_i$.
 * @param nu The site shifts $\tilde{\nu}_i$.
 * @returns $q$ in natural parameters and in moments.
 */
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
 * EP (and power EP) as a traceable algorithm: one site update per step, visiting the sites in `order`. The run is
 * `converged` once a full sweep moves no site parameter by more than `tolerance` and skips none. Improper cavities (and
 * tilted distributions without a positive variance) are skipped and counted, never hidden. Throws `DomainError` when
 * `damping` is not in $[0, 1)$.
 *
 * @param o The prior, the number of factors, the tilted-moment function, and the damping, power, order and tolerance.
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; a sweep is $n$ steps.
 *
 * @example Gaussian factors: EP is exact
 * // θ ~ N(0, 1) and three observations x ~ N(θ, 1): the posterior is N(Σx / 4, 1/4). The tilted moments here come
 * // from quadrature, so any factor would do.
 * const x = [1, 2, 3]
 * const tilted = (i, cavity, power) =>
 *   tiltedByQuadrature(cavity.mean, cavity.variance, (t) => -0.5 * (x[i] - t) ** 2, { power })
 * const s = run(expectationPropagation({ prior: { mean: 0, variance: 1 }, factors: 3, tilted }), undefined, 100)
 * print('posterior:', s.posterior.mean, s.posterior.variance)
 * print('sweeps =', s.sweep, 'converged =', s.converged)
 *
 * @example Probit sites: the first sweep is ADF, later sweeps refine it
 * // θ ~ N(0, 1) with three probit observations Φ(yᵢ θ), y = [1, 1, -1].
 * const y = [1, 1, -1]
 * const tilted = (i, c) => probitTilted(c.mean, c.variance, y[i])
 * const options = { prior: { mean: 0, variance: 1 }, factors: 3, tilted }
 * print('ADF:', run(assumedDensityFiltering(options), undefined, 3).posterior)
 * const one = run(expectationPropagation(options), undefined, 3).posterior
 * print('EP, one sweep:', one.mean, one.variance)
 * const s = run(expectationPropagation(options), undefined, 300)
 * print('EP, converged:', s.posterior.mean, s.posterior.variance, 'after', s.sweep, 'sweeps')
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

/**
 * The Gaussian log-normaliser
 * $\log \int \exp(-\tfrac{1}{2} \tau \theta^2 + \nu \theta)\,d\theta = \nu^2/(2\tau) + \tfrac{1}{2} \log(2\pi/\tau)$.
 *
 * @param tau The precision $\tau$ (positive).
 * @param nu The shift $\nu$.
 * @returns The log-normaliser.
 */
const logNormaliser = (tau: number, nu: number) => (nu * nu) / (2 * tau) + 0.5 * Math.log((2 * Math.PI) / tau)

/**
 * EP's estimate of the log evidence $\log \int p_0(\theta) \prod_i f_i(\theta)\,d\theta$ (Minka 2001, thesis eq.
 * 3.30; Rasmussen & Williams 2006, eq. 3.65):
 * $\sum_i [\log Z_i + A(q^{\setminus i}) - A(q^{\setminus i} \tilde{t}_i)] + A(q) - A(p_0)$, with $A$ the Gaussian
 * log-normaliser and each site's scale fixed at its last update. Exact when every factor is Gaussian. Defined for
 * $\alpha = 1$ only (NaN otherwise, or before every site has been updated).
 *
 * @param s An EP state, best a converged one.
 * @returns The estimate of the log evidence, or NaN.
 *
 * @example One probit factor: EP is exact
 * // ∫ N(θ; 0, 1) Φ(θ) dθ = 1/2 by symmetry.
 * const tilted = (i, c) => probitTilted(c.mean, c.variance, 1)
 * const s = run(expectationPropagation({ prior: { mean: 0, variance: 1 }, factors: 1, tilted }), undefined, 10)
 * print('EP log evidence =', epLogEvidence(s))
 * print('log(1/2) =', Math.log(0.5))
 *
 * @example Gaussian factors: EP and ADF agree
 * const x = [1, 2, 3]
 * const tilted = (i, c, power) => tiltedByQuadrature(c.mean, c.variance, (t) => -0.5 * (x[i] - t) ** 2, { power })
 * const options = { prior: { mean: 0, variance: 1 }, factors: 3, tilted }
 * print('EP:', epLogEvidence(run(expectationPropagation(options), undefined, 100)))
 * print('ADF:', run(assumedDensityFiltering(options), undefined, 3).logEvidence)
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
  /** The Gaussian prior $p_0(\theta)$. */
  prior: GaussianMoments
  /** The number of factors. */
  factors: number
  /** The tilted moments of each factor against the running Gaussian (called with power 1). */
  tilted: TiltedFn
  /** The order the factors are absorbed in (default $0, \dots, n - 1$); its length is the number of steps. */
  order?: readonly number[]
}

/** The state of ADF after absorbing `absorbed` factors. */
export interface AdfState extends Status {
  /** The factors absorbed so far. */
  absorbed: number
  /** The current Gaussian approximation. */
  posterior: GaussianMoments
  /** $\sum_i \log Z_i$ so far: ADF's estimate of the log evidence. */
  logEvidence: number
  /** The factor absorbed last ($-1$ at the start). */
  factor: number
  /** Whether every factor in the order has been absorbed. */
  done: boolean
}

/**
 * Assumed density filtering (Maybeck 1982; Opper 1998; Minka 2001, §3.1): one pass over the factors, each absorbed
 * into the running Gaussian by moment matching, $q \leftarrow \operatorname{proj}(q f_i)$. The result depends on the
 * order. Nothing checks the tilted moments: a factor that leaves no mass gives a NaN posterior.
 *
 * @param o The prior, the number of factors, the tilted-moment function and the order.
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; one step per factor.
 *
 * @example The order matters
 * // θ ~ N(0, 1), a step factor θ > 1 and a probit observation pulling θ down.
 * const tilted = (i, c) => (i === 0 ? stepTilted(c.mean, c.variance, 1) : probitTilted(c.mean, c.variance, -1))
 * const options = { prior: { mean: 0, variance: 1 }, factors: 2, tilted }
 * print('step first:', run(assumedDensityFiltering(options), undefined, 2).posterior)
 * print('probit first:', run(assumedDensityFiltering({ ...options, order: [1, 0] }), undefined, 2).posterior)
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
