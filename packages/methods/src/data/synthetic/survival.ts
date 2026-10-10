/**
 * Right-censored survival data with known truth: covariates (a 0/1 treatment and standard-normal features), event times
 * from a proportional-hazards Weibull model (hazard $h_0(t) e^{\xvec^\top\betavec}$ with
 * $h_0(t) = (k/\lambda)(t/\lambda)^{k-1}$), and independent exponential censoring plus an administrative end of
 * follow-up. The Weibull proportional-hazards model is also an AFT model with log time ratios $-\betavec/k$, so both
 * Cox and AFT fits have a known target.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

/** Options of `censoredSurvival`. */
export interface CensoredSurvivalOptions {
  /** Subjects (default 200). */
  n?: number
  /**
   * Log hazard ratios $\betavec$, one per covariate; the first covariate is a 0/1 treatment, the rest $\Gauss(0, 1)$
   * (default $[-0.7, 0.5]$).
   */
  coefficients?: readonly number[]
  /** Weibull shape $k$ (default 1.5; 1 is exponential). */
  shape?: number
  /**
   * Weibull scale $\lambda$ (default 10): the time by which a subject with $\xvec^\top\betavec = 0$ has failed with
   * probability $1 - e^{-1}$.
   */
  scale?: number
  /** Rate of exponential censoring (default 0.03; 0 for none). */
  censoringRate?: number
  /** End of follow-up: later times are censored there (default Infinity). */
  followUp?: number
  /** Round times up to whole units, which creates ties (default false). */
  round?: boolean
}

/** Survival data from `censoredSurvival`. */
export interface CensoredSurvival {
  /** The covariates, $n \times p$: column 0 the 0/1 treatment, the others standard normal. */
  readonly x: Tensor
  /** The observed times, $n$ values: the event time or the censoring time, whichever came first. */
  readonly time: Tensor
  /** The event flags, $n$ values: 1 when the event was observed, 0 when the subject was censored. */
  readonly event: Tensor
  /** The latent event times, $n$ values, observed or not (unrounded). */
  readonly latent: Float64Array
  /** The true model: the log hazard ratios, the Weibull shape $k$ and scale $\lambda$. */
  readonly truth: { coefficients: number[]; shape: number; scale: number }
  /** The covariate names: `'treatment'`, then `'x1'`, `'x2'`, and so on. */
  readonly featureNames: string[]
}

/**
 * Seeded right-censored data from a Weibull proportional-hazards model (see the file comment). Each subject is treated
 * with probability 1/2, its event time $T$ is drawn by inverting $S(t \mid \xvec)$ at a uniform draw, its censoring
 * time $C$ is exponential with rate `censoringRate` and capped at `followUp`, and it is observed at $\min(T, C)$ with
 * an event when $T \le C$. With `round`, observed times are rounded up to whole units (at least 1); the event flags
 * still compare the unrounded times.
 *
 * @param s The stream the treatments, features, event times and censoring times are drawn from (children
 *   `'treatment'`, `'features'`, `'events'` and `'censoring'`).
 * @param options The number of subjects, the true model and the censoring.
 * @returns The covariates, the observed times and event flags, the latent event times, and the true model.
 *
 * @example Shapes, the first subjects and the share censored
 * const d = censoredSurvival(stream(1), { n: 500 })
 * print('x:', d.x.shape, ' time:', d.time.shape, ' names:', d.featureNames)
 * print('first times:', toArray(d.time).slice(0, 4), ' events:', toArray(d.event).slice(0, 4))
 * print('share censored:', 1 - toArray(d.event).reduce((a, v) => a + v, 0) / 500)
 * print('share treated:', toArray(d.x).filter((r) => r[0] === 1).length / 500)
 */
export function censoredSurvival(s: Stream, options: CensoredSurvivalOptions = {}): CensoredSurvival {
  const { n = 200, coefficients = [-0.7, 0.5], shape = 1.5, scale = 10, censoringRate = 0.03 } = options
  const { followUp = Infinity, round = false } = options
  const p = coefficients.length
  const x = new Float64Array(n * p)
  const treat = units(child(s, 'treatment'), n)
  const z = standardNormals(child(s, 'features'), n * p)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < p; k++) x[i * p + k] = k === 0 ? (treat[i] < 0.5 ? 1 : 0) : z[i * p + k]
  const u = units(child(s, 'events'), n)
  const c = units(child(s, 'censoring'), n)
  const latent = new Float64Array(n)
  const time = new Float64Array(n)
  const event = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let eta = 0
    for (let k = 0; k < p; k++) eta += x[i * p + k] * coefficients[k]
    // S(t | x) = exp(−(t/λ)^k e^η), inverted at a uniform draw.
    latent[i] = scale * Math.pow(-Math.log(1 - u[i]) / Math.exp(eta), 1 / shape)
    const censor = Math.min(followUp, censoringRate > 0 ? -Math.log(1 - c[i]) / censoringRate : Infinity)
    let t = Math.min(latent[i], censor)
    if (round) t = Math.max(1, Math.ceil(t))
    time[i] = t
    event[i] = latent[i] <= censor ? 1 : 0
  }
  return {
    x: fromData(x, [n, p]),
    time: fromData(time, [n]),
    event: fromData(event, [n]),
    latent,
    truth: { coefficients: [...coefficients], shape, scale },
    featureNames: Array.from({ length: p }, (_, k) => (k === 0 ? 'treatment' : `x${k}`)),
  }
}

/**
 * The true survival $S(t \mid \xvec) = \exp(-(t/\lambda)^k e^{\xvec^\top\betavec})$ of `censoredSurvival`'s model at
 * the given times.
 *
 * @param truth The true model, as `censoredSurvival` returns it in `truth`.
 * @param x One subject's covariates $\xvec$, a row of `x` (treatment first); one value per coefficient is read.
 * @param times The times $t$ at which to evaluate the survival.
 * @returns $S(t \mid \xvec)$ at each time, in the order of `times`.
 *
 * @example At the scale, an untreated subject at the mean has survived with probability 1/e
 * const truth = { coefficients: [-0.7, 0.5], shape: 1.5, scale: 10 }
 * print('untreated:', weibullPhSurvival(truth, [0, 0], [0, 5, 10, 20]))
 * print('treated:', weibullPhSurvival(truth, [1, 0], [0, 5, 10, 20]))
 */
export function weibullPhSurvival(
  truth: CensoredSurvival['truth'],
  x: ArrayLike<number>,
  times: ArrayLike<number>,
): Float64Array {
  let eta = 0
  truth.coefficients.forEach((b, k) => (eta += b * x[k]))
  return Float64Array.from(times, (t) => Math.exp(-Math.pow(t / truth.scale, truth.shape) * Math.exp(eta)))
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'censoredSurvival',
    name: 'Censored survival data',
    summary:
      'Weibull proportional-hazards event times with a treatment and features, censored at random and at follow-up.',
    role: 'simulation',
    random: true,
    notes: ['censoring', 'cox-proportional-hazards', 'kaplan-meier-estimator', 'parametric-survival-models'],
    cite: ['kalbfleisch2002'],
  },
  censoredSurvival,
)
fn(
  {
    key: 'weibullPhSurvival',
    name: 'Weibull proportional-hazards survival',
    summary: 'The true survival curve of the censored-data generator for given covariates.',
    role: 'property',
    notes: ['survival-and-hazard-functions', 'parametric-survival-models'],
  },
  weibullPhSurvival,
)
