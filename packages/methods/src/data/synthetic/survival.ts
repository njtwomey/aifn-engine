/**
 * Right-censored survival data with known truth: covariates (a 0/1 treatment and standard-normal features), event times
 * from a proportional-hazards Weibull model (hazard h₀(t) e^{xᵀβ} with h₀(t) = (k/λ)(t/λ)^{k−1}), and independent
 * exponential censoring plus an administrative end of follow-up. The Weibull proportional-hazards model is also an AFT
 * model with log time ratios −β/k, so both Cox and AFT fits have a known target.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

/** Options of `censoredSurvival`. */
export interface CensoredSurvivalOptions {
  /** Subjects (default 200). */
  n?: number
  /** Log hazard ratios β; the first covariate is a 0/1 treatment, the rest N(0, 1) (default [−0.7, 0.5]). */
  coefficients?: readonly number[]
  /** Weibull shape k (default 1.5; 1 is exponential) and scale λ (default 10). */
  shape?: number
  scale?: number
  /** Rate of exponential censoring (default 0.03; 0 for none). */
  censoringRate?: number
  /** End of follow-up: later times are censored there (default Infinity). */
  followUp?: number
  /** Round times up to whole units, which creates ties (default false). */
  round?: boolean
}

/** Survival data: covariates [n, p], observed times and event flags [n], and the latent event times. */
export interface CensoredSurvival {
  readonly x: Tensor
  readonly time: Tensor
  readonly event: Tensor
  readonly latent: Float64Array
  readonly truth: { coefficients: number[]; shape: number; scale: number }
  readonly featureNames: string[]
}

/** Seeded right-censored data from a Weibull proportional-hazards model (module docs). */
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

/** The true survival S(t | x) = exp(−(t/λ)^k e^{xᵀβ}) of `censoredSurvival`'s model at the given times. */
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
