/**
 * The forward (noising) process: the closed-form marginal q(x_t | x₀) = N(√ᾱₜ x₀, (1 − ᾱₜ)I), the posterior
 * q(x_{t−1} | x_t, x₀) that DDPM's reverse steps imitate (Ho et al., 2020, eqs. 4, 6 and 7), and the step-by-step
 * Markov chain x_t = √αₜ x_{t−1} + √βₜ ε as a traceable algorithm.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import { add, mul, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { alphaBarAt, betaAt, type NoiseSchedule } from './schedules'

/** A noised sample and the noise that made it. */
export type Noised = { x: Tensor; noise: Tensor }

/**
 * x = √ᾱ·x₀ + √(1 − ᾱ)·ε with ε ~ N(0, I) drawn from `s`: a draw from q(x_t | x₀) at signal level ᾱ, for x₀ of any
 * shape. Returns the noise too, which is the regression target of an ε-prediction network.
 */
export function forwardNoise(s: Stream, x0: Tensor, alphaBar: number): Noised {
  const noise = normals(s, x0.shape)
  return { x: add(mul(x0, Math.sqrt(alphaBar)), mul(noise, Math.sqrt(1 - alphaBar))), noise }
}

/**
 * The Gaussian posterior q(x_{t−1} | x_t, x₀) (Ho et al., 2020, eqs. 6–7): mean
 * √ᾱ_{t−1}βₜ/(1 − ᾱₜ)·x₀ + √αₜ(1 − ᾱ_{t−1})/(1 − ᾱₜ)·x_t and variance β̃ₜ = (1 − ᾱ_{t−1})/(1 − ᾱₜ)·βₜ.
 */
export function forwardPosterior(
  schedule: NoiseSchedule,
  x0: Tensor,
  xt: Tensor,
  t: number,
): { mean: Tensor; variance: number } {
  const ab = alphaBarAt(schedule, t)
  const abPrev = alphaBarAt(schedule, t - 1)
  const beta = betaAt(schedule, t)
  const a = (Math.sqrt(abPrev) * beta) / (1 - ab)
  const b = (Math.sqrt(1 - beta) * (1 - abPrev)) / (1 - ab)
  return { mean: add(mul(x0, a), mul(xt, b)), variance: ((1 - abPrev) / (1 - ab)) * beta }
}

/** The state of `forwardProcess`: the points after t noising steps. */
export type ForwardState = Status & {
  /** Noising steps taken. */
  t: number
  x: Tensor
  /** ᾱₜ: the fraction of the data's variance still in x. */
  alphaBar: number
}

/**
 * The forward Markov chain x_t = √αₜ·x_{t−1} + √βₜ·ε, one step per `step`, from `{ x0 }` (points of any shape) until
 * t = T. Step t draws from the runner's step stream. Its marginal at every t is q(x_t | x₀).
 */
export function forwardProcess(schedule: NoiseSchedule): Algorithm<{ x0: Tensor }, ForwardState> {
  return {
    name: 'diffusion-forward',
    init: ({ x0 }) => ({ t: 0, x: x0, alphaBar: 1 }),
    step: (st, ctx) => {
      const t = st.t + 1
      const beta = betaAt(schedule, t)
      const eps = normals(ctx.stream, st.x.shape)
      const x = add(mul(st.x, Math.sqrt(1 - beta)), mul(eps, Math.sqrt(beta)))
      return { t, x, alphaBar: alphaBarAt(schedule, t) }
    },
    done: (st) => st.t >= schedule.steps,
  }
}
