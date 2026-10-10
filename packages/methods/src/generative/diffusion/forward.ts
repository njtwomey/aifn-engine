/**
 * The forward (noising) process of DDPM (Ho et al., 2020, eqs. 2, 4, 6 and 7): the closed-form marginal
 * $q(\xvec_t \mid \xvec_0) = \Gauss(\sqrt{\bar\alpha_t}\,\xvec_0, (1 - \bar\alpha_t)\Imat)$, the posterior
 * $q(\xvec_{t-1} \mid \xvec_t, \xvec_0)$ that DDPM's reverse steps imitate, and the step-by-step Markov chain
 * $\xvec_t = \sqrt{\alpha_t}\,\xvec_{t-1} + \sqrt{\beta_t}\,\epsilonvec$ as a traceable algorithm. Points may have
 * any shape; every operation is elementwise.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import { add, mul, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { alphaBarAt, betaAt, type NoiseSchedule } from './schedules'

/** A noised sample `x` and the noise `noise` $\epsilonvec$ that made it, both of the clean point's shape. */
export type Noised = { x: Tensor; noise: Tensor }

/**
 * $\xvec = \sqrt{\bar\alpha}\,\xvec_0 + \sqrt{1 - \bar\alpha}\,\epsilonvec$ with
 * $\epsilonvec \sim \Gauss(\zeros, \Imat)$ drawn from `s`: a draw from $q(\xvec_t \mid \xvec_0)$ at signal level
 * $\bar\alpha$, in one step rather than $t$. Returns the noise too, which is the regression target of an
 * $\epsilonvec$-prediction network.
 *
 * @param s The stream the noise is drawn from (advanced by the draw).
 * @param x0 The clean points $\xvec_0$, of any shape.
 * @param alphaBar The signal level $\bar\alpha \in [0, 1]$, as `alphaBarAt` gives it for a step.
 * @returns `x`, the noised points, and `noise`, the $\epsilonvec$ drawn, both of the shape of `x0`.
 *
 * @example Noise a point to $\bar\alpha = 0.64$, so $\xvec = 0.8\,\xvec_0 + 0.6\,\epsilonvec$
 * const x0 = tensor([[1, -2]])
 * const { x, noise } = forwardNoise(stream(1), x0, 0.64)
 * print('x =', x, ' noise =', noise)
 * print('0.8 x0 + 0.6 noise =', add(mul(x0, 0.8), mul(noise, 0.6)))
 *
 * @example The mean of many draws is the closed-form mean $\sqrt{\bar\alpha}\,\xvec_0$
 * const x0 = broadcastTo(tensor([1, -2]), [2000, 2])
 * const { x } = forwardNoise(stream(1), x0, 0.64)
 * print('mean of 2000 draws:', mean(x, 0), ' closed form:', [0.8, -1.6])
 * print('variance:', variance(x, 0), ' closed form:', 1 - 0.64)
 */
export function forwardNoise(s: Stream, x0: Tensor, alphaBar: number): Noised {
  const noise = normals(s, x0.shape)
  return { x: add(mul(x0, Math.sqrt(alphaBar)), mul(noise, Math.sqrt(1 - alphaBar))), noise }
}

/**
 * The Gaussian posterior $q(\xvec_{t-1} \mid \xvec_t, \xvec_0)$ (Ho et al., 2020, eqs. 6 and 7): mean
 * $a\,\xvec_0 + b\,\xvec_t$ with $a = \sqrt{\bar\alpha_{t-1}}\,\beta_t / (1 - \bar\alpha_t)$ and
 * $b = \sqrt{\alpha_t}(1 - \bar\alpha_{t-1}) / (1 - \bar\alpha_t)$, and variance
 * $\tilde\beta_t = \beta_t (1 - \bar\alpha_{t-1}) / (1 - \bar\alpha_t)$ in every coordinate. Throws
 * `DomainError` (from `betaAt`) for a step outside $1, \dots, T$.
 *
 * @param schedule The noise schedule.
 * @param x0 The clean points $\xvec_0$.
 * @param xt The noised points $\xvec_t$, of the same shape as `x0` (or broadcastable with it).
 * @param t The step $t$, from 1 to $T$.
 * @returns `mean`, the posterior mean of $\xvec_{t-1}$ with the shape of `x0` and `xt`, and `variance`,
 *   $\tilde\beta_t$ (0 at $t = 1$, where $\xvec_0$ is known).
 *
 * @example The posterior of one step back from $t = 50$ of 100
 * const s = linearSchedule(100)
 * const post = forwardPosterior(s, tensor([[1, -2]]), tensor([[0.5, 0.5]]), 50)
 * print('mean', post.mean, ' variance', post.variance, ' beta_50', betaAt(s, 50))
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
  /** The points $\xvec_t$, of the shape of `x0`. */
  x: Tensor
  /** $\bar\alpha_t$: the fraction of the data's variance still in `x`. */
  alphaBar: number
}

/**
 * The forward Markov chain $\xvec_t = \sqrt{\alpha_t}\,\xvec_{t-1} + \sqrt{\beta_t}\,\epsilonvec$, one step per `step`,
 * from `{ x0 }` (points of any shape) until $t = T$. Step $t$ draws from the runner's step stream. Its marginal at
 * every $t$ is $q(\xvec_t \mid \xvec_0)$, which `forwardNoise` draws in one go.
 *
 * @param schedule The noise schedule; the chain is done after its $T$ steps.
 * @returns The algorithm, started from `{ x0 }`.
 *
 * @example The chain's marginal after 200 steps matches the closed form
 * const s = linearSchedule(200)
 * const x0 = broadcastTo(tensor([2, -2]), [1000, 2])
 * const end = run(forwardProcess(s), { x0 }, 200)
 * print('t =', end.t, ' alpha bar =', end.alphaBar)
 * print('mean', mean(end.x, 0), ' closed form', mul(Math.sqrt(end.alphaBar), tensor([2, -2])))
 * print('variance', variance(end.x, 0), ' closed form', 1 - end.alphaBar)
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
