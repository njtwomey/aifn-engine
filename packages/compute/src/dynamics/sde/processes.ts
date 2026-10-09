/**
 * Named processes of `aifn-compute/dynamics/sde` with exact solutions: Brownian motion, Ornstein–Uhlenbeck and
 * geometric Brownian motion, each with its moments and an exact sampler on the integrators' streams (Uhlenbeck &
 * Ornstein, 1930, "On the theory of the Brownian motion", Phys. Rev. 36; Kloeden & Platen, 1992, §4.4).
 *
 * Each is an `ExactSde`: the SDE to hand to a scheme, the mean and variance of $X_t$ given $X_0 = x_0$, and `exact`,
 * an algorithm with the schemes' state and streams that samples the solution itself, so a scheme's error can be
 * measured path by path or in distribution.
 */

import { dense, mul, sub, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar } from 'aifn-compute/foundation/contracts'
import { scheme, type Sde, type SdeInitial, type SdeOptions, type SdeState } from './integrators'
import { DomainError } from 'aifn-compute/foundation/errors'

// ---------------------------------------------------------------------------------------------------------------------
// Exact solutions

/**
 * An SDE with a known exact solution: its moments at time $t$ given $x_0$, and an exact sampler on the schemes'
 * streams.
 */
export type ExactSde = {
  /** The SDE, to integrate with a scheme. */
  sde: Sde
  /** $\expect[X_t \mid X_0 = x_0]$. */
  mean: (t: Scalar, x0: Scalar) => Scalar
  /** $\var[X_t \mid X_0 = x_0]$. */
  variance: (t: Scalar, x0: Scalar) => Scalar
  /**
   * Exact transitions on a grid of step $h$, drawing the same Brownian increments as the schemes (`init` takes
   * `{ x0, paths, t0 }`, as theirs does).
   */
  exact: (options: SdeOptions) => Algorithm<SdeInitial, SdeState>
}

/**
 * An exact sampler as a scheme: each step maps every state through `update`, with the step's Brownian increment.
 *
 * @param name The sampler's name, used for the algorithm and in error messages.
 * @param options The step size and end time, as for the schemes.
 * @param update The exact transition of one state: from $x$ at time $t$ over a step $h$, given the increment
 *   $\Delta W \sim \Gauss(0, h)$.
 * @returns The algorithm, with the schemes' state (it makes no drift or diffusion evaluations).
 */
function exactScheme(
  name: string,
  options: SdeOptions,
  update: (x: Scalar, t: Scalar, h: Scalar, dW: Scalar) => Scalar,
): Algorithm<SdeInitial, SdeState> {
  return scheme(name, { drift: () => 0, diffusion: () => 0 }, options, (_p, t, x, hk, dW) => {
    const w = dense.data(dW)
    return { next: Float64Array.from(dense.data(x), (v, i) => update(v, t, hk, w[i])), evaluations: 0 }
  })
}

/**
 * Standard Brownian motion scaled by $\sigma$: $dX = \sigma \, dW$, with $X_t \sim \Gauss(x_0, \sigma^2 t)$. Its exact
 * sampler adds $\sigma \Delta W$ at each step, the same path that Euler–Maruyama takes.
 *
 * @param sigma The scale $\sigma$ of the noise.
 * @returns The SDE, its mean and variance, and the exact sampler.
 *
 * @example Mean and variance, and samples from them
 * const bm = brownianMotion(2)
 * print('mean at t = 1 from 0:', bm.mean(1, 0), ' variance:', bm.variance(1, 0))
 * const end = run(bm.exact({ stepSize: 0.5, tEnd: 1 }), { x0: 0, paths: 4000 }, 2, { stream: stream(0) })
 * print('sample mean:', mean(end.x), ' sample variance:', variance(end.x))
 */
export function brownianMotion(sigma: Scalar = 1): ExactSde {
  return {
    sde: { drift: () => 0, diffusion: () => sigma },
    mean: (_t, x0) => x0,
    variance: (t) => sigma * sigma * t,
    exact: (o) => exactScheme('brownian-motion', o, (x, _t, _h, dW) => x + sigma * dW),
  }
}

/**
 * The Ornstein–Uhlenbeck process $dX = \theta(\mu - X) \, dt + \sigma \, dW$ ($\theta > 0$):
 * $X_t \mid x_0 \sim \Gauss(\mu + (x_0 - \mu)e^{-\theta t}, \sigma^2(1 - e^{-2\theta t})/(2\theta))$, stationary
 * $\Gauss(\mu, \sigma^2/(2\theta))$. The exact sampler uses the transition over each step with the step's increment
 * rescaled to the right variance,
 * $X_{t+h} = \mu + (X - \mu)e^{-\theta h} + \sigma\sqrt{(1 - e^{-2\theta h})/(2\theta h)} \, \Delta W$, so it shares
 * the schemes' streams (though not their exact Brownian path). A $\theta$ that is not positive throws `DomainError`.
 *
 * @param options The process's parameters.
 * @param options.theta The rate $\theta > 0$ at which $X$ reverts to the mean.
 * @param options.mu The long-run mean $\mu$ (default 0).
 * @param options.sigma The noise scale $\sigma$.
 * @returns The SDE (with its diffusion's derivative, 0), its mean and variance, and the exact sampler.
 *
 * @example Relaxing to the stationary law
 * // θ = 2, μ = 1, σ = 1 from x₀ = 3: the variance tends to σ² / (2θ) = 0.25.
 * const ou = ornsteinUhlenbeck({ theta: 2, mu: 1, sigma: 1 })
 * print('mean at t = 0.5:', ou.mean(0.5, 3), ' variance:', ou.variance(0.5, 3))
 * print('variance at t = 10:', ou.variance(10, 3))
 *
 * @example The exact sampler takes large steps without error
 * // One step of 0.5 is exact; Euler–Maruyama with the same step is not.
 * const ou = ornsteinUhlenbeck({ theta: 2, mu: 1, sigma: 1 })
 * const start = { x0: 3, paths: 4000 }
 * const exact = run(ou.exact({ stepSize: 0.5, tEnd: 0.5 }), start, 1, { stream: stream(1) })
 * const euler = run(eulerMaruyama(ou.sde, { stepSize: 0.5, tEnd: 0.5 }), start, 1, { stream: stream(1) })
 * print('exact sampler: mean', mean(exact.x), ' variance', variance(exact.x))
 * print('Euler–Maruyama: mean', mean(euler.x), ' variance', variance(euler.x))
 */
export function ornsteinUhlenbeck({ theta, mu = 0, sigma }: { theta: Scalar; mu?: Scalar; sigma: Scalar }): ExactSde {
  if (!(theta > 0)) throw new DomainError('ornsteinUhlenbeck', 'ornsteinUhlenbeck: θ must be positive')
  return {
    sde: {
      drift: (_t, x) => mul(theta, sub(mu, x)),
      diffusion: () => sigma,
      diffusionDerivative: () => 0,
    },
    mean: (t, x0) => mu + (x0 - mu) * Math.exp(-theta * t),
    variance: (t) => (sigma * sigma * -Math.expm1(-2 * theta * t)) / (2 * theta),
    exact: (o) =>
      exactScheme('ornstein-uhlenbeck-exact', o, (x, _t, h, dW) => {
        const decay = Math.exp(-theta * h)
        const scale = sigma * Math.sqrt(-Math.expm1(-2 * theta * h) / (2 * theta * h))
        return mu + (x - mu) * decay + scale * dW
      }),
  }
}

/**
 * Geometric Brownian motion $dX = \mu X \, dt + \sigma X \, dW$:
 * $X_t = x_0 \exp((\mu - \sigma^2/2)t + \sigma W_t)$, with $\expect[X_t] = x_0 e^{\mu t}$ and
 * $\var[X_t] = x_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$. The exact sampler applies the solution step by step with the
 * same increments the schemes draw, so it gives the true path that a scheme's path approximates (for strong-error
 * measurements).
 *
 * @param options The process's parameters.
 * @param options.mu The drift rate $\mu$.
 * @param options.sigma The volatility $\sigma$.
 * @returns The SDE (with its diffusion's derivative, $\sigma$), its mean and variance, and the exact sampler.
 *
 * @example The exact path and the Euler–Maruyama path on one stream
 * const gbm = geometricBrownianMotion({ mu: 0.1, sigma: 0.4 })
 * const options = { stepSize: 0.25, tEnd: 1 }
 * const exact = run(gbm.exact(options), { x0: 1, paths: 3 }, 4, { stream: stream(2) })
 * const euler = run(eulerMaruyama(gbm.sde, options), { x0: 1, paths: 3 }, 4, { stream: stream(2) })
 * print('exact X_1 =', exact.x)
 * print('Euler X_1 =', euler.x)
 * print('E[X_1] =', gbm.mean(1, 1), ' Var[X_1] =', gbm.variance(1, 1))
 */
export function geometricBrownianMotion({ mu, sigma }: { mu: Scalar; sigma: Scalar }): ExactSde {
  const scaled = (c: Scalar) => (_t: Scalar, x: Tensor) => mul(c, x)
  return {
    sde: { drift: scaled(mu), diffusion: scaled(sigma), diffusionDerivative: () => sigma },
    mean: (t, x0) => x0 * Math.exp(mu * t),
    variance: (t, x0) => x0 * x0 * Math.exp(2 * mu * t) * Math.expm1(sigma * sigma * t),
    exact: (o) =>
      exactScheme('gbm-exact', o, (x, _t, h, dW) => x * Math.exp((mu - 0.5 * sigma * sigma) * h + sigma * dW)),
  }
}
