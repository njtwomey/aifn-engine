/**
 * Named processes of `aifn-compute/dynamics/sde` with exact solutions: Brownian motion, Ornstein–Uhlenbeck and geometric
 * Brownian motion, each with its moments and an exact sampler on the integrators' streams (Uhlenbeck & Ornstein, 1930,
 * "On the theory of the Brownian motion", Phys. Rev. 36; Kloeden & Platen, 1992, §4.4).
 */

import { dense, mul, sub, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar } from 'aifn-compute/foundation/contracts'
import { scheme, type Sde, type SdeInitial, type SdeOptions, type SdeState } from './integrators'
import { DomainError } from 'aifn-compute/foundation/errors'

// ---------------------------------------------------------------------------------------------------------------------
// Exact solutions

/** An SDE with a known exact solution: its law at time t given x₀, and an exact sampler on the scheme's streams. */
export type ExactSde = {
  sde: Sde
  /** E[X_t | X_0 = x₀]. */
  mean: (t: Scalar, x0: Scalar) => Scalar
  /** Var[X_t | X_0 = x₀]. */
  variance: (t: Scalar, x0: Scalar) => Scalar
  /** Exact transitions on a grid of step h, drawing the same Brownian increments as the schemes. */
  exact: (options: SdeOptions) => Algorithm<SdeInitial, SdeState>
}

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

/** Standard Brownian motion scaled by σ: dX = σ dW, with X_t ~ N(x₀, σ²t). */
export function brownianMotion(sigma: Scalar = 1): ExactSde {
  return {
    sde: { drift: () => 0, diffusion: () => sigma },
    mean: (_t, x0) => x0,
    variance: (t) => sigma * sigma * t,
    exact: (o) => exactScheme('brownian-motion', o, (x, _t, _h, dW) => x + sigma * dW),
  }
}

/**
 * The Ornstein–Uhlenbeck process dX = θ(μ − X) dt + σ dW (θ > 0): X_t | x₀ ~ N(μ + (x₀ − μ)e^{−θt},
 * σ²(1 − e^{−2θt})/(2θ)), stationary N(μ, σ²/(2θ)). The exact sampler uses the transition over each step with the
 * step's increment rescaled to the right variance, X_{t+h} = μ + (X − μ)e^{−θh} + σ√((1 − e^{−2θh})/(2θh)) ΔW, so it
 * shares the schemes' streams (though not their exact Brownian path).
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
 * Geometric Brownian motion dX = μX dt + σX dW: X_t = x₀ exp((μ − σ²/2)t + σW_t), with E[X_t] = x₀e^{μt} and
 * Var[X_t] = x₀²e^{2μt}(e^{σ²t} − 1). The exact sampler applies the solution step by step with the same increments
 * the schemes draw, so it gives the true path that a scheme's path approximates (for strong-error measurements).
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
