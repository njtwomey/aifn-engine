/**
 * Coordinate-ascent variational inference (CAVI) for the univariate Gaussian with unknown mean $\mu$ and precision
 * $\tau$, beside the exact normal-gamma posterior it approximates.
 *
 * The model is $x_n \sim \Gauss(\mu, \tau^{-1})$, $\mu \mid \tau \sim \Gauss(\mu_0, (\lambda_0\tau)^{-1})$ and
 * $\tau \sim \GammaD(a_0, b_0)$ (with rate $b_0$), and the mean-field family is
 * $q(\mu)q(\tau) = \Gauss(\mu \mid \mu_N, \lambda_N^{-1}) \GammaD(\tau \mid a_N, b_N)$ (Bishop, 2006, §10.1.3). The
 * exact posterior is normal-gamma (Murphy, 2007, §3), so the result can be compared with the truth: the factorised $q$
 * gets the marginal means right and under-states the spread of $\mu$, and the gap $\log p(\xvec) - \text{ELBO}$ is
 * exactly $\KL(q \,\Vert\, p(\mu, \tau \mid \xvec))$.
 */

import { digamma, logGamma } from 'aifn-compute/numerics/special'
import type { Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/**
 * The data as a Float64Array, with errors reported under the name `caviNormalGamma`.
 *
 * @param x The observations: a rank-1 tensor or an array of numbers.
 */
const toF64 = (x: VectorLike) => dense.toF64(x, 'caviNormalGamma')

const LOG_2PI = Math.log(2 * Math.PI)
/**
 * The digamma function $\psi(x)$ of a number.
 *
 * @param x The argument, positive here (a gamma shape).
 */
const dg = (x: number) => digamma(x) as number
/**
 * The log-gamma function $\log\Gamma(x)$ of a number.
 *
 * @param x The argument, positive here (a gamma shape).
 */
const lg = (x: number) => logGamma(x) as number

/**
 * The normal-gamma prior $\mu \mid \tau \sim \Gauss(\mu_0, (\lambda_0\tau)^{-1})$,
 * $\tau \sim \GammaD(a_0, b_0)$: `mean` is $\mu_0$, `precisionScale` is $\lambda_0$ (the prior's worth in
 * observations), `shape` is $a_0$ and `rate` is $b_0$.
 */
export type NormalGammaPrior = { mean: number; precisionScale: number; shape: number; rate: number }

/** The default prior: $\mu_0 = 0$, $\lambda_0 = 1$, $a_0 = 1$, $b_0 = 1$. */
export const defaultNormalGammaPrior: NormalGammaPrior = { mean: 0, precisionScale: 1, shape: 1, rate: 1 }

/**
 * The exact posterior: its normal-gamma parameters (`mean` $\mu_N$, `precisionScale` $\lambda_N$, `shape` $a_N$,
 * `rate` $b_N$), the log evidence and the marginal moments of $\mu$ and $\tau$.
 */
export type NormalGammaPosterior = NormalGammaPrior & {
  /** The log marginal likelihood $\log p(\xvec)$. */
  logEvidence: number
  /**
   * The posterior mean of $\mu$, which is $\mu_N$ (the marginal of $\mu$ is a Student t with $2a_N$ degrees of
   * freedom).
   */
  meanOfMu: number
  /** The posterior variance of $\mu$, $b_N / (\lambda_N (a_N - 1))$, or infinity when $a_N \le 1$. */
  varianceOfMu: number
  /** The posterior mean of $\tau$, $a_N / b_N$. */
  meanOfTau: number
  /** The posterior variance of $\tau$, $a_N / b_N^2$. */
  varianceOfTau: number
}

/**
 * The exact normal-gamma posterior (Murphy, 2007, §3): $\lambda_N = \lambda_0 + N$,
 * $\mu_N = (\lambda_0\mu_0 + N\bar{x}) / \lambda_N$, $a_N = a_0 + N/2$,
 * $b_N = b_0 + \tfrac{1}{2}\sum_n (x_n - \bar{x})^2 + \lambda_0 N (\bar{x} - \mu_0)^2 / (2\lambda_N)$, and
 * $\log p(\xvec) = \log\Gamma(a_N) - \log\Gamma(a_0) + a_0 \log b_0 - a_N \log b_N
 * + \tfrac{1}{2}\log(\lambda_0 / \lambda_N) - \tfrac{N}{2}\log 2\pi$.
 *
 * @param x The observations $x_1, \dots, x_N$: a rank-1 tensor or an array of numbers (at least one, or the result
 *   is NaN).
 * @param prior The normal-gamma prior (default $\mu_0 = 0$, $\lambda_0 = a_0 = b_0 = 1$).
 * @returns The posterior's parameters, the log evidence and the marginal means and variances of $\mu$ and $\tau$.
 *
 * @example The posterior after five measurements
 * const post = normalGammaPosterior([4.8, 5.1, 5.3, 4.9, 5.4])
 * print('mu_N, lambda_N, a_N, b_N:', post.mean, post.precisionScale, post.shape, post.rate)
 * print('E[mu], var[mu]:', post.meanOfMu, post.varianceOfMu)
 * print('E[tau], var[tau]:', post.meanOfTau, post.varianceOfTau)
 * print('log p(x):', post.logEvidence)
 */
export function normalGammaPosterior(
  x: VectorLike,
  prior: NormalGammaPrior = defaultNormalGammaPrior,
): NormalGammaPosterior {
  const xs = toF64(x)
  const N = xs.length
  const xbar = xs.reduce((a, b) => a + b, 0) / N
  const ss = xs.reduce((a, b) => a + (b - xbar) ** 2, 0)
  const precisionScale = prior.precisionScale + N
  const mean = (prior.precisionScale * prior.mean + N * xbar) / precisionScale
  const shape = prior.shape + N / 2
  const rate = prior.rate + ss / 2 + (prior.precisionScale * N * (xbar - prior.mean) ** 2) / (2 * precisionScale)
  const logEvidence =
    lg(shape) -
    lg(prior.shape) +
    prior.shape * Math.log(prior.rate) -
    shape * Math.log(rate) +
    0.5 * Math.log(prior.precisionScale / precisionScale) -
    (N / 2) * LOG_2PI
  return {
    mean,
    precisionScale,
    shape,
    rate,
    logEvidence,
    meanOfMu: mean,
    varianceOfMu: shape > 1 ? rate / (precisionScale * (shape - 1)) : Infinity,
    meanOfTau: shape / rate,
    varianceOfTau: shape / (rate * rate),
  }
}

/**
 * The state of `caviNormalGamma`: the two factors $q(\mu) = \Gauss(\mu_N, \lambda_N^{-1})$ and
 * $q(\tau) = \GammaD(a_N, b_N)$, the ELBO and its gap to the evidence.
 */
export type CaviNormalGammaState = Status & {
  /** Coordinate sweeps done (0 at the start). */
  t: number
  /** The mean $\mu_N$ of $q(\mu)$. */
  muMean: number
  /** The precision $\lambda_N$ of $q(\mu)$, so its variance is $1 / \lambda_N$. */
  muPrecision: number
  /** The shape $a_N$ of $q(\tau)$, which is $a_0 + (N + 1)/2$ throughout. */
  tauShape: number
  /** The rate $b_N$ of $q(\tau)$. */
  tauRate: number
  /** $\expect_q[\tau] = a_N / b_N$. */
  expectedTau: number
  /**
   * The half step after updating $q(\mu)$ and before $q(\tau)$: $(\expect[\mu], \expect[\tau])$, for drawing the
   * coordinate zig-zag.
   */
  halfStep: { muMean: number; expectedTau: number }
  /** The evidence lower bound $\expect_q[\log p(\xvec, \mu, \tau)] + \entropy[q(\mu)] + \entropy[q(\tau)]$. */
  elbo: number
  /** $\text{ELBO}(t) - \text{ELBO}(t - 1)$ (never negative for coordinate ascent; NaN at $t = 0$). */
  elboChange: number
  /** $\log p(\xvec) - \text{ELBO} = \KL(q \,\Vert\, p(\mu, \tau \mid \xvec))$, exact here. */
  kl: number
  /** Whether the ELBO changed by less than `tolerance` in the last sweep (false at $t = 0$). */
  converged: boolean
  /** Whether the ELBO is not finite. */
  diverged: boolean
}

/**
 * CAVI for the Gaussian with unknown mean and precision (Bishop, 2006, eqs. 10.26–10.30). Each step updates
 * $q(\mu) \gets \Gauss(\mu_N, ((\lambda_0 + N)\expect[\tau])^{-1})$ with
 * $\mu_N = (\lambda_0\mu_0 + N\bar{x}) / (\lambda_0 + N)$, and then
 * $q(\tau) \gets \GammaD(a_0 + (N + 1)/2,\ b_0 + \tfrac{1}{2}\expect_\mu[\sum_n (x_n - \mu)^2
 * + \lambda_0(\mu - \mu_0)^2])$.
 * Each update maximises the ELBO in its factor, so the ELBO never decreases. The mean $\mu_N$ does not depend on
 * $\expect[\tau]$, so after the first step only the two precisions move.
 *
 * `init` takes a starting `expectedTau0` $= \expect[\tau]$ (default 1) and `muMean0` $= \expect[\mu]$ (default the
 * prior mean), which only the starting state shows; run it with `run(alg, start, steps)`, where `start` may be
 * `undefined`.
 *
 * @param x The observations $x_1, \dots, x_N$: a rank-1 tensor or an array of numbers.
 * @param prior The normal-gamma prior (default $\mu_0 = 0$, $\lambda_0 = a_0 = b_0 = 1$).
 * @param options The stopping rule.
 * @param options.tolerance The state is `converged` when the ELBO changes by less than this in a step (default
 *   1e-10).
 * @returns The algorithm, whose state holds both factors, the ELBO and the exact KL gap to the posterior.
 *
 * @example CAVI against the exact posterior
 * const x = [4.8, 5.1, 5.3, 4.9, 5.4]
 * const s = run(caviNormalGamma(x), undefined, 20)
 * const exact = normalGammaPosterior(x)
 * print('converged:', s.converged, 'after', s.t, 'sweeps')
 * print('E[mu]: q', s.muMean, 'exact', exact.meanOfMu)
 * print('E[tau]: q', s.expectedTau, 'exact', exact.meanOfTau)
 * print('var[mu]: q', 1 / s.muPrecision, 'exact', exact.varianceOfMu)
 * print('KL(q || p):', s.kl)
 *
 * @example The ELBO rises at every sweep
 * // Twenty draws from N(2, 0.5^2), starting from a poor guess E[tau] = 0.01.
 * const x = add(2, mul(0.5, normals(stream(0), [20])))
 * const alg = caviNormalGamma(x)
 * print('ELBO:', [1, 2, 3, 4, 5].map((n) => run(alg, { expectedTau0: 0.01 }, n).elbo))
 */
export function caviNormalGamma(
  x: VectorLike,
  prior: NormalGammaPrior = defaultNormalGammaPrior,
  options: { tolerance?: number } = {},
): Algorithm<{ expectedTau0?: number; muMean0?: number }, CaviNormalGammaState> {
  const xs = toF64(x)
  const N = xs.length
  const xbar = xs.reduce((a, b) => a + b, 0) / N
  const sumSq = xs.reduce((a, b) => a + b * b, 0)
  const { mean: m0, precisionScale: l0, shape: a0, rate: b0 } = prior
  const tolerance = options.tolerance ?? 1e-10
  const logEvidence = normalGammaPosterior(xs, prior).logEvidence
  const aN = a0 + (N + 1) / 2
  const muMeanOpt = (l0 * m0 + N * xbar) / (l0 + N)

  // ELBO = E[log p(x, μ, τ)] + H[q(μ)] + H[q(τ)] (Bishop, 2006, Exercise 10.7).
  const elboOf = (mN: number, lN: number, a: number, b: number) => {
    const eTau = a / b
    const eLogTau = dg(a) - Math.log(b)
    const eMu2 = mN * mN + 1 / lN
    const sumSqDev = sumSq - 2 * N * xbar * mN + N * eMu2
    const lik = (N / 2) * (eLogTau - LOG_2PI) - (eTau / 2) * sumSqDev
    const priorMu = 0.5 * (Math.log(l0) + eLogTau - LOG_2PI) - ((l0 * eTau) / 2) * ((mN - m0) ** 2 + 1 / lN)
    const priorTau = a0 * Math.log(b0) - lg(a0) + (a0 - 1) * eLogTau - b0 * eTau
    const hMu = 0.5 * (1 + LOG_2PI - Math.log(lN))
    const hTau = a - Math.log(b) + lg(a) + (1 - a) * dg(a)
    return lik + priorMu + priorTau + hMu + hTau
  }
  const rateFor = (mN: number, lN: number) => {
    const eMu2 = mN * mN + 1 / lN
    return b0 + 0.5 * (sumSq - 2 * N * xbar * mN + N * eMu2 + l0 * (eMu2 - 2 * m0 * mN + m0 * m0))
  }
  const state = (
    t: number,
    mN: number,
    lN: number,
    b: number,
    half: CaviNormalGammaState['halfStep'],
    previous: number,
  ) => {
    const value = elboOf(mN, lN, aN, b)
    const change = value - previous
    return {
      t,
      muMean: mN,
      muPrecision: lN,
      tauShape: aN,
      tauRate: b,
      expectedTau: aN / b,
      halfStep: half,
      elbo: value,
      elboChange: change,
      kl: logEvidence - value,
      converged: Math.abs(change) < tolerance,
      diverged: !Number.isFinite(value),
    }
  }
  return {
    name: 'cavi-normal-gamma',
    init: ({ expectedTau0 = 1, muMean0 = m0 } = {}) =>
      state(
        0,
        muMean0,
        (l0 + N) * expectedTau0,
        aN / expectedTau0,
        { muMean: muMean0, expectedTau: expectedTau0 },
        NaN,
      ),
    step: (s) => {
      const lN = (l0 + N) * s.expectedTau
      const b = rateFor(muMeanOpt, lN)
      return state(s.t + 1, muMeanOpt, lN, b, { muMean: muMeanOpt, expectedTau: s.expectedTau }, s.elbo)
    },
  }
}
