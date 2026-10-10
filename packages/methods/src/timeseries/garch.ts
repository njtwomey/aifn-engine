/**
 * GARCH(1,1) (Bollerslev, 1986): $r_t = \mu + \sigma_t \varepsilon_t$, $\varepsilon_t \sim \Gauss(0, 1)$,
 * $\sigma_t^2 = \omega + \alpha (r_{t-1} - \mu)^2 + \beta \sigma_{t-1}^2$.
 *
 * The model's properties, simulation, the Gaussian likelihood, a maximum-likelihood fit and variance forecasts. The
 * process is covariance-stationary when $\alpha + \beta < 1$, with unconditional variance
 * $\bar\sigma^2 = \omega / (1 - \alpha - \beta)$. The likelihood is Gaussian, with the variance recursion started at
 * the mean squared deviation of the returns.
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import { mean as meanOf, tensor, toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { logit, sigmoid } from 'aifn-compute/numerics/special'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'

/**
 * GARCH(1,1) parameters: `omega` $\omega > 0$, `alpha` $\alpha \ge 0$, `beta` $\beta \ge 0$ (not checked), and the
 * return mean `mean` $\mu$ (default 0).
 */
export type GarchSpec = { omega: number; alpha: number; beta: number; mean?: number }

/**
 * Summary quantities of a GARCH(1,1): the persistence $p = \alpha + \beta$, the unconditional variance
 * $\omega / (1 - p)$ and the kurtosis $3(1 - p^2) / (1 - p^2 - 2\alpha^2)$ (Bollerslev, 1986, Theorem 2), each
 * Infinity where it does not exist ($p \ge 1$, or $1 - p^2 - 2\alpha^2 \le 0$ for the kurtosis).
 *
 * @param options The parameters; the mean is not used.
 * @param options.omega The constant $\omega$ of the variance recursion.
 * @param options.alpha The weight $\alpha$ of the last squared shock.
 * @param options.beta The weight $\beta$ of the last variance.
 * @returns The persistence, whether it is below 1 (`stationary`), the unconditional variance and the kurtosis.
 *
 * @example Persistence, variance and fat tails
 * print(garchProperties({ omega: 0.1, alpha: 0.1, beta: 0.8 }))
 *
 * @example An integrated GARCH has no unconditional variance
 * print(garchProperties({ omega: 0.1, alpha: 0.3, beta: 0.7 }))
 */
export function garchProperties({ omega, alpha, beta }: GarchSpec): {
  persistence: number
  stationary: boolean
  unconditionalVariance: number
  kurtosis: number
} {
  const p = alpha + beta
  const denom = 1 - p * p - 2 * alpha * alpha
  return {
    persistence: p,
    stationary: p < 1,
    unconditionalVariance: p < 1 ? omega / (1 - p) : Infinity,
    // E[r⁴]/E[r²]² = 3(1 − p²)/(1 − p² − 2α²) when the fourth moment exists (Bollerslev, 1986, Theorem 2).
    kurtosis: p < 1 && denom > 0 ? (3 * (1 - p * p)) / denom : Infinity,
  }
}

/**
 * Simulate $n$ returns and their conditional variances, starting the recursion with $\sigma^2$ and the squared shock
 * both at the unconditional variance (or $\omega$ when the model is not stationary), and discarding `burn` values.
 *
 * @param s The random stream the shocks $\varepsilon_t$ are drawn from; advanced in place.
 * @param spec The parameters, the mean included.
 * @param n The number of returns kept.
 * @param options Simulation options.
 * @param options.burn Values simulated and discarded before the $n$ kept (default 200).
 * @returns The returns $r_t$, their conditional variances $\sigma_t^2$ and whether $\alpha + \beta < 1$.
 *
 * @example The sample variance is near $\omega / (1 - \alpha - \beta)$
 * const spec = { omega: 0.1, alpha: 0.1, beta: 0.8 }
 * const sim = simulateGarch(stream(3), spec, 2000)
 * print('sample variance:', variance(sim.returns))
 * print('unconditional variance:', garchProperties(spec).unconditionalVariance)
 * print('first conditional variances:', toFlat(sim.variance).slice(0, 4))
 */
export function simulateGarch(
  s: Stream,
  spec: GarchSpec,
  n: number,
  { burn = 200 }: { burn?: number } = {},
): { returns: Vector; variance: Vector; stationary: boolean } {
  const { omega, alpha, beta, mean = 0 } = spec
  const props = garchProperties(spec)
  const e = toFlat(normals(s, n + burn))
  let v = props.stationary ? props.unconditionalVariance : omega
  let prev2 = v
  const r: number[] = []
  const vs: number[] = []
  for (let t = 0; t < n + burn; t++) {
    v = omega + alpha * prev2 + beta * v
    const x = Math.sqrt(v) * e[t]
    prev2 = x * x
    if (t >= burn) {
      r.push(mean + x)
      vs.push(v)
    }
  }
  return { returns: tensor(r), variance: tensor(vs), stationary: props.stationary }
}

/**
 * The conditional variances $\sigma_t^2$ of a return series under a GARCH(1,1), started (backcast) with $\sigma_0^2$
 * and $(r_0 - \mu)^2$ both at $\frac{1}{n} \sum_t (r_t - \mu)^2$, and the Gaussian log-likelihood
 * $-\frac{1}{2} \sum_t \left(\log 2\pi + \log \sigma_t^2 + (r_t - \mu)^2 / \sigma_t^2\right)$.
 *
 * @param r The returns $r_1, \dots, r_n$.
 * @param spec The parameters, the mean included.
 * @returns The log-likelihood and the conditional variances $\sigma_1^2, \dots, \sigma_n^2$.
 *
 * @example The likelihood peaks near the true $\beta$
 * const { returns } = simulateGarch(stream(3), { omega: 0.1, alpha: 0.1, beta: 0.8 }, 1000)
 * for (const beta of [0.5, 0.8, 0.85]) {
 *   print('β =', beta, 'log L =', garchLogLikelihood(returns, { omega: 0.1, alpha: 0.1, beta }).logLikelihood)
 * }
 */
export function garchLogLikelihood(r: VectorLike, spec: GarchSpec): { logLikelihood: number; variance: Vector } {
  const xs = toVec(r, 'garchLogLikelihood')
  const { omega, alpha, beta, mean = 0 } = spec
  const d = xs.map((x) => x - mean)
  const s0 = d.reduce((a, v) => a + v * v, 0) / d.length
  let v = s0
  let prev2 = s0
  let ll = 0
  const vs: number[] = []
  for (const x of d) {
    v = omega + alpha * prev2 + beta * v
    vs.push(v)
    ll += -0.5 * (Math.log(2 * Math.PI) + Math.log(v) + (x * x) / v)
    prev2 = x * x
  }
  return { logLikelihood: ll, variance: tensor(vs) }
}

/**
 * A maximum-likelihood GARCH(1,1) fitter as a traceable algorithm: Nelder–Mead over $\omega = e^{u_0}$, persistence
 * $\alpha + \beta = \sigma(u_1) < 1$ and $\alpha = (\alpha + \beta) \sigma(u_2)$, with $\sigma$ the logistic function,
 * so every iterate is stationary. $\mu$ is fixed at the sample mean. It starts at $\alpha = 0.1$, $\beta = 0.8$ and the
 * $\omega$ whose unconditional variance is the sample variance. `init` takes no start (`run(alg, undefined, steps)`).
 *
 * @param r The returns.
 * @returns The algorithm; its state is a `FitState` whose `params` are the parameters with their log-likelihood.
 *
 * @example The fit moves little from a good start
 * const { returns } = simulateGarch(stream(3), { omega: 0.1, alpha: 0.1, beta: 0.8 }, 1000)
 * const alg = garchFitSteps(returns)
 * for (const steps of [0, 20, 200]) {
 *   const { params } = run(alg, undefined, steps)
 *   print(steps, 'steps: α =', params.alpha, 'β =', params.beta, 'log L =', params.logLikelihood)
 * }
 */
export function garchFitSteps(r: VectorLike): Algorithm<void, FitState<GarchSpec & { logLikelihood: number }>> {
  const xs = toVec(r, 'garchFitSteps')
  const mean = meanOf(tensor(xs))
  const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length
  const decode = (u: number[]) => {
    const p = sigmoid(u[1])
    const alpha = p * sigmoid(u[2])
    const spec = { omega: Math.exp(u[0]), alpha, beta: p - alpha, mean }
    return { ...spec, logLikelihood: garchLogLikelihood(xs, spec).logLikelihood }
  }
  // Start at persistence 0.9 with α = 0.1, and ω matching the sample variance.
  const u0 = [Math.log(variance * 0.1), logit(0.9), logit(0.1 / 0.9)]
  return simplexFit('garch', (u) => -decode(u).logLikelihood, decode, u0, 1e-10)
}

/**
 * Fit a GARCH(1,1) model by maximum likelihood, running `garchFitSteps` until Nelder–Mead converges or `maxSteps`
 * steps have run. Check `converged` before trusting the result.
 *
 * @param r The returns.
 * @param options Fit options.
 * @param options.maxSteps The most Nelder–Mead steps to run (default 4000).
 * @returns The fitted parameters and log-likelihood, with whether the optimiser converged and the steps taken.
 *
 * @example Recover the parameters of a simulated series
 * const { returns } = simulateGarch(stream(3), { omega: 0.1, alpha: 0.1, beta: 0.8 }, 1000)
 * const fit = fitGarch(returns)
 * print('ω =', fit.omega, 'α =', fit.alpha, 'β =', fit.beta)
 * print('converged:', fit.converged, 'in', fit.steps, 'steps')
 */
export function fitGarch(
  r: VectorLike,
  { maxSteps = 4000 }: { maxSteps?: number } = {},
): GarchSpec & { logLikelihood: number; converged: boolean; steps: number } {
  const s = run(garchFitSteps(r), undefined, maxSteps)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/**
 * Variance forecasts $\sigma^2_{T+h}$, $h = 1, \dots, \mathrm{horizon}$, from the next-step variance
 * $\sigma^2_{T+1}$: since
 * $\expect[\sigma^2_{t+1} \mid \mathcal{F}_T] = \omega + (\alpha + \beta) \expect[\sigma^2_t \mid \mathcal{F}_T]$,
 * $\sigma^2_{T+h} = \bar\sigma^2 + (\alpha + \beta)^{h-1}(\sigma^2_{T+1} - \bar\sigma^2)$ when $\alpha + \beta < 1$,
 * with $\bar\sigma^2$ the unconditional variance. The recursion is run as it stands for any persistence.
 *
 * @param spec The parameters; the mean is not used.
 * @param next The variance $\sigma^2_{T+1}$ of the next return, known at time $T$:
 *   $\omega + \alpha (r_T - \mu)^2 + \beta \sigma_T^2$.
 * @param horizon The number of steps to forecast.
 * @returns $\sigma^2_{T+1}, \dots, \sigma^2_{T+\mathrm{horizon}}$, the first equal to `next`.
 *
 * @example Forecasts revert to the unconditional variance, 1 here
 * const spec = { omega: 0.1, alpha: 0.1, beta: 0.8 }
 * print('from a calm day:', garchForecast(spec, 0.5, 6))
 * print('from a volatile day:', garchForecast(spec, 3, 6))
 *
 * @example Forecast from the end of a series
 * const spec = { omega: 0.1, alpha: 0.1, beta: 0.8 }
 * const r = toFlat(simulateGarch(stream(3), spec, 500).returns)
 * const v = toFlat(garchLogLikelihood(r, spec).variance)
 * const next = spec.omega + spec.alpha * r[499] ** 2 + spec.beta * v[499]
 * print(garchForecast(spec, next, 5))
 */
export function garchForecast(spec: GarchSpec, next: number, horizon: number): Vector {
  const p = spec.alpha + spec.beta
  const out = [next]
  for (let h = 1; h < horizon; h++) out.push(spec.omega + p * out[h - 1])
  return tensor(out.slice(0, horizon))
}
