/**
 * Improvement-based and optimistic acquisition functions of Bayesian optimisation: the probability of improvement
 * (Kushner, 1964), the expected improvement (Mockus, Tiesis and Zilinskas, 1978; Jones, Schonlau and Welch, 1998), its
 * log-normal form for a surrogate of the log cost (Hutter, Hoos and Leyton-Brown, 2011) and the upper confidence bound
 * with the GP-UCB schedule of Srinivas, Krause, Kakade and Seeger (2010).
 *
 * Each reads the surrogate's posterior mean $\mu(\xvec)$ and standard deviation $\sigma(\xvec)$ at candidate points,
 * elementwise, and returns a score that is **maximised**: the next point is where the acquisition is largest. The
 * objective is **minimised** by default (`minimise: true`), with the incumbent $f^+$ the smallest value seen, so the
 * improvement of a value $f$ is $f^+ - f$; with `minimise: false` the objective is maximised and the improvement is
 * $f - f^+$. With $\delta = f^+ - \mu - \xi$ in minimisation ($\mu - f^+ - \xi$ in maximisation) and
 * $z = \delta / \sigma$,
 *
 * $$ \mathrm{PI} = \Phi(z), \qquad \mathrm{EI} = \delta\,\Phi(z) + \sigma\,\phi(z), \qquad
 * \mathrm{UCB} = \mp\mu + \sqrt{\beta}\,\sigma $$
 *
 * where $\Phi$ and $\phi$ are the standard normal cdf and density and $\xi \ge 0$ is the margin by which a point must
 * improve on $f^+$. In minimisation UCB is the lower confidence bound $\mu - \sqrt{\beta}\sigma$, negated so that it
 * too is maximised. Where $\sigma = 0$ the posterior is a point mass: PI is 1 when $\delta > 0$ and 0 otherwise, EI is
 * $\max(\delta, 0)$ and UCB is $\mp\mu$. None is differentiable (they are evaluated on a grid).
 */

import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `probabilityOfImprovement` and `expectedImprovement`. */
export type ImprovementOptions = {
  /** The margin $\xi \ge 0$ a point must improve on the incumbent by (default 0), in the objective's units. */
  xi?: number
  /** Minimise the objective (default true); false maximises it. */
  minimise?: boolean
}

/** Options of `upperConfidenceBound`. */
export type ConfidenceBoundOptions = {
  /** The exploration weight $\beta \ge 0$ (default 4): the bound is $\sqrt{\beta}$ standard deviations wide. */
  beta?: number
  /** Minimise the objective (default true), scoring the negated lower bound; false scores the upper bound. */
  minimise?: boolean
}

/**
 * The means and standard deviations as float64 arrays of equal length, checked. A standard deviation that is a number
 * is shared by every mean. Throws `ShapeError` when the two have different sizes and `DomainError` for a negative or
 * NaN standard deviation.
 *
 * @param mu The posterior means, a number or a tensor (read flat).
 * @param sd The posterior standard deviations, a number or a tensor with as many elements as `mu`.
 * @param where The caller's name, for error messages.
 * @returns `m`, the means, and `s`, the standard deviations, as new arrays.
 */
export function moments(mu: number | Tensor, sd: number | Tensor, where: string): { m: Float64Array; s: Float64Array } {
  const m = Float64Array.from(typeof mu === 'number' ? [mu] : toFlat(mu))
  const s = typeof sd === 'number' ? new Float64Array(m.length).fill(sd) : Float64Array.from(toFlat(sd))
  if (s.length !== m.length)
    throw new ShapeError(where, `${where}: ${m.length} means but ${s.length} standard deviations`)
  for (const v of s) if (!(v >= 0)) throw new DomainError(where, `${where}: standard deviations must be non-negative`)
  return { m, s }
}

/**
 * Apply a function of the mean and standard deviation elementwise. Numbers give a number; a tensor gives a tensor of
 * its shape. Checks its inputs as `moments` does.
 *
 * @param mu The posterior means, a number or a tensor.
 * @param sd The posterior standard deviations, a number or a tensor with as many elements as `mu`.
 * @param where The caller's name, for error messages.
 * @param f The score of one mean and standard deviation.
 * @returns The scores, with the kind and shape of `mu`.
 */
export function mapMoments(
  mu: number | Tensor,
  sd: number | Tensor,
  where: string,
  f: (m: number, s: number) => number,
): number | Tensor {
  const { m, s } = moments(mu, sd, where)
  const out = Float64Array.from(m, (mi, i) => f(mi, s[i]))
  return isTensor(mu) ? fromData(out, mu.shape) : out[0]
}

/**
 * The probability of improvement $\Phi(\delta / \sigma)$ (Kushner, 1964): the posterior probability that the
 * objective at a point beats the incumbent $f^+$ by more than $\xi$, with $\delta = f^+ - \mu - \xi$ when minimising
 * and $\mu - f^+ - \xi$ when maximising. Where $\sigma = 0$ it is 1 if $\delta > 0$ and 0 otherwise. Throws
 * `DomainError` for a negative standard deviation or $\xi < 0$, and `ShapeError` when `mu` and `sd` differ in size.
 *
 * @param mu The posterior means $\mu(\xvec)$ at the candidates: a number or a tensor of any shape.
 * @param sd The posterior standard deviations $\sigma(\xvec) \ge 0$, as many as `mu` (or one number for all).
 * @param best The incumbent $f^+$: the best value observed so far (the smallest when minimising).
 * @param options The margin $\xi$ and the direction, as `ImprovementOptions`.
 * @returns The probabilities, with the kind and shape of `mu`.
 *
 * @example Near the incumbent with a small spread, PI is high; far above it, PI vanishes
 * const mu = tensor([-1, 0, 1, 3])
 * const sd = tensor([0.5, 0.5, 0.5, 0.5])
 * print('PI with best = 0:', probabilityOfImprovement(mu, sd, 0))
 * print('with xi = 0.5:', probabilityOfImprovement(mu, sd, 0, { xi: 0.5 }))
 * print('where sd = 0:', probabilityOfImprovement(tensor([-1, 1]), 0, 0))
 */
export function probabilityOfImprovement(mu: number, sd: number, best: number, options?: ImprovementOptions): number
export function probabilityOfImprovement(
  mu: Tensor,
  sd: number | Tensor,
  best: number,
  options?: ImprovementOptions,
): Tensor
export function probabilityOfImprovement(
  mu: number | Tensor,
  sd: number | Tensor,
  best: number,
  options?: ImprovementOptions,
): number | Tensor
export function probabilityOfImprovement(
  mu: number | Tensor,
  sd: number | Tensor,
  best: number,
  { xi = 0, minimise = true }: ImprovementOptions = {},
): number | Tensor {
  if (!(xi >= 0)) throw new DomainError('probabilityOfImprovement', 'probabilityOfImprovement: xi must be ≥ 0')
  return mapMoments(mu, sd, 'probabilityOfImprovement', (m, s) => {
    const delta = minimise ? best - m - xi : m - best - xi
    if (s === 0) return delta > 0 ? 1 : 0
    return normalCdf(delta / s)
  })
}

/**
 * The expected improvement $\expect[\max(\delta_f, 0)] = \delta\Phi(z) + \sigma\phi(z)$ with $z = \delta / \sigma$
 * (Mockus, Tiesis and Zilinskas, 1978; Jones, Schonlau and Welch, 1998, eq. 15), where $\delta_f$ is the improvement
 * of the objective over the incumbent less the margin $\xi$ and $\delta$ its posterior mean: $f^+ - \mu - \xi$ when
 * minimising and $\mu - f^+ - \xi$ when maximising. It trades the mean (the first term) against the spread (the
 * second). Where $\sigma = 0$ it is $\max(\delta, 0)$. The value is never negative (rounding below 0 is set to 0).
 * Throws `DomainError` for a negative standard deviation or $\xi < 0$, and `ShapeError` when `mu` and `sd` differ in
 * size.
 *
 * @param mu The posterior means $\mu(\xvec)$ at the candidates: a number or a tensor of any shape.
 * @param sd The posterior standard deviations $\sigma(\xvec) \ge 0$, as many as `mu` (or one number for all).
 * @param best The incumbent $f^+$: the best value observed so far (the smallest when minimising).
 * @param options The margin $\xi$ and the direction, as `ImprovementOptions`.
 * @returns The expected improvements, with the kind and shape of `mu`.
 *
 * @example Equal means: the wider posterior has the larger expected improvement
 * const ei = expectedImprovement(tensor([0, 0, 0]), tensor([0.1, 1, 2]), 0)
 * print('EI at mu = best:', ei, ' sigma / sqrt(2 pi):', [0.1, 1, 2].map((s) => s / Math.sqrt(2 * Math.PI)))
 *
 * @example Maximising instead, and the point-mass limit
 * print('maximise:', expectedImprovement(tensor([1, 2]), tensor([1, 1]), 1.5, { minimise: false }))
 * print('sd = 0:', expectedImprovement(tensor([-1, 1]), 0, 0))
 */
export function expectedImprovement(mu: number, sd: number, best: number, options?: ImprovementOptions): number
export function expectedImprovement(mu: Tensor, sd: number | Tensor, best: number, options?: ImprovementOptions): Tensor
export function expectedImprovement(
  mu: number | Tensor,
  sd: number | Tensor,
  best: number,
  options?: ImprovementOptions,
): number | Tensor
export function expectedImprovement(
  mu: number | Tensor,
  sd: number | Tensor,
  best: number,
  { xi = 0, minimise = true }: ImprovementOptions = {},
): number | Tensor {
  if (!(xi >= 0)) throw new DomainError('expectedImprovement', 'expectedImprovement: xi must be ≥ 0')
  return mapMoments(mu, sd, 'expectedImprovement', (m, s) => {
    const delta = minimise ? best - m - xi : m - best - xi
    if (s === 0) return Math.max(delta, 0)
    const z = delta / s
    return Math.max(delta * normalCdf(z) + s * normalPdf(z), 0)
  })
}

/**
 * The expected improvement of a log-normal cost (Hutter, Hoos and Leyton-Brown, 2011, section 5.2): when the surrogate
 * models $\ln f$ with mean $\mu$ and standard deviation $\sigma$, the cost $f = e^{\ln f}$ is log-normal and the
 * expected improvement over the incumbent cost $f^+ > 0$ (minimised) is
 *
 * $$ \expect[\max(f^+ - f, 0)] = f^+\,\Phi(v) - e^{\mu + \sigma^2/2}\,\Phi(v - \sigma), \qquad
 * v = \frac{\ln f^+ - \mu}{\sigma}. $$
 *
 * It is the improvement in the cost itself, not in its logarithm, as SMAC uses for runtimes. Where $\sigma = 0$ it is
 * $\max(f^+ - e^{\mu}, 0)$. The value is never negative (rounding below 0 is set to 0). Throws `DomainError` for a
 * negative standard deviation or an incumbent that is not positive, and `ShapeError` when `mu` and `sd` differ in size.
 *
 * @param mu The posterior means $\mu(\xvec)$ of the log cost at the candidates: a number or a tensor of any shape.
 * @param sd The posterior standard deviations $\sigma(\xvec) \ge 0$ of the log cost, as many as `mu` (or one number).
 * @param best The incumbent cost $f^+ > 0$, in the cost's own units (not its logarithm).
 * @returns The expected improvements in the cost's units, with the kind and shape of `mu`.
 *
 * @example Against the expected improvement of the log cost: the log-normal one is in the cost's units
 * const mu = tensor([Math.log(0.5), Math.log(1), Math.log(2)])
 * const sd = tensor([0.5, 0.5, 0.5])
 * print('log-normal EI, best = 1:', expectedImprovementLogNormal(mu, sd, 1))
 * print('EI of ln f, best = ln 1:', expectedImprovement(mu, sd, 0))
 * print('sd = 0:', expectedImprovementLogNormal(tensor([Math.log(0.5), 1]), 0, 1))
 */
export function expectedImprovementLogNormal(mu: number, sd: number, best: number): number
export function expectedImprovementLogNormal(mu: Tensor, sd: number | Tensor, best: number): Tensor
export function expectedImprovementLogNormal(mu: number | Tensor, sd: number | Tensor, best: number): number | Tensor
export function expectedImprovementLogNormal(mu: number | Tensor, sd: number | Tensor, best: number): number | Tensor {
  const where = 'expectedImprovementLogNormal'
  if (!(best > 0)) throw new DomainError(where, `${where}: the incumbent cost must be positive`)
  const lb = Math.log(best)
  return mapMoments(mu, sd, where, (m, s) => {
    if (s === 0) return Math.max(best - Math.exp(m), 0)
    const v = (lb - m) / s
    return Math.max(best * normalCdf(v) - Math.exp(m + (s * s) / 2) * normalCdf(v - s), 0)
  })
}

/**
 * The upper confidence bound $\mu + \sqrt{\beta}\sigma$ of GP-UCB (Srinivas et al., 2010), optimism in the face of
 * uncertainty: a point scores well when its mean is good or its spread is wide. When minimising (the default) the
 * score is the negated lower confidence bound, $-(\mu - \sqrt{\beta}\sigma)$, so it is maximised as the others are.
 * $\beta = 0$ is pure exploitation (the posterior mean); a large $\beta$ explores. Throws `DomainError` for a negative
 * standard deviation or $\beta < 0$, and `ShapeError` when `mu` and `sd` differ in size.
 *
 * @param mu The posterior means $\mu(\xvec)$ at the candidates: a number or a tensor of any shape.
 * @param sd The posterior standard deviations $\sigma(\xvec) \ge 0$, as many as `mu` (or one number for all).
 * @param options The exploration weight $\beta$ and the direction, as `ConfidenceBoundOptions`.
 * @returns The scores, with the kind and shape of `mu`.
 *
 * @example A larger beta prefers the uncertain point
 * const mu = tensor([0, 0.5])
 * const sd = tensor([0.1, 1])
 * print('beta = 0.1:', upperConfidenceBound(mu, sd, { beta: 0.1 }))
 * print('beta = 4:', upperConfidenceBound(mu, sd, { beta: 4 }))
 */
export function upperConfidenceBound(mu: number, sd: number, options?: ConfidenceBoundOptions): number
export function upperConfidenceBound(mu: Tensor, sd: number | Tensor, options?: ConfidenceBoundOptions): Tensor
export function upperConfidenceBound(
  mu: number | Tensor,
  sd: number | Tensor,
  options?: ConfidenceBoundOptions,
): number | Tensor
export function upperConfidenceBound(
  mu: number | Tensor,
  sd: number | Tensor,
  { beta = 4, minimise = true }: ConfidenceBoundOptions = {},
): number | Tensor {
  if (!(beta >= 0)) throw new DomainError('upperConfidenceBound', 'upperConfidenceBound: beta must be ≥ 0')
  const w = Math.sqrt(beta)
  return mapMoments(mu, sd, 'upperConfidenceBound', (m, s) => (minimise ? -m : m) + w * s)
}

/**
 * The exploration weight $\beta_t = 2 \log(\lvert D \rvert t^2 \pi^2 / 6\delta)$ of GP-UCB on a finite candidate set
 * $D$ (Srinivas et al., 2010, Theorem 1): with it, the cumulative regret after $t$ steps grows as
 * $O^*(\sqrt{t\,\gamma_t \log\lvert D \rvert})$ with probability at least $1 - \delta$, $\gamma_t$ being the maximum
 * information gain. It grows slowly with $t$, so exploration never quite stops. Throws `DomainError` unless $t \ge 1$,
 * $\lvert D \rvert \ge 1$ and $0 < \delta < 1$.
 *
 * @param t The step $t \ge 1$ (the number of the evaluation about to be chosen).
 * @param options The size of the candidate set and the failure probability.
 * @param options.candidates The number $\lvert D \rvert$ of candidate points (a grid's size).
 * @param options.delta The failure probability $\delta$ of the regret bound (default 0.1).
 * @returns $\beta_t$.
 *
 * @example The schedule on a grid of 201 points
 * print('beta_t for t = 1, 10, 100:', [1, 10, 100].map((t) => ucbBeta(t, { candidates: 201 }).toFixed(3)))
 */
export function ucbBeta(t: number, { candidates, delta = 0.1 }: { candidates: number; delta?: number }): number {
  if (!(t >= 1)) throw new DomainError('ucbBeta', 'ucbBeta: t must be ≥ 1')
  if (!(candidates >= 1)) throw new DomainError('ucbBeta', 'ucbBeta: needs at least one candidate')
  if (!(delta > 0 && delta < 1)) throw new DomainError('ucbBeta', 'ucbBeta: delta must lie in (0, 1)')
  return 2 * Math.log((candidates * t * t * Math.PI * Math.PI) / (6 * delta))
}
