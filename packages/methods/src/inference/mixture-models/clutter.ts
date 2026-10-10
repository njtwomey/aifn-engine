/**
 * Minka's clutter problem (Minka, 2001, "A family of algorithms for approximate Bayesian inference", §3.3): its
 * structure in the model language, its tilted moments for EP, its likelihood, sampling, and the exact posterior on a
 * grid.
 *
 * A scalar $\theta \sim \Gauss(0, v_0)$ is observed through points that are each signal,
 * $x_i \sim \Gauss(\theta, 1)$, with probability $1 - w$, and clutter, $x_i \sim \Gauss(0, c^2)$, with probability
 * $w$. The posterior is a mixture of $2^n$ Gaussians, which is what makes it a test of approximate inference: EP
 * (`clutterEp`) is compared with the exact posterior (`clutterPosterior`). The defaults are Minka's, $v_0 = 100$ and
 * $c^2 = 10$.
 */

import { lift, type Out } from 'aifn-compute/inference/expectation-propagation'
import { normal, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { normalLogPdf } from 'aifn-compute/numerics/special'
import { fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { EpOptions } from 'aifn-compute/inference/expectation-propagation'
import { tiltedByQuadrature, type Tilted } from 'aifn-compute/inference/expectation-propagation'
import { dist, model, type Model } from 'aifn-compute/inference/model'

/** Options of {@link clutterTilted}: the parameters of Minka's clutter factor. */
export interface ClutterFactorOptions {
  /** $w$, the prior probability that a point is clutter, in $(0, 1)$. */
  weight: number
  /** $c^2$, the variance of the clutter distribution (it has mean 0). Default 10. */
  clutterVariance?: number
  /** $\sigma^2$, the variance of a signal point around $\theta$. Default 1. */
  noiseVariance?: number
}

/**
 * The tilted moments of Minka's clutter factor $t(\theta) = (1 - w)\Gauss(x; \theta, \sigma^2) + w\Gauss(x; 0, c^2)$
 * against a Gaussian cavity $\Gauss(\theta; m, v)$ (Minka, 2001, §3.3.1), in closed form and elementwise over
 * broadcast arguments. With $s = v + \sigma^2$, $d = x - m$ and $r$ the probability under the cavity that $x$ is
 * signal, the tilted mean is $m + r v d / s$ and the variance $v - r v^2 / s + r(1 - r) v^2 d^2 / s^2$. Not
 * differentiable.
 *
 * @param x The observation $x$: a number, or a tensor broadcast against `mean` and `variance`.
 * @param mean The cavity mean $m$.
 * @param variance The cavity variance $v$.
 * @param options The factor's parameters.
 * @param options.weight $w$, the prior probability that a point is clutter, in $(0, 1)$.
 * @param options.clutterVariance $c^2$, the variance of the zero-mean clutter distribution.
 * @param options.noiseVariance $\sigma^2$, the variance of a signal point around $\theta$.
 * @returns `logZ`, the log normaliser $\log \int t(\theta)\Gauss(\theta; m, v)\,d\theta$, the tilted `mean` and
 *   `variance`, and `signal`, the probability $r$: numbers when every argument is a number, otherwise tensors of the
 *   broadcast shape.
 *
 * @example A point near the cavity mean is signal; a far one is clutter
 * const near = clutterTilted(0.5, 0, 1, { weight: 0.2 })
 * const far = clutterTilted(8, 0, 1, { weight: 0.2 })
 * print('near: signal', near.signal, 'mean', near.mean, 'variance', near.variance)
 * print('far:  signal', far.signal, 'mean', far.mean, 'variance', far.variance)
 * print('over a tensor of points', clutterTilted(tensor([0.5, 2, 8]), 0, 1, { weight: 0.2 }).mean)
 */
export function clutterTilted<X extends number | Tensor, M extends number | Tensor, S extends number | Tensor>(
  x: X,
  mean: M,
  variance: S,
  { weight, clutterVariance = 10, noiseVariance = 1 }: ClutterFactorOptions,
): Tilted<Out<X | M | S>> & { signal: Out<X | M | S> } {
  return lift([x, mean, variance], ['logZ', 'mean', 'variance', 'signal'], (xi, m, s2) => {
    const sv = s2 + noiseVariance
    const ls = Math.log1p(-weight) + logNormal(xi, m, sv)
    const lc = Math.log(weight) + logNormal(xi, 0, clutterVariance)
    const top = Math.max(ls, lc)
    const logZ = top + Math.log(Math.exp(ls - top) + Math.exp(lc - top))
    const r = Math.exp(ls - logZ)
    const d = xi - m
    return {
      logZ,
      signal: r,
      mean: m + (r * s2 * d) / sv,
      variance: s2 - (r * s2 * s2) / sv + (r * (1 - r) * s2 * s2 * d * d) / (sv * sv),
    }
  }) as Tilted<Out<X | M | S>> & { signal: Out<X | M | S> }
}

// ── The clutter problem ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Minka's clutter problem: $\theta \sim \Gauss(0, v_0)$, and each $x_i$ is $\Gauss(\theta, 1)$ with probability
 * $1 - w$, else $\Gauss(0, c^2)$.
 */
export interface ClutterProblem {
  /** $w$, the probability that a point is clutter, in $(0, 1)$. */
  weight: number
  /** $v_0$, the variance of the zero-mean Gaussian prior on $\theta$. Default 100. */
  priorVariance?: number
  /** $c^2$, the variance of the zero-mean clutter distribution. Default 10. */
  clutterVariance?: number
}

/**
 * The clutter problem's structure in the model language (Minka, 2001, §3.3), for diagrams and model-driven inference:
 * $\theta \sim \Gauss(0, v_0)$ and a plate of $N$ points, each with a signal indicator $s_n \sim \Bern(1 - w)$ and
 * $x_n \sim \Gauss(s_n\theta, \sigma_n^2)$, where $\sigma_n$ is 1 for signal and $c$ for clutter (a table indexed by
 * $s_n$).
 *
 * @param problem The clutter problem; `priorVariance` and `clutterVariance` default to 100 and 10.
 * @returns The model, named `'clutter problem'`, with $N$ left symbolic.
 *
 * @example The nodes and plate of the model
 * const m = clutterModel({ weight: 0.2 })
 * print('nodes:', m.attributes.map((n) => `${n.name} (${n.role}, in ${n.group})`))
 * print('plates:', m.groups.map((g) => `${g.name} of size ${g.size}`))
 */
export function clutterModel(problem: ClutterProblem): Model {
  const { weight, priorVariance, clutterVariance } = defaults(problem)
  return model('clutter problem', (m) => {
    const theta = m.variable('θ', dist.Normal(0, Math.sqrt(priorVariance)), { label: '\\theta' })
    const points = m.plate('points', 'N', { label: 'N', index: 'n' })
    const signal = points.variable('s', dist.Bernoulli(1 - weight), { label: 's_n' })
    const mean = points.deterministic('μ', 'product', [signal, theta], { label: '\\mu_n' })
    const sd = points.deterministic('σ', 'index', [[Math.sqrt(clutterVariance), 1], signal], { label: '\\sigma_n' })
    points.observed('x', dist.Normal(mean, sd), { label: 'x_n' })
  })
}

/**
 * The Gaussian log density $\log\Gauss(x; m, v)$.
 *
 * @param x The point.
 * @param m The mean.
 * @param v The variance (not the standard deviation).
 * @returns $\log\Gauss(x; m, v)$.
 */
const logNormal = (x: number, m: number, v: number): number => normalLogPdf((x - m) / Math.sqrt(v)) - 0.5 * Math.log(v)

/**
 * A clutter problem with its defaults filled in.
 *
 * @param p The problem as given.
 * @returns `weight`, `priorVariance` (default 100) and `clutterVariance` (default 10).
 */
const defaults = (p: ClutterProblem) => ({
  weight: p.weight,
  priorVariance: p.priorVariance ?? 100,
  clutterVariance: p.clutterVariance ?? 10,
})

/**
 * The log-likelihood of the clutter problem,
 * $\log p(\xvec \mid \theta) = \sum_i \log[(1 - w)\Gauss(x_i; \theta, 1) + w\Gauss(x_i; 0, c^2)]$, each term
 * computed by log-sum-exp.
 *
 * @param theta The value of $\theta$.
 * @param x The observations $x_i$.
 * @param problem The clutter problem; its `weight` and `clutterVariance` are used (the prior is not).
 * @returns The log-likelihood, 0 for no observations.
 *
 * @example The likelihood peaks near the signal, whatever the clutter
 * const x = [1.8, 2.1, 2.3, -6, 9]
 * print('theta = 0:', clutterLogLikelihood(0, x, { weight: 0.3 }))
 * print('theta = 2:', clutterLogLikelihood(2, x, { weight: 0.3 }))
 * print('theta = 5:', clutterLogLikelihood(5, x, { weight: 0.3 }))
 */
export function clutterLogLikelihood(theta: number, x: ArrayLike<number>, problem: ClutterProblem): number {
  const { weight, clutterVariance } = defaults(problem)
  let total = 0
  for (let i = 0; i < x.length; i++) {
    const ls = Math.log1p(-weight) + logNormal(x[i], theta, 1)
    const lc = Math.log(weight) + logNormal(x[i], 0, clutterVariance)
    const top = Math.max(ls, lc)
    total += top + Math.log(Math.exp(ls - top) + Math.exp(lc - top))
  }
  return total
}

/**
 * Draw observations of the clutter problem at a true $\theta$: each is clutter, $\Gauss(0, c^2)$, with probability
 * $w$, and otherwise signal, $\Gauss(\theta, 1)$. Observation $i$ is drawn from its own child stream `child(s, i)`, so
 * the first $n$ draws do not depend on how many are asked for.
 *
 * @param s The stream the draws derive from.
 * @param n The number of observations.
 * @param theta The true value of $\theta$.
 * @param problem The clutter problem; its `weight` and `clutterVariance` are used (the prior is not).
 * @returns The $n$ observations, a float64 vector. The functions here that take observations take an
 *   `ArrayLike<number>`, so pass them `toArray` of it.
 *
 * @example Mostly near the true value, with some clutter
 * const x = sampleClutter(stream(0), 10, 2, { weight: 0.3 })
 * print('x', x)
 * print('within 3 of theta', toArray(x).filter((v) => Math.abs(v - 2) < 3).length, 'of 10')
 */
export function sampleClutter(s: Stream, n: number, theta: number, problem: ClutterProblem): Vector {
  const { weight, clutterVariance } = defaults(problem)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = child(s, i)
    out[i] =
      uniform(child(r, 'which')) < weight
        ? normal(child(r, 'x'), 0, Math.sqrt(clutterVariance))
        : normal(child(r, 'x'), theta, 1)
  }
  return fromData(out, [n])
}

/**
 * EP options for the clutter problem, ready for `expectationPropagation` (in
 * `aifn-compute/inference/expectation-propagation`): the prior $\Gauss(0, v_0)$, one factor per observation, and
 * their tilted moments, in closed form (`clutterTilted`) for power $\alpha = 1$ and by quadrature for power EP.
 *
 * @param x The observations $x_i$, one EP factor each; copied.
 * @param problem The clutter problem; `priorVariance` and `clutterVariance` default to 100 and 10.
 * @param options EP settings passed through unchanged: `damping`, `power`, `order` and `tolerance`.
 * @returns The `EpOptions`: `prior`, `factors` (the number of observations), `tilted`, and the settings of `options`.
 *
 * @example The options, and one factor's tilted moments against the prior
 * const ep = clutterEp([1.8, 2.1, -6], { weight: 0.3 })
 * print('prior', ep.prior, 'factors', ep.factors)
 * print('factor 0, closed form', ep.tilted(0, ep.prior, 1))
 * print('factor 0, power 0.5', ep.tilted(0, ep.prior, 0.5))
 */
export function clutterEp(
  x: ArrayLike<number>,
  problem: ClutterProblem,
  options: Pick<EpOptions, 'damping' | 'power' | 'order' | 'tolerance'> = {},
): EpOptions {
  const p = defaults(problem)
  const xs = Array.from(x)
  const factor = (i: number) => (theta: number) => clutterLogLikelihood(theta, [xs[i]], p)
  return {
    prior: { mean: 0, variance: p.priorVariance },
    factors: xs.length,
    tilted: (i, c, power) =>
      power === 1
        ? clutterTilted(xs[i], c.mean, c.variance, { weight: p.weight, clutterVariance: p.clutterVariance })
        : tiltedByQuadrature(c.mean, c.variance, factor(i), { power }),
    ...options,
  }
}

/**
 * The exact posterior of the clutter problem on an even grid of $\theta$, by the trapezoid rule: its density, mean,
 * variance and log evidence, the reference EP is compared with. The grid must hold the posterior mass; the default
 * $[-40, 40]$ does for the usual prior.
 *
 * @param x The observations $x_i$.
 * @param problem The clutter problem; `priorVariance` and `clutterVariance` default to 100 and 10.
 * @param options The grid.
 * @param options.lower The smallest $\theta$ of the grid.
 * @param options.upper The largest $\theta$ of the grid.
 * @param options.points The number of grid points, ends included.
 * @returns `grid`, the values of $\theta$; `density`, the posterior density at each (it integrates to 1 on the grid);
 *   the posterior `mean` and `variance`; and `logEvidence`, $\log \int p(\theta) p(\xvec \mid \theta)\,d\theta$.
 *
 * @example The posterior of twenty points drawn at theta = 2
 * const x = sampleClutter(stream(1), 20, 2, { weight: 0.3 })
 * const exact = clutterPosterior(toArray(x), { weight: 0.3 })
 * print('mean', exact.mean, 'variance', exact.variance)
 * print('log evidence', exact.logEvidence)
 * const density = toArray(exact.density)
 * print('peak at theta =', toArray(exact.grid)[density.indexOf(Math.max(...density))])
 */
export function clutterPosterior(
  x: ArrayLike<number>,
  problem: ClutterProblem,
  { lower = -40, upper = 40, points = 4001 }: { lower?: number; upper?: number; points?: number } = {},
): { grid: Vector; density: Vector; mean: number; variance: number; logEvidence: number } {
  const p = defaults(problem)
  const h = (upper - lower) / (points - 1)
  const grid = Float64Array.from({ length: points }, (_, i) => lower + i * h)
  const logs = grid.map(
    (t) =>
      -0.5 * Math.log(2 * Math.PI * p.priorVariance) - (0.5 * t * t) / p.priorVariance + clutterLogLikelihood(t, x, p),
  )
  const top = Math.max(...logs)
  const f = logs.map((l) => Math.exp(l - top))
  const trap = (g: ArrayLike<number>) => {
    let s = 0
    for (let i = 0; i < g.length; i++) s += g[i] * (i === 0 || i === g.length - 1 ? 0.5 : 1)
    return s * h
  }
  const Z = trap(f)
  const density = f.map((v) => v / Z)
  const mean = trap(density.map((d, i) => d * grid[i]))
  const variance = trap(density.map((d, i) => d * (grid[i] - mean) ** 2))
  return {
    grid: fromData(grid, [points]),
    density: fromData(density, [points]),
    mean,
    variance,
    logEvidence: top + Math.log(Z),
  }
}
