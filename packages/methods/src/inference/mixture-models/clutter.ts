/**
 * Minka's clutter problem (Minka, 2001, "A family of algorithms for approximate Bayesian inference", §3.3), part of
 * `aifn-methods/inference/mixture-models`: its structure in the model language, its tilted moments for EP, sampling,
 * and the exact posterior on a grid.
 */

import { lift, type Out } from 'aifn-compute/inference/expectation-propagation'
import { normal, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { normalLogPdf } from 'aifn-compute/numerics/special'
import { fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { EpOptions } from 'aifn-compute/inference/expectation-propagation'
import { tiltedByQuadrature, type Tilted } from 'aifn-compute/inference/expectation-propagation'
import { dist, model, type Model } from 'aifn-compute/inference/model'

/** Options of {@link clutterTilted}: Minka's clutter model. */
export interface ClutterFactorOptions {
  /** w, the prior probability that a point is clutter. */
  weight: number
  /** The clutter distribution's variance (it has mean 0). Default 10. */
  clutterVariance?: number
  /** The signal's noise variance around θ. Default 1. */
  noiseVariance?: number
}

/**
 * Minka's clutter factor t(θ) = (1 − w) N(x; θ, σ²) + w N(x; 0, c²) times a Gaussian cavity (Minka 2001, §3.3.1).
 * r is the probability, under the cavity, that x is signal; with d = x − m and s = v + σ²:
 * mean = m + r v d/s, variance = v − r v²/s + r(1 − r) v² d²/s².
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

/** Minka's clutter problem: θ ~ N(0, priorVariance); each xᵢ is N(θ, 1) with probability 1 − w, else N(0, c²). */
export interface ClutterProblem {
  /** w, the clutter probability. */
  weight: number
  /** Default 100. */
  priorVariance?: number
  /** Default 10. */
  clutterVariance?: number
}

/**
 * The clutter problem's structure in the model language (Minka 2001, §3.3): θ ~ N(0, priorVariance) and a plate of N
 * points, each with a signal indicator s_n ~ Bernoulli(1 − w) and x_n ~ N(s_n θ, σ_n²), where σ_n = 1 for signal and
 * √c² for clutter (a table indexed by s_n).
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

/** log N(x; m, v) for variance v. */
const logNormal = (x: number, m: number, v: number): number => normalLogPdf((x - m) / Math.sqrt(v)) - 0.5 * Math.log(v)

const defaults = (p: ClutterProblem) => ({
  weight: p.weight,
  priorVariance: p.priorVariance ?? 100,
  clutterVariance: p.clutterVariance ?? 10,
})

/** log p(x | θ) = Σᵢ log[(1 − w) N(xᵢ; θ, 1) + w N(xᵢ; 0, c²)]. */
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

/** Draw n observations of the clutter problem at a true θ (float64 vector). */
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
 * EP options for the clutter problem, ready for `expectationPropagation`: closed-form tilted moments for α = 1,
 * quadrature for power EP.
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
 * The exact posterior of the clutter problem on an even grid (trapezoid rule): density, mean, variance and
 * log evidence. The grid must hold the posterior mass; the default ±40 does for the usual prior.
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
