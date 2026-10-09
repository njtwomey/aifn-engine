/**
 * Tilted distributions: a Gaussian cavity $\Gauss(\theta; m, v)$ times one exact factor, summarised by its log
 * normaliser and first two moments (the moment-matching step of EP and ADF). Closed forms for step, probit and
 * interval factors (truncated Gaussian, the TrueSkill draw) (Minka 2001, thesis §3.2–3.3, §5; Herbrich, Minka &
 * Graepel 2007, "TrueSkill", NIPS), and a quadrature fallback for any factor raised to a power. The clutter factor's
 * closed form is `clutterTilted` in `aifn-methods/inference/mixture-models`.
 *
 * Every closed form broadcasts over its arguments (numbers in, numbers out; tensors elementwise). The truncated-normal
 * ratios $v(t) = \phi(t)/\Phi(t)$ and $w(t) = v(t)(v(t) + t)$ come from `aifn-compute/numerics/special`, accurate far
 * into both tails, so extreme cavities do not give $0/0$. These functions are not differentiable primitives: EP uses
 * them for updates, not under autodiff.
 */

import {
  normalLogCdf,
  normalLogIntervalProbability,
  normalLogPdf,
  truncatedNormalV,
  truncatedNormalW,
} from 'aifn-compute/numerics/special'
import {
  broadcastShapes,
  broadcastTo,
  fromData,
  isTensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

/**
 * The tilted distribution's log normaliser $\log Z = \log \int \Gauss(\theta; m, v) f(\theta)\,d\theta$ and its mean
 * and variance.
 */
export interface Tilted<T extends Value = number> {
  /** The log normaliser $\log Z$. */
  logZ: T
  /** The mean of the tilted distribution. */
  mean: T
  /** The variance of the tilted distribution. */
  variance: T
}

type Numbers = readonly (number | Tensor)[]

/**
 * Apply a scalar kernel returning several named numbers elementwise over broadcast arguments: numbers in give numbers
 * out, otherwise each named output is a tensor of the broadcast shape. The tilted moments are built on it. Not
 * differentiable: the kernel sees plain numbers.
 *
 * @param args The arguments, numbers or tensors, broadcast against each other.
 * @param keys The names of the kernel's outputs, which become the result's fields.
 * @param f The kernel: one number per argument in, one number per key out.
 * @returns For each key, a number (all arguments numbers) or a float64 tensor of the broadcast shape.
 *
 * @example A kernel with two outputs over a broadcast pair
 * print(lift([1, tensor([0, 1, 2])], ['sum', 'product'], (a, b) => ({ sum: a + b, product: a * b })))
 * print(lift([2, 3], ['sum'], (a, b) => ({ sum: a + b })))
 */
export function lift<K extends string>(
  args: Numbers,
  keys: readonly K[],
  f: (...x: number[]) => Record<K, number>,
): Record<K, number | Tensor> {
  if (args.every((a) => typeof a === 'number')) return f(...(args as number[]))
  const shape = broadcastShapes(...args.map((a) => (isTensor(a) ? a.shape : [])))
  const flats = args.map((a) => (isTensor(a) ? toFlat(broadcastTo(a, shape)) : null))
  const n = shape.reduce((p, s) => p * s, 1)
  const out = Object.fromEntries(keys.map((k) => [k, new Float64Array(n)])) as Record<K, Float64Array>
  for (let i = 0; i < n; i++) {
    const r = f(...args.map((a, j) => (flats[j] ? flats[j]![i] : (a as number))))
    for (const k of keys) out[k][i] = r[k]
  }
  return Object.fromEntries(keys.map((k) => [k, fromData(out[k], shape)])) as Record<K, Tensor>
}

/** The output type of a lifted kernel: a number when every argument is a number, else a tensor. */
export type Out<T> = [T] extends [number] ? number : Tensor

/**
 * The truncated-normal mean function $v(t) = \phi(t)/\Phi(t)$.
 *
 * @param t The standardised distance of the cavity mean above the truncation point.
 * @returns $v(t)$.
 */
const v = (t: number) => truncatedNormalV(t) as number
/**
 * The truncated-normal variance function $w(t) = v(t)(v(t) + t)$, in $(0, 1)$.
 *
 * @param t The standardised distance of the cavity mean above the truncation point.
 * @returns $w(t)$.
 */
const w = (t: number) => truncatedNormalW(t) as number

/**
 * Step factor $\indicator(\theta > e)$, $e$ the threshold: the cavity truncated below. With $t = (m - e)/\sqrt{v}$:
 * $\log Z = \log \Phi(t)$, mean $m + \sqrt{v}\,v(t)$ and variance $v (1 - w(t))$.
 *
 * @param mean The cavity mean $m$ (a number, or a tensor for a batch).
 * @param variance The cavity variance $v$ (positive).
 * @param threshold The threshold $e$ below which the factor is zero (default 0).
 * @returns $\log Z$ and the tilted mean and variance, numbers or tensors of the broadcast shape.
 *
 * @example A standard normal truncated at zero: the half-normal
 * print(stepTilted(0, 1))
 * print('half-normal mean:', Math.sqrt(2 / Math.PI), ' variance:', 1 - 2 / Math.PI)
 *
 * @example A batch of thresholds
 * print(stepTilted(0, 1, tensor([-2, 0, 2])))
 */
export function stepTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  threshold: number | Tensor = 0,
): Tilted<Out<M | S>> {
  return lift([mean, variance, threshold], ['logZ', 'mean', 'variance'], (m, s2, e) => {
    const s = Math.sqrt(s2)
    const t = (m - e) / s
    return { logZ: normalLogCdf(t) as number, mean: m + s * v(t), variance: s2 * (1 - w(t)) }
  }) as Tilted<Out<M | S>>
}

/** Options of {@link probitTilted}. */
export interface ProbitOptions {
  /** $c$ in $\Phi(y(\theta - c)/s)$. Default 0. */
  offset?: number | Tensor
  /** $s^2$, the variance of the Gaussian noise the probit integrates out. Default 1. */
  noiseVariance?: number | Tensor
}

/**
 * Probit factor $\Phi(y(\theta - c)/s)$, $y = \pm 1$: the step factor on $\theta$ plus $\Gauss(0, s^2)$ noise. With
 * $d = \sqrt{s^2 + v}$ and $z = y(m - c)/d$: $\log Z = \log \Phi(z)$, mean $m + y\,v\,v(z)/d$ and variance
 * $v - v^2 w(z)/(s^2 + v)$. The site of GP classification and probit regression.
 *
 * @param mean The cavity mean $m$ (a number, or a tensor for a batch).
 * @param variance The cavity variance $v$ (positive).
 * @param y The label $y$, $+1$ or $-1$ (default $+1$); other values scale the argument of $\Phi$.
 * @param options The offset $c$ and the noise variance $s^2$.
 * @param options.offset $c$, where the probit is one half (default 0).
 * @param options.noiseVariance $s^2$, the variance of the noise the probit integrates out (default 1).
 * @returns $\log Z$ and the tilted mean and variance, numbers or tensors of the broadcast shape.
 *
 * @example A positive label pulls the cavity up
 * print('y = +1:', probitTilted(0, 1, 1))
 * print('y = -1:', probitTilted(0, 1, -1))
 *
 * @example Without noise the probit is a step
 * print('probit:', probitTilted(0.5, 1, 1, { noiseVariance: 1e-12 }))
 * print('step:', stepTilted(0.5, 1))
 */
export function probitTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  y: number | Tensor = 1,
  { offset = 0, noiseVariance = 1 }: ProbitOptions = {},
): Tilted<Out<M | S>> {
  return lift([mean, variance, y, offset, noiseVariance], ['logZ', 'mean', 'variance'], (m, s2, yy, c, n2) => {
    const d = Math.sqrt(n2 + s2)
    const z = (yy * (m - c)) / d
    return {
      logZ: normalLogCdf(z) as number,
      mean: m + (yy * s2 * v(z)) / d,
      variance: s2 - (s2 * s2 * w(z)) / (n2 + s2),
    }
  }) as Tilted<Out<M | S>>
}

/**
 * Interval factor $\indicator(l < \theta < u)$: the cavity truncated to an interval (either end may be infinite).
 * With $a = (l - m)/\sqrt{v}$, $b = (u - m)/\sqrt{v}$, $Z = \Phi(b) - \Phi(a)$ (computed without cancellation),
 * $r_a = \phi(a)/Z$ and $r_b = \phi(b)/Z$: mean $m + \sqrt{v}(r_a - r_b)$ and variance
 * $v [1 + a r_a - b r_b - (r_a - r_b)^2]$. The TrueSkill draw is the interval $(-\varepsilon, \varepsilon)$.
 *
 * @param mean The cavity mean $m$ (a number, or a tensor for a batch).
 * @param variance The cavity variance $v$ (positive).
 * @param lower The lower end $l$ (may be $-\infty$).
 * @param upper The upper end $u$ (may be $+\infty$).
 * @returns $\log Z$ and the tilted mean and variance, numbers or tensors of the broadcast shape.
 *
 * @example A standard normal truncated to (-1, 1)
 * const t = intervalTilted(0, 1, -1, 1)
 * print(t)
 * print('Z =', Math.exp(t.logZ))
 *
 * @example A half-infinite interval is a step
 * print('interval:', intervalTilted(0.5, 2, 0, Infinity))
 * print('step:', stepTilted(0.5, 2))
 */
export function intervalTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  lower: number | Tensor,
  upper: number | Tensor,
): Tilted<Out<M | S>> {
  return lift([mean, variance, lower, upper], ['logZ', 'mean', 'variance'], (m, s2, lo, hi) => {
    const s = Math.sqrt(s2)
    const a = (lo - m) / s
    const b = (hi - m) / s
    const logZ = normalLogIntervalProbability(a, b) as number
    const ra = Number.isFinite(a) ? Math.exp((normalLogPdf(a) as number) - logZ) : 0
    const rb = Number.isFinite(b) ? Math.exp((normalLogPdf(b) as number) - logZ) : 0
    const ara = Number.isFinite(a) ? a * ra : 0
    const brb = Number.isFinite(b) ? b * rb : 0
    return { logZ, mean: m + s * (ra - rb), variance: s2 * (1 + ara - brb - (ra - rb) ** 2) }
  }) as Tilted<Out<M | S>>
}

/**
 * $\log \Gauss(x; m, s^2)$, through the one standard normal log-density of `aifn-compute/numerics/special`.
 *
 * @param x The point.
 * @param m The mean.
 * @param s2 The variance (positive).
 * @returns The log-density.
 */
const logNormal = (x: number, m: number, s2: number) =>
  (normalLogPdf((x - m) / Math.sqrt(s2)) as number) - 0.5 * Math.log(s2)

/** Options of {@link tiltedByQuadrature}. */
export interface QuadratureTiltOptions {
  /** Raise the factor to this power (power EP). Default 1. */
  power?: number
  /** Grid points (default 801), at least 2, spanning the grid. */
  points?: number
  /** The half-width of the grid, in cavity standard deviations either side of the mean (default 10). */
  width?: number
}

/**
 * Moments of $\Gauss(\theta; m, v) f(\theta)^p$ ($p$ the power) by the trapezoid rule on a grid of `width` cavity
 * standard deviations either side of the mean, for factors without closed forms. Accurate when the tilted mass lies
 * within the grid and $f$ is smooth on it (a jump costs accuracy of the order of the grid spacing). Numbers only.
 *
 * @param mean The cavity mean $m$.
 * @param variance The cavity variance $v$ (positive).
 * @param logFactor $\log f(\theta)$; $-\infty$ where the factor is zero.
 * @param options The power and the grid.
 * @param options.power The power $p$ the factor is raised to (default 1; power EP uses a fraction).
 * @param options.points The number of grid points (default 801).
 * @param options.width The grid's half-width in cavity standard deviations (default 10).
 * @returns $\log Z$ and the tilted mean and variance.
 *
 * @example A Gaussian factor, against its closed form
 * // The cavity N(0, 1) times f(θ) = exp(−(θ − 2)²/2), a Gaussian in θ, so the product has a closed form; its
 * // normaliser is ∫ N(θ; 0, 1) exp(−(θ − 2)²/2) dθ = e^{−1}/√2.
 * print(tiltedByQuadrature(0, 1, (t) => -((t - 2) ** 2) / 2))
 * print('closed form:', gaussianMoments(multiplyGaussians(naturalGaussian(0, 1), naturalGaussian(2, 1))))
 * print('closed-form log Z =', -1 - Math.log(2) / 2)
 *
 * @example Power EP takes a fraction of the factor
 * print(tiltedByQuadrature(0, 1, (t) => -((t - 2) ** 2) / 2, { power: 0.5 }))
 */
export function tiltedByQuadrature(
  mean: number,
  variance: number,
  logFactor: (theta: number) => number,
  { power = 1, points = 801, width = 10 }: QuadratureTiltOptions = {},
): Tilted {
  const s = Math.sqrt(variance)
  const lo = mean - width * s
  const h = (2 * width * s) / (points - 1)
  const logs = new Float64Array(points)
  let top = -Infinity
  for (let i = 0; i < points; i++) {
    const t = lo + i * h
    logs[i] = logNormal(t, mean, variance) + power * logFactor(t)
    if (logs[i] > top) top = logs[i]
  }
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i < points; i++) {
    const t = lo + i * h
    const f = Math.exp(logs[i] - top) * (i === 0 || i === points - 1 ? 0.5 : 1)
    z += f
    m1 += f * t
    m2 += f * t * t
  }
  const m = m1 / z
  return { logZ: top + Math.log(z * h), mean: m, variance: m2 / z - m * m }
}
