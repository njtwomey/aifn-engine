/** Seeded regression problems: a named 1-D function with noise, a linear model in $d$ dimensions, and Friedman #1. */

import { normal, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { regressionTruth } from '../truth'
import { checkCount, type Dataset, generatorRecipe, matrix, vector } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The named 1-D regression functions. */
export type RegressionFunction = 'sine' | 'linear' | 'cubic' | 'step' | 'sinc' | 'bump' | 'doppler'

/** The noise-free functions, each on its default input range. */
const FUNCTIONS: Record<RegressionFunction, { f: (x: number) => number; range: [number, number]; formula: string }> = {
  sine: { f: (x) => Math.sin(x), range: [0, 2 * Math.PI], formula: 'sin x' },
  linear: { f: (x) => 0.5 + 0.8 * x, range: [-2, 2], formula: '0.5 + 0.8x' },
  cubic: { f: (x) => x * x * x - x, range: [-1.5, 1.5], formula: 'x³ − x' },
  step: { f: (x) => (x < 0 ? -1 : 1), range: [-2, 2], formula: 'sign x' },
  sinc: { f: (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)), range: [-4, 4], formula: 'sin(πx)/(πx)' },
  bump: { f: (x) => Math.exp(-8 * x * x), range: [-1, 1], formula: 'exp(−8x²)' },
  // Donoho and Johnstone (1994), "Ideal spatial adaptation by wavelet shrinkage", Biometrika 81(3).
  doppler: {
    f: (x) => Math.sqrt(x * (1 - x)) * Math.sin((2.1 * Math.PI) / (x + 0.05)),
    range: [0, 1],
    formula: '√(x(1 − x)) sin(2.1π / (x + 0.05))',
  },
}

/** Options for `regression1d`. */
export interface Regression1dOptions {
  /** Points (default 50). */
  n?: number
  /** A named function or any function of one number. Default `sine`. */
  fn?: RegressionFunction | ((x: number) => number)
  /** Noise standard deviation $\sigma_0$ (at the left end of the range when it grows). Default 0.2. */
  noise?: number
  /** Input range $[a, b]$; defaults to the named function's, or $[0, 1]$ for a custom one. */
  range?: readonly [number, number]
  /**
   * `random`: x uniform on the range (sorted); `even`: evenly spaced including both ends; `gapped`: non-uniform with a
   * gap (sorted): three quarters of the inputs uniform on the first 45% of the range, the rest on the last 35%, none in
   * between. Default `random`.
   */
  spacing?: 'random' | 'even' | 'gapped'
  /**
   * $h$: the noise standard deviation grows linearly from $\sigma_0$ at the left end to $\sigma_0(1 + h)$ at the right.
   * Default 0.
   */
  heteroscedastic?: number
}

/**
 * A 1-D regression problem $y = f(x) + \varepsilon$ with $\varepsilon \sim \Gauss(0, \sigma(x)^2)$ and
 * $\sigma(x) = \sigma_0(1 + h(x - a)/(b - a))$ on the range $[a, b]$. The named functions are `sine` ($\sin x$ on
 * $[0, 2\pi]$), `linear` ($0.5 + 0.8x$ on $[-2, 2]$), `cubic` ($x^3 - x$ on $[-1.5, 1.5]$), `step` ($\sgn x$ on
 * $[-2, 2]$), `sinc` ($\sin(\pi x)/(\pi x)$ on $[-4, 4]$), `bump` ($e^{-8x^2}$ on $[-1, 1]$) and `doppler`
 * ($\sqrt{x(1 - x)} \sin(2.1\pi/(x + 0.05))$ on $[0, 1]$, Donoho and Johnstone, 1994). The truth (`meta.truth`) knows
 * $f$ and $\sigma(x)$. Throws `DomainError` when $n$ is not a non-negative integer or the function name is unknown.
 *
 * @param s The stream the inputs (child `'x'`) and the noise (child `'noise'`) are drawn from.
 * @param options The size, the function and its range, the noise and its growth, and the spacing of the inputs.
 * @returns A regression dataset: `x` ($n \times 1$, sorted), `y`, and `f`, the noise-free values at `x`.
 *
 * @example Noisy sine: the residuals have the noise's standard deviation
 * const d = regression1d(stream(1), { n: 500, fn: 'sine', noise: 0.2 })
 * const [x, y, f] = [toArray(d.x).map((r) => r[0]), toArray(d.y), toArray(d.f)]
 * print('x:', d.x.shape, ' first x:', x.slice(0, 3), ' first y:', y.slice(0, 3))
 * print('sd of y - f:', Math.sqrt(y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0) / y.length))
 *
 * @example Gapped inputs leave the middle of the range empty
 * const x = toArray(regression1d(stream(1), { n: 200, fn: 'bump', spacing: 'gapped' }).x).map((r) => r[0])
 * // On [-1, 1] the gap is from -0.1 to 0.3.
 * print('left of the gap:', x.filter((v) => v < -0.1).length, ' in it:', x.filter((v) => v > -0.1 && v < 0.3).length)
 */
export function regression1d(s: Stream, options: Regression1dOptions = {}): Dataset {
  const { n = 50, fn = 'sine', noise = 0.2, spacing = 'random', heteroscedastic = 0 } = options
  checkCount(n, 'regression1d')
  const named = typeof fn === 'string' ? FUNCTIONS[fn] : undefined
  if (typeof fn === 'string' && !named) throw new DomainError('regression1d', `regression1d: unknown function ${fn}`)
  const f = named ? named.f : (fn as (x: number) => number)
  const [lo, hi] = options.range ?? named?.range ?? [0, 1]
  const xs = new Float64Array(n)
  const inputs = child(s, 'x')
  const position = (): number => {
    if (spacing !== 'gapped') return uniform(inputs)
    return uniform(inputs) < 0.75 ? 0.45 * uniform(inputs) : 0.65 + 0.35 * uniform(inputs)
  }
  for (let i = 0; i < n; i++)
    xs[i] = spacing === 'even' ? (n > 1 ? lo + ((hi - lo) * i) / (n - 1) : lo) : lo + (hi - lo) * position()
  if (spacing !== 'even') xs.sort()
  const eps = child(s, 'noise')
  const fx = xs.map(f)
  const sdAt = (x: number) => noise * (1 + (heteroscedastic * (x - lo)) / (hi - lo || 1))
  const y = fx.map((v, i) => v + sdAt(xs[i]) * normal(eps))
  // E[σ(x)²] for x uniform on the range: σ² ∫₀¹ (1 + h u)² du = σ² (1 + h + h²/3).
  const h = heteroscedastic
  const truth = regressionTruth((x) => f(x[0]), noise, {
    sdAt: (x) => sdAt(x[0]),
    bayesRisk: noise * noise * (1 + h + (h * h) / 3),
  })
  return {
    kind: 'dataset',
    x: matrix(xs, n, 1),
    y: vector(y),
    f: vector(fx),
    meta: {
      name: typeof fn === 'string' ? fn : 'custom function',
      description: `${n} noisy observations of ${named ? `y = ${named.formula}` : 'a function'} on [${lo}, ${hi}]${spacing === 'gapped' ? ' (dense on the left, sparse on the right, a gap between)' : ''} with noise sd ${noise}${heteroscedastic ? ' growing to the right' : ''}.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 'y',
      key: s.key,
      truth,
      recipe: generatorRecipe('regression1d', s.key, {
        n,
        fn: typeof fn === 'string' ? fn : 'custom',
        noise,
        spacing,
        heteroscedastic,
        range: [lo, hi],
      }),
    },
  }
}

/** Options for `linearRegressionData`. */
export interface LinearRegressionOptions {
  /** Points (default 100). */
  n?: number
  /** Number of features $d$. Default 3 (ignored when `weights` is given). */
  d?: number
  /** True weights $\wvec$; drawn $\Gauss(0, 1)$ from the substream `weights` when omitted. */
  weights?: readonly number[]
  /** The intercept $b$ (default 0). */
  bias?: number
  /** The standard deviation of the Gaussian noise (default 0.5). */
  noise?: number
  /**
   * Correlation $\rho$ between neighbouring features (an AR(1) design), in $(-1, 1)$; 0 (the default) gives independent
   * $\Gauss(0, 1)$ features.
   */
  correlation?: number
}

/**
 * A linear model $y = \xvec^\top\wvec + b + \varepsilon$ with standard normal features (optionally AR(1)-correlated
 * along the feature index: $x_j = \rho x_{j-1} + \sqrt{1 - \rho^2} z_j$) and Gaussian noise. `f` holds
 * $\xvec^\top\wvec + b$; the true weights are returned as `weights` and `bias`, and given in the description. Throws
 * `DomainError` when $n$ is not a non-negative integer; $\rho$ is not checked.
 *
 * @param s The stream the weights (child `'weights'`, when not given), the features (child `'x'`) and the noise (child
 *   `'noise'`) are drawn from.
 * @param options The size, the true weights or their number, the intercept, the noise and the feature correlation.
 * @returns A regression dataset (`x` $n \times d$, `y`, `f`) with the true `weights` and `bias`.
 *
 * @example Correlated features and the noise's standard deviation
 * const d = linearRegressionData(stream(1), { n: 1000, d: 3, correlation: 0.8, noise: 0.5 })
 * const [x, y, f] = [toArray(d.x), toArray(d.y), toArray(d.f)]
 * print('x:', d.x.shape, ' weights:', d.weights, ' bias:', d.bias)
 * print('first row:', x[0], ' y:', y[0])
 * print('mean of x1 x2 (the correlation):', x.reduce((a, r) => a + r[0] * r[1], 0) / 1000)
 * print('sd of y - f:', Math.sqrt(y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0) / 1000))
 */
export function linearRegressionData(
  s: Stream,
  options: LinearRegressionOptions = {},
): Dataset & { weights: Float64Array; bias: number } {
  const { n = 100, bias = 0, noise = 0.5, correlation = 0 } = options
  checkCount(n, 'linearRegression')
  const ws = child(s, 'weights')
  const w = Float64Array.from(options.weights ?? Array.from({ length: options.d ?? 3 }, () => normal(ws)))
  const d = w.length
  const x = new Float64Array(n * d)
  const fx = new Float64Array(n)
  const y = new Float64Array(n)
  const xs = child(s, 'x')
  const eps = child(s, 'noise')
  const scale = Math.sqrt(1 - correlation * correlation)
  for (let i = 0; i < n; i++) {
    let prev = 0
    let v = bias
    for (let j = 0; j < d; j++) {
      const z = normal(xs)
      const xij = j === 0 ? z : correlation * prev + scale * z
      prev = xij
      x[i * d + j] = xij
      v += xij * w[j]
    }
    fx[i] = v
    y[i] = v + noise * normal(eps)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: vector(y),
    f: vector(fx),
    weights: w,
    bias,
    meta: {
      truth: regressionTruth((r) => {
        let v = bias
        for (let j = 0; j < d; j++) v += r[j] * w[j]
        return v
      }, noise),
      recipe: generatorRecipe('linearRegressionData', s.key, { n, weights: Array.from(w), bias, noise, correlation }),
      name: 'linear regression',
      description: `${n} points from y = xᵀw + ${bias} + ε with ${d} features, w = (${Array.from(w, (v) => v.toFixed(2)).join(', ')}) and noise sd ${noise}.`,
      task: 'regression',
      featureNames: Array.from({ length: d }, (_, j) => `x${j + 1}`),
      targetName: 'y',
      key: s.key,
    },
  }
}

/**
 * Friedman's first benchmark (Friedman, 1991, "Multivariate adaptive regression splines", Annals of Statistics 19(1)):
 * $\xvec$ uniform on $[0, 1]^d$ ($d \ge 5$) and
 * $y = 10 \sin(\pi x_1x_2) + 20(x_3 - 1/2)^2 + 10x_4 + 5x_5 + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$;
 * features beyond the fifth are noise, as in `sklearn.datasets.make_friedman1`. Throws `DomainError` when $n$ is not a
 * non-negative integer or $d < 5$.
 *
 * @param s The stream the features (child `'x'`) and the noise (child `'noise'`) are drawn from.
 * @param options `n` (default 200), the number of points; `d` (default 10), the number of features, at least 5;
 *   `noise` (default 1), the standard deviation $\sigma$ of the noise.
 * @returns A regression dataset: `x` ($n \times d$), `y`, and `f` the noise-free values.
 *
 * @example The formula gives `f`, and the noise has standard deviation 1
 * const d = friedman1(stream(1), { n: 1000 })
 * const [x, y, f] = [toArray(d.x), toArray(d.y), toArray(d.f)]
 * print('x:', d.x.shape, ' first row:', x[0])
 * const [a, b, c, e, g] = x[0]
 * print('formula:', 10 * Math.sin(Math.PI * a * b) + 20 * (c - 0.5) ** 2 + 10 * e + 5 * g, ' f:', f[0])
 * print('sd of y - f:', Math.sqrt(y.reduce((t, v, i) => t + (v - f[i]) ** 2, 0) / 1000))
 */
export function friedman1(s: Stream, options: { n?: number; d?: number; noise?: number } = {}): Dataset {
  const { n = 200, d = 10, noise = 1 } = options
  checkCount(n, 'friedman1')
  if (d < 5) throw new DomainError('friedman1', 'friedman1: needs at least five features')
  const x = new Float64Array(n * d)
  const fx = new Float64Array(n)
  const y = new Float64Array(n)
  const xs = child(s, 'x')
  const eps = child(s, 'noise')
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) x[i * d + j] = uniform(xs)
    const r = x.subarray(i * d, i * d + 5)
    fx[i] = 10 * Math.sin(Math.PI * r[0] * r[1]) + 20 * (r[2] - 0.5) ** 2 + 10 * r[3] + 5 * r[4]
    y[i] = fx[i] + noise * normal(eps)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: vector(y),
    f: vector(fx),
    meta: {
      truth: regressionTruth(
        (r) => 10 * Math.sin(Math.PI * r[0] * r[1]) + 20 * (r[2] - 0.5) ** 2 + 10 * r[3] + 5 * r[4],
        noise,
      ),
      recipe: generatorRecipe('friedman1', s.key, { n, d, noise }),
      name: 'Friedman #1',
      description: `Friedman's first regression benchmark: ${n} points, ${d} uniform features of which five matter.`,
      task: 'regression',
      featureNames: Array.from({ length: d }, (_, j) => `x${j + 1}`),
      targetName: 'y',
      source: 'Friedman (1991), Annals of Statistics 19(1)',
      key: s.key,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'regression1d',
    name: 'One-dimensional regression',
    summary: 'y = f(x) + noise for a named function f of one input.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 50 }),
      fn: oneOf(['sine', 'linear', 'cubic', 'step', 'sinc', 'bump', 'doppler']),
      noise: real(0, 2, { default: 0.2 }),
      spacing: oneOf(['random', 'even', 'gapped']),
      heteroscedastic: real(0, 10, { default: 0 }),
    }),
    truth: true,
    random: true,
    notes: ['local-regression', 'regression-splines'],
  },
  regression1d,
)

dataset(
  {
    key: 'linearRegressionData',
    name: 'Linear regression data',
    summary: 'A linear model with Gaussian noise and AR(1)-correlated features.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 100 }),
      d: int(1, 50, { default: 3 }),
      bias: real(-10, 10, { default: 0 }),
      noise: real(0, 5, { default: 0.5 }),
      correlation: real(-0.99, 0.99, { default: 0 }),
    }),
    truth: true,
    random: true,
    notes: ['linear-regression'],
  },
  linearRegressionData,
)

dataset(
  {
    key: 'friedman1',
    name: 'Friedman #1',
    summary: "Friedman's first benchmark: a nonlinear function of five uniform inputs plus uninformative ones.",
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 200 }),
      d: int(5, 50, { default: 10 }),
      noise: real(0, 5, { default: 1 }),
    }),
    truth: true,
    random: true,
  },
  friedman1,
)
