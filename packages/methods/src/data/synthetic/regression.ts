/** Seeded regression problems: a named 1-D function with noise, a linear model in d dimensions, and Friedman #1. */

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
  n?: number
  /** A named function or any function of one number. Default `sine`. */
  fn?: RegressionFunction | ((x: number) => number)
  /** Noise standard deviation. Default 0.2. */
  noise?: number
  /** Input range; defaults to the named function's. */
  range?: readonly [number, number]
  /**
   * `random`: x uniform on the range (sorted); `even`: evenly spaced including both ends; `gapped`: non-uniform with a
   * gap (sorted): three quarters of the inputs uniform on the first 45% of the range, the rest on the last 35%, none in
   * between. Default `random`.
   */
  spacing?: 'random' | 'even' | 'gapped'
  /** Noise sd grows linearly from `noise` at the left end to `noise · (1 + heteroscedastic)` at the right. Default 0. */
  heteroscedastic?: number
}

/**
 * A 1-D regression problem y = f(x) + ε with ε ~ N(0, σ(x)²). Returns x (n × 1, sorted), y, and `f`, the noise-free
 * values at x.
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
  n?: number
  /** Number of features. Default 3 (ignored when `weights` is given). */
  d?: number
  /** True weights; drawn N(0, 1) from the substream `weights` when omitted. */
  weights?: readonly number[]
  bias?: number
  noise?: number
  /** Correlation between neighbouring features (an AR(1) design); 0 gives independent N(0, 1) features. */
  correlation?: number
}

/**
 * A linear model y = xᵀw + b + ε with standard normal features (optionally AR(1)-correlated along the feature index)
 * and Gaussian noise. `f` holds xᵀw + b; the true weights are in the description and in `meta`.
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
 * x uniform on [0, 1]^d (d ≥ 5) and y = 10 sin(π x₁x₂) + 20(x₃ − 1/2)² + 10x₄ + 5x₅ + ε; features beyond the fifth are
 * noise, as in `sklearn.datasets.make_friedman1`.
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
