/**
 * Integration in several dimensions: tensor-product rules in 2-D, Monte Carlo integration with its standard error (a
 * traceable algorithm that adds a batch of points per step), low-discrepancy Halton and Sobol sequences, and randomised
 * quasi–Monte Carlo with a standard error from independent random shifts. Sources: Owen (2013), "Monte Carlo theory,
 * methods and examples", ch. 2 and 17; Halton (1960); Sobol' (1967); Joe & Kuo (2008); Cranley & Patterson (1976).
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { gaussLegendre, type QuadratureRule } from './gauss'

/** A function on ℝᵈ, given the point as a vector of length d. */
export type MultivariateIntegrand = (x: Vector) => Scalar

const flat = (v: VectorLike, where: string): number[] => {
  const out = Array.from(isTensor(v) ? toFlat(v) : (v as ArrayLike<number>))
  if (!out.every(Number.isFinite)) throw new ShapeError(where, `${where}: the box corners must be finite`)
  return out
}

/**
 * The tensor-product rule of two 1-D rules: points (xᵢ, yⱼ) as rows of an (nx·ny)×2 matrix, x varying slowest, with
 * weights wᵢvⱼ. Exact for products of polynomials each rule integrates exactly.
 */
export function productRule(ruleX: QuadratureRule, ruleY: QuadratureRule): { points: Tensor; weights: Tensor } {
  const xs = toFlat(ruleX.nodes)
  const wx = toFlat(ruleX.weights)
  const ys = toFlat(ruleY.nodes)
  const wy = toFlat(ruleY.weights)
  const points = new Float64Array(xs.length * ys.length * 2)
  const weights = new Float64Array(xs.length * ys.length)
  let k = 0
  for (let i = 0; i < xs.length; i++)
    for (let j = 0; j < ys.length; j++, k++) {
      points[2 * k] = xs[i]
      points[2 * k + 1] = ys[j]
      weights[k] = wx[i] * wy[j]
    }
  return { points: fromData(points, [k, 2]), weights: fromData(weights, [k]) }
}

/** Equal-panel Newton–Cotes rules as nodes and weights, for product rules. */
function newtonCotesRule(kind: 'trapezoid' | 'simpson', n: number, a: number, b: number): QuadratureRule {
  if (kind === 'simpson' && n % 2 !== 0)
    throw new DomainError('integrate2d', 'integrate2d: Simpson needs an even number of panels')
  const h = (b - a) / n
  const nodes = Float64Array.from({ length: n + 1 }, (_, i) => a + i * h)
  const weights = Float64Array.from({ length: n + 1 }, (_, i) => {
    if (kind === 'trapezoid') return i === 0 || i === n ? h / 2 : h
    return ((i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2) * h) / 3
  })
  return { nodes: fromData(nodes, [n + 1]), weights: fromData(weights, [n + 1]) }
}

/**
 * ∬ f(x, y) dx dy over the rectangle [ax, bx]×[ay, by] by a tensor-product rule: `'gauss-legendre'` (default, n points
 * per axis, default 20), `'simpson'` or `'trapezoid'` (n panels per axis).
 */
export function integrate2d(
  f: (x: number, y: number) => number,
  [ax, bx]: readonly [number, number],
  [ay, by]: readonly [number, number],
  { n = 20, rule = 'gauss-legendre' }: { n?: number; rule?: 'gauss-legendre' | 'simpson' | 'trapezoid' } = {},
): number {
  const make = (a: number, b: number) =>
    rule === 'gauss-legendre' ? gaussLegendre(n, [a, b]) : newtonCotesRule(rule, n, a, b)
  const { points, weights } = productRule(make(ax, bx), make(ay, by))
  let total = 0
  for (let k = 0; k < weights.shape[0]; k++) total += weights.data[k] * f(points.data[2 * k], points.data[2 * k + 1])
  return total
}

// ---------------------------------------------------------------------------------------------------------------------
// Monte Carlo.

/** The state of `monteCarlo` (plain data: the draws of step t come from the runner's `ctx.stream`). */
export type MonteCarloState = Status & {
  /** Points drawn so far. */
  n: Size
  /** The running mean of f and the sum of squared deviations (Welford, 1962). */
  mean: number
  sumSquares: number
  /** The estimate volume·mean of the integral over the box. */
  value: number
  /** Its standard error volume·√(s²/n), with s² the sample variance of f (NaN below two points). */
  standardError: number
  /** The volume of the box. */
  volume: number
  /** The last batch of points (rows) and f at each, for drawing. */
  batch: Tensor
  batchValues: Tensor
  /** True once the running mean is not finite (f returned ±∞ or NaN). */
  diverged: boolean
}

/** Options for `monteCarlo`. */
export type MonteCarloOptions = {
  /** Lower and upper corners of the box (length d each). */
  lo: VectorLike
  hi: VectorLike
  /** Points per step. Default 100. */
  batch?: Size
}

/**
 * Plain Monte Carlo integration over the box [lo, hi] ⊂ ℝᵈ: draw points uniformly, estimate the integral by the
 * volume times the mean of f, with standard error volume·s/√n, so the error falls as n^{−1/2} in any dimension. Each
 * step adds a batch drawn from the step's stream. `init` takes no start. It never converges: run it for a budget of
 * steps and read `standardError`.
 */
export function monteCarlo(f: MultivariateIntegrand, options: MonteCarloOptions): Algorithm<unknown, MonteCarloState> {
  const lo = flat(options.lo, 'monteCarlo')
  const hi = flat(options.hi, 'monteCarlo')
  if (lo.length !== hi.length) throw new ShapeError('monteCarlo', 'monteCarlo: lo and hi differ in length')
  const d = lo.length
  const batchSize = options.batch ?? 100
  const volume = lo.reduce((v, l, i) => v * (hi[i] - l), 1)
  return {
    name: 'monte-carlo',
    init: () => ({
      t: 0,
      diverged: false,
      n: 0,
      mean: 0,
      sumSquares: 0,
      value: NaN,
      standardError: NaN,
      volume,
      batch: fromData(new Float64Array(0), [0, d]),
      batchValues: fromData(new Float64Array(0), [0]),
    }),
    step: (s, ctx) => {
      const u = uniform(ctx.stream, 0, 1, { shape: [batchSize, d] }).data
      const points = new Float64Array(batchSize * d)
      const values = new Float64Array(batchSize)
      let { n, mean, sumSquares } = s
      for (let k = 0; k < batchSize; k++) {
        const x = new Float64Array(d)
        for (let j = 0; j < d; j++) x[j] = lo[j] + (hi[j] - lo[j]) * u[k * d + j]
        points.set(x, k * d)
        const fx = f(fromData(x, [d]))
        values[k] = fx
        n++
        const delta = fx - mean
        mean += delta / n
        sumSquares += delta * (fx - mean)
      }
      return {
        ...s,
        t: s.t + 1,
        n,
        mean,
        sumSquares,
        value: volume * mean,
        standardError: n > 1 ? volume * Math.sqrt(sumSquares / (n - 1) / n) : NaN,
        batch: fromData(points, [batchSize, d]),
        batchValues: fromData(values, [batchSize]),
        diverged: !Number.isFinite(mean),
      }
    },
  }
}

/** The result of `integrateMonteCarlo` and `quasiMonteCarlo`. */
export type MonteCarloResult = { value: number; standardError: number; n: Size }

/**
 * ∫ over [lo, hi] of f by plain Monte Carlo with n points (default 10 000), drawn from stream `s` (first, as every
 * random function). See `monteCarlo`; the points are the draws of its step 0 under root `s`.
 */
export function integrateMonteCarlo(
  s: Stream,
  f: MultivariateIntegrand,
  lo: VectorLike,
  hi: VectorLike,
  { n = 10000 }: { n?: Size } = {},
): MonteCarloResult {
  const alg = monteCarlo(f, { lo, hi, batch: n })
  const state = alg.step(alg.init(undefined, child(s, 'init')), { t: 0, stream: child(s, 'step', 0) })
  return { value: state.value, standardError: state.standardError, n: state.n }
}

// ---------------------------------------------------------------------------------------------------------------------
// Low-discrepancy sequences.

const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

/** The radical inverse of i in base b: the digits of i mirrored about the radix point. */
function radicalInverse(i: number, b: number): number {
  let result = 0
  let f = 1 / b
  while (i > 0) {
    result += f * (i % b)
    i = Math.floor(i / b)
    f /= b
  }
  return result
}

/**
 * The first n points of the d-dimensional Halton sequence (Halton, 1960), coordinate j the radical inverse in the j-th
 * prime base, starting at index `skip` (default 0, whose point is the origin), as an n×d matrix. d ≤ 25. Unscrambled,
 * so it matches scipy's `qmc.Halton(d, scramble=False)`.
 */
export function halton(n: Size, d: Size, { skip = 0 }: { skip?: Size } = {}): Tensor {
  if (d > PRIMES.length) throw new ShapeError('halton', `halton: at most ${PRIMES.length} dimensions`)
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = radicalInverse(i + skip, PRIMES[j])
  return fromData(out, [n, d])
}

// Sobol direction numbers of Joe & Kuo (2008), "Constructing Sobol sequences with better two-dimensional projections",
// file new-joe-kuo-6.21201, for dimensions 2–21 (as scipy ships them): the primitive polynomial (with its leading and
// trailing 1 bits) and the initial m values. Dimension 1 is the van der Corput sequence.
const SOBOL_POLY = [3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97, 103, 109, 115, 131, 137]
const SOBOL_M = [
  [1],
  [1, 3],
  [1, 3, 1],
  [1, 1, 1],
  [1, 1, 3, 3],
  [1, 3, 5, 13],
  [1, 1, 5, 5, 17],
  [1, 1, 5, 5, 5],
  [1, 1, 7, 11, 19],
  [1, 1, 5, 1, 1],
  [1, 1, 1, 3, 11],
  [1, 3, 5, 5, 31],
  [1, 3, 3, 9, 7, 49],
  [1, 1, 1, 15, 21, 21],
  [1, 3, 1, 13, 27, 49],
  [1, 1, 1, 15, 7, 5],
  [1, 3, 1, 15, 13, 25],
  [1, 1, 5, 5, 19, 61],
  [1, 3, 7, 11, 23, 15, 103],
  [1, 3, 7, 13, 13, 15, 69],
]
const SOBOL_BITS = 32

/** Direction numbers V[1..32] (1-based, left-aligned in 32 bits) for one dimension (0-based). */
function directions(dim: number): Uint32Array {
  const V = new Uint32Array(SOBOL_BITS + 1)
  if (dim === 0) {
    for (let i = 1; i <= SOBOL_BITS; i++) V[i] = 2 ** (SOBOL_BITS - i)
    return V
  }
  const poly = SOBOL_POLY[dim - 1]
  const m = SOBOL_M[dim - 1]
  const s = Math.floor(Math.log2(poly))
  const a = (poly >> 1) & ((1 << (s - 1)) - 1)
  for (let i = 1; i <= s; i++) V[i] = m[i - 1] * 2 ** (SOBOL_BITS - i)
  // Bratley & Fox (1988), Algorithm 659: v_i = v_{i−s} ⊕ (v_{i−s} >> s) ⊕ Σ_k a_k v_{i−k}.
  for (let i = s + 1; i <= SOBOL_BITS; i++) {
    let v = V[i - s] ^ (V[i - s] >>> s)
    for (let k = 1; k <= s - 1; k++) if ((a >> (s - 1 - k)) & 1) v ^= V[i - k]
    V[i] = v >>> 0
  }
  return V
}

/**
 * The first n points of the d-dimensional Sobol sequence (Sobol', 1967) in Gray-code order (Antonov & Saleev, 1979),
 * starting at index `skip` (default 0, the origin), as an n×d matrix; d ≤ 21. Unscrambled, so it matches scipy's
 * `qmc.Sobol(d, scramble=False)`. Balance properties hold for n a power of 2.
 */
export function sobol(n: Size, d: Size, { skip = 0 }: { skip?: Size } = {}): Tensor {
  if (d > SOBOL_POLY.length + 1) throw new ShapeError('sobol', `sobol: at most ${SOBOL_POLY.length + 1} dimensions`)
  const V = Array.from({ length: d }, (_, j) => directions(j))
  const out = new Float64Array(n * d)
  const X = new Uint32Array(d)
  const scale = 2 ** -SOBOL_BITS
  for (let i = 0; i < skip + n; i++) {
    if (i > 0) {
      // The index (1-based) of the lowest zero bit of i − 1.
      let c = 1
      let value = i - 1
      while (value & 1) {
        value >>>= 1
        c++
      }
      for (let j = 0; j < d; j++) X[j] = (X[j] ^ V[j][c]) >>> 0
    }
    if (i >= skip) for (let j = 0; j < d; j++) out[(i - skip) * d + j] = X[j] * scale
  }
  return fromData(out, [n, d])
}

/**
 * Randomised quasi–Monte Carlo over the box [lo, hi]: `replicates` (default 8) copies of the first n points of a
 * Halton or Sobol sequence (default Sobol), each shifted by an independent uniform vector modulo 1 (Cranley &
 * Patterson, 1976). Each shifted copy gives an unbiased estimate; their mean is the estimate and their standard
 * deviation / √replicates its standard error. For smooth f the error falls nearly as n^{−1}. Shift r comes from
 * `child(s, 'shift', r)`.
 */
export function quasiMonteCarlo(
  s: Stream,
  f: MultivariateIntegrand,
  lo: VectorLike,
  hi: VectorLike,
  { n = 1024, sequence = 'sobol', replicates = 8 }: { n?: Size; sequence?: 'sobol' | 'halton'; replicates?: Size } = {},
): MonteCarloResult {
  const l = flat(lo, 'quasiMonteCarlo')
  const h = flat(hi, 'quasiMonteCarlo')
  const d = l.length
  const volume = l.reduce((v, li, i) => v * (h[i] - li), 1)
  const points = sequence === 'sobol' ? sobol(n, d) : halton(n, d)
  const estimates: number[] = []
  for (let r = 0; r < replicates; r++) {
    const shift = uniform(child(s, 'shift', r), 0, 1, { shape: [d] }).data
    let total = 0
    for (let i = 0; i < n; i++) {
      const x = new Float64Array(d)
      for (let j = 0; j < d; j++) {
        const u = (points.data[i * d + j] + shift[j]) % 1
        x[j] = l[j] + (h[j] - l[j]) * u
      }
      total += f(fromData(x, [d]))
    }
    estimates.push((volume * total) / n)
  }
  const mean = estimates.reduce((a, b) => a + b, 0) / replicates
  const variance = replicates > 1 ? estimates.reduce((a, e) => a + (e - mean) ** 2, 0) / (replicates - 1) : NaN
  return { value: mean, standardError: Math.sqrt(variance / replicates), n: n * replicates }
}
