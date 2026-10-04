/**
 * Double descent with random features (Belkin et al., 2019; Nakkiran et al., 2020). Fix n noisy training points
 * xᵢ ~ N(0, I_d) with yᵢ = f(xᵢ) + σεᵢ, map each x to p random features (ReLU units max(0, aⱼ·x/√d) or Fourier
 * features cos(aⱼ·x/√d + bⱼ), with aⱼ ~ N(0, I_d) and bⱼ uniform on [0, 2π)), and fit the output weights by least
 * squares, taking the minimum-norm solution once p ≥ n and the system interpolates. As p grows the test error first
 * follows the classical U, then peaks where p = n, where the one interpolating solution must bend hard to pass
 * through the noise (its weight norm blows up), and then falls again: among the many interpolating solutions the
 * minimum-norm one is ever smoother. A ridge penalty λ removes the peak.
 *
 * The inputs are d-dimensional because random features of a one-dimensional input are nearly linearly dependent (smooth
 * functions of one variable), which makes the interpolating solution numerically meaningless; in d ≥ 5 dimensions the
 * feature matrix is well conditioned away from p = n. The regression function is f(x) = (β·x + sin(2γ·x))/√2 for
 * fixed random unit directions β and γ.
 */

import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { applyWeights, leastSquaresWeights } from '../regression'

/** The random feature map. */
export type RandomFeatureKind = 'relu' | 'fourier'

/** A random feature map in d dimensions: directions [p × d] and phases [p]. */
export interface FeatureMap {
  readonly kind: RandomFeatureKind
  readonly d: number
  readonly p: number
  readonly directions: Float64Array
  readonly phases: Float64Array
}

/** A random feature map drawn from a stream (the same stream gives the same map). */
export function randomFeatureMap(s: Stream, d: number, p: number, kind: RandomFeatureKind = 'relu'): FeatureMap {
  const directions = standardNormals(child(s, 'directions'), p * d)
  const phases = Float64Array.from(units(child(s, 'phases'), p), (u) => 2 * Math.PI * u)
  return { kind, d, p, directions, phases }
}

/** The features of points x [m × d] (row-major), [m × p], scaled by 1/√p. */
export function randomFeatures(map: FeatureMap, x: Float64Array): Float64Array {
  const { d, p, directions, phases, kind } = map
  const m = x.length / d
  const out = new Float64Array(m * p)
  const scale = 1 / Math.sqrt(p)
  const rd = 1 / Math.sqrt(d)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < p; j++) {
      let z = 0
      for (let k = 0; k < d; k++) z += directions[j * d + k] * x[i * d + k]
      z *= rd
      out[i * p + j] = scale * (kind === 'relu' ? Math.max(0, z) : Math.SQRT2 * Math.cos(z + phases[j]))
    }
  return out
}

/** Options of `doubleDescent`. */
export interface DoubleDescentOptions {
  /** Input dimension d (default 8). */
  dimension?: number
  /** Training points n (default 40). */
  n?: number
  /** Noise standard deviation σ (default 0.3). */
  noise?: number
  /** Feature counts to try (default 1 … 6n, dense near n). */
  features?: readonly number[]
  /** The feature map (default ReLU). */
  kind?: RandomFeatureKind
  /** Ridge penalty λ (default 0: least squares, minimum-norm when p ≥ n). */
  ridge?: number
  /** Training sets and feature maps averaged per p (default 10). */
  repeats?: number
  /** Test points (default 1000). */
  test?: number
  /** Points of the slice x = t·u, t ∈ [−3, 3], along which the first repeat's fits are kept (default 121). */
  slice?: number
}

/** Test and training error, and the weight norm, against the number of features. */
export interface DoubleDescentResult {
  readonly features: number[]
  readonly n: number
  /** Mean squared error against f on fresh inputs, averaged over the repeats, and its median over the repeats. */
  readonly testError: Float64Array
  readonly testErrorMedian: Float64Array
  readonly trainError: Float64Array
  /** Mean ‖w‖₂ of the fitted output weights. */
  readonly weightNorm: Float64Array
  /** The error of always predicting 0, for scale (E f²). */
  readonly nullError: number
  /** A slice through input space: t, f along it, and the first repeat's fit along it at every p ([features × slice]). */
  readonly slice: { t: Float64Array; truth: Float64Array; fits: Float64Array }
}

/** The default feature counts: 1 … 6n, every count near the threshold n. */
export function featureCounts(n: number, max = 6 * n): number[] {
  const out = new Set<number>()
  for (let p = 1; p <= max; p = Math.max(p + 1, Math.round(p * 1.12))) out.add(p)
  for (let p = Math.max(1, n - 5); p <= n + 5; p++) out.add(p)
  out.add(max)
  return [...out].filter((p) => p <= max).sort((a, b) => a - b)
}

/** Random-features regression across the interpolation threshold (module docs). */
export function doubleDescent(s: Stream, options: DoubleDescentOptions = {}): DoubleDescentResult {
  const { dimension: d = 8, n = 40, noise = 0.3, kind = 'relu', ridge = 0, repeats = 10 } = options
  const { test = 1000, slice = 121 } = options
  const features = [...(options.features ?? featureCounts(n))]
  const unit = (v: Float64Array) => {
    const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0))
    return v.map((x) => x / norm)
  }
  const beta = unit(standardNormals(child(s, 'beta'), d))
  const gamma = unit(standardNormals(child(s, 'gamma'), d))
  const f = (x: Float64Array, i: number) => {
    let a = 0
    let b = 0
    for (let k = 0; k < d; k++) {
      a += beta[k] * x[i * d + k]
      b += gamma[k] * x[i * d + k]
    }
    return (a + Math.sin(2 * b)) / Math.SQRT2
  }
  const xTest = standardNormals(child(s, 'test'), test * d)
  const yTest = Float64Array.from({ length: test }, (_, i) => f(xTest, i))
  const nullError = yTest.reduce((a, v) => a + v * v, 0) / test
  const t = Float64Array.from({ length: slice }, (_, i) => -3 + (6 * i) / (slice - 1))
  // The slice runs along (β + γ)/‖β + γ‖, where both parts of f vary.
  const u = unit(Float64Array.from(beta, (b, k) => b + gamma[k]))
  const xSlice = new Float64Array(slice * d)
  for (let i = 0; i < slice; i++) for (let k = 0; k < d; k++) xSlice[i * d + k] = t[i] * u[k]
  const truth = Float64Array.from({ length: slice }, (_, i) => f(xSlice, i))
  const sets = Array.from({ length: repeats }, (_, r) => {
    const x = standardNormals(child(s, 'train', r), n * d)
    const e = standardNormals(child(s, 'noise', r), n)
    return { x, y: Float64Array.from({ length: n }, (_, i) => f(x, i) + noise * e[i]) }
  })
  const testError = new Float64Array(features.length)
  const testErrorMedian = new Float64Array(features.length)
  const trainError = new Float64Array(features.length)
  const weightNorm = new Float64Array(features.length)
  const fits = new Float64Array(features.length * slice)
  features.forEach((p, k) => {
    const errors: number[] = []
    for (let r = 0; r < repeats; r++) {
      const map = randomFeatureMap(child(s, 'map', r), d, p, kind)
      const { x, y } = sets[r]
      const phi = randomFeatures(map, x)
      const w = leastSquaresWeights(phi, n, p, y, ridge)
      const pred = applyWeights(randomFeatures(map, xTest), test, p, w)
      const fitted = applyWeights(phi, n, p, w)
      let te = 0
      for (let i = 0; i < test; i++) te += (pred[i] - yTest[i]) ** 2 / test
      let tr = 0
      for (let i = 0; i < n; i++) tr += (fitted[i] - y[i]) ** 2 / n
      errors.push(te)
      testError[k] += te / repeats
      trainError[k] += tr / repeats
      weightNorm[k] += Math.sqrt(w.reduce((a, v) => a + v * v, 0)) / repeats
      if (r === 0) fits.set(applyWeights(randomFeatures(map, xSlice), slice, p, w), k * slice)
    }
    errors.sort((a, b) => a - b)
    testErrorMedian[k] = errors[Math.floor((errors.length - 1) / 2)]
  })
  return {
    features,
    n,
    testError,
    testErrorMedian,
    trainError,
    weightNorm,
    nullError,
    slice: { t, truth, fits },
  }
}
