/**
 * Double descent with random features (Belkin et al., 2019; Nakkiran et al., 2020). Fix $n$ noisy training points
 * $\xvec_i \sim \Gauss(\zeros, \Imat_d)$ with $y_i = f(\xvec_i) + \sigma\varepsilon_i$, map each $\xvec$ to $p$
 * random features (ReLU units $\max(0, \avec_j^\top\xvec/\sqrt{d})$ or Fourier features
 * $\sqrt{2} \cos(\avec_j^\top\xvec/\sqrt{d} + b_j)$, with $\avec_j \sim \Gauss(\zeros, \Imat_d)$ and $b_j$ uniform on
 * $[0, 2\pi)$, all scaled by $1/\sqrt{p}$), and fit the output weights by least squares, taking the minimum-norm
 * solution once $p \ge n$ and the system interpolates. As $p$ grows the test error first follows the classical U, then
 * peaks where $p = n$, where the one interpolating solution must bend hard to pass through the noise (its weight norm
 * blows up), and then falls again: among the many interpolating solutions the minimum-norm one is ever smoother. A
 * ridge penalty $\lambda$ removes the peak.
 *
 * The inputs are $d$-dimensional because random features of a one-dimensional input are nearly linearly dependent
 * (smooth functions of one variable), which makes the interpolating solution numerically meaningless; in $d \ge 5$
 * dimensions the feature matrix is well conditioned away from $p = n$. The regression function is
 * $f(\xvec) = (\betavec^\top\xvec + \sin(2\gammavec^\top\xvec))/\sqrt{2}$ for fixed random unit directions
 * $\betavec$ and $\gammavec$.
 */

import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { applyWeights, leastSquaresWeights } from '../regression'

/** The random feature map: ReLU units, or Fourier features with random phases. */
export type RandomFeatureKind = 'relu' | 'fourier'

/** A random feature map in $d$ dimensions: directions $[p, d]$ and phases $[p]$. */
export interface FeatureMap {
  /** The kind of feature. */
  readonly kind: RandomFeatureKind
  /** The input dimension $d$. */
  readonly d: number
  /** The number of features $p$. */
  readonly p: number
  /** The directions $\avec_j$, row-major $[p, d]$: row $j$ is feature $j$'s direction. */
  readonly directions: Float64Array
  /** The phases $b_j$ in $[0, 2\pi)$, one per feature (used by Fourier features only). */
  readonly phases: Float64Array
}

/**
 * A random feature map drawn from a stream (the same stream gives the same map): directions
 * $\avec_j \sim \Gauss(\zeros, \Imat_d)$ and phases uniform on $[0, 2\pi)$.
 *
 * @param s The stream; the directions and phases come from its children `directions` and `phases`, so `s` is not
 *   advanced.
 * @param d The input dimension $d$.
 * @param p The number of features $p$.
 * @param kind The kind of feature.
 * @returns The feature map.
 *
 * @example A map of three ReLU features in two dimensions
 * const map = randomFeatureMap(stream(0), 2, 3)
 * print('directions =', map.directions)
 * print('phases =', map.phases)
 */
export function randomFeatureMap(s: Stream, d: number, p: number, kind: RandomFeatureKind = 'relu'): FeatureMap {
  const directions = standardNormals(child(s, 'directions'), p * d)
  const phases = Float64Array.from(units(child(s, 'phases'), p), (u) => 2 * Math.PI * u)
  return { kind, d, p, directions, phases }
}

/**
 * The features of points, scaled by $1/\sqrt{p}$: $\max(0, z_j)$ for ReLU and $\sqrt{2}\cos(z_j + b_j)$ for Fourier
 * features, with $z_j = \avec_j^\top\xvec/\sqrt{d}$.
 *
 * @param map The feature map.
 * @param x The points, row-major $[m, d]$: row `i` occupies entries `i * d` to `i * d + d - 1`.
 * @returns The features, row-major $[m, p]$.
 *
 * @example Features of two points, with their sum of squares
 * const map = randomFeatureMap(stream(0), 2, 4, 'fourier')
 * const phi = randomFeatures(map, new Float64Array([1, 0, 0, 1]))
 * print('features =', phi)
 * print('squared norm of the first point =', phi.slice(0, 4).reduce((a, v) => a + v * v, 0))
 */
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
  /** Input dimension $d$ (default 8). */
  dimension?: number
  /** Training points $n$ (default 40). */
  n?: number
  /** Noise standard deviation $\sigma$ (default 0.3). */
  noise?: number
  /** Feature counts to try (default `featureCounts(n)`: 1 to $6n$, every count near $n$). */
  features?: readonly number[]
  /** The feature map (default ReLU). */
  kind?: RandomFeatureKind
  /** Ridge penalty $\lambda$ (default 0: least squares, minimum-norm when $p \ge n$). */
  ridge?: number
  /** Training sets and feature maps averaged per $p$ (default 10). */
  repeats?: number
  /** Test points, drawn as the training inputs (default 1000). */
  test?: number
  /**
   * Points of the slice $\xvec = t\uvec$, $t \in [-3, 3]$, along which the first repeat's fits are kept (default 121);
   * $\uvec$ is the unit vector along $\betavec + \gammavec$, where both parts of $f$ vary.
   */
  slice?: number
}

/** Test and training error, and the weight norm, against the number of features. */
export interface DoubleDescentResult {
  /** The feature counts $p$ tried, in the order given. */
  readonly features: number[]
  /** The number of training points $n$, the interpolation threshold. */
  readonly n: number
  /** Mean squared error against $f$ (noise-free) on the test inputs, averaged over the repeats, per $p$. */
  readonly testError: Float64Array
  /** The (lower) median of the test error over the repeats, per $p$: less swayed by one ill-conditioned fit. */
  readonly testErrorMedian: Float64Array
  /** Mean squared error on the noisy training targets, averaged over the repeats, per $p$. */
  readonly trainError: Float64Array
  /** Mean $\lVert \wvec \rVert_2$ of the fitted output weights, per $p$. */
  readonly weightNorm: Float64Array
  /** The error of always predicting 0, for scale ($\expect f^2$ on the test inputs). */
  readonly nullError: number
  /**
   * A slice through input space: `t`, $f$ along it (`truth`), and the first repeat's fit along it at every $p$
   * (`fits`, row-major $[\text{features}, \text{slice}]$).
   */
  readonly slice: { t: Float64Array; truth: Float64Array; fits: Float64Array }
}

/**
 * The default feature counts: from 1 to `max` in steps of about 12%, with every count within 5 of the threshold $n$,
 * and `max` itself.
 *
 * @param n The number of training points, the interpolation threshold.
 * @param max The largest count (default $6n$).
 * @returns The counts, increasing and distinct.
 *
 * @example The counts for 20 training points
 * print(featureCounts(20))
 */
export function featureCounts(n: number, max = 6 * n): number[] {
  const out = new Set<number>()
  for (let p = 1; p <= max; p = Math.max(p + 1, Math.round(p * 1.12))) out.add(p)
  for (let p = Math.max(1, n - 5); p <= n + 5; p++) out.add(p)
  out.add(max)
  return [...out].filter((p) => p <= max).sort((a, b) => a - b)
}

/**
 * Random-features regression across the interpolation threshold (see the file's introduction): for each feature count,
 * fit `repeats` training sets, each with its own feature map, and average the errors and weight norms. Costs one
 * least-squares solve per count and repeat, so keep the sizes small on a page.
 *
 * @param s The stream; the target directions, test inputs, training sets and feature maps come from its children, so
 *   `s` is not advanced, and repeat $r$ uses the same training set at every $p$.
 * @param options The problem, the feature counts, the model and the study size (see `DoubleDescentOptions`).
 * @returns The errors and weight norms per feature count, and the fits along a slice (see `DoubleDescentResult`).
 *
 * @example The test error peaks where the feature count reaches the 20 training points
 * const r = doubleDescent(stream(0), {
 *   n: 20,
 *   features: [5, 10, 15, 18, 20, 22, 25, 40, 80, 160],
 *   repeats: 4,
 *   test: 200,
 *   slice: 2,
 * })
 * print('features   ', r.features)
 * print('test error ', r.testErrorMedian)
 * print('train error', r.trainError)
 * print('peak at p =', r.features[r.testErrorMedian.indexOf(Math.max(...r.testErrorMedian))])
 *
 * @example A ridge penalty removes the peak
 * const options = { n: 20, features: [10, 20, 40], repeats: 4, test: 200, slice: 2 }
 * print('minimum-norm', doubleDescent(stream(0), options).testErrorMedian)
 * print('ridge 0.1   ', doubleDescent(stream(0), { ...options, ridge: 0.1 }).testErrorMedian)
 */
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
