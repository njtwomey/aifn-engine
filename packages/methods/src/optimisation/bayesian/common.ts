/**
 * Internal helpers shared by the Bayesian optimisation loops of this module: the search box and its scaling to the
 * unit cube, candidate grids, initial designs, the standardisation of targets, the Gaussian-process surrogate (refitted
 * by marginal likelihood) and the choice of a maximiser.
 *
 * Every loop works on inputs scaled to the unit cube, $u_j = (x_j - \mathrm{lo}_j) / (\mathrm{hi}_j - \mathrm{lo}_j)$,
 * and on targets standardised by their mean and standard deviation, $z = (y - \mathrm{shift}) / \mathrm{scale}$, so a
 * kernel's hyperparameters are on that scale. Points are row-major float64 arrays, $n \times d$, row $i$ at entries
 * $i d$ to $i d + d - 1$.
 */

import type { Vector } from 'aifn-compute/foundation/contracts'
import { child, permutation, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Kernel } from 'aifn-compute/learning/kernels'
import { sobol } from 'aifn-compute/numerics/quadrature'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fitGp, gpPosterior } from 'aifn-methods/learning/gaussian-processes'

/** The search box: the lower and upper corners, $d$ values each (a `TestFunction` is one). */
export type SearchBox = {
  /** The lower corner $\mathrm{lo}$. */
  lo: Vector | readonly number[]
  /** The upper corner $\mathrm{hi}$, each coordinate above the lower one's. */
  hi: Vector | readonly number[]
}

/** A search box read and checked, with the maps between its coordinates and the unit cube. */
export type Box = {
  /** The lower corner, $d$ values. */
  lo: Float64Array
  /** The upper corner, $d$ values. */
  hi: Float64Array
  /** The dimension $d$. */
  d: number
  /** Row-major points in the box's coordinates to the unit cube. */
  toUnit: (v: Float64Array) => Float64Array
  /** Row-major points on the unit cube to the box's coordinates. */
  fromUnit: (v: Float64Array) => Float64Array
}

/**
 * Read and check a search box. Throws `ShapeError` when the corners differ in length or are empty, and `DomainError`
 * when an upper coordinate is not above the lower one.
 *
 * @param box The search box: its corners as tensors or arrays.
 * @param where The caller's name, for error messages.
 * @returns The corners as arrays, the dimension and the maps to and from the unit cube.
 */
export function readBox(box: SearchBox, where: string): Box {
  const lo = Float64Array.from(isTensor(box.lo) ? toFlat(box.lo) : box.lo)
  const hi = Float64Array.from(isTensor(box.hi) ? toFlat(box.hi) : box.hi)
  const d = lo.length
  if (d === 0 || hi.length !== d)
    throw new ShapeError(where, `${where}: the box's corners must have the same, positive length`)
  for (let j = 0; j < d; j++)
    if (!(hi[j] > lo[j])) throw new DomainError(where, `${where}: the box's upper corner must be above its lower one`)
  return {
    lo,
    hi,
    d,
    toUnit: (v) => v.map((x, i) => (x - lo[i % d]) / (hi[i % d] - lo[i % d])),
    fromUnit: (v) => v.map((u, i) => lo[i % d] + u * (hi[i % d] - lo[i % d])),
  }
}

/**
 * A regular grid on the unit cube with `counts[j]` points along coordinate $j$, corners included, the last coordinate
 * varying fastest.
 *
 * @param counts The number of points per coordinate (each at least 2).
 * @returns The $\prod_j n_j \times d$ points, row-major.
 */
export function unitGrid(counts: readonly number[]): Float64Array {
  const d = counts.length
  const m = counts.reduce((a, b) => a * b, 1)
  const out = new Float64Array(m * d)
  for (let i = 0; i < m; i++) {
    let rest = i
    for (let j = d - 1; j >= 0; j--) {
      const k = rest % counts[j]
      rest = Math.floor(rest / counts[j])
      out[i * d + j] = k / (counts[j] - 1)
    }
  }
  return out
}

/**
 * Points given in the box's coordinates as the row-major values of an $n \times d$ matrix. Throws `ShapeError` for
 * points of the wrong shape.
 *
 * @param points A `[n, d]` tensor, or `[n]` when $d = 1$.
 * @param d The dimension.
 * @param what What the points are, for error messages.
 * @param where The caller's name, for error messages.
 * @returns The $nd$ values, a new array.
 */
export function pointRows(points: Tensor, d: number, what: string, where: string): Float64Array {
  const shape = points.shape
  if (!((shape.length === 2 && shape[1] === d) || (shape.length === 1 && d === 1)))
    throw new ShapeError(where, `${where}: ${what} must be [n, ${d}], got [${shape.join(', ')}]`)
  return Float64Array.from(toFlat(points))
}

/** The candidates of a loop: on the unit cube and in the box's coordinates, with the grid's counts. */
export type Candidates = {
  /** The $m \times d$ candidates on the unit cube, row-major. */
  unit: Float64Array
  /** The same candidates as an `[m, d]` tensor. */
  unitTensor: Tensor
  /** The candidates in the box's coordinates, an `[m, d]` tensor. */
  grid: Tensor
  /** The counts per coordinate of a regular grid, or `[m]` for given candidates. */
  shape: number[]
  /** The number $m$ of candidates. */
  m: number
}

/**
 * The candidates an acquisition is maximised over: a regular grid of a number of points per coordinate (default 201
 * in one dimension, 41 in two and $\lfloor 4096^{1/d} \rfloor$ beyond) or one count per coordinate, corners included;
 * or given points in the box's coordinates. Throws `DomainError` for a count below 2 and `ShapeError` for points of
 * the wrong dimension.
 *
 * @param option The counts, or the $m \times d$ candidates (`[m]` when $d = 1$), or undefined for the default.
 * @param box The search box, read by `readBox`.
 * @param where The caller's name, for error messages.
 * @returns The candidates on the unit cube and in the box, with the grid's shape.
 */
export function candidateGrid(
  option: number | readonly number[] | Tensor | undefined,
  box: Box,
  where: string,
): Candidates {
  const { d } = box
  let shape: number[]
  let unit: Float64Array
  if (isTensor(option)) {
    unit = box.toUnit(pointRows(option, d, 'the grid', where))
    shape = [unit.length / d]
  } else {
    const fallback = d === 1 ? 201 : d === 2 ? 41 : Math.max(2, Math.floor(4096 ** (1 / d)))
    shape = Array.isArray(option)
      ? [...(option as readonly number[])]
      : Array.from({ length: d }, () => (option as number | undefined) ?? fallback)
    if (shape.length !== d || shape.some((c) => !(Number.isInteger(c) && c >= 2)))
      throw new DomainError(where, `${where}: a grid needs an integer count of at least 2 per coordinate`)
    unit = unitGrid(shape)
  }
  const m = unit.length / d
  return { unit, unitTensor: fromData(unit, [m, d]), grid: fromData(box.fromUnit(unit), [m, d]), shape, m }
}

/**
 * An initial design in the box's coordinates: given points, or $n_0$ points placed by a Latin hypercube (`'lhs'`, one
 * point in each of $n_0$ equal slices of every coordinate), the first $n_0$ Sobol points under a random shift modulo 1
 * (`'sobol'`), or independent uniform points (`'uniform'`), all drawn from `s`. Throws `DomainError` for a count that
 * is not a positive integer and `ShapeError` for points of the wrong dimension.
 *
 * @param init The number $n_0$ of points, or the points themselves (`[n0, d]`, or `[n0]` when $d = 1$).
 * @param design How a number of points is placed.
 * @param box The search box, read by `readBox`.
 * @param s The stream the design draws from.
 * @param where The caller's name, for error messages.
 * @returns The $n_0 \times d$ points in the box's coordinates, row-major.
 */
export function designPoints(
  init: number | Tensor,
  design: 'lhs' | 'sobol' | 'uniform',
  box: Box,
  s: Stream,
  where: string,
): Float64Array {
  const { d } = box
  if (isTensor(init)) return pointRows(init, d, 'the initial points', where)
  if (!(Number.isInteger(init) && init >= 1))
    throw new DomainError(where, `${where}: init must be a positive integer or the points`)
  const n0 = init
  const u = new Float64Array(n0 * d)
  if (design === 'uniform') u.set(toFlat(uniform(child(s, 'uniform'), 0, 1, { shape: [n0, d] }) as Tensor))
  else if (design === 'sobol') {
    const base = toFlat(sobol(n0, d))
    const shift = toFlat(uniform(child(s, 'shift'), 0, 1, { shape: [d] }) as Tensor)
    for (let i = 0; i < n0 * d; i++) u[i] = (base[i] + shift[i % d]) % 1
  } else {
    for (let j = 0; j < d; j++) {
      const order = toFlat(permutation(child(s, 'slice', j), n0))
      const jitter = toFlat(uniform(child(s, 'jitter', j), 0, 1, { shape: [n0] }) as Tensor)
      for (let i = 0; i < n0; i++) u[i * d + j] = (order[i] + jitter[i]) / n0
    }
  }
  return box.fromUnit(u)
}

/**
 * The mean and standard deviation targets are standardised by: the standard deviation is 1 when it is 0 or there is
 * one value.
 *
 * @param y The targets.
 * @returns `shift`, the mean, `scale`, the population standard deviation (or 1), and `z`, the standardised targets.
 */
export function standardise(y: ArrayLike<number>): { shift: number; scale: number; z: Float64Array } {
  const n = y.length
  let shift = 0
  for (let i = 0; i < n; i++) shift += y[i] / n
  let variance = 0
  for (let i = 0; i < n; i++) variance += (y[i] - shift) ** 2 / n
  const scale = n > 1 && variance > 0 ? Math.sqrt(variance) : 1
  return { shift, scale, z: Float64Array.from({ length: n }, (_, i) => (y[i] - shift) / scale) }
}

/** Options of `gpSurrogate`. */
export type GpSurrogateOptions = {
  /** Refit the kernel by marginal likelihood (once there are at least three points), or keep it (`false`). */
  refit: 'marginal-likelihood' | false
  /** The observation-noise variance on the standardised scale, fixed, or `'fit'` to refit it with the kernel. */
  noise: number | 'fit'
}

/** A Gaussian-process surrogate: the kernel and noise used, and the posterior they give. */
export type GpSurrogate = {
  /** The kernel used, refitted or the starting one. */
  kernel: Kernel
  /** The observation-noise variance used. */
  noiseVariance: number
  /** The posterior of `gpPosterior`. */
  post: ReturnType<typeof gpPosterior>
}

/**
 * Fit a Gaussian-process surrogate to points on the unit cube and standardised targets: when `refit` is set and there
 * are at least three points, the kernel's hyperparameters (and with `'fit'` the noise) are refitted from the starting
 * kernel by maximising the log marginal likelihood (`fitGp`, at most 100 steps), and kept unless the fit fails.
 *
 * @param kernel The starting kernel, on the unit cube.
 * @param X The points, `[n, d]` on the unit cube.
 * @param Z The standardised targets, `[n]`.
 * @param options Whether to refit, and the noise, as `GpSurrogateOptions`.
 * @returns The kernel and noise used, and the posterior.
 */
export function gpSurrogate(kernel: Kernel, X: Tensor, Z: Tensor, options: GpSurrogateOptions): GpSurrogate {
  const { refit, noise } = options
  let k = kernel
  let noiseVariance = noise === 'fit' ? 0.1 : noise
  if (refit && X.shape[0] >= 3) {
    const fit = fitGp(kernel, X, Z, {
      noiseVariance: noise === 'fit' ? 0.1 : Math.max(noise, 1e-12),
      fitNoise: noise === 'fit',
      maxSteps: 100,
    })
    if (Number.isFinite(fit.logMarginal)) {
      k = fit.kernel
      if (noise === 'fit') noiseVariance = fit.noiseVariance
    }
  }
  return { kernel: k, noiseVariance, post: gpPosterior(k, X, Z, { noiseVariance }) }
}

/**
 * The index of the largest entry, the first of any ties; a NaN entry loses to any number.
 *
 * @param a The scores.
 * @returns The index of the largest.
 */
export function argmax(a: ArrayLike<number>): number {
  let k = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[k] || Number.isNaN(a[k])) k = i
  return k
}

/**
 * The index of the best value, the first of any ties: the smallest when minimising, the largest otherwise.
 *
 * @param y The values.
 * @param minimise Whether smaller is better.
 * @returns The index of the best.
 */
export function bestIndexOf(y: ArrayLike<number>, minimise: boolean): number {
  let k = 0
  for (let i = 1; i < y.length; i++) if (minimise ? y[i] < y[k] : y[i] > y[k]) k = i
  return k
}
