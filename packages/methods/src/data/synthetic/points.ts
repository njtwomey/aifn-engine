/**
 * Seeded point clouds in two and three dimensions: blobs, moons, circles and rings, spirals, XOR, checkerboard,
 * anisotropic and general Gaussians, the Swiss roll and the S-curve. Recipes follow scikit-learn's `make_*` functions
 * (Pedregosa et al., 2011, JMLR 12) where one exists. Points are grouped by class in label order; use `shuffleDataset`
 * for a random order.
 *
 * Every labelled generator takes class sizes the same way (`ClassSizeOptions`: a total `n` with `prevalence` or
 * `classWeights`, or per-class counts), with exact counts by largest remainder, and attaches its known truth in
 * `meta.truth` where the generating process has a closed form.
 */

import { multivariateNormal } from 'aifn-compute/probability/samplers'
import { normal, permutation, type Stream, child, integers, uniform } from 'aifn-compute/foundation/random'
import { selectRows } from '../rows'
import { resolveClassSizes, type ClassSizeOptions, type ClassSizes } from '../sizes'
import { appendStep, checkCount, generatorRecipe, labels, matrix, values, vector, type Dataset } from '../types'
import {
  REFERENCE_SIZE,
  classColumns,
  classificationTruth,
  curveLogDensity,
  equalReference,
  gaussianClasses,
  points,
  twoGaussianBayesError,
  type ClassModel,
  type Reference,
} from '../truth'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { mahalanobisDistance } from 'aifn-compute/learning/metrics'
import { MultivariateNormal, Normal } from 'aifn-compute/probability/distributions'
import type { DatasetInfo, ModifierInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space, when } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const TAU = 2 * Math.PI

type Sizes = ClassSizes

// Reference samples for Monte Carlo Bayes errors are drawn by calling the generator again; inside that call the truth
// is not attached, so references do not recurse.
let referenceDepth = 0

/**
 * A lazily drawn reference sample of the population, from a generator call on a child stream, for the Monte Carlo
 * Bayes error. It is drawn once, on first use, with the truth switched off for the inner call.
 *
 * @param make Draws the reference dataset: the generator called again on a child stream, with `REFERENCE_SIZE` points
 *   in the population's class proportions. Called at most once.
 * @returns A function returning the reference sample: the points of `make()` with equal weights, cached.
 */
function referenceFrom(make: () => Dataset): () => Reference {
  let cached: Reference | undefined
  return () => {
    if (cached) return cached
    referenceDepth++
    try {
      cached = equalReference(make().x)
      return cached
    } finally {
      referenceDepth--
    }
  }
}

/**
 * Whether a generator should attach its truth: false inside a reference draw, so that references do not recurse.
 *
 * @returns True outside every reference draw.
 */
const wantTruth = () => referenceDepth === 0

/**
 * Plain parameters for the recipe step (functions and undefined dropped).
 *
 * @param options A generator's options object; not modified.
 * @returns A copy of its fields without those that are `undefined` or functions.
 */
function params(options: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(options).filter(([, v]) => v !== undefined && typeof v !== 'function'))
}

/**
 * Whether two matrices, given as rows, are equal entry by entry. Only the entries of `a` are compared, so `b` must be
 * at least as large.
 *
 * @param a The first matrix, as rows.
 * @param b The second matrix, as rows.
 * @returns True when every entry of `a` equals the same entry of `b`.
 */
function equalMatrices(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): boolean {
  return a.every((row, i) => row.every((v, j) => v === b[i][j]))
}

/**
 * The model of Gaussian classes $\Gauss(\muvec_j, \Sigmamat_j)$: closed-form Bayes error for two classes with a
 * shared covariance (from their Mahalanobis distance), Monte Carlo otherwise.
 *
 * @param means The class means $\muvec_j$, $k$ rows of $d$ values.
 * @param covariances The class covariances $\Sigmamat_j$, $k$ matrices of $d \times d$ as rows; each must be positive
 *   definite.
 * @param priors The population class proportions $\pi_j$, $k$ values summing to one.
 * @param reference The lazy reference sample used for the Monte Carlo Bayes error.
 * @param family The name of the model, as the truth reports it.
 * @returns The class model: priors, the class log densities $\log p(\xvec \mid j)$ and the Bayes error's route.
 */
function gaussianModel(
  means: readonly (readonly number[])[],
  covariances: readonly (readonly (readonly number[])[])[],
  priors: number[],
  reference: () => Reference,
  family: string,
): ClassModel {
  const shared = means.length === 2 && equalMatrices(covariances[0], covariances[1])
  const delta = shared ? mahalanobisDistance(means[0], means[1], { covariance: covariances[0] }) : 0
  return {
    classes: means.length,
    priors,
    logDensity: gaussianClasses(means, covariances),
    ops: [],
    reference,
    closedForm: shared ? (p) => twoGaussianBayesError(delta, p) : undefined,
    family,
  }
}

/**
 * The isotropic covariance $\sigma^2\Imat$ as rows.
 *
 * @param d The dimension $d$: the matrix is $d \times d$.
 * @param sd The standard deviation $\sigma$ of every coordinate.
 * @returns The $d \times d$ matrix $\sigma^2\Imat$, as rows.
 */
function isotropic(d: number, sd: number): number[][] {
  return Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? sd * sd : 0)))
}

/**
 * Concentric rings in polar coordinates: $p(\rho, \theta \mid k) = \Gauss(\rho; r_k, \sigma^2)/(2\pi)$, so
 * $p(\xvec \mid k) = \Gauss(\lVert\xvec\rVert; r_k, \sigma^2)/(2\pi\lVert\xvec\rVert)$. The
 * $-\log(2\pi\lVert\xvec\rVert)$ term is shared by every class and cancels in the posterior; it is kept so the
 * densities are true ones.
 *
 * @param radii The ring radii $r_k$, one per class.
 * @param noise The standard deviation $\sigma$ of the radial noise; positive.
 * @returns A function mapping $n$ points ($n \times 2$) to their class log densities $\log p(\xvec_i \mid k)$
 *   ($n \times k$).
 */
function ringDensities(radii: readonly number[], noise: number): (x: Tensor) => Tensor {
  const k = radii.length
  const law = Normal(fromData(Float64Array.from(radii), [k]), noise)
  return (x) => {
    const { data, n } = points(x)
    const rho = Float64Array.from({ length: n }, (_, i) => Math.hypot(data[2 * i], data[2 * i + 1]))
    const out = values(law.logProb(fromData(rho, [n, 1])) as Tensor)
    for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[i * k + j] -= Math.log(TAU * rho[i])
    return fromData(out, [n, k])
  }
}

/**
 * Classes uniform on disjoint tiles of the plane: `tile(p)` names the class whose tiles contain the point ($-1$ outside
 * every tile) and class $j$'s tiles have total area $a_j$ = `area[j]`, so $p(\xvec \mid j) = 1/a_j$ on them.
 *
 * @param tile Maps a point (its two coordinates) to the class whose tiles contain it, or $-1$ outside every tile.
 * @param area The total area $a_j$ of each class's tiles, one per class.
 * @returns A function mapping $n$ points ($n \times 2$) to their class log densities ($n \times k$): $-\log a_j$ in
 *   class $j$'s tiles and $-\infty$ elsewhere.
 */
function tileDensities(tile: (p: Float64Array) => number, area: readonly number[]): (x: Tensor) => Tensor {
  const k = area.length
  return (x) => {
    const { data, n } = points(x)
    const out = new Float64Array(n * k).fill(-Infinity)
    for (let i = 0; i < n; i++) {
      const j = tile(data.subarray(2 * i, 2 * i + 2))
      if (j >= 0) out[i * k + j] = -Math.log(area[j])
    }
    return fromData(out, [n, k])
  }
}

/**
 * Gaussian XOR: each class is an equal mixture of the two blobs $\Gauss(\cvec, \sigma^2\Imat)$ at its quadrants'
 * corners $\cvec$: $(1, 1)$ and $(-1, -1)$ for class 0, $(-1, 1)$ and $(1, -1)$ for class 1.
 *
 * @param sd The standard deviation $\sigma$ of every blob.
 * @returns A function mapping $n$ points ($n \times 2$) to their class log densities ($n \times 2$).
 */
function cornerDensities(sd: number): (x: Tensor) => Tensor {
  const law = MultivariateNormal(fromData(Float64Array.from(CORNERS.flat()), [4, 2]), {
    covariance: fromData(Float64Array.of(sd * sd, 0, 0, sd * sd), [2, 2]),
  })
  return (x) => {
    const { data, n } = points(x)
    const blob = values(law.logProb(fromData(data, [n, 1, 2])) as Tensor)
    const out = new Float64Array(2 * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < 2; j++) {
        const [u, v] = [blob[4 * i + j], blob[4 * i + j + 2]]
        const m = Math.max(u, v)
        out[2 * i + j] = m === -Infinity ? -Infinity : m + Math.log(Math.exp(u - m) + Math.exp(v - m)) - Math.LN2
      }
    return fromData(out, [n, 2])
  }
}

// ── Blobs ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `blobs`. */
export interface BlobsOptions extends ClassSizeOptions {
  /** Total points (split over the centres by `classWeights`, evenly by default) or one count per centre. Default 300. */
  n?: Sizes
  /** Centres ($k \times d$), or a number of centres drawn uniformly in `box` (or placed by `separation`). Default 3. */
  centers?: number | readonly (readonly number[])[]
  /** Standard deviation of every blob, or one per blob. Default 1. */
  sd?: number | readonly number[]
  /** Dimension when centres are drawn. Default 2. */
  dim?: number
  /** The box centres are drawn in, per coordinate. Default $[-10, 10]$ (scikit-learn's `center_box`). */
  box?: readonly [number, number]
  /**
   * With a number of centres: `random` draws them uniformly in `box` (scikit-learn); `polygon` places them by
   * `separation`. Default `polygon` when `separation` is given, else `random`.
   */
  layout?: 'random' | 'polygon'
  /**
   * The overlap knob of the `polygon` layout (default 4): place the centres on a regular polygon in the first two
   * coordinates so that neighbouring centres are $d'$ = `separation` blob standard deviations apart (two centres at
   * $\pm d'\sigma/2$ on $x_1$; with one sd per blob, $\sigma$ is their mean). For two blobs this is the Mahalanobis
   * distance $d'$, and the Bayes error with equal classes is $\Phi(-d'/2)$.
   */
  separation?: number
}

/**
 * Isotropic Gaussian blobs, as `sklearn.datasets.make_blobs`: blob $j$ has `n[j]` points around centre $j$ with
 * standard deviation `sd[j]`, in blob order. Centres drawn at random come from the substream `centers`, points from
 * `points`. The truth is the Gaussian mixture: exact Bayes error for two blobs of equal sd, Monte Carlo otherwise.
 * Throws `ShapeError` when `sd` has a length other than the number of centres.
 *
 * @param s The random stream; its children `centers`, `points` and `truth` (the reference sample) are used.
 * @param options The centres, spreads and class sizes; see `BlobsOptions`.
 * @returns The dataset: `x` ($n \times d$), `y` the blob of each point, and the mixture in `meta.truth`. The recipe
 *   records the centres actually used.
 *
 * @example Three blobs around given centres
 * const d = blobs(stream(0), { n: 150, centers: [[0, 0], [6, 0], [0, 6]] })
 * print('x:', d.x.shape, ' y:', d.y.shape)
 * print('first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', [0, 1, 2].map((j) => y.filter((v) => v === j).length))
 * const rows = toArray(d.x).filter((_, i) => y[i] === 1)
 * print('mean of blob 2:', [0, 1].map((c) => rows.reduce((a, r) => a + r[c], 0) / rows.length))
 *
 * @example Two blobs placed by their separation
 * const d = blobs(stream(1), { centers: 2, separation: 2 })
 * print('centres on x1:', d.meta.recipe.knobs.centers.map((c) => c[0]))
 * print('Bayes error:', d.meta.truth.bayesError, ' Phi(-1) = 0.158655')
 */
export function blobs(s: Stream, options: BlobsOptions = {}): Dataset {
  const { centers = 3, dim = 2, box = [-10, 10] } = options
  const layout = options.layout ?? (options.separation !== undefined ? 'polygon' : 'random')
  const separation = options.separation ?? 4
  const sdList = typeof options.sd === 'object' ? [...options.sd] : undefined
  let centres: number[][]
  if (typeof centers === 'number') {
    if (layout === 'polygon') {
      const sd = sdList ? sdList.reduce((a, b) => a + b, 0) / sdList.length : ((options.sd as number | undefined) ?? 1)
      const radius = centers > 1 ? (separation * sd) / (2 * Math.sin(Math.PI / centers)) : 0
      centres = Array.from({ length: centers }, (_, j) => {
        const c = new Array<number>(Math.max(dim, 2)).fill(0)
        // Two blobs sit on the x₁ axis; more go round a polygon starting at the top.
        const angle = centers === 2 ? Math.PI * (1 - j) : Math.PI / 2 + (TAU * j) / centers
        c[0] = radius * Math.cos(angle)
        c[1] = radius * Math.sin(angle)
        return c.slice(0, dim)
      })
    } else {
      const c = child(s, 'centers')
      centres = Array.from({ length: centers }, () =>
        Array.from({ length: dim }, () => box[0] + (box[1] - box[0]) * uniform(c)),
      )
    }
  } else centres = centers.map((row) => [...row])
  const k = centres.length
  const d = centres[0]?.length ?? dim
  const { sizes, priors } = resolveClassSizes(options, k, 300, 'blobs')
  const sds = sdList ?? Array<number>(k).fill((options.sd as number | undefined) ?? 1)
  if (sds.length !== k) throw new ShapeError('blobs', `blobs: ${sds.length} standard deviations for ${k} centres`)
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(total * d)
  const y = new Int32Array(total)
  const p = child(s, 'points')
  let row = 0
  for (let j = 0; j < k; j++)
    for (let i = 0; i < sizes[j]; i++, row++) {
      for (let c = 0; c < d; c++) x[row * d + c] = centres[j][c] + sds[j] * normal(p)
      y[row] = j
    }
  const truth = wantTruth()
    ? classificationTruth(
        gaussianModel(
          centres,
          sds.map((v) => isotropic(d, v)),
          priors,
          referenceFrom(() =>
            blobs(child(s, 'truth'), {
              ...options,
              centers: centres,
              n: REFERENCE_SIZE,
              classWeights: priors,
              prevalence: undefined,
            }),
          ),
          `${k} isotropic Gaussian blobs`,
        ),
      )
    : undefined
  return {
    kind: 'dataset',
    x: matrix(x, total, d),
    y: labels(y),
    meta: {
      name: 'blobs',
      description: `${k} isotropic Gaussian blobs of ${total} points in ${d} dimensions.`,
      task: 'clustering',
      featureNames: Array.from({ length: d }, (_, c) => `x${c + 1}`),
      labelNames: centres.map((_, j) => `blob ${j + 1}`),
      source: 'scikit-learn make_blobs (Pedregosa et al., 2011)',
      key: s.key,
      truth,
      recipe: generatorRecipe('blobs', s.key, params({ ...options, centers: centres, n: sizes })),
    },
  }
}

// ── Curves: moons, circles, rings, spirals ───────────────────────────────────────────────────────────────────────────

/**
 * The truth of classes spread uniformly along curves and blurred by isotropic Gaussian noise, or `undefined` when the
 * noise is not positive (the densities would be singular) or inside a reference draw.
 *
 * @param curves One curve per class, mapping a position $u \in [0, 1]$ to a point of the plane; class $j$ is uniform
 *   in $u$ along curve $j$.
 * @param noise The standard deviation of the isotropic Gaussian noise.
 * @param priors The population class proportions, one per curve.
 * @param reference The lazy reference sample used for the Monte Carlo Bayes error.
 * @param family The name of the model, as the truth reports it.
 * @returns The classification truth, whose class densities are integrals along the curves, or `undefined`.
 */
function curveTruth(
  curves: ((u: number) => [number, number])[],
  noise: number,
  priors: number[],
  reference: () => Reference,
  family: string,
) {
  if (!(noise > 0) || !wantTruth()) return undefined
  const densities = curves.map((c) => curveLogDensity(c, noise))
  return classificationTruth({
    classes: curves.length,
    priors,
    logDensity: (x) => classColumns(densities.map((f) => f(x))),
    ops: [],
    reference,
    family,
  })
}

/** Options for `moons`. */
export interface MoonsOptions extends ClassSizeOptions {
  /** Total points, or `[outer, inner]`. Default 200. */
  n?: Sizes
  /**
   * Standard deviation of Gaussian noise added to each coordinate. Default 0.1. The overlap knob: the moons are 1 apart
   * at their closest, so the classes start to overlap near noise 0.25 and are mixed by 0.5.
   */
  noise?: number
  /** `even`: angles evenly spaced on $[0, \pi]$ (scikit-learn); `random`: uniform angles. Default `even`. */
  spacing?: 'even' | 'random'
}

/**
 * Two interleaving half circles, as `sklearn.datasets.make_moons`: the outer moon $(\cos t, \sin t)$ (label 0) and
 * the inner moon $(1 - \cos t, \tfrac12 - \sin t)$ (label 1) for $t \in [0, \pi]$, plus Gaussian noise. The usual
 * test that linear and centroid methods fail. The truth (attached when the noise is positive) takes $t$ uniform on
 * $[0, \pi]$, the limit of even spacing, and integrates over $t$ numerically.
 *
 * @param s The random stream: random angles come from its child `angles`, the noise from `noise`, the truth's
 *   reference sample from `truth`.
 * @param options The class sizes, noise and spacing; see `MoonsOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` 0 for the outer moon and 1 for the inner, and the truth.
 *
 * @example Two moons
 * const d = moons(stream(0), { n: 100 })
 * print('x:', d.x.shape, ' labels:', d.meta.labelNames)
 * print('first rows:', toArray(d.x).slice(0, 3))
 * const y = toArray(d.y)
 * print('class counts:', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * // The mean height of the outer moon is about 2 / pi = 0.637.
 * const top = toArray(d.x).filter((_, i) => y[i] === 0)
 * print('outer moon mean x2:', top.reduce((a, r) => a + r[1], 0) / top.length)
 *
 * @example Exact class sizes from a prevalence
 * const y = toArray(moons(stream(0), { n: 100, prevalence: 0.2 }).y)
 * print('inner moon points:', y.filter((v) => v === 1).length, 'of', y.length)
 */
export function moons(s: Stream, options: MoonsOptions = {}): Dataset {
  const { noise = 0.1, spacing = 'even' } = options
  const { sizes, priors } = resolveClassSizes(options, 2, 200, 'moons')
  const [nOut, nIn] = sizes
  const total = nOut + nIn
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  const angles = child(s, 'angles')
  const eps = child(s, 'noise')
  const angle = (i: number, m: number) =>
    spacing === 'even' ? (m > 1 ? (Math.PI * i) / (m - 1) : 0) : Math.PI * uniform(angles)
  for (let i = 0; i < total; i++) {
    const outer = i < nOut
    const t = outer ? angle(i, nOut) : angle(i - nOut, nIn)
    const px = outer ? Math.cos(t) : 1 - Math.cos(t)
    const py = outer ? Math.sin(t) : 0.5 - Math.sin(t)
    x[2 * i] = px + (noise > 0 ? noise * normal(eps) : 0)
    x[2 * i + 1] = py + (noise > 0 ? noise * normal(eps) : 0)
    y[i] = outer ? 0 : 1
  }
  const truth = curveTruth(
    [
      (u) => [Math.cos(Math.PI * u), Math.sin(Math.PI * u)],
      (u) => [1 - Math.cos(Math.PI * u), 0.5 - Math.sin(Math.PI * u)],
    ],
    noise,
    priors,
    referenceFrom(() =>
      moons(child(s, 'truth'), { noise, spacing: 'random', n: REFERENCE_SIZE, classWeights: priors }),
    ),
    'two moons',
  )
  return {
    kind: 'dataset',
    x: matrix(x, total, 2),
    y: labels(y),
    meta: {
      name: 'moons',
      description: `Two interleaving half circles (${total} points, noise sd ${noise}).`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['outer moon', 'inner moon'],
      source: 'scikit-learn make_moons (Pedregosa et al., 2011)',
      key: s.key,
      truth,
      recipe: generatorRecipe('moons', s.key, params({ ...options, n: sizes })),
    },
  }
}

/** Options for `circles`. */
export interface CirclesOptions extends ClassSizeOptions {
  /** Total points, or `[outer, inner]`. Default 200. */
  n?: Sizes
  /** Radius of the inner circle relative to the outer (radius 1). Default 0.5. */
  factor?: number
  /**
   * Gaussian noise standard deviation. Default 0.05. The overlap knob: the circles are $1 - f$ apart ($f$ the
   * `factor`), so the classes overlap once the noise nears $(1 - f)/2$.
   */
  noise?: number
}

/**
 * Two concentric circles, as `sklearn.datasets.make_circles`: points evenly spaced in angle on the unit circle
 * (label 0) and on a circle of radius `factor` (label 1), plus Gaussian noise. The truth (attached when the noise is positive)
 * takes the angle uniform. Throws `DomainError` unless `factor` is in $[0, 1)$.
 *
 * @param s The random stream: the noise comes from its child `noise`, the truth's reference sample from `truth`.
 * @param options The class sizes, inner radius and noise; see `CirclesOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` 0 for the outer circle and 1 for the inner, and the truth.
 *
 * @example Two circles
 * const d = circles(stream(0), { n: 100, factor: 0.4 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * // The inner circle's mean radius is close to the factor.
 * const inner = toArray(d.x).filter((_, i) => y[i] === 1)
 * print('inner mean radius:', inner.reduce((a, [u, v]) => a + Math.hypot(u, v), 0) / inner.length)
 */
export function circles(s: Stream, options: CirclesOptions = {}): Dataset {
  const { factor = 0.5, noise = 0.05 } = options
  if (!(factor >= 0 && factor < 1)) throw new DomainError('circles', `circles: factor must be in [0, 1), got ${factor}`)
  const { sizes, priors } = resolveClassSizes(options, 2, 200, 'circles')
  const [nOut, nIn] = sizes
  const total = nOut + nIn
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  const eps = child(s, 'noise')
  for (let i = 0; i < total; i++) {
    const outer = i < nOut
    const m = outer ? nOut : nIn
    const t = (TAU * (outer ? i : i - nOut)) / m
    const r = outer ? 1 : factor
    x[2 * i] = r * Math.cos(t) + (noise > 0 ? noise * normal(eps) : 0)
    x[2 * i + 1] = r * Math.sin(t) + (noise > 0 ? noise * normal(eps) : 0)
    y[i] = outer ? 0 : 1
  }
  const truth = curveTruth(
    [(u) => [Math.cos(TAU * u), Math.sin(TAU * u)], (u) => [factor * Math.cos(TAU * u), factor * Math.sin(TAU * u)]],
    noise,
    priors,
    referenceFrom(() => {
      // Uniform angles for the reference: rotate each circle's evenly spaced points by a random offset per point.
      const r = child(s, 'truth')
      const out = circles(r, { factor, noise, n: REFERENCE_SIZE, classWeights: priors })
      const xs = values(out.x)
      const turn = child(r, 'turn')
      for (let i = 0; i < xs.length; i += 2) {
        const a = TAU * uniform(turn)
        const [u, v] = [xs[i], xs[i + 1]]
        xs[i] = u * Math.cos(a) - v * Math.sin(a)
        xs[i + 1] = u * Math.sin(a) + v * Math.cos(a)
      }
      return { ...out, x: matrix(xs, xs.length / 2, 2) }
    }),
    'two concentric circles',
  )
  return {
    kind: 'dataset',
    x: matrix(x, total, 2),
    y: labels(y),
    meta: {
      name: 'circles',
      description: `Two concentric circles of radius 1 and ${factor} (${total} points, noise sd ${noise}).`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['outer circle', 'inner circle'],
      source: 'scikit-learn make_circles (Pedregosa et al., 2011)',
      key: s.key,
      truth,
      recipe: generatorRecipe('circles', s.key, params({ ...options, n: sizes })),
    },
  }
}

/** Options for `rings`. */
export interface RingsOptions extends ClassSizeOptions {
  /** Total points, or one count per ring. Default 300. */
  n?: Sizes
  /** Ring radii. Default `[1, 2, 3]`. */
  radii?: readonly number[]
  /** Radial Gaussian noise standard deviation. Default 0.1. The overlap knob, against the gap between radii. */
  noise?: number
}

/**
 * Concentric rings with uniformly random angles and Gaussian noise on the radius; label $k$ is ring $k$. The truth
 * (attached when the noise is positive) is in closed form: in polar coordinates
 * $p(\rho, \theta \mid k) = \Gauss(\rho; r_k, \sigma^2)/(2\pi)$, so
 * $p(\xvec \mid k) = \Gauss(\lVert\xvec\rVert; r_k, \sigma^2)/(2\pi\lVert\xvec\rVert)$.
 *
 * @param s The random stream: ring $k$ is drawn from its child `ring` $k$, the truth's reference sample from `truth`.
 * @param options The class sizes, radii and noise; see `RingsOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the ring of each point, and the truth.
 *
 * @example Three rings
 * const d = rings(stream(0), { n: 90 })
 * print('x:', d.x.shape, ' labels:', d.meta.labelNames)
 * const y = toArray(d.y)
 * const r = toArray(d.x).map(([u, v]) => Math.hypot(u, v))
 * print('class counts:', [0, 1, 2].map((k) => y.filter((v) => v === k).length))
 * // Each ring's mean radius is close to its radius 1, 2 or 3.
 * const ring = (k) => r.filter((_, i) => y[i] === k)
 * print('mean radius per ring:', [0, 1, 2].map((k) => ring(k).reduce((a, v) => a + v, 0) / ring(k).length))
 */
export function rings(s: Stream, options: RingsOptions = {}): Dataset {
  const { radii = [1, 2, 3], noise = 0.1 } = options
  const { sizes, priors } = resolveClassSizes(options, radii.length, 300, 'rings')
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  let row = 0
  radii.forEach((radius, k) => {
    const r = child(s, 'ring', k)
    for (let i = 0; i < sizes[k]; i++, row++) {
      const t = TAU * uniform(r)
      const rho = radius + noise * normal(r)
      x[2 * row] = rho * Math.cos(t)
      x[2 * row + 1] = rho * Math.sin(t)
      y[row] = k
    }
  })
  const truth =
    noise > 0 && wantTruth()
      ? classificationTruth({
          classes: radii.length,
          priors,
          logDensity: ringDensities(radii, noise),
          ops: [],
          reference: referenceFrom(() =>
            rings(child(s, 'truth'), { radii, noise, n: REFERENCE_SIZE, classWeights: priors }),
          ),
          family: 'concentric rings',
        })
      : undefined
  return {
    kind: 'dataset',
    x: matrix(x, total, 2),
    y: labels(y),
    meta: {
      name: 'rings',
      description: `${radii.length} concentric rings of radii ${radii.join(', ')} (${total} points, radial noise sd ${noise}).`,
      task: 'clustering',
      featureNames: ['x1', 'x2'],
      labelNames: radii.map((r) => `radius ${r}`),
      key: s.key,
      truth,
      recipe: generatorRecipe('rings', s.key, params({ ...options, n: sizes })),
    },
  }
}

/** Options for `spirals`. */
export interface SpiralsOptions extends ClassSizeOptions {
  /** Total points, or one count per arm. Default 300. */
  n?: Sizes
  /** Number of arms (classes). Default 2. */
  arms?: number
  /** Turns of each arm. Default 1.5. */
  turns?: number
  /**
   * Gaussian noise standard deviation. Default 0.05. The overlap knob: neighbouring arms are $1/(a\tau)$ apart along
   * a radius ($a$ the `arms`, $\tau$ the `turns`), so the arms blur together once the noise nears a quarter of that.
   */
  noise?: number
}

/**
 * Interleaved Archimedean spirals: arm $j$ holds points at radius $u$ and angle $2\pi(\tau u + j/a)$ ($\tau$ the
 * `turns`, $a$ the `arms`) for $u$ uniform on $[0.05, 1)$, plus Gaussian noise. `t` is the position $u$ along the arm.
 * The truth (attached when the noise is positive) integrates over $u$ numerically.
 *
 * @param s The random stream: arm $j$ is drawn from its child `arm` $j$, the truth's reference sample from `truth`.
 * @param options The class sizes, arms, turns and noise; see `SpiralsOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the arm of each point, `t` its position $u$ along the arm, and the
 *   truth.
 *
 * @example Two spirals
 * const d = spirals(stream(0), { n: 100, noise: 0 })
 * print('x:', d.x.shape, ' t:', d.t.shape)
 * print('first rows:', toArray(d.x).slice(0, 2), ' their t:', toArray(d.t).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * // Without noise the radius of a point is its position t along the arm.
 * const t = toArray(d.t)
 * print('largest |radius - t|:', Math.max(...toArray(d.x).map(([u, v], i) => Math.abs(Math.hypot(u, v) - t[i]))))
 */
export function spirals(s: Stream, options: SpiralsOptions = {}): Dataset {
  const { arms = 2, turns = 1.5, noise = 0.05 } = options
  const { sizes, priors } = resolveClassSizes(options, arms, 300, 'spirals')
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  const t = new Float64Array(total)
  let row = 0
  for (let j = 0; j < arms; j++) {
    const r = child(s, 'arm', j)
    for (let i = 0; i < sizes[j]; i++, row++) {
      const u = 0.05 + 0.95 * uniform(r)
      const angle = TAU * (turns * u + j / arms)
      x[2 * row] = u * Math.cos(angle) + noise * normal(r)
      x[2 * row + 1] = u * Math.sin(angle) + noise * normal(r)
      y[row] = j
      t[row] = u
    }
  }
  const arm =
    (j: number) =>
    (v: number): [number, number] => {
      const u = 0.05 + 0.95 * v
      const angle = TAU * (turns * u + j / arms)
      return [u * Math.cos(angle), u * Math.sin(angle)]
    }
  const truth = curveTruth(
    Array.from({ length: arms }, (_, j) => arm(j)),
    noise,
    priors,
    referenceFrom(() => spirals(child(s, 'truth'), { arms, turns, noise, n: REFERENCE_SIZE, classWeights: priors })),
    `${arms} spirals`,
  )
  return {
    kind: 'dataset',
    x: matrix(x, total, 2),
    y: labels(y),
    t: vector(t),
    meta: {
      name: 'spirals',
      description: `${arms} interleaved spirals of ${turns} turns (${total} points, noise sd ${noise}).`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: Array.from({ length: arms }, (_, j) => `arm ${j + 1}`),
      key: s.key,
      truth,
      recipe: generatorRecipe('spirals', s.key, params({ ...options, n: sizes })),
    },
  }
}

// ── XOR and checkerboard ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `xor`. */
export interface XorOptions extends ClassSizeOptions {
  /**
   * Total points (default 200), or `[same sign, opposite signs]`. With a total alone, `uniform` points are iid on the
   * square, so the class counts are random; pass `prevalence`, `classWeights` or counts for exact class sizes (points
   * are then drawn within each class's quadrants).
   */
  n?: Sizes
  /** `uniform`: points uniform on $[-1, 1]^2$; `gaussian`: blobs at $(\pm 1, \pm 1)$ with standard deviation `sd`. */
  kind?: 'uniform' | 'gaussian'
  /**
   * Blob standard deviation for `gaussian`. Default 0.4. The overlap knob: blobs are 2 apart, so neighbouring
   * quadrants overlap once sd nears 1. Uniform XOR has Bayes error 0; use label noise to blur it.
   */
  sd?: number
}

const CORNERS = [
  [1, 1],
  [-1, 1],
  [-1, -1],
  [1, -1],
]

/**
 * The XOR problem: label 1 when $x_1$ and $x_2$ have opposite signs (the second and fourth quadrants), 0 otherwise. No
 * linear classifier separates it. For `gaussian`, the label is that of the blob's quadrant, so blobs overlap; each
 * class is an equal mixture of its two blobs. With a total `n` alone the original recipe runs on `s` itself: iid
 * uniform points (random class counts), or the four blobs in turn. Exact class sizes draw class $j$ on the child
 * `class` $j$, alternating between its two quadrants.
 *
 * @param s The random stream; also its children `class` $j$ (exact sizes) and `truth` (the reference sample).
 * @param options The class sizes, kind and blob spread; see `XorOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` 1 for opposite signs and 0 for the same sign, and the truth (Bayes
 *   error 0 for `uniform`).
 *
 * @example Uniform XOR
 * const d = xor(stream(0), { n: 100 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 3), ' their labels:', toArray(d.y).slice(0, 3))
 * const y = toArray(d.y)
 * print('class counts (random):', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * print('labels match the signs:', toArray(d.x).every(([u, v], i) => (u * v < 0 ? 1 : 0) === y[i]))
 *
 * @example Gaussian XOR with exact class sizes
 * const d = xor(stream(0), { n: 100, kind: 'gaussian', prevalence: 0.3 })
 * const y = toArray(d.y)
 * print('class counts:', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * print('Bayes error:', d.meta.truth.bayesError, '(' + d.meta.truth.bayesErrorMethod + ')')
 */
export function xor(s: Stream, options: XorOptions = {}): Dataset {
  const { kind = 'uniform', sd = 0.4 } = options
  const { sizes, priors, controlled } = resolveClassSizes(options, 2, 200, 'xor')
  const n = sizes[0] + sizes[1]
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  if (!controlled) {
    // The original recipe: iid points (uniform), or the four blobs in turn (gaussian).
    for (let i = 0; i < n; i++) {
      if (kind === 'uniform') {
        x[2 * i] = 2 * uniform(s) - 1
        x[2 * i + 1] = 2 * uniform(s) - 1
        y[i] = x[2 * i] * x[2 * i + 1] < 0 ? 1 : 0
      } else {
        const q = i % 4
        x[2 * i] = CORNERS[q][0] + sd * normal(s)
        x[2 * i + 1] = CORNERS[q][1] + sd * normal(s)
        y[i] = q % 2
      }
    }
  } else {
    // Class j lives in quadrants j and j + 2 (corners above); alternate between them.
    let row = 0
    for (let j = 0; j < 2; j++) {
      const r = child(s, 'class', j)
      for (let i = 0; i < sizes[j]; i++, row++) {
        const corner = CORNERS[j + 2 * (i % 2)]
        if (kind === 'uniform') {
          x[2 * row] = corner[0] * uniform(r)
          x[2 * row + 1] = corner[1] * uniform(r)
        } else {
          x[2 * row] = corner[0] + sd * normal(r)
          x[2 * row + 1] = corner[1] + sd * normal(r)
        }
        y[row] = j
      }
    }
  }
  let truth
  if (wantTruth()) {
    const reference = referenceFrom(() => xor(child(s, 'truth'), { kind, sd, n: REFERENCE_SIZE, classWeights: priors }))
    if (kind === 'uniform')
      truth = classificationTruth({
        classes: 2,
        priors,
        // Each class is uniform on its two quadrants of [−1, 1]² (area 2).
        logDensity: tileDensities(
          (p) => (Math.abs(p[0]) <= 1 && Math.abs(p[1]) <= 1 ? (p[0] * p[1] < 0 ? 1 : 0) : -1),
          [2, 2],
        ),
        ops: [],
        reference,
        closedForm: () => 0,
        family: 'uniform XOR',
      })
    else {
      truth = classificationTruth({
        classes: 2,
        priors,
        logDensity: cornerDensities(sd),
        ops: [],
        reference,
        family: 'Gaussian XOR',
      })
    }
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name: 'xor',
      description: `The XOR problem: ${n} points labelled by whether their coordinates have opposite signs.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['same sign', 'opposite signs'],
      key: s.key,
      truth,
      recipe: generatorRecipe('xor', s.key, params({ ...options, n: controlled ? sizes : n })),
    },
  }
}

/** Options for `checkerboard`. */
export interface CheckerboardOptions extends ClassSizeOptions {
  /**
   * Total points (default 400), or `[even, odd]`. A total alone draws iid points on the board (random class counts);
   * `prevalence`, `classWeights` or counts draw within each class's tiles for exact sizes.
   */
  n?: Sizes
  /** Tiles per side, at least 2 (one tile has a single colour, so a single class). Default 4. */
  tiles?: number
}

/**
 * Points uniform on the square $[0, T]^2$ ($T$ the `tiles`), labelled by the parity of their tile,
 * $(\lfloor x_1 \rfloor + \lfloor x_2 \rfloor) \bmod 2$. The Bayes error is 0; there is no overlap knob, so blur it
 * with label noise. With a total `n` alone the points are iid on the board, drawn from `s` itself, so the class counts
 * are random and the class proportions are the tiles' areas. Exact class sizes draw class $j$ on the child `class` $j$,
 * each point in a random tile of its colour. Throws `DomainError` unless `tiles` is an integer of at least 2.
 *
 * @param s The random stream; also its children `class` $j$ (exact sizes) and `truth` (the reference sample).
 * @param options The class sizes and tiles per side; see `CheckerboardOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` 0 for an even tile and 1 for an odd one, and the truth.
 *
 * @example A 4 by 4 board
 * const d = checkerboard(stream(0), { n: 200 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2), ' their labels:', toArray(d.y).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts (random):', [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length])
 * print('labels match the tiles:', toArray(d.x).every(([u, v], i) => (Math.floor(u) + Math.floor(v)) % 2 === y[i]))
 */
export function checkerboard(s: Stream, options: CheckerboardOptions = {}): Dataset {
  const { tiles = 4 } = options
  if (!(Number.isInteger(tiles) && tiles >= 2))
    throw new DomainError('checkerboard', `checkerboard: tiles must be an integer of at least 2, got ${tiles}`)
  const cells: [number, number][][] = [[], []]
  for (let a = 0; a < tiles; a++) for (let b = 0; b < tiles; b++) cells[(a + b) % 2].push([a, b])
  const area = cells.map((c) => c.length)
  const explicit =
    typeof options.n === 'object' || options.prevalence !== undefined || options.classWeights !== undefined
  // Without explicit sizes the classes' shares are their areas, as for iid points on the board.
  const { sizes, priors } = resolveClassSizes(
    explicit ? options : { ...options, classWeights: area },
    2,
    400,
    'checkerboard',
  )
  const n = sizes[0] + sizes[1]
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  if (!explicit) {
    for (let i = 0; i < n; i++) {
      const a = tiles * uniform(s)
      const b = tiles * uniform(s)
      x[2 * i] = a
      x[2 * i + 1] = b
      y[i] = (Math.floor(a) + Math.floor(b)) % 2
    }
  } else {
    let row = 0
    for (let j = 0; j < 2; j++) {
      const r = child(s, 'class', j)
      for (let i = 0; i < sizes[j]; i++, row++) {
        const [a, b] = cells[j][integers(r, cells[j].length)]
        x[2 * row] = a + uniform(r)
        x[2 * row + 1] = b + uniform(r)
        y[row] = j
      }
    }
  }
  const truth = wantTruth()
    ? classificationTruth({
        classes: 2,
        priors,
        logDensity: tileDensities(
          (p) =>
            p[0] >= 0 && p[0] < tiles && p[1] >= 0 && p[1] < tiles ? (Math.floor(p[0]) + Math.floor(p[1])) % 2 : -1,
          area,
        ),
        ops: [],
        reference: referenceFrom(() =>
          checkerboard(child(s, 'truth'), { tiles, n: REFERENCE_SIZE, classWeights: priors }),
        ),
        closedForm: () => 0,
        family: 'checkerboard',
      })
    : undefined
  const counted = Array.from(y).reduce((c, v) => c + v, 0)
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name: 'checkerboard',
      description: `${n} points uniform on a ${tiles} × ${tiles} checkerboard, labelled by tile colour.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['even tile', 'odd tile'],
      key: s.key,
      truth,
      recipe: generatorRecipe('checkerboard', s.key, params({ ...options, n: [n - counted, counted] })),
    },
  }
}

// ── Gaussians ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `gaussians`. */
export interface GaussiansOptions extends ClassSizeOptions {
  /** Means, $k \times d$. Default two classes at $(-1, 0)$ and $(1, 0)$. */
  means?: readonly (readonly number[])[]
  /** Covariances, $k$ matrices of $d \times d$. Default $\sigma^2\Imat$ ($\sigma$ the `sd`) for every class. */
  covariances?: readonly (readonly (readonly number[])[])[]
  /** Without `covariances`: the standard deviation of every coordinate of every class. Default 1. */
  sd?: number
  /** Total points or one count per component. Default 300. */
  n?: Sizes
  /**
   * The overlap knob $d'$: scale the means about their centroid so that the closest pair is `separation` apart in
   * Mahalanobis distance under the average covariance. For two classes with a shared covariance and equal priors the
   * Bayes error is $\Phi(-d'/2)$.
   */
  separation?: number
}

/**
 * Samples from $k$ Gaussians $\Gauss(\muvec_j, \Sigmamat_j)$ with full covariances; label $j$ is component $j$. The
 * truth is exact: the posterior from the Gaussian densities, and a closed-form Bayes error for two classes with a
 * shared covariance (Monte Carlo otherwise). Throws `ShapeError` when the number of covariances differs from the number
 * of means, and `DomainError` when `separation` is asked of coincident means. A covariance that is not positive
 * definite throws.
 *
 * @param s The random stream: component $j$ is drawn from its child `component` $j$, the truth's reference sample
 *   from `truth`.
 * @param options The means, covariances, separation and class sizes; see `GaussiansOptions`.
 * @returns The dataset: `x` ($n \times d$), `y` the component of each point, and the truth. The recipe records the
 *   means after any rescaling by `separation`.
 *
 * @example Two correlated classes
 * const covariances = [[[1, 0.8], [0.8, 1]], [[1, 0], [0, 0.5]]]
 * const d = gaussians(stream(0), { n: 400, means: [[0, 0], [3, 1]], covariances })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * const second = toArray(d.x).filter((_, i) => y[i] === 1)
 * print('class counts:', [y.length - second.length, second.length])
 * print('mean of class 1:', [0, 1].map((c) => second.reduce((a, r) => a + r[c], 0) / second.length))
 *
 * @example The separation sets the Bayes error
 * const d = gaussians(stream(0), { separation: 2 })
 * print('means:', d.meta.recipe.knobs.means)
 * print('Bayes error:', d.meta.truth.bayesError, ' Phi(-1) = 0.158655')
 */
export function gaussians(s: Stream, options: GaussiansOptions = {}): Dataset {
  let means = (
    options.means ?? [
      [-1, 0],
      [1, 0],
    ]
  ).map((m) => [...m])
  const k = means.length
  const d = means[0].length
  const covariances = options.covariances ?? means.map(() => isotropic(d, options.sd ?? 1))
  if (covariances.length !== k)
    throw new ShapeError('gaussians', `gaussians: ${covariances.length} covariances for ${k} means`)
  if (options.separation !== undefined && k > 1) {
    const pooled = isotropic(d, 0).map((row, i) => row.map((_, j) => covariances.reduce((a, c) => a + c[i][j], 0) / k))
    let closest = Infinity
    for (let a = 0; a < k; a++)
      for (let b = a + 1; b < k; b++)
        closest = Math.min(closest, mahalanobisDistance(means[a], means[b], { covariance: pooled }))
    if (!(closest > 0)) throw new DomainError('gaussians', 'gaussians: separation needs distinct means')
    const centroid = means[0].map((_, c) => means.reduce((a, m) => a + m[c], 0) / k)
    const scale = options.separation / closest
    means = means.map((m) => m.map((v, c) => centroid[c] + scale * (v - centroid[c])))
  }
  const { sizes, priors } = resolveClassSizes(options, k, 300, 'gaussians')
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(total * d)
  const y = new Int32Array(total)
  let row = 0
  for (let j = 0; j < k; j++) {
    if (sizes[j] === 0) continue
    const draws = values(
      multivariateNormal(
        child(s, 'component', j),
        means[j],
        { covariance: matrixOf(covariances[j]) },
        { shape: [sizes[j]] },
      ),
    )
    x.set(draws, row * d)
    y.fill(j, row, row + sizes[j])
    row += sizes[j]
  }
  const truth = wantTruth()
    ? classificationTruth(
        gaussianModel(
          means,
          covariances,
          priors,
          referenceFrom(() =>
            gaussians(child(s, 'truth'), { means, covariances, n: REFERENCE_SIZE, classWeights: priors }),
          ),
          `${k} Gaussian classes`,
        ),
      )
    : undefined
  return {
    kind: 'dataset',
    x: matrix(x, total, d),
    y: labels(y),
    meta: {
      name: 'gaussians',
      description: `${k} Gaussian components with full covariances (${total} points).`,
      task: 'clustering',
      featureNames: Array.from({ length: d }, (_, c) => `x${c + 1}`),
      labelNames: Array.from({ length: k }, (_, j) => `component ${j + 1}`),
      key: s.key,
      truth,
      recipe: generatorRecipe('gaussians', s.key, params({ ...options, means, covariances, n: sizes })),
    },
  }
}

/**
 * A square matrix, given as rows, as a tensor.
 *
 * @param rows The $d$ rows of $d$ values each.
 * @returns The $d \times d$ tensor.
 */
function matrixOf(rows: readonly (readonly number[])[]): Tensor {
  const d = rows.length
  const out = new Float64Array(d * d)
  rows.forEach((r, i) => out.set(r, i * d))
  return matrix(out, d, d)
}

/** Options for `anisotropicBlobs`. */
export interface AnisotropicOptions extends BlobsOptions {
  /**
   * A $2 \times 2$ linear map $\Amat$, as rows, applied to every point as $\xvec \mapsto \xvec\Amat$. Default
   * `[[0.6, -0.6], [-0.4, 0.8]]` (scikit-learn's k-means example).
   */
  transform?: readonly (readonly number[])[]
}

/**
 * Blobs stretched by a shared linear map $\xvec \mapsto \xvec\Amat$ (points as rows), as in scikit-learn's
 * "demonstration of k-means assumptions": the clusters are elongated and tilted, which breaks k-means' spherical
 * assumption. The blobs are those of `blobs` on the same stream (300 points round 3 centres unless `options` says
 * otherwise, always in two dimensions). The truth is Gaussian with means $\Amat^\top\muvec_j$ and covariances
 * $\sigma_j^2\Amat^\top\Amat$.
 *
 * @param s The random stream, passed to `blobs`; its child `truth` draws the reference sample.
 * @param options The options of `blobs` and the map `transform`; see `AnisotropicOptions`.
 * @returns The dataset: `x` ($n \times 2$) the mapped points, `y` the blob of each point, and the truth.
 *
 * @example One blob's covariance under the map
 * const d = anisotropicBlobs(stream(0), { n: 2000, centers: [[0, 0]] })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * // A^T A for the default map is [[0.52, -0.68], [-0.68, 1]].
 * const x = toArray(d.x)
 * const cov = (a, b) => x.reduce((t, r) => t + r[a] * r[b], 0) / x.length
 * print('sample covariance:', [[cov(0, 0), cov(0, 1)], [cov(1, 0), cov(1, 1)]])
 */
export function anisotropicBlobs(s: Stream, options: AnisotropicOptions = {}): Dataset {
  const base = blobs(s, { n: 300, centers: 3, ...options, dim: 2 })
  const [[a, b], [c, d]] = options.transform ?? [
    [0.6, -0.6],
    [-0.4, 0.8],
  ]
  const map = (u: number, v: number): [number, number] => [u * a + v * c, u * b + v * d]
  const x = values(base.x)
  for (let i = 0; i < x.length; i += 2) [x[i], x[i + 1]] = map(x[i], x[i + 1])
  let truth
  const bt = base.meta.truth
  if (bt?.task === 'classification' && 'priors' in bt && wantTruth()) {
    const recipeParams = base.meta.recipe!.knobs as { centers: number[][]; sd?: number | number[] }
    const centres = recipeParams.centers
    const sds =
      typeof recipeParams.sd === 'object' ? recipeParams.sd : centres.map(() => (recipeParams.sd as number) ?? 1)
    // Cov of xA for x ~ N(μ, σ² I) is σ² AᵀA.
    const ata = [
      [a * a + c * c, a * b + c * d],
      [a * b + c * d, b * b + d * d],
    ]
    truth = classificationTruth(
      gaussianModel(
        centres.map(([u, v]) => map(u, v)),
        sds.map((sd) => ata.map((row) => row.map((v) => sd * sd * v))),
        [...bt.priors],
        referenceFrom(() =>
          anisotropicBlobs(child(s, 'truth'), {
            ...options,
            centers: centres,
            n: REFERENCE_SIZE,
            classWeights: [...bt.priors],
            prevalence: undefined,
          }),
        ),
        'anisotropic Gaussian blobs',
      ),
    )
  }
  return {
    ...base,
    x: matrix(x, x.length / 2, 2),
    meta: {
      ...base.meta,
      name: 'anisotropic blobs',
      description: `${base.meta.description} Stretched by a linear map.`,
      truth,
      recipe: generatorRecipe(
        'anisotropicBlobs',
        s.key,
        params({
          ...base.meta.recipe!.knobs,
          transform: [
            [a, b],
            [c, d],
          ],
        }),
      ),
    },
  }
}

/** Options for `swissRoll` and `sCurve`. */
export interface ManifoldOptions {
  /** Points. Default 500. */
  n?: number
  /** Gaussian noise standard deviation in every coordinate. Default 0. */
  noise?: number
}

/**
 * The Swiss roll, as `sklearn.datasets.make_swiss_roll`: $t = 1.5\pi(1 + 2u)$ and height $h = Hv$ ($H$ the `height`,
 * 21 by default) for $u$, $v$ uniform on $[0, 1)$, mapped to $(t\cos t, h, t\sin t)$. `t` is the position along the
 * roll, the coordinate a good embedding recovers. No truth is attached. Throws `DomainError` unless `n` is a
 * non-negative integer.
 *
 * @param s The random stream all draws come from.
 * @param options `n` the number of points (default 500), `noise` the standard deviation of Gaussian noise added to
 *   every coordinate (default 0), and `height` the height $H$ of the roll (default 21).
 * @returns The dataset: `x` ($n \times 3$) and `t`, each point's position along the roll, in
 *   $[1.5\pi, 4.5\pi)$.
 *
 * @example The roll and its coordinate
 * const d = swissRoll(stream(0), { n: 200 })
 * print('x:', d.x.shape, ' t:', d.t.shape)
 * print('first rows:', toArray(d.x).slice(0, 2), ' their t:', toArray(d.t).slice(0, 2))
 * const t = toArray(d.t)
 * print('t from', Math.min(...t), 'to', Math.max(...t), ' (1.5 pi = 4.712, 4.5 pi = 14.137)')
 * // Without noise the first coordinate is t cos t.
 * print('largest |x1 - t cos t|:', Math.max(...toArray(d.x).map((r, i) => Math.abs(r[0] - t[i] * Math.cos(t[i])))))
 */
export function swissRoll(s: Stream, options: ManifoldOptions & { height?: number } = {}): Dataset {
  const { n = 500, noise = 0, height = 21 } = options
  checkCount(n, 'swissRoll')
  const x = new Float64Array(3 * n)
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const a = 1.5 * Math.PI * (1 + 2 * uniform(s))
    const h = height * uniform(s)
    x[3 * i] = a * Math.cos(a) + (noise > 0 ? noise * normal(s) : 0)
    x[3 * i + 1] = h + (noise > 0 ? noise * normal(s) : 0)
    x[3 * i + 2] = a * Math.sin(a) + (noise > 0 ? noise * normal(s) : 0)
    t[i] = a
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 3),
    t: vector(t),
    meta: {
      name: 'Swiss roll',
      description: `${n} points on a rectangle rolled into a spiral in three dimensions.`,
      task: 'manifold',
      featureNames: ['x', 'y', 'z'],
      source: 'scikit-learn make_swiss_roll (Marsland, 2009, "Machine Learning: An Algorithmic Perspective")',
      key: s.key,
    },
  }
}

/**
 * The S-curve, as `sklearn.datasets.make_s_curve`: $t = 3\pi(u - \tfrac12)$ for $u$, $v$ uniform on $[0, 1)$, mapped
 * to $(\sin t, 2v, \sgn(t)(\cos t - 1))$. `t` is the position along the S. No truth is attached. Throws
 * `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The random stream all draws come from.
 * @param options The number of points and the standard deviation of Gaussian noise added to every coordinate; see
 *   `ManifoldOptions`.
 * @returns The dataset: `x` ($n \times 3$) and `t`, each point's position along the S, in $[-1.5\pi, 1.5\pi)$.
 *
 * @example The S and its coordinate
 * const d = sCurve(stream(0), { n: 200 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const t = toArray(d.t)
 * print('t from', Math.min(...t), 'to', Math.max(...t), ' (1.5 pi = 4.712)')
 * // Without noise the first coordinate is sin t.
 * print('largest |x1 - sin t|:', Math.max(...toArray(d.x).map((r, i) => Math.abs(r[0] - Math.sin(t[i])))))
 */
export function sCurve(s: Stream, options: ManifoldOptions = {}): Dataset {
  const { n = 500, noise = 0 } = options
  checkCount(n, 'sCurve')
  const x = new Float64Array(3 * n)
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const a = 3 * Math.PI * (uniform(s) - 0.5)
    const h = 2 * uniform(s)
    x[3 * i] = Math.sin(a) + (noise > 0 ? noise * normal(s) : 0)
    x[3 * i + 1] = h + (noise > 0 ? noise * normal(s) : 0)
    x[3 * i + 2] = Math.sign(a) * (Math.cos(a) - 1) + (noise > 0 ? noise * normal(s) : 0)
    t[i] = a
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 3),
    t: vector(t),
    meta: {
      name: 'S-curve',
      description: `${n} points on a sheet bent into an S in three dimensions.`,
      task: 'manifold',
      featureNames: ['x', 'y', 'z'],
      source: 'scikit-learn make_s_curve',
      key: s.key,
    },
  }
}

/**
 * The same dataset with its rows in a random order drawn from `s` (x, y, t, f and the per-row metadata permuted
 * together), and a `shuffleDataset` step appended to its recipe.
 *
 * @param s The random stream the permutation is drawn from.
 * @param data The dataset to shuffle; not modified.
 * @returns A new dataset with the same rows in a random order.
 *
 * @example Generators group points by class; shuffling mixes them
 * const d = blobs(stream(0), { n: 12, centers: 3 })
 * print('labels before:', toArray(d.y))
 * const shuffled = shuffleDataset(stream(1), d)
 * print('labels after: ', toArray(shuffled.y))
 * print('recipe modifiers:', shuffled.meta.recipe.modifiers.map((m) => m.op))
 */
export function shuffleDataset(s: Stream, data: Dataset): Dataset {
  const n = data.x.shape[0]
  const out = selectRows(data, Array.from(permutation(s, n).data))
  return { ...out, meta: { ...out.meta, recipe: appendStep(data.meta.recipe, { op: 'shuffleDataset', params: {} }) } }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'blobs',
    name: 'Gaussian blobs',
    summary: 'Isotropic Gaussian blobs around random or evenly spaced centres (scikit-learn make_blobs).',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      centers: int(1, 10, { default: 3 }),
      sd: real(0.05, 10, { default: 1 }),
      dim: int(1, 20, { default: 2 }),
      layout: oneOf(['random', 'polygon']),
      separation: real(0, 20, { default: 4, when: when('layout', 'polygon') }),
      prevalence: real(0.01, 0.99, { default: 0.5, when: when('centers', 2) }),
    }),
    truth: true,
    random: true,
    notes: ['k-means', 'gaussian-mixture-model'],
  },
  blobs,
)

dataset(
  {
    key: 'moons',
    name: 'Two moons',
    summary: 'Two interleaving half circles with Gaussian noise (scikit-learn make_moons).',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 200 }),
      noise: real(0, 1, { default: 0.1 }),
      spacing: oneOf(['even', 'random']),
      prevalence: real(0.01, 0.99, { default: 0.5, doc: 'The share of class 1.' }),
    }),
    truth: true,
    random: true,
    notes: ['spectral-clustering'],
  },
  moons,
)

dataset(
  {
    key: 'circles',
    name: 'Concentric circles',
    summary: 'A small circle inside a large one, with Gaussian noise (scikit-learn make_circles).',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 200 }),
      factor: real(0.05, 0.95, { default: 0.5 }),
      noise: real(0, 1, { default: 0.05 }),
      prevalence: real(0.01, 0.99, { default: 0.5, doc: 'The share of class 1.' }),
    }),
    truth: true,
    random: true,
    notes: ['kernel-principal-component-analysis'],
  },
  circles,
)

dataset(
  {
    key: 'rings',
    name: 'Rings',
    summary: 'Concentric rings of radii 1, 2 and 3 with radial noise.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({ n: int(2, 5000, { default: 300 }), noise: real(0, 1, { default: 0.1 }) }),
    truth: true,
    random: true,
    notes: ['spectral-clustering'],
  },
  rings,
)

dataset(
  {
    key: 'spirals',
    name: 'Spirals',
    summary: 'Interleaved spiral arms, one class per arm.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      arms: int(1, 6, { default: 2 }),
      turns: real(0.25, 5, { default: 1.5 }),
      noise: real(0, 0.5, { default: 0.05 }),
      prevalence: real(0.01, 0.99, { default: 0.5, when: when('arms', 2) }),
    }),
    truth: true,
    random: true,
  },
  spirals,
)

dataset(
  {
    key: 'xor',
    name: 'XOR',
    summary: 'Opposite quadrants share a class: uniform on the square or Gaussian blobs at (±1, ±1).',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 200 }),
      kind: oneOf(['uniform', 'gaussian']),
      sd: real(0.05, 2, { default: 0.4, when: when('kind', 'gaussian') }),
      prevalence: real(0.01, 0.99, { default: 0.5, doc: 'The share of class 1.' }),
    }),
    truth: true,
    random: true,
    notes: ['multilayer-perceptron'],
  },
  xor,
)

dataset(
  {
    key: 'checkerboard',
    name: 'Checkerboard',
    summary: 'Points on a board of alternating tiles, the class given by the tile colour.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 400 }),
      tiles: int(2, 12, { default: 4 }),
      prevalence: real(0.01, 0.99, { default: 0.5, doc: 'The share of class 1.' }),
    }),
    truth: true,
    random: true,
    notes: ['decision-tree'],
  },
  checkerboard,
)

dataset(
  {
    key: 'gaussians',
    name: 'Gaussian classes',
    summary:
      'Gaussian classes with given means and covariances; separation sets d′, the Mahalanobis distance of the closest means.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      sd: real(0.05, 10, { default: 1 }),
      separation: real(0, 10, { default: 2, label: 'd′' }),
      prevalence: real(0.01, 0.99, { default: 0.5, doc: 'The share of class 1.' }),
    }),
    truth: true,
    random: true,
    notes: ['linear-discriminant-analysis', 'bayes-decision-rule'],
  },
  gaussians,
)

dataset(
  {
    key: 'anisotropicBlobs',
    name: 'Anisotropic blobs',
    summary:
      'Gaussian blobs sheared by one linear map, so the clusters are elongated (the scikit-learn k-means example).',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      centers: int(1, 10, { default: 3 }),
      sd: real(0.05, 10, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: ['k-means'],
  },
  anisotropicBlobs,
)

dataset(
  {
    key: 'swissRoll',
    name: 'Swiss roll',
    summary: 'A rolled-up rectangle in three dimensions; `t` is the position along the roll.',
    task: 'manifold',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 500 }),
      noise: real(0, 2, { default: 0 }),
      height: real(1, 50, { default: 21 }),
    }),
    truth: false,
    random: true,
    notes: ['isomap', 'locally-linear-embedding'],
  },
  swissRoll,
)

dataset(
  {
    key: 'sCurve',
    name: 'S-curve',
    summary: 'An S-shaped surface in three dimensions; `t` is the position along the S.',
    task: 'manifold',
    output: 'dataset',
    knobs: space({ n: int(2, 5000, { default: 500 }), noise: real(0, 2, { default: 0 }) }),
    truth: false,
    random: true,
    notes: ['locally-linear-embedding'],
  },
  sCurve,
)

const modifier = definer<ModifierInfo>('modifier', 'data/synthetic')

modifier(
  {
    key: 'shuffleDataset',
    name: 'Shuffle',
    summary: 'Put the rows in a random order (generators group points by class).',
    params: space({}),
    random: true,
  },
  shuffleDataset,
)
