/**
 * Seeded two-dimensional densities for generative models, each with its exact density: a ring of Gaussians, a grid of
 * Gaussians, a pinwheel, a Swiss-roll slice, and a uniform annulus for out-of-distribution points. Every point is
 * labelled with the mode it came from (a Gaussian, an arm, a stretch of the roll), so that `meta.truth` is a
 * classification truth whose class-conditional densities and priors give the data density
 * $p(\xvec) = \sum_j \pi_j p(\xvec \mid j)$ exactly, and a generated point can be assigned to its mode by the Bayes
 * posterior (`decide`). The rings and grids follow the GAN mode-collapse benchmarks (Metz et al., 2017; Srivastava
 * et al., 2017); the pinwheel after Johnson et al. (2016) and the roll after scikit-learn's `make_swiss_roll` are drawn
 * here as curves blurred by isotropic Gaussian noise, so their densities are one-dimensional integrals along the curve.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { resolveClassSizes, type ClassSizeOptions, type ClassSizes } from '../sizes'
import {
  REFERENCE_SIZE,
  classColumns,
  classificationTruth,
  curveLogDensity,
  equalReference,
  gaussianClasses,
  points,
  type ClassModel,
  type Reference,
} from '../truth'
import { generatorRecipe, labels, matrix, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

const TAU = 2 * Math.PI

// Reference samples (for Monte Carlo Bayes errors) call the generator again with the truth switched off.
let referenceDepth = 0
/**
 * A lazily drawn reference sample of the population, for the Monte Carlo Bayes error: drawn once, on first use, with
 * the truth switched off for the inner call.
 *
 * @param make Draws the reference dataset: the generator called again on a child stream with `REFERENCE_SIZE` points.
 *   Called at most once.
 * @returns A function returning the reference sample: the points of `make()` with equal weights, cached.
 */
function referenceOf(make: () => Dataset): () => Reference {
  let cached: Reference | undefined
  return () => {
    if (cached) return cached
    referenceDepth++
    try {
      return (cached = equalReference(make().x))
    } finally {
      referenceDepth--
    }
  }
}

/**
 * The $2 \times 2$ isotropic covariance $\sigma^2\Imat$ as rows.
 *
 * @param sd The standard deviation $\sigma$ of each coordinate.
 * @returns The matrix $\sigma^2\Imat$, as two rows.
 */
const isotropic = (sd: number) => [
  [sd * sd, 0],
  [0, sd * sd],
]

/**
 * A labelled 2-d dataset with the truth built from `model` (unless inside a reference draw), task `clustering`.
 *
 * @param name The dataset's name, also the recipe's generator key.
 * @param description The one-line description in `meta`.
 * @param x The points, row-major: $2n$ values, point $i$ at entries `2 * i` and `2 * i + 1`. Used as is, not copied.
 * @param y The label of each point, $n$ values.
 * @param model Builds the class model the truth is made from; called only outside a reference draw.
 * @param labelNames The name of each class.
 * @param s The stream the data was drawn from, whose key is recorded.
 * @param knobs The generator's parameters, recorded in the recipe.
 * @param source Where the dataset comes from, for `meta.source`; left out when there is none.
 * @returns The dataset: `x` ($n \times 2$), `y` and `meta`.
 */
function assemble(
  name: string,
  description: string,
  x: Float64Array,
  y: Int32Array,
  model: () => ClassModel,
  labelNames: string[],
  s: Stream,
  knobs: Record<string, unknown>,
  source?: string,
): Dataset {
  const n = y.length
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name,
      description,
      task: 'clustering',
      featureNames: ['x1', 'x2'],
      labelNames,
      source,
      key: s.key,
      truth: referenceDepth === 0 ? classificationTruth(model()) : undefined,
      recipe: generatorRecipe(name, s.key, knobs),
    },
  }
}

/**
 * Points from $k$ Gaussian modes $\Gauss(\mvec_j, \sigma^2\Imat)$, `sizes[j]` from mode $j$, on the substream
 * `mode` $j$, in mode order.
 *
 * @param s The random stream; mode $j$ is drawn from its child `mode` $j$.
 * @param means The mode centres $\mvec_j$, $k$ rows of two values.
 * @param sd The standard deviation $\sigma$ of every mode.
 * @param sizes The number of points of each mode, $k$ counts.
 * @returns `x`, the points row-major ($2n$ values), and `y`, the mode of each.
 */
function drawGaussians(s: Stream, means: readonly (readonly number[])[], sd: number, sizes: readonly number[]) {
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  let row = 0
  means.forEach((m, j) => {
    const r = child(s, 'mode', j)
    for (let i = 0; i < sizes[j]; i++, row++) {
      x[2 * row] = m[0] + sd * normal(r)
      x[2 * row + 1] = m[1] + sd * normal(r)
      y[row] = j
    }
  })
  return { x, y }
}

// ── Ring and grid of Gaussians ───────────────────────────────────────────────────────────────────────────────────────

/** Options of `gaussianRing`. */
export interface GaussianRingOptions extends ClassSizeOptions {
  /** Total points, or one count per mode. Default 1000. */
  n?: ClassSizes
  /** Number of modes on the ring. Default 8. */
  modes?: number
  /** Radius of the ring. Default 2. */
  radius?: number
  /** Standard deviation of every mode. Default 0.05. */
  sd?: number
}

/**
 * Gaussians spaced evenly on a circle, mode $j$ at $r(\cos 2\pi j/k, \sin 2\pi j/k)$ with standard deviation
 * $\sigma$ ($r$ the `radius`, $k$ the `modes`, $\sigma$ the `sd`): the "8 Gaussians" benchmark of GAN mode collapse
 * (Metz et al., 2017, use radius 2 and sd 0.02). Label $j$ is mode $j$; the modes are equally likely unless the class
 * sizes say otherwise.
 *
 * @param s The random stream: mode $j$ is drawn from its child `mode` $j$, the truth's reference sample from `truth`.
 * @param options The class sizes, number of modes, radius and spread; see `GaussianRingOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the mode of each point, and the exact mixture in `meta.truth`.
 *
 * @example Eight modes on a ring of radius 2
 * const d = gaussianRing(stream(0), { n: 80 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', Array.from({ length: 8 }, (_, j) => y.filter((v) => v === j).length))
 * // Mode 2 is centred at (0, 2).
 * const mode = toArray(d.x).filter((_, i) => y[i] === 2)
 * print('mean of mode 2:', [0, 1].map((c) => mode.reduce((a, r) => a + r[c], 0) / mode.length))
 * print('modes of (0, 2) and (-2, 0):', d.meta.truth.decide(tensor([[0, 2], [-2, 0]])))
 */
export function gaussianRing(s: Stream, options: GaussianRingOptions = {}): Dataset {
  const { modes = 8, radius = 2, sd = 0.05 } = options
  const { sizes, priors } = resolveClassSizes(options, modes, 1000, 'gaussianRing')
  const means = Array.from({ length: modes }, (_, j) => [
    radius * Math.cos((TAU * j) / modes),
    radius * Math.sin((TAU * j) / modes),
  ])
  const { x, y } = drawGaussians(s, means, sd, sizes)
  const model = (): ClassModel => ({
    classes: modes,
    priors,
    logDensity: gaussianClasses(
      means,
      means.map(() => isotropic(sd)),
    ),
    ops: [],
    reference: referenceOf(() => gaussianRing(child(s, 'truth'), { modes, radius, sd, n: REFERENCE_SIZE })),
    family: `${modes} Gaussians on a ring`,
  })
  return assemble(
    'gaussianRing',
    `${modes} Gaussians of sd ${sd} evenly spaced on a circle of radius ${radius} (${y.length} points).`,
    x,
    y,
    model,
    means.map((_, j) => `mode ${j + 1}`),
    s,
    { modes, radius, sd, n: sizes },
    'Metz et al. (2017), unrolled GANs',
  )
}

/** Options of `gaussianGrid`. */
export interface GaussianGridOptions extends ClassSizeOptions {
  /** Total points, or one count per mode. Default 1000. */
  n?: ClassSizes
  /** Modes per side ($m^2$ modes for $m$ per side). Default 5. */
  side?: number
  /** Distance between neighbouring modes. Default 1. */
  spacing?: number
  /** Standard deviation of every mode. Default 0.05. */
  sd?: number
}

/**
 * Gaussians on a square grid centred at the origin, $m \times m$ modes ($m$ the `side`) `spacing` apart with standard
 * deviation $\sigma$ = `sd`: the "25 Gaussians" benchmark (Srivastava et al., 2017, VEEGAN). Mode $j$ sits at row
 * $\lfloor j/m \rfloor$ (the $x_2$ coordinate), column $j \bmod m$ (the $x_1$ coordinate), both counted from the
 * lowest.
 *
 * @param s The random stream: mode $j$ is drawn from its child `mode` $j$, the truth's reference sample from `truth`.
 * @param options The class sizes, grid size, spacing and spread; see `GaussianGridOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the mode of each point, and the exact mixture in `meta.truth`.
 *
 * @example A 3 by 3 grid
 * const d = gaussianGrid(stream(0), { n: 90, side: 3 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', Array.from({ length: 9 }, (_, j) => y.filter((v) => v === j).length))
 * // Mode 0 is the lower-left corner, (-1, -1).
 * const mode = toArray(d.x).filter((_, i) => y[i] === 0)
 * print('mean of mode 0:', [0, 1].map((c) => mode.reduce((a, r) => a + r[c], 0) / mode.length))
 * print('mode of (1, 0):', d.meta.truth.decide(tensor([[1, 0]])))
 */
export function gaussianGrid(s: Stream, options: GaussianGridOptions = {}): Dataset {
  const { side = 5, spacing = 1, sd = 0.05 } = options
  const k = side * side
  const { sizes, priors } = resolveClassSizes(options, k, 1000, 'gaussianGrid')
  const half = ((side - 1) * spacing) / 2
  const means = Array.from({ length: k }, (_, j) => [
    (j % side) * spacing - half,
    Math.floor(j / side) * spacing - half,
  ])
  const { x, y } = drawGaussians(s, means, sd, sizes)
  const model = (): ClassModel => ({
    classes: k,
    priors,
    logDensity: gaussianClasses(
      means,
      means.map(() => isotropic(sd)),
    ),
    ops: [],
    reference: referenceOf(() => gaussianGrid(child(s, 'truth'), { side, spacing, sd, n: REFERENCE_SIZE })),
    family: `a ${side} × ${side} grid of Gaussians`,
  })
  return assemble(
    'gaussianGrid',
    `A ${side} × ${side} grid of Gaussians of sd ${sd}, ${spacing} apart (${y.length} points).`,
    x,
    y,
    model,
    means.map((_, j) => `mode ${j + 1}`),
    s,
    { side, spacing, sd, n: sizes },
    'Srivastava et al. (2017), VEEGAN',
  )
}

// ── Curves: pinwheel and Swiss-roll slice ────────────────────────────────────────────────────────────────────────────

/**
 * Points spread uniformly in $u$ along each curve and blurred by $\Gauss(\zeros, \sigma^2\Imat)$; `sizes[j]` on
 * curve $j$, drawn on the substream `mode` $j$, in curve order.
 *
 * @param s The random stream; curve $j$ is drawn from its child `mode` $j$.
 * @param curves One curve per mode, mapping a position $u \in [0, 1)$ to a point of the plane.
 * @param noise The standard deviation $\sigma$ of the noise.
 * @param sizes The number of points on each curve.
 * @returns `x`, the points row-major ($2n$ values), and `y`, the curve of each.
 */
function drawCurves(s: Stream, curves: ((u: number) => [number, number])[], noise: number, sizes: readonly number[]) {
  const total = sizes.reduce((a, b) => a + b, 0)
  const x = new Float64Array(2 * total)
  const y = new Int32Array(total)
  let row = 0
  curves.forEach((curve, j) => {
    const r = child(s, 'mode', j)
    for (let i = 0; i < sizes[j]; i++, row++) {
      const [a, b] = curve(uniform(r))
      x[2 * row] = a + noise * normal(r)
      x[2 * row + 1] = b + noise * normal(r)
      y[row] = j
    }
  })
  return { x, y }
}

/**
 * The class model of modes spread uniformly along curves and blurred by isotropic Gaussian noise: each class density is
 * an integral along its curve.
 *
 * @param curves One curve per class, mapping a position $u \in [0, 1]$ to a point of the plane.
 * @param noise The standard deviation of the noise; positive.
 * @param priors The population class proportions, one per curve.
 * @param reference The lazy reference sample used for the Monte Carlo Bayes error.
 * @param family The name of the model, as the truth reports it.
 * @returns The class model.
 */
function curveModel(
  curves: ((u: number) => [number, number])[],
  noise: number,
  priors: number[],
  reference: () => Reference,
  family: string,
): ClassModel {
  const densities = curves.map((c) => curveLogDensity(c, noise))
  return {
    classes: curves.length,
    priors,
    logDensity: (x) => classColumns(densities.map((f) => f(x))),
    ops: [],
    reference,
    family,
  }
}

/** Options of `pinwheel`. */
export interface PinwheelOptions extends ClassSizeOptions {
  /** Total points, or one count per arm. Default 1000. */
  n?: ClassSizes
  /** Number of arms. Default 5. */
  arms?: number
  /** How far each arm turns, in radians per unit of radius. Default 1. */
  twist?: number
  /** Standard deviation of the isotropic Gaussian noise; positive. Default 0.1. */
  noise?: number
}

/**
 * A pinwheel: $a$ curved arms ($a$ the `arms`), arm $j$ the points at radius $\rho \in [0.4, 2.4)$ (uniform) and angle
 * $2\pi j/a + w\rho$ ($w$ the `twist`), blurred by isotropic noise. After the pinwheel of Johnson et al. (2016), with
 * the radial spread made uniform along the arm so that the density is an exact integral along it. Label $j$ is arm $j$.
 *
 * @param s The random stream: arm $j$ is drawn from its child `mode` $j$, the truth's reference sample from `truth`.
 * @param options The class sizes, number of arms, twist and noise; see `PinwheelOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the arm of each point, and the exact density in `meta.truth`.
 *
 * @example Five arms
 * const d = pinwheel(stream(0), { n: 500 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', Array.from({ length: 5 }, (_, j) => y.filter((v) => v === j).length))
 * // Without noise the radii would lie in [0.4, 2.4); the mean is about 1.4.
 * const r = toArray(d.x).map(([u, v]) => Math.hypot(u, v))
 * print('mean radius:', r.reduce((a, v) => a + v, 0) / r.length)
 */
export function pinwheel(s: Stream, options: PinwheelOptions = {}): Dataset {
  const { arms = 5, twist = 1, noise = 0.1 } = options
  const { sizes, priors } = resolveClassSizes(options, arms, 1000, 'pinwheel')
  const curves = Array.from({ length: arms }, (_, j) => (u: number): [number, number] => {
    const rho = 0.4 + 2 * u
    const angle = (TAU * j) / arms + twist * rho
    return [rho * Math.cos(angle), rho * Math.sin(angle)]
  })
  const { x, y } = drawCurves(s, curves, noise, sizes)
  return assemble(
    'pinwheel',
    `A pinwheel of ${arms} arms with twist ${twist} and noise sd ${noise} (${y.length} points).`,
    x,
    y,
    () =>
      curveModel(
        curves,
        noise,
        priors,
        referenceOf(() => pinwheel(child(s, 'truth'), { arms, twist, noise, n: REFERENCE_SIZE })),
        `a ${arms}-arm pinwheel`,
      ),
    curves.map((_, j) => `arm ${j + 1}`),
    s,
    { arms, twist, noise, n: sizes },
    'after Johnson et al. (2016)',
  )
}

/** Options of `swissRoll2d`. */
export interface SwissRoll2dOptions extends ClassSizeOptions {
  /** Total points, or one count per stretch. Default 1000. */
  n?: ClassSizes
  /** The roll cut into this many stretches of equal length in $t$, the modes. Default 4. */
  stretches?: number
  /** Standard deviation of the isotropic Gaussian noise; positive. Default 0.08. */
  noise?: number
}

/**
 * The Swiss roll's cross-section as a 2-d density: points $(t\cos t, t\sin t)/5$ for $t$ uniform on
 * $[1.5\pi, 4.5\pi)$ (the slice in $x$ and $z$ of `sklearn.datasets.make_swiss_roll`), blurred by isotropic
 * noise, so it fits in $[-3, 3]^2$. The roll is cut into `stretches` equal ranges of $t$, each a mode with its own
 * label (from the inside out), so that coverage of the roll can be counted.
 *
 * @param s The random stream: stretch $j$ is drawn from its child `mode` $j$, the truth's reference sample from
 *   `truth`.
 * @param options The class sizes, number of stretches and noise; see `SwissRoll2dOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` the stretch of each point, and the exact density in `meta.truth`.
 *
 * @example The roll in four stretches
 * const d = swissRoll2d(stream(0), { n: 100 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const y = toArray(d.y)
 * print('class counts:', Array.from({ length: 4 }, (_, j) => y.filter((v) => v === j).length))
 * // The radius is t / 5, so it grows from about 0.94 (inner stretch) to 2.83 (outer).
 * const r = toArray(d.x).map(([u, v]) => Math.hypot(u, v))
 * const stretch = (j) => r.filter((_, i) => y[i] === j)
 * print('mean radius per stretch:', [0, 1, 2, 3].map((j) => stretch(j).reduce((a, v) => a + v, 0) / stretch(j).length))
 */
export function swissRoll2d(s: Stream, options: SwissRoll2dOptions = {}): Dataset {
  const { stretches = 4, noise = 0.08 } = options
  const { sizes, priors } = resolveClassSizes(options, stretches, 1000, 'swissRoll2d')
  const curves = Array.from({ length: stretches }, (_, j) => (u: number): [number, number] => {
    const t = 1.5 * Math.PI * (1 + (2 * (j + u)) / stretches)
    return [(t * Math.cos(t)) / 5, (t * Math.sin(t)) / 5]
  })
  const { x, y } = drawCurves(s, curves, noise, sizes)
  return assemble(
    'swissRoll2d',
    `The Swiss roll's 2-d cross-section in ${stretches} stretches, noise sd ${noise} (${y.length} points).`,
    x,
    y,
    () =>
      curveModel(
        curves,
        noise,
        priors,
        referenceOf(() => swissRoll2d(child(s, 'truth'), { stretches, noise, n: REFERENCE_SIZE })),
        'a Swiss-roll slice',
      ),
    curves.map((_, j) => `stretch ${j + 1}`),
    s,
    { stretches, noise, n: sizes },
    'scikit-learn make_swiss_roll (Pedregosa et al., 2011)',
  )
}

// ── Out-of-distribution points ───────────────────────────────────────────────────────────────────────────────────────

/** Options of `annulus`. */
export interface AnnulusOptions {
  /** Points. Default 200. */
  n?: number
  /** Inner radius, at least 0. Default 3. */
  inner?: number
  /** Outer radius, above the inner. Default 4. */
  outer?: number
}

/**
 * Points uniform on the annulus $r_0 \le \lVert\xvec\rVert \le r_1$ ($r_0$ the `inner` radius, $r_1$ the `outer`),
 * a shell around data in a smaller disc: out-of-distribution points for density and energy scores. Radius by inverse
 * CDF, $r = \sqrt{r_0^2 + u(r_1^2 - r_0^2)}$. The truth is the uniform density $1/(\pi(r_1^2 - r_0^2))$ on the
 * annulus, one class (every label is 0). Throws `DomainError` unless $0 \le r_0 < r_1$.
 *
 * @param s The random stream: the points come from its child `points`, the truth's reference sample from `truth`.
 * @param options The number of points and the two radii; see `AnnulusOptions`.
 * @returns The dataset: `x` ($n \times 2$), `y` all 0, and the uniform density in `meta.truth`.
 *
 * @example A shell between radii 3 and 4
 * const d = annulus(stream(0), { n: 200 })
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * const r = toArray(d.x).map(([u, v]) => Math.hypot(u, v))
 * print('radii from', Math.min(...r), 'to', Math.max(...r))
 * // Uniform on the annulus, the mean squared radius is (9 + 16) / 2 = 12.5.
 * print('mean squared radius:', r.reduce((a, v) => a + v * v, 0) / r.length)
 * // The density is 1 / (7 pi) = 0.0455 inside and 0 outside.
 * const logp = toArray(d.meta.truth.logDensity(tensor([[3.5, 0], [0, 0]])))
 * print('density at radius 3.5 and 0:', logp.map(([v]) => Math.exp(v)))
 */
export function annulus(s: Stream, options: AnnulusOptions = {}): Dataset {
  const { n = 200, inner = 3, outer = 4 } = options
  if (!(outer > inner && inner >= 0)) throw new DomainError('annulus', 'annulus: need 0 ≤ inner < outer')
  const x = new Float64Array(2 * n)
  const r = child(s, 'points')
  for (let i = 0; i < n; i++) {
    const rho = Math.sqrt(inner * inner + uniform(r) * (outer * outer - inner * inner))
    const a = TAU * uniform(r)
    x[2 * i] = rho * Math.cos(a)
    x[2 * i + 1] = rho * Math.sin(a)
  }
  const logArea = Math.log(Math.PI * (outer * outer - inner * inner))
  const model = (): ClassModel => ({
    classes: 1,
    priors: [1],
    logDensity: (q: Tensor) => {
      const { data, n: m } = points(q)
      const out = new Float64Array(m)
      for (let i = 0; i < m; i++) {
        const rho = Math.hypot(data[2 * i], data[2 * i + 1])
        out[i] = rho >= inner && rho <= outer ? -logArea : -Infinity
      }
      return fromData(out, [m, 1])
    },
    ops: [],
    reference: referenceOf(() => annulus(child(s, 'truth'), { n: REFERENCE_SIZE, inner, outer })),
    family: 'a uniform annulus',
  })
  return assemble(
    'annulus',
    `${n} points uniform on the annulus ${inner} ≤ |x| ≤ ${outer}.`,
    x,
    new Int32Array(n),
    model,
    ['shell'],
    s,
    { n, inner, outer },
  )
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'gaussianRing',
    name: 'Ring of Gaussians',
    summary: 'Gaussian modes evenly spaced on a circle, the GAN mode-collapse benchmark; exact density.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 1000 }),
      modes: int(2, 16, { default: 8 }),
      radius: real(0.5, 5, { default: 2 }),
      sd: real(0.01, 1, { default: 0.05 }),
    }),
    truth: true,
    random: true,
    notes: ['generative-adversarial-network'],
  },
  gaussianRing,
)

dataset(
  {
    key: 'gaussianGrid',
    name: 'Grid of Gaussians',
    summary: 'Gaussian modes on a square grid (25 by default); exact density.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 1000 }),
      side: int(2, 7, { default: 5 }),
      spacing: real(0.2, 3, { default: 1 }),
      sd: real(0.01, 1, { default: 0.05 }),
    }),
    truth: true,
    random: true,
    notes: ['generative-adversarial-network'],
  },
  gaussianGrid,
)

dataset(
  {
    key: 'pinwheel',
    name: 'Pinwheel',
    summary: 'Curved arms turning about the origin, one mode per arm; exact density as an integral along each arm.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 1000 }),
      arms: int(2, 8, { default: 5 }),
      twist: real(0, 3, { default: 1 }),
      noise: real(0.02, 0.5, { default: 0.1 }),
    }),
    truth: true,
    random: true,
  },
  pinwheel,
)

dataset(
  {
    key: 'swissRoll2d',
    name: 'Swiss-roll slice',
    summary: "The Swiss roll's 2-d cross-section, cut into stretches that serve as modes; exact density.",
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 1000 }),
      stretches: int(1, 8, { default: 4 }),
      noise: real(0.02, 0.5, { default: 0.08 }),
    }),
    truth: true,
    random: true,
  },
  swissRoll2d,
)

dataset(
  {
    key: 'annulus',
    name: 'Uniform annulus',
    summary: 'Points uniform on a shell around the data: out-of-distribution points for density and energy scores.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({
      n: int(1, 5000, { default: 200 }),
      inner: real(0, 10, { default: 3 }),
      outer: real(0.1, 12, { default: 4 }),
    }),
    truth: true,
    random: true,
  },
  annulus,
)
