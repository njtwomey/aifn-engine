/**
 * Seeded two-dimensional densities for generative models, each with its exact density: a ring of Gaussians, a grid of
 * Gaussians, a pinwheel, a Swiss-roll slice, and a uniform annulus for out-of-distribution points. Every point is
 * labelled with the mode it came from (a Gaussian, an arm, a stretch of the roll), so that `meta.truth` is a
 * classification truth whose class-conditional densities and priors give the data density
 * p(x) = Σⱼ πⱼ p(x | j) exactly (`marginalLogDensity`), and a generated point can be assigned to its mode by the Bayes
 * posterior. The rings and grids follow the GAN mode-collapse benchmarks (Metz et al., 2017; Srivastava et al., 2017);
 * the pinwheel after Johnson et al. (2016) and the roll after scikit-learn's `make_swiss_roll` are drawn here as
 * curves blurred by isotropic Gaussian noise, so their densities are one-dimensional integrals along the curve.
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

const isotropic = (sd: number) => [
  [sd * sd, 0],
  [0, sd * sd],
]

/** A labelled 2-d dataset with the truth built from `model` (unless inside a reference draw). */
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

/** Points from k Gaussian modes N(mⱼ, sd²I), `sizes[j]` from mode j, on the substream `mode j`. */
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
 * Gaussians spaced evenly on a circle, mode j at radius·(cos 2πj/k, sin 2πj/k) with standard deviation sd: the
 * "8 Gaussians" benchmark of GAN mode collapse (Metz et al., 2017, use radius 2 and sd 0.02). Label j is mode j.
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
  /** Modes per side (side² modes). Default 5. */
  side?: number
  /** Distance between neighbouring modes. Default 1. */
  spacing?: number
  /** Standard deviation of every mode. Default 0.05. */
  sd?: number
}

/**
 * Gaussians on a square grid centred at the origin, side × side modes `spacing` apart with standard deviation sd: the
 * "25 Gaussians" benchmark (Srivastava et al., 2017, VEEGAN). Mode j sits at row ⌊j/side⌋, column j mod side.
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

/** Points spread uniformly in u along each curve and blurred by N(0, noise²I); `sizes[j]` on curve j. */
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
  /** Isotropic Gaussian noise. Default 0.1. */
  noise?: number
}

/**
 * A pinwheel: `arms` curved arms, arm j the points at radius ρ ∈ [0.4, 2.4] (uniform) and angle 2πj/arms + twist·ρ,
 * blurred by isotropic noise. After the pinwheel of Johnson et al. (2016), with the radial spread made uniform along
 * the arm so that the density is an exact integral along it. Label j is arm j.
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
  /** The roll cut into this many stretches of equal length in t, the modes. Default 4. */
  stretches?: number
  /** Isotropic Gaussian noise. Default 0.08. */
  noise?: number
}

/**
 * The Swiss roll's cross-section as a 2-d density: points (t cos t, t sin t)/5 for t uniform on [1.5π, 4.5π] (the
 * x–z slice of `sklearn.datasets.make_swiss_roll`), blurred by isotropic noise, so it fits in [−3, 3]². The roll is
 * cut into `stretches` equal ranges of t, each a mode with its own label, so that coverage of the roll can be counted.
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
  /** Inner and outer radius. Default 3 and 4. */
  inner?: number
  outer?: number
}

/**
 * Points uniform on the annulus inner ≤ |x| ≤ outer, a shell around data in a smaller disc: out-of-distribution
 * points for density and energy scores. Radius by inverse CDF, r = √(inner² + u(outer² − inner²)). The truth is the
 * uniform density 1/(π(outer² − inner²)) on the annulus, one class.
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
