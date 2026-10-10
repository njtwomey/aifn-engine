/**
 * Seeded data from a generalised additive model: features uniform on $[0, 1]$, $\eta = \alpha + c\sum_j f_j(x_j)$ with
 * one named shape $f_j$ per feature and an effect scale $c$, $\mu = g^{-1}(\eta)$, and $y$ drawn from the family at
 * $\mu$. The truth carries each centred partial effect, so a figure draws a fitted GAM's terms against the true ones on
 * the same (link) scale.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { checkLink, family as familyByName, type LinkName } from 'aifn-compute/probability/likelihoods'
import { additiveTruth } from '../truth'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The named effect shapes of `additiveData`, each on $[0, 1]$ with mean zero there. */
export type AdditiveShape = 'periodic' | 'monotone' | 'wiggly' | 'smooth' | 'linear' | 'none'

/** The families `additiveData` draws from. */
export type AdditiveFamily = 'gaussian' | 'binomial' | 'poisson' | 'gamma'

const MIDPOINTS = 2000
/**
 * $f$ minus its mean over $\Unif(0, 1)$ (midpoint rule, 2000 points; exact to about $10^{-7}$ for these shapes).
 *
 * @param f The shape to centre, a function on $[0, 1]$.
 * @returns The centred function $x \mapsto f(x) - \bar{f}$.
 */
const centred = (f: (x: number) => number) => {
  let m = 0
  for (let i = 0; i < MIDPOINTS; i++) m += f((i + 0.5) / MIDPOINTS) / MIDPOINTS
  return (x: number) => f(x) - m
}

/** Each shape on $[0, 1]$, centred, with a formula for captions. Peak sizes are about 1. */
export const ADDITIVE_SHAPES: Readonly<Record<AdditiveShape, { f: (x: number) => number; formula: string }>> = {
  // Its value and slope agree at 0 and 1, so a cyclic smooth fits it exactly at the ends.
  periodic: {
    f: centred((x) => Math.sin(2 * Math.PI * x) + 0.5 * Math.cos(4 * Math.PI * x)),
    formula: 'sin 2πx + ½ cos 4πx',
  },
  monotone: { f: centred((x) => 2 / (1 + Math.exp(-8 * (x - 0.5))) - 1), formula: '2σ(8(x − ½)) − 1' },
  // A chirp: the frequency grows to the right, so a small basis misses the last oscillations.
  wiggly: { f: centred((x) => 0.8 * Math.sin(6 * Math.PI * x * x)), formula: '0.8 sin 6πx²' },
  smooth: { f: centred((x) => 3 * (x - 0.5) ** 2), formula: '3(x − ½)²' },
  linear: { f: centred((x) => x - 0.5), formula: 'x − ½' },
  none: { f: () => 0, formula: '0' },
}

/**
 * The intercept $\alpha$ and effect scale $c$ for each family and link: chosen so that $\mu$ stays inside the family's
 * mean space for any $\xvec$ (the shapes sum to at most about 3.2 in size) and the effects are clearly visible against
 * the noise.
 */
const SCALES: Readonly<Record<AdditiveFamily, Partial<Record<LinkName, { intercept: number; scale: number }>>>> = {
  gaussian: {
    identity: { intercept: 0, scale: 1 },
    log: { intercept: Math.log(3), scale: 0.3 },
    inverse: { intercept: 0.5, scale: 0.1 },
  },
  binomial: {
    logit: { intercept: 0, scale: 1.5 },
    probit: { intercept: 0, scale: 0.9 },
    cloglog: { intercept: -0.4, scale: 0.9 },
    log: { intercept: Math.log(0.3), scale: 0.25 },
  },
  poisson: {
    log: { intercept: Math.log(4), scale: 0.6 },
    identity: { intercept: 6, scale: 1.6 },
    sqrt: { intercept: 2.5, scale: 0.5 },
  },
  gamma: {
    inverse: { intercept: 0.6, scale: 0.12 },
    log: { intercept: 0, scale: 0.5 },
    identity: { intercept: 4, scale: 0.8 },
  },
}

/** Options of `additiveData`. */
export interface AdditiveOptions {
  /** Rows. Default 300. */
  n?: number
  /** The family $y$ is drawn from (default `'gaussian'`). */
  family?: AdditiveFamily
  /**
   * The link $g$ (default the family's default link). One the family does not take, or one without a setting of
   * $\alpha$ and $c$ here, is rejected.
   */
  link?: LinkName
  /** One shape per feature, so their number is $d$. Default `'periodic'`, `'monotone'`, `'wiggly'`. */
  shapes?: readonly AdditiveShape[]
  /**
   * Gaussian: the noise standard deviation (default 0.4); gamma: the coefficient of variation (default 0.3); ignored
   * otherwise.
   */
  noise?: number
}

/**
 * A GAM dataset: $\Xmat$ ($n \times d$) uniform on $[0, 1]$, and $y$ from the family at
 * $\mu = g^{-1}(\alpha + c\sum_j f_j(x_j))$, with $\alpha$ and $c$ set per family and link. The truth (`meta.truth`, an
 * `AdditiveTruth`) gives each effect $c f_j$ centred over $\Unif(0, 1)$, the mean and the law of $y$. Throws
 * `DomainError` when $n$ is not a non-negative integer, the link does not suit the family or has no setting, a shape is
 * unknown, or the noise of a Gaussian or gamma family is not positive.
 *
 * @param s The stream the features (child `'x'`) and the responses (child `'y'`) are drawn from.
 * @param options The size, the family and link, the shapes and the noise.
 * @returns A regression dataset: `x` ($n \times d$), `y`, `f` the true mean $\mu$, and `meta.truth`.
 *
 * @example The residuals have the noise's standard deviation
 * const d = additiveData(stream(1), { n: 500, noise: 0.4 })
 * const [y, f] = [toArray(d.y), toArray(d.f)]
 * print('x:', d.x.shape, ' y:', d.y.shape, ' features:', d.meta.featureNames)
 * print('first row:', toArray(d.x)[0], ' y:', y[0], ' mu:', f[0])
 * print('sd of y - mu:', Math.sqrt(y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0) / y.length))
 *
 * @example Poisson counts through a log link
 * const d = additiveData(stream(1), { n: 300, family: 'poisson', shapes: ['monotone', 'none'] })
 * print('first counts:', toArray(d.y).slice(0, 8))
 * const { link, intercept } = d.meta.truth
 * print('link:', link, ' intercept:', intercept, ' exp(intercept), the typical count:', Math.exp(intercept))
 */
export function additiveData(s: Stream, options: AdditiveOptions = {}): Dataset {
  const { n = 300, family = 'gaussian', shapes = ['periodic', 'monotone', 'wiggly'] } = options
  checkCount(n, 'additiveData')
  const fam = familyByName(family)
  const link = checkLink(fam, options.link, 'additiveData').name
  const setting = SCALES[family]?.[link]
  if (!setting)
    throw new DomainError('additiveData', `additiveData: no setting for the ${family} family with the ${link} link`)
  const noise = options.noise ?? (family === 'gamma' ? 0.3 : 0.4)
  if (!(noise > 0) && (family === 'gaussian' || family === 'gamma'))
    throw new DomainError('additiveData', 'additiveData: the noise must be positive')
  const dispersion = family === 'gaussian' || family === 'gamma' ? noise * noise : 1
  const d = shapes.length
  const effects = shapes.map((name) => {
    const shape = ADDITIVE_SHAPES[name]
    if (!shape) throw new DomainError('additiveData', `additiveData: unknown shape "${name}"`)
    return { name, f: (x: number) => setting.scale * shape.f(x) }
  })
  const truth = additiveTruth({ family, link, intercept: setting.intercept, dispersion, effects })
  const x = uniform(child(s, 'x'), 0, 1, { shape: [n, d] }) as Tensor
  const xs = Float64Array.from(toFlat(x))
  const X = matrix(xs, n, d)
  const mu = truth.mean(X)
  const y = Float64Array.from(toFlat(truth.predictive(X).sample(child(s, 'y')) as Tensor))
  const named = shapes.map((sh) => `${sh} (${ADDITIVE_SHAPES[sh].formula})`).join(', ')
  return {
    kind: 'dataset',
    x: X,
    y: vector(y),
    f: fromData(Float64Array.from(toFlat(mu)), [n]),
    meta: {
      name: `additive ${family}`,
      description: `${n} draws from a ${family} GAM with the ${link} link: x uniform on [0, 1]^${d}, effects ${named}.`,
      task: 'regression',
      featureNames: shapes.map((_, j) => `x${j + 1}`),
      targetName: 'y',
      key: s.key,
      truth,
      recipe: generatorRecipe('additiveData', s.key, { n, family, link, shapes: [...shapes], noise }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'additiveData',
    name: 'Additive model data',
    summary:
      'Draws from a GAM: a named shape per uniform feature, through a link, with a Gaussian, binomial, Poisson or gamma response.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      family: oneOf(['gaussian', 'binomial', 'poisson', 'gamma']),
      noise: real(0.01, 3, { default: 0.4 }),
    }),
    truth: true,
    random: true,
    notes: ['generalised-additive-model'],
  },
  additiveData,
)
