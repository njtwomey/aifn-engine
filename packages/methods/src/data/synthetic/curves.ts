/**
 * Seeded one-dimensional smooth regressions for GAM and expectile figures, each with its whole law known: a sine wave
 * whose noise spreads and pinches, a bump with skewed noise growing to the right, Poisson counts with a log link,
 * Bernoulli outcomes with a logit link and gamma responses with a log link. x is uniform on [0, 1] and sorted; the
 * truth (`meta.truth`, a `Curve1dTruth`) gives the mean, the link scale and the true expectile curves.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import type { FamilyName, LinkName } from 'aifn-compute/probability/likelihoods'
import { curve1dTruth, type Curve1dLaw } from '../truth'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The cases of `curve1d`. */
export type Curve1dCase = 'sine' | 'skewed' | 'counts' | 'binary' | 'gamma'

/** Each case: its family and natural link, η(x), and a formula for captions. */
export const CURVE1D_CASES: Readonly<
  Record<Curve1dCase, { family: FamilyName; link: LinkName; eta: (x: number) => number; formula: string }>
> = {
  sine: { family: 'gaussian', link: 'identity', eta: (x) => Math.sin(2 * Math.PI * x), formula: 'μ = sin 2πx' },
  skewed: {
    family: 'gaussian',
    link: 'identity',
    eta: (x) => 0.8 * x + 1.2 * Math.exp(-(((x - 0.3) / 0.15) ** 2)),
    formula: 'μ = 0.8x + 1.2 exp(−((x − 0.3)/0.15)²)',
  },
  counts: { family: 'poisson', link: 'log', eta: (x) => 1 + Math.sin(2 * Math.PI * x), formula: 'log μ = 1 + sin 2πx' },
  binary: {
    family: 'binomial',
    link: 'logit',
    eta: (x) => -2 + 4 * Math.exp(-(((x - 0.5) / 0.15) ** 2)),
    formula: 'logit μ = −2 + 4 exp(−((x − ½)/0.15)²)',
  },
  gamma: {
    family: 'gamma',
    link: 'log',
    eta: (x) => 0.5 + 0.8 * x + 0.6 * Math.sin(3 * Math.PI * x),
    formula: 'log μ = 0.5 + 0.8x + 0.6 sin 3πx',
  },
}

/** Options of `curve1d`. */
export interface Curve1dOptions {
  /** Rows. Default 300. */
  n?: number
  /** Default `sine`. */
  case?: Curve1dCase
  /**
   * sine and skewed: σ₀, the noise sd where it is smallest (default 0.15); gamma: the coefficient of variation
   * (default 0.5); ignored otherwise.
   */
  noise?: number
  /** sine and skewed: the noise shape (default normal for sine, skewed for skewed). */
  noiseShape?: 'normal' | 'skewed'
  /**
   * sine and skewed: h ≥ 0, how far the noise sd varies: σ(x) = σ₀(1 + h sin²(3πx/2)) for sine (it spreads and pinches
   * twice) and σ₀(1 + hx) for skewed (it grows to the right). Default 2.
   */
  heteroscedastic?: number
}

/** A one-dimensional smooth regression (see the module comment); `f` holds μ(x). */
export function curve1d(s: Stream, options: Curve1dOptions = {}): Dataset {
  const { n = 300, heteroscedastic: h = 2 } = options
  const which = options.case ?? 'sine'
  checkCount(n, 'curve1d')
  const spec = CURVE1D_CASES[which]
  if (!spec) throw new DomainError('curve1d', `curve1d: unknown case "${which}"`)
  if (!(h >= 0)) throw new DomainError('curve1d', 'curve1d: heteroscedastic must be non-negative')
  const continuous = which === 'sine' || which === 'skewed'
  const noise = options.noise ?? (which === 'gamma' ? 0.5 : 0.15)
  if (!(noise > 0)) throw new DomainError('curve1d', 'curve1d: the noise must be positive')
  const noiseShape = options.noiseShape ?? (which === 'skewed' ? 'skewed' : 'normal')
  const law: Curve1dLaw = continuous
    ? {
        kind: 'location-scale',
        noise: noiseShape,
        sd: which === 'sine' ? (x) => noise * (1 + h * Math.sin(1.5 * Math.PI * x) ** 2) : (x) => noise * (1 + h * x),
      }
    : { kind: 'family', dispersion: which === 'gamma' ? noise * noise : 1 }
  const truth = curve1dTruth({ name: `curve1d ${which}`, family: spec.family, link: spec.link, eta: spec.eta, law })
  const xs = Float64Array.from(toFlat(uniform(child(s, 'x'), 0, 1, { shape: [n] }) as Tensor)).sort()
  const X = matrix(xs, n, 1)
  const y = Float64Array.from(toFlat(truth.predictive(X).sample(child(s, 'y')) as Tensor))
  const spread = continuous
    ? `, ${truth.law} with sd ${noise}·(1 + ${h}${which === 'sine' ? ' sin²(3πx/2)' : 'x'})`
    : `, ${truth.law}`
  return {
    kind: 'dataset',
    x: X,
    y: vector(y),
    f: truth.mean(X),
    meta: {
      name: `curve1d ${which}`,
      description: `${n} draws with x uniform on [0, 1]: ${spec.formula} (${spec.family}, ${spec.link} link)${spread}.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 'y',
      key: s.key,
      truth,
      recipe: generatorRecipe('curve1d', s.key, { n, case: which, noise, noiseShape, heteroscedastic: h }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'curve1d',
    name: 'One-dimensional smooth regression',
    summary:
      'A smooth curve on [0, 1] with heteroscedastic normal or skewed noise, Poisson counts, Bernoulli outcomes or gamma responses; the truth knows the whole law, so the true expectile curves.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      case: oneOf(['sine', 'skewed', 'counts', 'binary', 'gamma']),
      noise: real(0.01, 2, { default: 0.15 }),
      noiseShape: oneOf(['normal', 'skewed']),
      heteroscedastic: real(0, 10, { default: 2 }),
    }),
    truth: true,
    random: true,
    notes: ['expectile-generalised-additive-models', 'generalised-additive-model'],
  },
  curve1d,
)
