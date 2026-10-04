/**
 * Binary data corrupted by class-conditional label noise, with the true posterior known, for the anchor-point tests of
 * Poyiadzi et al. (2022, "Statistical hypothesis testing for class-conditional label noise", ECML-PKDD) and Yang et al.
 * (2024, AAAI). Notation as the papers: the clean label y ∈ {0, 1} (the papers' −1, +1), the noisy label ỹ, and the
 * noise rates α = P(ỹ = 0 | y = 1), β = P(ỹ = 1 | y = 0); uniform noise is α = β = τ, class-conditional noise α ≠ β.
 * The noisy posterior is η̃(x) = (1 − α − β) η(x) + β, so at an anchor point, where η(x) = ½, it is (1 − α + β)/2: ½
 * exactly when α = β.
 *
 * Layouts, each with equal class priors:
 * - `gaussians`: N([1, 1], I) against N([−1, −1], I) (Poyiadzi et al. 2022, §5); η(x) = σ(2(x₁ + x₂)) is logistic,
 *   and the anchors η = ½ are the line x₂ = −x₁, drawn with x₁ ~ U[−4, 4].
 * - `xor`: class 1 an equal mixture of N([2, 2], I) and N([−2, −2], I), class 0 of N([−2, 2], I) and N([2, −2], I)
 *   (Yang et al. 2024, symmetric XOR); the anchors are the two axes.
 * - `asymmetric-xor`: class 1 at [4, 4] and [−2, −2], class 0 at [−1, 1] and [1, −1] (Yang et al. 2024); the anchors are
 *   found numerically.
 *
 * Anchors lie in [−4, 4]²: strict ones (δ = 0) by bisection of η − ½ along random segments, relaxed ones
 * (η ∈ [½ − δ, ½ + δ]) by rejection sampling.
 */

import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { classificationTruth, points, REFERENCE_SIZE, type ClassificationTruth } from '../truth'
import { checkCount, labels, matrix, type DatasetMeta } from '../types'

/** The class layouts of {@link classConditionalNoise}. */
export type NoiseLayout = 'gaussians' | 'xor' | 'asymmetric-xor'

/** Each layout's class-1 and class-0 mixture centres (unit covariance). */
const CENTRES: Record<NoiseLayout, { one: number[][]; zero: number[][] }> = {
  gaussians: { one: [[1, 1]], zero: [[-1, -1]] },
  xor: {
    one: [
      [2, 2],
      [-2, -2],
    ],
    zero: [
      [-2, 2],
      [2, -2],
    ],
  },
  'asymmetric-xor': {
    one: [
      [4, 4],
      [-2, -2],
    ],
    zero: [
      [-1, 1],
      [1, -1],
    ],
  },
}

const density = (x: readonly number[], centres: number[][]) =>
  centres.reduce((s, c) => s + Math.exp(-0.5 * ((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)), 0) / centres.length

/** The clean posterior η(x) = P(y = 1 | x) of a layout (equal priors). */
export function noiseLayoutPosterior(layout: NoiseLayout): (x: readonly number[]) => number {
  const { one, zero } = CENTRES[layout]
  return (x) => {
    const p = density(x, one)
    const q = density(x, zero)
    return p + q > 0 ? p / (p + q) : 0.5
  }
}

/** log p(x | y = j) of a layout's two mixtures, [n, 2] (up to the shared Gaussian constant). */
function layoutLogDensity(layout: NoiseLayout): (x: Tensor) => Tensor {
  const { one, zero } = CENTRES[layout]
  return (x) => {
    const { data, n } = points(x)
    const out = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      const p = [data[2 * i], data[2 * i + 1]]
      out[2 * i] = Math.log(density(p, zero))
      out[2 * i + 1] = Math.log(density(p, one))
    }
    return fromData(out, [n, 2])
  }
}

/**
 * The truth of a noisy layout as a classification model: clean densities and equal priors, then the noise matrix
 * T = [[1 − β, β], [α, 1 − α]] (T[i][j] = P(ỹ = j | y = i)), so `posterior` is η̃ and `cleanPosterior` is η.
 */
export function classConditionalNoiseTruth(
  s: Stream,
  layout: NoiseLayout,
  alpha: number,
  beta: number,
): ClassificationTruth {
  return classificationTruth({
    classes: 2,
    priors: [0.5, 0.5],
    logDensity: layoutLogDensity(layout),
    ops: [
      {
        kind: 'noise',
        matrix: [
          [1 - beta, beta],
          [alpha, 1 - alpha],
        ],
      },
    ],
    reference: () => ({
      x: classConditionalNoise(child(s, 'reference'), { n: REFERENCE_SIZE, layout, alpha: 0, beta: 0, truth: false }).x,
      weights: new Float64Array(REFERENCE_SIZE).fill(1 / REFERENCE_SIZE),
    }),
    family: `${layout} with class-conditional label noise`,
  })
}

/** The noisy posterior η̃ = (1 − α − β) η + β. */
export const noisyPosterior = (eta: number, alpha: number, beta: number): number => (1 - alpha - beta) * eta + beta

const BOX = 4

/**
 * k anchor points of a layout in [−4, 4]²: |η(x) − ½| ≤ δ. With δ = 0 each is a root of η − ½ on a random segment
 * (bisection to machine precision); with δ > 0 points are drawn uniformly on the box and kept when they qualify. The
 * `gaussians` layout uses the papers' recipe directly: x = (u, −u), u ~ U[−4, 4] (plus a uniform offset along [1, 1]
 * inside the band for δ > 0, found by rejection).
 */
export function noiseLayoutAnchors(s: Stream, layout: NoiseLayout, k: number, delta = 0): number[][] {
  checkCount(k, 'noiseLayoutAnchors')
  if (!(delta >= 0 && delta < 0.5))
    throw new DomainError('noiseLayoutAnchors', 'noiseLayoutAnchors: δ must be in [0, ½)')
  const eta = noiseLayoutPosterior(layout)
  const draw = (r: Stream) => [BOX * (2 * uniform(child(r, 'a')) - 1), BOX * (2 * uniform(child(r, 'b')) - 1)]
  const out: number[][] = []
  for (let i = 0; out.length < k; i++) {
    if (i > 10000 * (k + 1)) throw new DomainError('noiseLayoutAnchors', 'noiseLayoutAnchors: no anchors found')
    const r = child(s, 'anchor', i)
    if (delta > 0) {
      const x = layout === 'gaussians' ? lineOffset(r) : draw(r)
      if (Math.abs(eta(x) - 0.5) <= delta) out.push(x)
      continue
    }
    if (layout === 'gaussians') {
      const u = BOX * (2 * uniform(child(r, 'u')) - 1)
      out.push([u, -u])
      continue
    }
    let a = draw(child(r, 'from'))
    let b = draw(child(r, 'to'))
    let fa = eta(a) - 0.5
    const fb = eta(b) - 0.5
    if (fa === 0) {
      out.push(a)
      continue
    }
    if (Math.sign(fa) === Math.sign(fb)) continue
    for (let it = 0; it < 80; it++) {
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
      const fm = eta(m) - 0.5
      if (Math.sign(fm) === Math.sign(fa)) {
        a = m
        fa = fm
      } else b = m
    }
    out.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
  }
  return out
}

/** A point near the anchor line of `gaussians`: (u, −u) plus an offset v(1, 1)/√2 with v ~ U[−1, 1]. */
function lineOffset(r: Stream): number[] {
  const u = BOX * (2 * uniform(child(r, 'u')) - 1)
  const v = 2 * uniform(child(r, 'v')) - 1
  return [u + v / Math.SQRT2, -u + v / Math.SQRT2]
}

/** A sample with class-conditional label noise. */
export interface ClassConditionalNoiseSample {
  kind: 'dataset'
  x: Tensor
  /** The noisy labels ỹ (int32, 0/1): what a learner sees. */
  y: Tensor
  /** The clean labels y and which labels were flipped. */
  clean: Tensor
  flipped: Tensor
  alpha: number
  beta: number
  layout: NoiseLayout
  meta: DatasetMeta
}

/** Options of {@link classConditionalNoise}. */
export interface ClassConditionalNoiseOptions {
  /** Training points N (default 1000), half of each class. */
  n?: number
  layout?: NoiseLayout
  /** α = P(ỹ = 0 | y = 1) and β = P(ỹ = 1 | y = 0) (default 0 and 0.1, as Yang et al.). */
  alpha?: number
  beta?: number
  /** Attach the truth (default true). */
  truth?: boolean
}

/**
 * Draw N points of a layout (module notes), half of each class, and flip each label independently: a class-1 label with
 * probability α, a class-0 label with probability β. The flips use their own substream (`child(s, 'flip')`), so the
 * same seed with α = β = 0 gives the clean sample.
 */
export function classConditionalNoise(
  s: Stream,
  options: ClassConditionalNoiseOptions = {},
): ClassConditionalNoiseSample {
  const { n = 1000, layout = 'gaussians', alpha = 0, beta = 0.1, truth = true } = options
  checkCount(n, 'classConditionalNoise')
  if (!(alpha >= 0 && beta >= 0 && alpha + beta < 1))
    throw new DomainError('classConditionalNoise', 'classConditionalNoise: the noise rates need α, β ≥ 0 and α + β < 1')
  const { one, zero } = CENTRES[layout]
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  const clean = new Int32Array(n)
  const flipped = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const c = i < Math.ceil(n / 2) ? 1 : 0
    const centres = c === 1 ? one : zero
    const m = centres[Math.floor(uniform(child(s, 'component', i)) * centres.length)]
    x[2 * i] = m[0] + normal(child(s, 'x', i, 0))
    x[2 * i + 1] = m[1] + normal(child(s, 'x', i, 1))
    clean[i] = c
    const flip = uniform(child(s, 'flip', i)) < (c === 1 ? alpha : beta)
    flipped[i] = flip ? 1 : 0
    y[i] = flip ? 1 - c : c
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    clean: labels(clean),
    flipped: labels(flipped),
    alpha,
    beta,
    layout,
    meta: {
      name: `label noise (${layout})`,
      description: `${n} points of the ${layout} layout with labels flipped at α = P(ỹ = 0 | y = 1) = ${alpha} and β = P(ỹ = 1 | y = 0) = ${beta}.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['0 (−1)', '1 (+1)'],
      key: s.key,
      ...(truth ? { truth: classConditionalNoiseTruth(child(s, 'truth'), layout, alpha, beta) } : {}),
    } as DatasetMeta,
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'classConditionalNoise',
    name: 'Class-conditional label noise',
    summary: 'Two classes with a known posterior whose labels are flipped at rates α (class 1) and β (class 0).',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(10, 100000, { default: 1000 }),
      alpha: real(0, 0.49, { default: 0 }),
      beta: real(0, 0.49, { default: 0.1 }),
    }),
    truth: true,
    random: true,
    notes: [
      'hypothesis-testing-for-class-conditional-label-noise',
      'local-maximum-likelihood-noise-test',
      'label-noise-models',
    ],
    cite: ['poyiadzi2022', 'yang2024'],
  },
  classConditionalNoise,
)
