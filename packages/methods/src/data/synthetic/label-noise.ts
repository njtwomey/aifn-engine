/**
 * Binary data corrupted by class-conditional label noise, with the true posterior known, for the anchor-point tests of
 * Poyiadzi et al. (2022, "Statistical hypothesis testing for class-conditional label noise", ECML-PKDD) and Yang et al.
 * (2024, AAAI). Notation as the papers: the clean label $y \in \{0, 1\}$ (the papers' $-1$, $+1$), the noisy label
 * $\tilde{y}$, and the noise rates $\alpha = \pr(\tilde{y} = 0 \mid y = 1)$, $\beta = \pr(\tilde{y} = 1 \mid y = 0)$;
 * uniform noise is $\alpha = \beta = \tau$, class-conditional noise $\alpha \ne \beta$. The noisy posterior is
 * $\tilde{\eta}(\xvec) = (1 - \alpha - \beta) \eta(\xvec) + \beta$, so at an anchor point, where $\eta(\xvec) = 1/2$,
 * it is $(1 - \alpha + \beta)/2$: $1/2$ exactly when $\alpha = \beta$.
 *
 * Layouts, each with equal class priors:
 * - `gaussians`: $\Gauss([1, 1], \Imat)$ against $\Gauss([-1, -1], \Imat)$ (Poyiadzi et al. 2022, §5);
 *   $\eta(\xvec) = \sigma(2(x_1 + x_2))$ is logistic, and the anchors $\eta = 1/2$ are the line $x_2 = -x_1$, drawn
 *   with $x_1 \sim \Unif(-4, 4)$.
 * - `xor`: class 1 an equal mixture of $\Gauss([2, 2], \Imat)$ and $\Gauss([-2, -2], \Imat)$, class 0 of
 *   $\Gauss([-2, 2], \Imat)$ and $\Gauss([2, -2], \Imat)$ (Yang et al. 2024, symmetric XOR); the anchors are the two
 *   axes.
 * - `asymmetric-xor`: class 1 at $[4, 4]$ and $[-2, -2]$, class 0 at $[-1, 1]$ and $[1, -1]$ (Yang et al. 2024); the
 *   anchors are found numerically.
 *
 * Anchors lie in $[-4, 4]^2$: strict ones ($\delta = 0$) by bisection of $\eta - 1/2$ along random segments, relaxed
 * ones ($\eta \in [1/2 - \delta, 1/2 + \delta]$) by rejection sampling.
 */

import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { classificationTruth, points, REFERENCE_SIZE, type ClassificationTruth } from '../truth'
import { checkCount, labels, matrix, type DatasetMeta } from '../types'

/** The class layouts of `classConditionalNoise`. */
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

/**
 * The density of an equal mixture of unit Gaussians at a point, up to the constant $1/(2\pi)$.
 *
 * @param x The point, two coordinates.
 * @param centres The mixture's centres, each two coordinates.
 * @returns The mean over the centres of $\exp(-\norm{\xvec - \cvec}^2/2)$.
 */
const density = (x: readonly number[], centres: number[][]) =>
  centres.reduce((s, c) => s + Math.exp(-0.5 * ((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)), 0) / centres.length

/**
 * The clean posterior $\eta(\xvec) = \pr(y = 1 \mid \xvec)$ of a layout (equal priors); $1/2$ where both densities
 * underflow.
 *
 * @param layout The class layout.
 * @returns $\eta$ as a function of a point (two coordinates).
 *
 * @example On the anchor line, and towards each class
 * const eta = noiseLayoutPosterior('gaussians')
 * print('eta(0, 0):', eta([0, 0]), ' eta(1, -1):', eta([1, -1]))
 * print('eta(1, 1):', eta([1, 1]), ' sigma(4):', 1 / (1 + Math.exp(-4)))
 * print('xor, eta(2, 2) and eta(-2, 2):', noiseLayoutPosterior('xor')([2, 2]), noiseLayoutPosterior('xor')([-2, 2]))
 */
export function noiseLayoutPosterior(layout: NoiseLayout): (x: readonly number[]) => number {
  const { one, zero } = CENTRES[layout]
  return (x) => {
    const p = density(x, one)
    const q = density(x, zero)
    return p + q > 0 ? p / (p + q) : 0.5
  }
}

/**
 * $\log p(\xvec \mid y = j)$ of a layout's two mixtures, $n \times 2$ (up to the shared Gaussian constant).
 *
 * @param layout The class layout.
 * @returns A function of $n \times 2$ points giving, per row, the log density under class 0 then class 1.
 */
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
 * $\Tmat = \begin{pmatrix} 1 - \beta & \beta \\ \alpha & 1 - \alpha \end{pmatrix}$
 * ($T_{ij} = \pr(\tilde{y} = j \mid y = i)$), so `posterior` is $\tilde{\eta}$ and `cleanPosterior` is $\eta$.
 *
 * @param s The stream of the clean reference sample the truth draws when asked for it (child `'reference'`).
 * @param layout The class layout.
 * @param alpha The rate $\alpha$ at which class-1 labels flip.
 * @param beta The rate $\beta$ at which class-0 labels flip.
 * @returns The classification truth, its posteriors over the noisy labels.
 *
 * @example At an anchor the noisy posterior is (1 - alpha + beta) / 2
 * const truth = classConditionalNoiseTruth(stream(1), 'gaussians', 0.1, 0.3)
 * const x = tensor([[0, 0], [2, 2]])
 * print('clean posterior:', truth.cleanPosterior(x))
 * print('noisy posterior:', truth.posterior(x))
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

/**
 * The noisy posterior $\tilde{\eta} = (1 - \alpha - \beta) \eta + \beta$.
 *
 * @param eta The clean posterior $\eta = \pr(y = 1 \mid \xvec)$.
 * @param alpha The rate $\alpha$ at which class-1 labels flip.
 * @param beta The rate $\beta$ at which class-0 labels flip.
 * @returns $\tilde{\eta} = \pr(\tilde{y} = 1 \mid \xvec)$.
 *
 * @example At an anchor point, uniform against class-conditional noise
 * print('alpha = beta = 0.2:', noisyPosterior(0.5, 0.2, 0.2))
 * print('alpha = 0, beta = 0.1:', noisyPosterior(0.5, 0, 0.1))
 * print('eta = 1, alpha = 0.2:', noisyPosterior(1, 0.2, 0.1))
 */
export const noisyPosterior = (eta: number, alpha: number, beta: number): number => (1 - alpha - beta) * eta + beta

/** Half the side of the box $[-4, 4]^2$ in which anchors are sought. */
const BOX = 4

/**
 * $k$ anchor points of a layout in $[-4, 4]^2$: $\abs{\eta(\xvec) - 1/2} \le \delta$. With $\delta = 0$ each is a root
 * of $\eta - 1/2$ on a random segment (bisection to machine precision); with $\delta > 0$ points are drawn uniformly
 * on the box and kept when they qualify. The `gaussians` layout uses the papers' recipe directly: $\xvec = (u, -u)$,
 * $u \sim \Unif(-4, 4)$ (plus, for $\delta > 0$, an offset $v(1, 1)/\sqrt{2}$ with $v \sim \Unif(-1, 1)$, kept by
 * rejection, which can reach just past the box). Throws `DomainError` when $k$ is not a non-negative integer, $\delta$
 * is not in $[0, 1/2)$, or no $k$ anchors are found in $10000(k + 1)$ tries.
 *
 * @param s The stream the candidates are drawn from (try $i$ from `child(s, 'anchor', i)`).
 * @param layout The class layout.
 * @param k The number of anchor points.
 * @param delta The half-width $\delta$ of the band of $\eta$ around $1/2$ that counts as an anchor; 0 for exact ones.
 * @returns The $k$ anchor points, each two coordinates.
 *
 * @example Exact anchors of the XOR layout lie on the axes
 * const anchors = noiseLayoutAnchors(stream(1), 'xor', 4)
 * print('anchors:', anchors)
 * print('eta there:', anchors.map(noiseLayoutPosterior('xor')))
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

/**
 * A point near the anchor line of `gaussians`: $(u, -u)$ plus an offset $v(1, 1)/\sqrt{2}$, with $u \sim \Unif(-4, 4)$
 * and $v \sim \Unif(-1, 1)$.
 *
 * @param r The stream $u$ and $v$ are drawn from (children `'u'` and `'v'`).
 * @returns The point, two coordinates.
 */
function lineOffset(r: Stream): number[] {
  const u = BOX * (2 * uniform(child(r, 'u')) - 1)
  const v = 2 * uniform(child(r, 'v')) - 1
  return [u + v / Math.SQRT2, -u + v / Math.SQRT2]
}

/** A sample with class-conditional label noise. */
export interface ClassConditionalNoiseSample {
  /** Marks a dataset. */
  kind: 'dataset'
  /** The points, $n \times 2$. */
  x: Tensor
  /** The noisy labels $\tilde{y}$ (int32, 0/1): what a learner sees. */
  y: Tensor
  /** The clean labels $y$ (int32, 0/1). */
  clean: Tensor
  /** Which labels were flipped (int32, 1 for a flip). */
  flipped: Tensor
  /** The rate $\alpha$ at which class-1 labels were flipped. */
  alpha: number
  /** The rate $\beta$ at which class-0 labels were flipped. */
  beta: number
  /** The class layout. */
  layout: NoiseLayout
  /** The name, description and, unless turned off, the truth (`classConditionalNoiseTruth`). */
  meta: DatasetMeta
}

/** Options of `classConditionalNoise`. */
export interface ClassConditionalNoiseOptions {
  /** Training points $N$ (default 1000), half of each class (class 1 gets the odd one). */
  n?: number
  /** The class layout (default `'gaussians'`). */
  layout?: NoiseLayout
  /** $\alpha = \pr(\tilde{y} = 0 \mid y = 1)$ (default 0, as Yang et al.). */
  alpha?: number
  /** $\beta = \pr(\tilde{y} = 1 \mid y = 0)$ (default 0.1, as Yang et al.); $\alpha + \beta$ must be below 1. */
  beta?: number
  /** Attach the truth (default true). */
  truth?: boolean
}

/**
 * Draw $N$ points of a layout (see the file comment), half of each class, and flip each label independently: a class-1
 * label with probability $\alpha$, a class-0 label with probability $\beta$. The flips use their own substream
 * (`child(s, 'flip')`), so the same seed with $\alpha = \beta = 0$ gives the clean sample. The first
 * $\lceil N/2 \rceil$ rows are class 1. Throws `DomainError` when $N$ is not a non-negative integer, a rate is negative
 * or $\alpha + \beta \ge 1$.
 *
 * @param s The stream the points are drawn from (per point, children `'component'`, `'x'` and `'flip'`; the truth's
 *   reference sample from `child(s, 'truth')`).
 * @param options The size, the layout, the noise rates, and whether to attach the truth.
 * @returns The points with their noisy and clean labels, which were flipped, the rates and the layout.
 *
 * @example The flip rate of each class matches its noise rate
 * const d = classConditionalNoise(stream(1), { n: 4000, alpha: 0.1, beta: 0.3 })
 * const [clean, flipped] = [toArray(d.clean), toArray(d.flipped)]
 * print('x:', d.x.shape, ' first rows:', toArray(d.x).slice(0, 2))
 * print('first noisy labels:', toArray(d.y).slice(0, 8), ' clean:', clean.slice(0, 8))
 * const rate = (c) => flipped.filter((f, i) => clean[i] === c && f === 1).length / clean.filter((v) => v === c).length
 * print('class-1 flip rate (alpha 0.1):', rate(1), ' class-0 flip rate (beta 0.3):', rate(0))
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
