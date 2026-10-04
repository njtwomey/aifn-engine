/**
 * Mixture density heads (Bishop, 1994, "Mixture density networks"; Bishop, 2006, "Pattern Recognition and Machine
 * Learning", §5.6): a network's raw outputs read as the parameters of a mixture of K Gaussians over a target in ℝᴰ with
 * diagonal covariances, and the mixture's negative log-likelihood as a loss.
 *
 * The head has K(1 + 2D) outputs per row, laid out [logits (K) | means (K·D) | raw scales (K·D)], component-major
 * (component k's means are columns K + kD … K + kD + D − 1). The mixing weights are π = softmax(logits), the means are
 * the outputs as they are, and the standard deviations are σ = floor + exp(s) (Bishop's choice) or floor + softplus(s).
 * The floor keeps the likelihood bounded: without it a component can shrink onto one target and send the
 * log-likelihood to +∞.
 *
 * - `mixtureDensityNll`: −log Σₖ πₖ Πⱼ N(yⱼ; μₖⱼ, σₖⱼ²) per row, by a stable log-sum-exp over components; differentiable.
 * - `mixtureDensityParams`: the differentiable split of a head into log π, μ and σ.
 * - `mixtureDensityHead`: the same as plain numbers per row, with the density, draws, the moments and the modes, and,
 *   the batch's law (its marginals for D > 1) as a compute `Mixture` of `Normal`s.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { categorical, child, normal, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  div,
  exp,
  fromData,
  log,
  logsumexp,
  mul,
  neg,
  reshape,
  shapeOfValue,
  slice,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { logSoftmax, softplus } from 'aifn-compute/numerics/special'
import { Mixture, Normal, type Univariate } from 'aifn-compute/probability/distributions'
import { constant, defineLoss, reduce, type ReductionOptions, type Target } from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)

/** The link from a raw scale output s to a standard deviation: `exp` (σ = e^s) or `softplus` (σ = log(1 + e^s)). */
export type ScaleLink = 'exp' | 'softplus'

/** The shape of a mixture density head. */
export type MixtureHeadOptions = {
  /** The number of components K. */
  components: Size
  /** The target's dimension D (default 1). */
  dims?: Size
  /** How a raw scale becomes a standard deviation (default `exp`). */
  scale?: ScaleLink
  /** A floor added to every standard deviation (default 1e-3; 0 allows a degenerate fit). */
  floor?: number
}

/** The number of outputs of a head with K components over ℝᴰ: K(1 + 2D). */
export const mixtureHeadSize = (components: Size, dims: Size = 1): Size => components * (1 + 2 * dims)

function headShape(outputs: Value, options: MixtureHeadOptions, where: string) {
  const { components: K, dims: D = 1 } = options
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError(where, `${where}: components must be a positive integer`)
  if (!(Number.isInteger(D) && D >= 1)) throw new DomainError(where, `${where}: dims must be a positive integer`)
  const shape = shapeOfValue(outputs)
  if (shape.length !== 2 || shape[1] !== mixtureHeadSize(K, D))
    throw new ShapeError(
      where,
      `${where}: expected outputs [n, ${mixtureHeadSize(K, D)}] for K = ${K}, D = ${D}, got [${shape.join(', ')}]`,
    )
  return { n: shape[0], K, D }
}

/** A head split into its parts (differentiable): log π [n, K], μ [n, K, D] and σ [n, K, D]. */
export type MixtureHeadParts = { logWeights: Value; means: Value; scales: Value }

/** Split raw head outputs [n, K(1 + 2D)] into log π, μ and σ (see the module's layout). */
export function mixtureDensityParams(outputs: Value, options: MixtureHeadOptions): MixtureHeadParts {
  const { n, K, D } = headShape(outputs, options, 'mixtureDensityParams')
  const { scale = 'exp', floor = 1e-3 } = options
  if (!(floor >= 0)) throw new DomainError('mixtureDensityParams', 'mixtureDensityParams: floor must be ≥ 0')
  const logWeights = logSoftmax(slice(outputs, null, [0, K]))
  const means = reshape(slice(outputs, null, [K, K + K * D]), [n, K, D])
  const raw = reshape(slice(outputs, null, [K + K * D, K + 2 * K * D]), [n, K, D])
  const sd = scale === 'exp' ? exp(raw) : softplus(raw)
  return { logWeights, means, scales: floor > 0 ? add(sd, floor) : sd }
}

/** Options of `mixtureDensityNll`. */
export type MixtureDensityNllOptions = MixtureHeadOptions & ReductionOptions

/**
 * The negative log-likelihood of targets y (shape [n] for D = 1, or [n, D]) under the mixture of Gaussians a head
 * predicts for each row: −log Σₖ πₖ Πⱼ N(yⱼ; μₖⱼ, σₖⱼ²), with log πₖ from a log-softmax and the sum over components
 * a log-sum-exp, so neither under- nor overflows. Averaged over rows by default. Its population minimiser is the
 * conditional density p(y | x) itself, multimodal where y given x is.
 */
export const mixtureDensityNll = defineLoss(
  {
    key: 'mixtureDensityNll',
    name: 'Mixture density negative log-likelihood',
    family: 'regression',
    inputs: 'distribution',
    notes: ['regression-losses', 'gaussian-mixture-model'],
    cite: ['bishop2006'],
    target: 'the conditional density p(y | x)',
  },
  (outputs: Value, targets: Target, options: MixtureDensityNllOptions): Value => {
    const { n, D } = headShape(outputs, options, 'mixtureDensityNll')
    const y = constant(targets)
    const count = typeof y === 'number' ? 1 : y.shape.reduce((a, b) => a * b, 1)
    if (count !== n * D)
      throw new ShapeError('mixtureDensityNll', `mixtureDensityNll: expected ${n * D} target values, got ${count}`)
    const { logWeights, means, scales } = mixtureDensityParams(outputs, options)
    const z = div(sub(reshape(y as Tensor, [n, 1, D]), means), scales)
    // log N(y; μ, σ²) summed over the D independent coordinates: [n, K].
    const logNormal = sum(sub(mul(-0.5, square(z)), add(log(scales), HALF_LOG_2PI)), -1)
    return reduce(neg(logsumexp(add(logWeights, logNormal), -1)), options.reduction)
  },
)

// ── Heads as numbers ─────────────────────────────────────────────────────────────────────────────────────────────────

/** One row's mixture: weights πₖ, means μₖ (length D each) and standard deviations σₖ. */
export type MixtureRow = {
  readonly weights: number[]
  readonly means: number[][]
  readonly scales: number[][]
}

/** A mode of a row's density, with the density's value there. */
export type MixtureMode = { readonly value: number[]; readonly density: number }

/** Options of `MixtureDensity.modes`. */
export type MixtureModeOptions = {
  /** Start from the means of components whose weight is at least this (default 1e-3). */
  minWeight?: number
  /** Iterations of the fixed point per start (default 200). */
  maxSteps?: Size
  /** Modes closer than this times the smallest σ are merged (default 0.5). */
  mergeScale?: number
}

/** A batch of mixtures as plain numbers, from `mixtureDensityHead`. */
export type MixtureDensity = {
  readonly rows: Size
  readonly components: Size
  readonly dims: Size
  /** πₖ per row, row-major [n, K]. */
  readonly weights: Float64Array
  /** μₖⱼ per row, row-major [n, K, D]. */
  readonly means: Float64Array
  /** σₖⱼ per row, row-major [n, K, D]. */
  readonly scales: Float64Array
  /** Row i's mixture. */
  row(i: Size): MixtureRow
  /**
   * The batch's law as a compute `Mixture` of `Normal`s: batch shape [n] for D = 1; for D > 1 the marginal law of each
   * coordinate, batch [n, D] (the marginals of a diagonal mixture are mixtures with the same weights; the joint is
   * `logDensity`).
   */
  distribution(): Univariate
  /** log p(y | row i) at a point y (length D). */
  logDensity(i: Size, y: ArrayLike<number>): number
  /** E[y | x] per row, row-major [n, D]: Σₖ πₖ μₖ. */
  mean(): Float64Array
  /** Var(yⱼ | x) per row, row-major [n, D], by the law of total variance. */
  variance(): Float64Array
  /** `count` draws from row i (each of length D). */
  sample(s: Stream, i: Size, count: Size): number[][]
  /** The modes of row i's density, most probable first. */
  modes(i: Size, options?: MixtureModeOptions): MixtureMode[]
}

const flatOf = (v: Value): Float64Array => {
  const r = unwrap(v)
  return typeof r === 'number' ? Float64Array.of(r) : Float64Array.from(toFlat(r))
}

/**
 * Read raw head outputs [n, K(1 + 2D)] as a batch of Gaussian mixtures in plain numbers: for a figure's densities and
 * modes, and for a model's predictive. Modes are found by the fixed point of Carreira-Perpiñán (2000, "Mode-finding for
 * mixtures of Gaussian distributions", IEEE TPAMI 22(11)), x ← (Σₖ p(k | x) Σₖ⁻¹)⁻¹ Σₖ p(k | x) Σₖ⁻¹ μₖ (coordinatewise
 * for diagonal Σₖ), started from every component mean of non-negligible weight; ends closer than `mergeScale` σ_min
 * are merged.
 */
export function mixtureDensityHead(outputs: Value, options: MixtureHeadOptions): MixtureDensity {
  const { n, K, D } = headShape(outputs, options, 'mixtureDensityHead')
  const parts = mixtureDensityParams(outputs, options)
  const weights = flatOf(exp(parts.logWeights))
  const means = flatOf(parts.means)
  const scales = flatOf(parts.scales)
  const row = (i: Size): MixtureRow => ({
    weights: Array.from(weights.subarray(i * K, (i + 1) * K)),
    means: Array.from({ length: K }, (_, k) => Array.from(means.subarray((i * K + k) * D, (i * K + k + 1) * D))),
    scales: Array.from({ length: K }, (_, k) => Array.from(scales.subarray((i * K + k) * D, (i * K + k + 1) * D))),
  })
  // log πₖ + log N(y; μₖ, σₖ²) for every component of row i.
  const joint = (i: Size, y: ArrayLike<number>): number[] =>
    Array.from({ length: K }, (_, k) => {
      let acc = Math.log(weights[i * K + k])
      for (let j = 0; j < D; j++) {
        const o = (i * K + k) * D + j
        const z = (y[j] - means[o]) / scales[o]
        acc -= 0.5 * z * z + Math.log(scales[o]) + HALF_LOG_2PI
      }
      return acc
    })
  const lse = (a: number[]) => {
    const m = Math.max(...a)
    if (m === -Infinity) return -Infinity
    return m + Math.log(a.reduce((acc, v) => acc + Math.exp(v - m), 0))
  }
  const logDensity = (i: Size, y: ArrayLike<number>) => lse(joint(i, y))
  const mean = () => {
    const out = new Float64Array(n * D)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++)
        for (let j = 0; j < D; j++) out[i * D + j] += weights[i * K + k] * means[(i * K + k) * D + j]
    return out
  }
  return {
    rows: n,
    components: K,
    dims: D,
    weights,
    means,
    scales,
    row,
    distribution: () => {
      // Component k's parameters as [n] (D = 1) or [n, D]; for D > 1 the weights [n, 1, K] broadcast over coordinates.
      const component = (v: Float64Array, k: number) =>
        fromData(
          Float64Array.from({ length: n * D }, (_, e) => v[(Math.floor(e / D) * K + k) * D + (e % D)]),
          D === 1 ? [n] : [n, D],
        )
      return Mixture(
        fromData(weights, D === 1 ? [n, K] : [n, 1, K]),
        Array.from({ length: K }, (_, k) => Normal(component(means, k), component(scales, k)) as Univariate),
      )
    },
    logDensity,
    mean,
    variance: () => {
      const m = mean()
      const out = new Float64Array(n * D)
      for (let i = 0; i < n; i++)
        for (let k = 0; k < K; k++)
          for (let j = 0; j < D; j++) {
            const o = (i * K + k) * D + j
            out[i * D + j] += weights[i * K + k] * (scales[o] ** 2 + (means[o] - m[i * D + j]) ** 2)
          }
      return out
    },
    sample: (s, i, count) => {
      const w = weights.subarray(i * K, (i + 1) * K)
      const picks = child(s, 'component')
      const noise = child(s, 'noise')
      return Array.from({ length: count }, () => {
        const k = categorical(picks, w) as number
        return Array.from({ length: D }, (_, j) => {
          const o = (i * K + k) * D + j
          return means[o] + scales[o] * (normal(noise) as number)
        })
      })
    },
    modes: (i, { minWeight = 1e-3, maxSteps = 200, mergeScale = 0.5 } = {}) => {
      const r = row(i)
      const minScale = Math.min(...r.scales.flat())
      const found: MixtureMode[] = []
      for (let start = 0; start < K; start++) {
        if (r.weights[start] < minWeight) continue
        const x = [...r.means[start]]
        for (let step = 0; step < maxSteps; step++) {
          const a = joint(i, x)
          const total = lse(a)
          const num = new Array<number>(D).fill(0)
          const den = new Array<number>(D).fill(0)
          for (let k = 0; k < K; k++) {
            const p = Math.exp(a[k] - total)
            for (let j = 0; j < D; j++) {
              const prec = 1 / r.scales[k][j] ** 2
              num[j] += p * prec * r.means[k][j]
              den[j] += p * prec
            }
          }
          let moved = 0
          for (let j = 0; j < D; j++) {
            const next = num[j] / den[j]
            moved = Math.max(moved, Math.abs(next - x[j]))
            x[j] = next
          }
          if (moved < 1e-9 * (1 + minScale)) break
        }
        const density = Math.exp(logDensity(i, x))
        const near = found.findIndex((m) => Math.hypot(...m.value.map((v, j) => v - x[j])) < mergeScale * minScale)
        if (near < 0) found.push({ value: x, density })
        else if (density > found[near].density) found[near] = { value: x, density }
      }
      return found.sort((a, b) => b.density - a.density)
    },
  }
}
