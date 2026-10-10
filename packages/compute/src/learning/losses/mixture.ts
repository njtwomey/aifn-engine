/**
 * Mixture density heads (Bishop, 1994, "Mixture density networks"; Bishop, 2006, "Pattern Recognition and Machine
 * Learning", §5.6): a network's raw outputs read as the parameters of a mixture of $K$ Gaussians over a target in
 * $\reals^D$ with diagonal covariances, and the mixture's negative log-likelihood as a loss.
 *
 * The head has $K(1 + 2D)$ outputs per row, laid out [logits ($K$) | means ($KD$) | raw scales ($KD$)],
 * component-major (component $k$'s means are columns $K + kD$ to $K + kD + D - 1$). The mixing weights are
 * $\pivec = \operatorname{softmax}(\text{logits})$, the means are the outputs as they are, and the standard deviations
 * are $\sigma = \text{floor} + \exp(s)$ (Bishop's choice) or $\sigma = \text{floor} + \operatorname{softplus}(s)$. The
 * floor keeps the likelihood bounded: without it a component can shrink onto one target and send the log-likelihood
 * to $+\infty$.
 *
 * - `mixtureDensityNll`: $-\log \sum_k \pi_k \prod_j \Gauss(y_j; \mu_{kj}, \sigma_{kj}^2)$ per row, by a stable
 *   log-sum-exp over components; differentiable.
 * - `mixtureDensityParams`: the differentiable split of a head into $\log \pivec$, $\muvec$ and $\sigmavec$.
 * - `mixtureDensityHead`: the same as plain numbers per row, with the density, draws, the moments and the modes, and
 *   the batch's law (its marginals for $D > 1$) as a compute `Mixture` of `Normal`s.
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

/**
 * The link from a raw scale output $s$ to a standard deviation: `exp` ($\sigma = e^s$) or `softplus`
 * ($\sigma = \log(1 + e^s)$), before the floor is added.
 */
export type ScaleLink = 'exp' | 'softplus'

/** The shape of a mixture density head. */
export type MixtureHeadOptions = {
  /** The number of components $K$. */
  components: Size
  /** The target's dimension $D$ (default 1). */
  dims?: Size
  /** How a raw scale becomes a standard deviation (default `exp`). */
  scale?: ScaleLink
  /** A floor added to every standard deviation (default 1e-3; 0 allows a degenerate fit). */
  floor?: number
}

/**
 * The number of outputs of a head with $K$ components over $\reals^D$: $K(1 + 2D)$, the width of the network's last
 * layer.
 *
 * @param components The number of components $K$.
 * @param dims The target's dimension $D$.
 * @returns $K(1 + 2D)$.
 *
 * @example Three components over the plane
 * print('outputs per row:', mixtureHeadSize(3, 2))
 */
export const mixtureHeadSize = (components: Size, dims: Size = 1): Size => components * (1 + 2 * dims)

/**
 * Check a head's shape against its options: `components` and `dims` must be positive integers (else `DomainError`)
 * and the outputs `[n, K(1 + 2D)]` (else `ShapeError`).
 *
 * @param outputs The raw head outputs.
 * @param options The head's shape: `components` and `dims`.
 * @param where The caller's name, for error messages.
 * @returns The number of rows `n`, with `K` and `D`.
 */
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

/**
 * A head split into its parts (differentiable): `logWeights` $\log \pivec$ `[n, K]`, `means` $\muvec$ `[n, K, D]`
 * and `scales` $\sigmavec$ `[n, K, D]`, the standard deviations with the floor added.
 */
export type MixtureHeadParts = { logWeights: Value; means: Value; scales: Value }

/**
 * Split raw head outputs `[n, K(1 + 2D)]` into $\log \pivec$ (a log-softmax of the logits), $\muvec$ and
 * $\sigmavec$ (the link of the raw scales plus the floor), following the layout in the file's comment;
 * differentiable. A shape that does not match the options throws `ShapeError`, a negative floor `DomainError`.
 *
 * @param outputs The raw head outputs, `[n, K(1 + 2D)]`.
 * @param options The head's shape (`components`, `dims`), the scale link and the floor.
 * @returns The parts `logWeights` `[n, K]`, `means` `[n, K, D]` and `scales` `[n, K, D]`.
 *
 * @example One row of a two-component head
 * // Logits 0 and 0, means -2 and 2, raw scales 0 (so sd = e^0 + 0.001).
 * const parts = mixtureDensityParams(tensor([[0, 0, -2, 2, 0, 0]]), { components: 2 })
 * print('weights =', exp(parts.logWeights))
 * print('means =', parts.means)
 * print('scales =', parts.scales)
 */
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
 * The negative log-likelihood of targets $\yvec$ (shape `[n]` for $D = 1$, or `[n, D]`) under the mixture of Gaussians
 * a head predicts for each row: $-\log \sum_k \pi_k \prod_j \Gauss(y_j; \mu_{kj}, \sigma_{kj}^2)$, with
 * $\log \pi_k$ from a log-softmax and the sum over components a log-sum-exp, so neither under- nor overflows.
 * Averaged over rows by default. Its population minimiser is the conditional density $p(\yvec \mid \xvec)$ itself,
 * multimodal where $\yvec$ given $\xvec$ is. A number of targets other than $nD$ throws `ShapeError`.
 *
 * @param outputs The raw head outputs, `[n, K(1 + 2D)]`.
 * @param targets The targets: $n$ values for $D = 1$, or `[n, D]`; constants.
 * @param options The head's shape (`components`, `dims`), the scale link, the floor and the reduction.
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example One component with no floor is the standard normal's negative log-density
 * print('nll =', mixtureDensityNll(tensor([[0, 0, 0]]), [0], { components: 1, floor: 0 }))
 * print('log(2 pi)/2 =', 0.5 * Math.log(2 * Math.PI))
 *
 * @example Two equal components at -2 and 2: a target at either mean costs log 2 more than under one of them
 * const head = tensor([[0, 0, -2, 2, 0, 0]])
 * print('nll =', mixtureDensityNll(head, [2], { components: 2, floor: 0 }))
 * print('about log 2 + log(2 pi)/2 =', Math.log(2) + 0.5 * Math.log(2 * Math.PI))
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

/** One row's mixture as plain arrays. */
export type MixtureRow = {
  /** The weights $\pi_k$, one per component. */
  readonly weights: number[]
  /** The means $\muvec_k$, one array of length $D$ per component. */
  readonly means: number[][]
  /** The standard deviations $\sigmavec_k$, one array of length $D$ per component. */
  readonly scales: number[][]
}

/** A mode of a row's density: `value`, the point (length $D$), and `density`, the density there. */
export type MixtureMode = { readonly value: number[]; readonly density: number }

/** Options of `MixtureDensity.modes`. */
export type MixtureModeOptions = {
  /** Start from the means of components whose weight is at least this (default 1e-3). */
  minWeight?: number
  /** Iterations of the fixed point per start (default 200). */
  maxSteps?: Size
  /** Modes closer than this times the row's smallest $\sigma$ are merged (default 0.5). */
  mergeScale?: number
}

/** A batch of mixtures as plain numbers, from `mixtureDensityHead`. */
export type MixtureDensity = {
  /** The number of rows $n$. */
  readonly rows: Size
  /** The number of components $K$. */
  readonly components: Size
  /** The target's dimension $D$. */
  readonly dims: Size
  /** $\pi_k$ per row, row-major `[n, K]`. */
  readonly weights: Float64Array
  /** $\mu_{kj}$ per row, row-major `[n, K, D]`. */
  readonly means: Float64Array
  /** $\sigma_{kj}$ per row, row-major `[n, K, D]`. */
  readonly scales: Float64Array
  /** Row `i`'s mixture. */
  row(i: Size): MixtureRow
  /**
   * The batch's law as a compute `Mixture` of `Normal`s: batch shape `[n]` for $D = 1$; for $D > 1$ the marginal law of
   * each coordinate, batch `[n, D]` (the marginals of a diagonal mixture are mixtures with the same weights; the joint
   * is `logDensity`).
   */
  distribution(): Univariate
  /** $\log p(\yvec \mid \text{row } i)$ at a point $\yvec$ (length $D$). */
  logDensity(i: Size, y: ArrayLike<number>): number
  /** $\expect[\yvec \mid \xvec]$ per row, row-major `[n, D]`: $\sum_k \pi_k \muvec_k$. */
  mean(): Float64Array
  /** $\var(y_j \mid \xvec)$ per row, row-major `[n, D]`, by the law of total variance. */
  variance(): Float64Array
  /** `count` draws from row `i` (each of length $D$), from the stream `s`. */
  sample(s: Stream, i: Size, count: Size): number[][]
  /** The modes of row `i`'s density, highest density first. */
  modes(i: Size, options?: MixtureModeOptions): MixtureMode[]
}

/**
 * A value's entries as a new flat `Float64Array`, row-major; a traced value is read as its current value.
 *
 * @param v A number, tensor or traced value.
 * @returns Its entries (one for a number).
 */
const flatOf = (v: Value): Float64Array => {
  const r = unwrap(v)
  return typeof r === 'number' ? Float64Array.of(r) : Float64Array.from(toFlat(r))
}

/**
 * Read raw head outputs `[n, K(1 + 2D)]` as a batch of Gaussian mixtures in plain numbers: for a figure's densities
 * and modes, and for a model's predictive. Not differentiable (traced outputs are read as values). Modes are found by
 * the fixed point of Carreira-Perpiñán (2000, "Mode-finding for mixtures of Gaussian distributions", IEEE TPAMI
 * 22(11)),
 * $\xvec \leftarrow (\sum_k p(k \mid \xvec) \Sigmamat_k^{-1})^{-1} \sum_k p(k \mid \xvec) \Sigmamat_k^{-1} \muvec_k$
 * (coordinatewise for diagonal $\Sigmamat_k$), started from every component mean of non-negligible weight; ends
 * closer than `mergeScale` times $\sigma_{\min}$ are merged, keeping the one of higher density.
 *
 * @param outputs The raw head outputs, `[n, K(1 + 2D)]`.
 * @param options The head's shape (`components`, `dims`), the scale link and the floor, as for `mixtureDensityNll`.
 * @returns The mixtures, with their density, moments, draws, modes and law.
 *
 * @example A two-component mixture: its mean, variance and two modes
 * const head = mixtureDensityHead(tensor([[0, 0, -2, 2, 0, 0]]), { components: 2, floor: 0 })
 * print('mean =', head.mean(), ' variance =', head.variance(), ' (1 + 4)')
 * print('modes =', head.modes(0))
 * print('density at 0 =', Math.exp(head.logDensity(0, [0])), ' N(0; 2, 1) =', Math.exp(-2) / Math.sqrt(2 * Math.PI))
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
