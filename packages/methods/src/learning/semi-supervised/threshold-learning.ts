/**
 * Supervised and semi-supervised learning of a threshold on the real line, the smallest setting in which the value of
 * unlabelled data can be computed exactly (Balcan and Blum, 2010, "A discriminative model for semi-supervised
 * learning", JACM 57(3), §2; Ben-David, Lu and Pál, 2008, "Does unlabeled data provably help?", COLT, §6.2).
 *
 * The class is $h_t(x) = \indicator[x > t]$ (class 1 to the right). The supervised learner is empirical risk
 * minimisation: among the thresholds with the fewest training errors it takes the first interval $(\ell, r)$ of such
 * thresholds, bounded by consecutive labelled points (or the domain), and returns its midpoint (`thresholdErm`). The
 * semi-supervised learner adds a compatibility notion in the sense of Balcan and Blum: a boundary should cross the
 * input density at a trough. It returns, among the troughs of a Gaussian kernel density estimate of the unlabelled
 * inputs, the one with the fewest training errors, the deepest on a tie (`lowDensityThreshold`). With no unlabelled
 * inputs, or no trough, it returns the supervised midpoint, so the two learners agree. Kääriäinen's learner of
 * Ben-David et al., which takes the median of the interval under the known input distribution, is `medianThreshold`;
 * the low-density rule replaces it to show a compatibility assumption that can be right or wrong.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { kde, type BandwidthRule } from 'aifn-compute/probability/stats'

/** A labelled sample on the line: inputs and classes 0 or 1. */
export interface LineSample {
  /** The inputs. */
  x: ArrayLike<number>
  /** The class of each input, 0 or 1. */
  y: ArrayLike<number>
}

/** The interval of empirical risk minimisers and its training error count. */
export interface ThresholdInterval {
  /** Left end $\ell$ of the first interval of thresholds with the fewest training errors. */
  lo: number
  /** Right end $r$. */
  hi: number
  /** Number of training errors of every threshold in $(\ell, r)$. */
  errors: number
}

/**
 * The first interval of empirical risk minimisers of $h_t(x) = \indicator[x > t]$, clipped to the domain. Throws
 * `DomainError` for a class other than 0 or 1.
 *
 * @param sample The labelled points; classes must be 0 or 1.
 * @param domain The input range $[a, b]$ that bounds an interval open to one side.
 * @returns The interval $(\ell, r)$ and its error count.
 *
 * @example Separable points, then one reversed pair
 * print(ermInterval({ x: [0.6, 0.1, 0.9, 0.3], y: [1, 0, 1, 0] }, [0, 1]))
 * print(ermInterval({ x: [0.2, 0.7], y: [1, 0] }, [0, 1]))
 */
export function ermInterval(sample: LineSample, domain: readonly [number, number]): ThresholdInterval {
  const n = sample.x.length
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => sample.x[a] - sample.x[b] || a - b)
  // Threshold below every point: every class-0 point is an error.
  let errors = 0
  for (let i = 0; i < n; i++) {
    const c = sample.y[i]
    if (c !== 0 && c !== 1) throw new DomainError('ermInterval', 'ermInterval: classes must be 0 or 1')
    if (c === 0) errors++
  }
  let best = errors
  let at = 0 // the gap index: 0 is below the first sorted point, n above the last
  for (let g = 1; g <= n; g++) {
    // Moving the threshold past sorted point g - 1 corrects it if class 0, breaks it if class 1.
    errors += sample.y[order[g - 1]] === 0 ? -1 : 1
    if (errors < best) {
      best = errors
      at = g
    }
  }
  const lo = at === 0 ? domain[0] : Math.max(domain[0], sample.x[order[at - 1]])
  const hi = at === n ? domain[1] : Math.min(domain[1], sample.x[order[at]])
  return { lo, hi, errors: best }
}

/**
 * The supervised learner: the midpoint of the first interval of empirical risk minimisers.
 *
 * @param sample The labelled points; classes must be 0 or 1.
 * @param domain The input range bounding an interval open to one side.
 * @returns The threshold $t = (\ell + r) / 2$.
 *
 * @example The midpoint of the gap between the classes
 * print(thresholdErm({ x: [0.6, 0.1, 0.9, 0.3], y: [1, 0, 1, 0] }, [0, 1]))
 */
export function thresholdErm(sample: LineSample, domain: readonly [number, number]): number {
  const { lo, hi } = ermInterval(sample, domain)
  return (lo + hi) / 2
}

/** A density estimate of the unlabelled inputs on an increasing grid. */
export interface GridDensity {
  /** The grid points, increasing. */
  x: number[]
  /** The estimated density at each grid point. */
  f: number[]
}

/**
 * A Gaussian kernel density estimate of the unlabelled inputs on an even grid over the domain, by `kde` of
 * `aifn-compute/probability/stats`. Returns null for fewer than two inputs or constant inputs.
 *
 * @param unlabelled The unlabelled inputs.
 * @param domain The grid's range.
 * @param options The number of grid points and the bandwidth rule.
 * @param options.points The number of grid points (default 200).
 * @param options.bandwidth The bandwidth rule of `kde` (default `'silverman'`).
 * @returns The grid and the density on it, or null.
 *
 * @example Two clusters: the density has a trough between them
 * const u = [0.15, 0.2, 0.25, 0.75, 0.8, 0.85]
 * const g = unlabelledDensity(u, [0, 1], { points: 5 })
 * print('grid:', g.x)
 * print('density:', g.f)
 */
export function unlabelledDensity(
  unlabelled: ArrayLike<number>,
  domain: readonly [number, number],
  options: { points?: number; bandwidth?: BandwidthRule } = {},
): GridDensity | null {
  if (unlabelled.length < 2) return null
  const G = options.points ?? 200
  const x = Array.from({ length: G }, (_, j) => domain[0] + (j * (domain[1] - domain[0])) / (G - 1))
  const { density, degenerate } = kde(Array.from(unlabelled), x, { bandwidth: options.bandwidth ?? 'silverman' })
  return degenerate ? null : { x, f: toFlat(density) }
}

/**
 * Training errors of $h_t(x) = \indicator[x > t]$ on a labelled sample.
 *
 * @param sample The labelled points.
 * @param t The threshold.
 * @returns The number of points on the wrong side of $t$.
 *
 * @example One point on the wrong side
 * print(thresholdErrors({ x: [0.2, 0.7], y: [1, 0] }, 0.5))
 */
export function thresholdErrors(sample: LineSample, t: number): number {
  let errors = 0
  for (let i = 0; i < sample.x.length; i++) if ((sample.x[i] > t ? 1 : 0) !== sample.y[i]) errors++
  return errors
}

/**
 * The semi-supervised learner: compatibility first, then the labels. The compatible thresholds are the troughs of the
 * unlabelled density, grid points strictly inside the grid whose density is no higher than at their two neighbours.
 * Among them the learner takes the one with the fewest training errors, and among those the lowest density: the
 * learner of Balcan and Blum's Theorem 4, empirical risk minimisation over the compatible class, with "the boundary
 * lies in a trough" as the compatibility notion. Without a density, or with no trough, it returns the supervised
 * midpoint.
 *
 * @param sample The labelled points.
 * @param density The unlabelled density on a grid (`unlabelledDensity`), or null.
 * @param domain The input range bounding an open interval.
 * @returns The threshold.
 *
 * @example Two labels far apart: the trough of the unlabelled clusters decides
 * const un = Array.from({ length: 100 }, (_, i) => (i % 2 ? 0.8 : 0.2) + (Math.floor(i / 2) - 25) * 0.002)
 * const sample = { x: [0.1, 0.9], y: [0, 1] }
 * print('low density:', lowDensityThreshold(sample, unlabelledDensity(un, [0, 1]), [0, 1]))
 * print('supervised:', thresholdErm(sample, [0, 1]))
 */
export function lowDensityThreshold(
  sample: LineSample,
  density: GridDensity | null,
  domain: readonly [number, number],
): number {
  if (!density) return thresholdErm(sample, domain)
  const { x, f } = density
  let best = -1
  let bestErrors = Infinity
  for (let j = 1; j < x.length - 1; j++) {
    if (!(f[j] <= f[j - 1] && f[j] <= f[j + 1])) continue
    const errors = thresholdErrors(sample, x[j])
    if (errors < bestErrors || (errors === bestErrors && f[j] < f[best])) {
      best = j
      bestErrors = errors
    }
  }
  return best < 0 ? thresholdErm(sample, domain) : x[best]
}

/**
 * Kääriäinen's semi-supervised learner for a known input distribution (Ben-David et al., 2008, §6.2): the median of the
 * interval of empirical risk minimisers under the input distribution function $F$, $t = F^{-1}((F(\ell) + F(r))/2)$.
 *
 * @param sample The labelled points.
 * @param cdf The input distribution function $F$.
 * @param quantile Its inverse $F^{-1}$.
 * @param domain The input range bounding an open interval.
 * @returns The threshold.
 *
 * @example Under a uniform input distribution the median is the midpoint
 * print(medianThreshold({ x: [0.6, 0.1, 0.9, 0.3], y: [1, 0, 1, 0] }, (x) => x, (p) => p, [0, 1]))
 */
export function medianThreshold(
  sample: LineSample,
  cdf: (x: number) => number,
  quantile: (p: number) => number,
  domain: readonly [number, number],
): number {
  const { lo, hi } = ermInterval(sample, domain)
  return quantile((cdf(lo) + cdf(hi)) / 2)
}
