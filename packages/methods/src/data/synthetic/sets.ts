/**
 * Sets for permutation-invariant networks: `numberSets` draws sets of real numbers with three targets per set (the sum,
 * the maximum and the variance), the regression tasks of Deep Sets (Zaheer et al. 2017, "Deep Sets", NeurIPS);
 * `shapeClouds` draws noisy 2-d point clouds sampled from the outlines of circles, squares and triangles, the
 * classification task of PointNet in miniature (Qi et al. 2017, "PointNet", CVPR). Pad them with `padSets` and pool
 * them with `setPool` (`aifn-compute/nn/sets`).
 *
 * Every element is drawn from its own child stream, so a set does not change when more sets are drawn.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'

/** A target of `numberSets`: the sum, the maximum or the population variance of a set. */
export type SetTarget = 'sum' | 'max' | 'variance'

/** Options of `numberSets`. */
export interface NumberSetsOptions {
  /** The number of sets. */
  count: number
  /** The smallest set size, at least 1. */
  minSize: number
  /** The largest set size, at least `minSize`. */
  maxSize: number
  /** The upper end of the uniform range $[0, \mathrm{high}]$ of the numbers (default 1). */
  high?: number
}

/**
 * Sets of numbers drawn uniformly on $[0, \mathrm{high}]$, with sizes uniform on `minSize` to `maxSize`, and three
 * targets per set: the sum, the maximum and the population variance $\frac{1}{n}\sum_k (x_k - \bar x)^2$. Throws
 * `DomainError` unless $1 \le$ `minSize` $\le$ `maxSize` are integers.
 *
 * @param s The random stream; set $b$ draws from its child $b$: its size from the child `'size'`, element $k$ from the
 *   child $k$.
 * @param options The number of sets, the size range and the range of the numbers.
 * @returns The sets and their targets.
 *
 * @example Three sets and their targets
 * const d = numberSets(stream(0), { count: 3, minSize: 2, maxSize: 4 })
 * print('sets:', d.sets)
 * print('sums:', d.targets.sum, ' maxima:', d.targets.max)
 */
export function numberSets(
  s: Stream,
  options: NumberSetsOptions,
): { sets: number[][]; targets: Record<SetTarget, number[]> } {
  const { count, minSize, maxSize, high = 1 } = options
  if (!(Number.isInteger(minSize) && Number.isInteger(maxSize) && minSize >= 1 && maxSize >= minSize))
    throw new DomainError('numberSets', 'numberSets: need integer sizes with 1 ≤ minSize ≤ maxSize')
  const sets = Array.from({ length: count }, (_, b) => {
    const e = child(s, b)
    const n = minSize + Math.floor(uniform(child(e, 'size')) * (maxSize - minSize + 1))
    return Array.from({ length: n }, (_, k) => high * uniform(child(e, k)))
  })
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
  return {
    sets,
    targets: {
      sum: sets.map((v) => v.reduce((a, b) => a + b, 0)),
      max: sets.map((v) => Math.max(...v)),
      variance: sets.map((v) => {
        const m = mean(v)
        return mean(v.map((x) => (x - m) ** 2))
      }),
    },
  }
}

/** The shapes of `shapeClouds`, in label order. */
export const CLOUD_SHAPES = ['circle', 'square', 'triangle'] as const

/** A shape of `shapeClouds`. */
export type CloudShape = (typeof CLOUD_SHAPES)[number]

/**
 * The point at fraction $t$ of the way round the outline of a shape of circumradius 1 centred at the origin.
 *
 * @param shape The shape.
 * @param t The fraction, in $[0, 1)$.
 * @returns The point.
 */
function outlinePoint(shape: CloudShape, t: number): [number, number] {
  if (shape === 'circle') return [Math.cos(2 * Math.PI * t), Math.sin(2 * Math.PI * t)]
  const sides = shape === 'square' ? 4 : 3
  const side = Math.min(sides - 1, Math.floor(t * sides))
  const u = t * sides - side
  const corner = (k: number): [number, number] => {
    const a = Math.PI / 2 + (2 * Math.PI * k) / sides + (shape === 'square' ? Math.PI / 4 : 0)
    return [Math.cos(a), Math.sin(a)]
  }
  const [a, b] = [corner(side), corner(side + 1)]
  return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])]
}

/** Options of `shapeClouds`. */
export interface ShapeCloudsOptions {
  /** The number of clouds. */
  count: number
  /** The points per cloud. */
  points: number
  /** The half-width of the uniform jitter of every point (default 0.05). */
  noise?: number
  /** Rotate each shape by a uniform angle (default true). */
  rotate?: boolean
}

/**
 * Noisy 2-d point clouds of circles, squares and triangles: `points` points uniformly along the outline, the shape
 * rotated by a uniform angle (unless `rotate` is false), scaled by a factor uniform on $[0.8, 1.2]$, and each point
 * jittered by uniform noise of half-width `noise`. Labels index `CLOUD_SHAPES`, cycling so that the classes are
 * balanced.
 *
 * @param s The random stream; cloud $b$ draws from its child $b$ (children `'angle'`, `'scale'`, and `'where'`,
 *   `'jx'`, `'jy'` with the point's index).
 * @param options The number of clouds, the points per cloud, the noise and whether to rotate.
 * @returns The clouds, each a list of points, and their labels.
 *
 * @example A noise-free circle has one radius
 * const { clouds, labels } = shapeClouds(stream(0), { count: 3, points: 8, noise: 0 })
 * const radii = clouds[0].map(([x, y]) => Math.hypot(x, y))
 * print('labels:', labels, ' radii of the circle:', Math.min(...radii), Math.max(...radii))
 */
export function shapeClouds(
  s: Stream,
  options: ShapeCloudsOptions,
): { clouds: [number, number][][]; labels: number[] } {
  const noise = options.noise ?? 0.05
  const clouds = Array.from({ length: options.count }, (_, b) => {
    const e = child(s, b)
    const shape = CLOUD_SHAPES[b % CLOUD_SHAPES.length]
    const angle = options.rotate === false ? 0 : 2 * Math.PI * uniform(child(e, 'angle'))
    const scale = 0.8 + 0.4 * uniform(child(e, 'scale'))
    const [c, sn] = [Math.cos(angle), Math.sin(angle)]
    return Array.from({ length: options.points }, (_, k) => {
      const [px, py] = outlinePoint(shape, uniform(child(e, 'where', k)))
      const jx = noise * (2 * uniform(child(e, 'jx', k)) - 1)
      const jy = noise * (2 * uniform(child(e, 'jy', k)) - 1)
      return [scale * (c * px - sn * py) + jx, scale * (sn * px + c * py) + jy] as [number, number]
    })
  })
  return { clouds, labels: clouds.map((_, b) => b % CLOUD_SHAPES.length) }
}
