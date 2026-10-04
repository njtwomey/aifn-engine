/**
 * The quality of a learned embedding on the unit hypersphere (Wang & Isola, 2020, "Understanding Contrastive
 * Representation Learning through Alignment and Uniformity on the Hypersphere", ICML): alignment, how close the two
 * embeddings of a positive pair are, and uniformity, how evenly the embeddings spread over the sphere. Contrastive
 * losses such as InfoNCE optimise both at once; each is a number of the embeddings, so they can be tracked during
 * training and compared across settings.
 */

import { defineMetric, dense, nonEmpty } from './core'
import type { Rows } from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `alignment`. */
export type AlignmentOptions = {
  /** The power α of the distance (default 2). */
  alpha?: number
}

/**
 * Alignment E‖f(x) − f(y)‖^α over positive pairs: row i of `x` [n, d] and row i of `y` [n, d] are the two embeddings of
 * pair i (Wang & Isola, 2020, eq. 4; α = 2 by default). Lower is better; for unit vectors and α = 2 it lies in [0, 4].
 * The embeddings are used as given: normalise them first to measure them on the sphere.
 */
export const alignment = defineMetric(
  {
    key: 'alignment',
    name: 'Alignment (positive-pair distance)',
    inputs: 'vectors',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['contrastive-learning'],
  },
  (x: Rows, y: Rows, { alpha = 2 }: AlignmentOptions = {}): number => {
    const a = dense(x, 'alignment x')
    const b = dense(y, 'alignment y')
    if (a.rows !== b.rows || a.cols !== b.cols)
      throw new ShapeError('metrics', `metrics: alignment: x is ${a.rows} × ${a.cols} but y is ${b.rows} × ${b.cols}`)
    nonEmpty(a.rows, 'alignment')
    let total = 0
    for (let i = 0; i < a.rows; i++) {
      let d2 = 0
      for (let k = 0; k < a.cols; k++) {
        const e = a.data[i * a.cols + k] - b.data[i * a.cols + k]
        d2 += e * e
      }
      total += d2 ** (alpha / 2)
    }
    return total / a.rows
  },
)

/** Options of `uniformity`. */
export type UniformityOptions = {
  /** The scale t of the Gaussian potential (default 2). */
  t?: number
}

/**
 * Uniformity log E exp(−t‖f(x) − f(x′)‖²) over distinct pairs of the rows of `x` [n, d] (Wang & Isola, 2020, eq. 5;
 * t = 2 by default): the log of the average Gaussian potential between embeddings. Lower is better (more uniform); for
 * unit vectors it is at least −4t, approached as the points spread out, and 0 when they all coincide. Computed with a
 * log-sum-exp, so it stays finite for widely spread points.
 */
export const uniformity = defineMetric(
  {
    key: 'uniformity',
    name: 'Uniformity (log mean Gaussian potential)',
    inputs: 'vectors',
    direction: 'lower',
    range: [-Infinity, 0],
    notes: ['contrastive-learning'],
  },
  (x: Rows, { t = 2 }: UniformityOptions = {}): number => {
    const a = dense(x, 'uniformity x')
    if (a.rows < 2) throw new DomainError('metrics', 'metrics: uniformity needs at least two rows')
    const logs: number[] = []
    for (let i = 0; i < a.rows; i++)
      for (let j = i + 1; j < a.rows; j++) {
        let d2 = 0
        for (let k = 0; k < a.cols; k++) {
          const e = a.data[i * a.cols + k] - a.data[j * a.cols + k]
          d2 += e * e
        }
        logs.push(-t * d2)
      }
    let m = -Infinity
    for (const v of logs) if (v > m) m = v
    let s = 0
    for (const v of logs) s += Math.exp(v - m)
    return m + Math.log(s / logs.length)
  },
)
