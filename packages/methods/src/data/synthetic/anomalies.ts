/**
 * Two-dimensional data with planted anomalies, for comparing anomaly detectors: inliers from one of four shapes (two
 * clusters of different density, two moons, a ring, an elongated line) and anomalies drawn uniformly over a box around
 * them, kept only where they are clearly away from the inliers. Labels mark the anomalies (1) against the inliers (0).
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { checkCount, labels, matrix, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The inlier shapes of `plantedAnomalies`. */
export const ANOMALY_SHAPES = ['clusters', 'moons', 'ring', 'line'] as const
/** One of the inlier shapes in `ANOMALY_SHAPES`. */
export type AnomalyShape = (typeof ANOMALY_SHAPES)[number]

/** Options of `plantedAnomalies`. */
export interface PlantedAnomaliesOptions {
  /** Points in total (default 300). */
  n?: number
  /** Share of anomalies, in $[0, 1)$ (default 0.05); their number is rounded to the nearest whole. */
  contamination?: number
  /** The inlier shape (default `clusters`). */
  shape?: AnomalyShape
  /**
   * Standard deviation of the Gaussian noise on the inlier shape (default 0.1, relative to its unit size; the two
   * clusters take 1.2 and 3.5 times it, and the ring is noisy in radius only).
   */
  noise?: number
  /** Anomalies are kept only at least this far from every inlier (default 0.35). */
  margin?: number
}

/**
 * One inlier point of a shape: `clusters` is a tight cluster at $(-1.2, 0.4)$ and a wide one at $(1, -0.3)$, half the
 * points each; `moons` two interleaved half-circles of radius 1; `ring` a circle of radius 1.2; `line` the segment from
 * $(-1.6, -1)$ to $(1.6, 1)$.
 *
 * @param shape The inlier shape.
 * @param s The point's own stream, with children `'u'` (the position along the shape, or which cluster or moon),
 *   `'e1'` and `'e2'` (the noise) and `'t'` (the angle on a moon).
 * @param noise The standard deviation of the noise (see `PlantedAnomaliesOptions`).
 * @returns The point's two coordinates.
 */
function inlier(shape: AnomalyShape, s: Stream, noise: number): [number, number] {
  const u = uniform(child(s, 'u'))
  const e1 = normal(child(s, 'e1'))
  const e2 = normal(child(s, 'e2'))
  switch (shape) {
    case 'clusters': {
      // A tight cluster and a wide one, three times as spread: a single global density threshold fits neither.
      if (u < 0.5) return [-1.2 + 1.2 * noise * e1, 0.4 + 1.2 * noise * e2]
      return [1.0 + 3.5 * noise * e1, -0.3 + 3.5 * noise * e2]
    }
    case 'moons': {
      const t = Math.PI * uniform(child(s, 't'))
      return u < 0.5
        ? [Math.cos(t) - 0.5 + noise * e1, Math.sin(t) - 0.25 + noise * e2]
        : [0.5 - Math.cos(t) + noise * e1, 0.25 - Math.sin(t) + noise * e2]
    }
    case 'ring': {
      const t = 2 * Math.PI * u
      const r = 1.2 + noise * e1
      return [r * Math.cos(t), r * Math.sin(t)]
    }
    case 'line': {
      const t = 4 * u - 2
      return [t * 0.8 + noise * e1, t * 0.5 + noise * e2]
    }
  }
}

/**
 * $n$ points in the plane: $n - m$ inliers from `shape` and $m$ anomalies, $m$ the nearest whole number to $cn$ ($c$
 * the `contamination`). Anomalies are uniform over the box $[-2.5, 2.5]^2$ and redrawn while closer than `margin` to an
 * inlier, at most 201 times (the last draw is then kept wherever it falls); they are not kept apart from each other.
 * The inliers come first in the rows, then the anomalies, and `y` is 1 for anomalies. Throws `DomainError` when $n$ is
 * not a non-negative integer or `contamination` is not in $[0, 1)$.
 *
 * @param s The stream the points are drawn from: inlier $i$ from `child(s, 'inlier', i)`, and anomaly $j$'s attempts
 *   from `child(s, 'anomaly', j, ...)`.
 * @param options The size, the share of anomalies, the inlier shape and noise, and the margin.
 * @returns A classification dataset: `x` ($n \times 2$) and `y` (0 inlier, 1 anomaly).
 *
 * @example Anomalies are kept clear of the inliers
 * const d = plantedAnomalies(stream(1), { n: 200, contamination: 0.1, shape: 'ring' })
 * const x = toArray(d.x)
 * const y = toArray(d.y)
 * print('x:', d.x.shape, ' anomalies:', y.filter((v) => v === 1).length)
 * print('first rows:', x.slice(0, 2), ' last row:', x[199])
 * const inliers = x.filter((_, i) => y[i] === 0)
 * const gap = (p) => Math.min(...inliers.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])))
 * print('closest anomaly to an inlier:', Math.min(...x.filter((_, i) => y[i] === 1).map(gap)))
 */
export function plantedAnomalies(s: Stream, options: PlantedAnomaliesOptions = {}): Dataset {
  const { n = 300, contamination = 0.05, shape = 'clusters', noise = 0.1, margin = 0.35 } = options
  checkCount(n, 'plantedAnomalies')
  if (!(contamination >= 0 && contamination < 1))
    throw new DomainError('plantedAnomalies', 'plantedAnomalies: contamination in [0, 1)')
  const anomalies = Math.round(n * contamination)
  const inliers = n - anomalies
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  for (let i = 0; i < inliers; i++) {
    const [a, b] = inlier(shape, child(s, 'inlier', i), noise)
    x[2 * i] = a
    x[2 * i + 1] = b
  }
  for (let j = 0; j < anomalies; j++) {
    const i = inliers + j
    for (let t = 0; ; t++) {
      const a = -2.5 + 5 * uniform(child(s, 'anomaly', j, t, 0))
      const b = -2.5 + 5 * uniform(child(s, 'anomaly', j, t, 1))
      let near = Infinity
      for (let r = 0; r < inliers; r++) near = Math.min(near, Math.hypot(a - x[2 * r], b - x[2 * r + 1]))
      if (near >= margin || t > 200) {
        x[2 * i] = a
        x[2 * i + 1] = b
        break
      }
    }
    y[i] = 1
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name: 'planted anomalies',
      description: `${inliers} inliers (${shape}, noise ${noise}) and ${anomalies} uniform anomalies at least ${margin} from them.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'plantedAnomalies',
    name: 'Planted anomalies',
    summary: 'Inliers from clusters, moons, a ring or a line, with uniform anomalies kept clear of them.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      contamination: real(0, 0.5, { default: 0.05 }),
      shape: oneOf([...ANOMALY_SHAPES]),
      noise: real(0, 1, { default: 0.1 }),
      margin: real(0, 2, { default: 0.35 }),
    }),
    truth: false,
    random: true,
    notes: ['anomaly-detection', 'evaluating-anomaly-detectors'],
  },
  plantedAnomalies,
)
