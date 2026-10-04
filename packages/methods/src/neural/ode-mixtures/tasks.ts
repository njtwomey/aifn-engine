/**
 * Run data from datasets (passed in by structure, since the data area sits above this one): labelled points, 1-d
 * start–target pairs (the failure cases of fig. 1) and sampled walks with the hour of day (the behaviour task, §4.4).
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { SvfmRunData } from './run'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The fields of a dataset a run reads. */
export type DatasetLike = { x: Tensor; y?: Tensor }

/** Labelled points: starts x, class labels y. */
export function classificationTask(d: DatasetLike): SvfmRunData {
  return { x: d.x, y: d.y }
}

/**
 * Start–target pairs (x [n, 1], y [n]); colours by the side of the start when starts lie on both sides of 0 (crossing),
 * else by the side of the target when targets do (splitting), else one group (scaling).
 */
export function endpointTask(d: DatasetLike): SvfmRunData {
  const x = Float64Array.from(toFlat(d.x))
  const y = Float64Array.from(toFlat(d.y!))
  const both = (a: Float64Array) => a.some((v) => v < 0) && a.some((v) => v > 0)
  const by = both(x) ? x : both(y) ? y : null
  const groups = Int32Array.from(x, (_, i) => (by ? (by[i] < 0 ? 0 : 1) : 0))
  return { x: d.x, y: fromData(y, [y.length, 1]), groups: fromData(groups) }
}

/** Options of {@link walkTask}. */
export type WalkTaskOptions = {
  /** Feed the time of day to every network as t = (cos 2πh/24, sin 2πh/24) (the paper's cyclic time, §4.2). */
  timeOfDay?: boolean
}

/**
 * Walks stored as rows x₀, y₀, …, x_{M−1}, y_{M−1}, hour (`floorplanWalks`): paths [n, M, 2] at regular times on
 * [0, 1], starts at their first sample, groups = the target labels, and the time-of-day context when asked for.
 */
export function walkTask(d: DatasetLike, options: WalkTaskOptions = {}): SvfmRunData {
  const [n, cols] = d.x.shape
  const M = (cols - 1) / 2
  if (!Number.isInteger(M) || M < 2) throw new ShapeError('walkTask', 'walkTask: expected rows x₀, y₀, …, hour')
  const a = toFlat(d.x)
  const paths = new Float64Array(n * M * 2)
  const starts = new Float64Array(n * 2)
  const context = new Float64Array(n * 2)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 2 * M; k++) paths[i * 2 * M + k] = a[i * cols + k]
    starts[2 * i] = a[i * cols]
    starts[2 * i + 1] = a[i * cols + 1]
    const hour = a[i * cols + 2 * M]
    context[2 * i] = Math.cos((2 * Math.PI * hour) / 24)
    context[2 * i + 1] = Math.sin((2 * Math.PI * hour) / 24)
  }
  return {
    x: fromData(starts, [n, 2]),
    paths: fromData(paths, [n, M, 2]),
    pathTimes: Array.from({ length: M }, (_, j) => j / (M - 1)),
    context: options.timeOfDay ? fromData(context, [n, 2]) : undefined,
    groups: d.y,
  }
}
