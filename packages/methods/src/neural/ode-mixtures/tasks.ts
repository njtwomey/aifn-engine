/**
 * Run data from datasets (passed in by structure, since the data area sits above this one): labelled points, 1-d
 * start–target pairs (the failure cases of fig. 1) and sampled walks with the hour of day (the behaviour task, §4.4).
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { SvfmRunData } from './run'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The fields of a dataset a run reads: the rows `x`, and the labels or targets `y`. */
export type DatasetLike = { x: Tensor; y?: Tensor }

/**
 * Labelled points as a classification run's data: the points are the starts, the labels the classes.
 *
 * @param d The points `x` $[n, D]$ and their int32 labels `y` $[n]$.
 * @returns The run data, the same tensors.
 *
 * @example Two points and their labels
 * const d = classificationTask({ x: tensor([[0, 1], [1, 0]]), y: fromData(Int32Array.of(0, 1)) })
 * print('starts:', d.x, ' labels:', d.y)
 */
export function classificationTask(d: DatasetLike): SvfmRunData {
  return { x: d.x, y: d.y }
}

/**
 * Start–target pairs as an endpoint run's data (the failure cases of fig. 1), coloured by the side of 0 the start lies
 * on when starts lie on both sides (crossing), else by the side of the target when targets do (splitting), else in one
 * group (scaling).
 *
 * @param d The starts `x` $[n, 1]$ and the end targets `y` $[n]$.
 * @returns The run data: the starts, the targets as $[n, 1]$ and the groups $[n]$ (0 for a negative start or target,
 *   1 otherwise).
 *
 * @example Crossing, splitting and scaling
 * for (const [x, y] of [[[-1, 1], [1, -1]], [[0.1, 0.2], [-1, 1]], [[1, 2], [2, 4]]]) {
 *   const d = endpointTask({ x: tensor(x.map((v) => [v])), y: tensor(y) })
 *   print('starts', x, 'targets', y, '-> groups', d.groups)
 * }
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
  /**
   * Feed the time of day to every network as $(\cos(2\pi h / 24), \sin(2\pi h / 24))$, $h$ the hour (the paper's
   * cyclic time, §4.2). Default false.
   */
  timeOfDay?: boolean
}

/**
 * Walks as a forecasting run's data: paths $[n, M, 2]$ at regular times on $[0, 1]$, starts at their first sample,
 * groups the target labels, and the time-of-day context when asked for. Throws `ShapeError` unless the rows hold at
 * least two points and an hour.
 *
 * @param d The walks `x`, one per row as $x_0, y_0, \dots, x_{M-1}, y_{M-1}$ and the hour (as `floorplanWalks` stores
 *   them), and their labels `y` $[n]$.
 * @param options Whether to add the time-of-day context.
 * @returns The run data: starts, paths, path times, context (or undefined) and groups.
 *
 * @example Two walks of three points, at 6:00 and 18:00
 * const rows = tensor([[0, 0, 1, 0, 2, 1, 6], [5, 5, 4, 4, 3, 3, 18]])
 * const d = walkTask({ x: rows, y: fromData(Int32Array.of(0, 1)) }, { timeOfDay: true })
 * print('starts:', d.x, ' path times:', d.pathTimes)
 * print('paths:', d.paths)
 * print('time of day (cos, sin):', d.context)
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
