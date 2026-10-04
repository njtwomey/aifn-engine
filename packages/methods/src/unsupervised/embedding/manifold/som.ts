/**
 * Self-organising maps (Kohonen, 1982; 2013): a grid of units, each with a weight vector in data space, fitted so that
 * neighbouring units model neighbouring data. This is the batch SOM (Kohonen, 2013, §4): each step assigns every row to
 * its best-matching unit (BMU), then sets every unit's weight to the neighbourhood-weighted mean of the rows,
 * w_u = Σᵢ h(bᵢ, u) xᵢ / Σᵢ h(bᵢ, u) with h(b, u) = exp(−‖g_b − g_u‖² / 2σ²) on grid positions g, while the radius σ
 * shrinks geometrically from `radius` to `finalRadius` over the `epochs` steps.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { integers, normals } from 'aifn-compute/foundation/random'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Dataset, Estimator, FitOptions, Trained, Transforms } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { eigh } from 'aifn-compute/numerics/linalg'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { mat, matrix, values } from '../util'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Grid and schedule of a SOM. */
export type SomParams = {
  /** Grid rows (default 8) and columns (default 8); units are numbered row by row. */
  rows?: number
  cols?: number
  /** Batch steps (default 30). */
  epochs?: number
  /** Initial neighbourhood radius σ₀ in grid units (default max(rows, cols) / 2). */
  radius?: number
  /** Final radius (default 0.5). */
  finalRadius?: number
  /** Initial weights: the plane of the top two principal axes (default) or random rows plus noise. */
  start?: 'pca' | 'random'
}

/** A state of `selfOrganisingMapSteps`. */
export interface SomState extends Status {
  t: number
  /** Unit weights [rows·cols, d]. */
  weights: Tensor
  /** The radius used in the step that produced this state (σ₀ at the start). */
  radius: number
  /** Mean distance from each row to its BMU's weight. */
  quantisationError: number
  /** Share of rows whose two best-matching units are not grid neighbours (8-neighbourhood). */
  topographicError: number
  /** Each row's BMU. */
  assignment: Int32Array
}

function bestTwo(v: Float64Array, i: number, W: Float64Array, units: number, d: number) {
  let b1 = -1
  let b2 = -1
  let d1 = Infinity
  let d2 = Infinity
  for (let u = 0; u < units; u++) {
    let s = 0
    for (let c = 0; c < d; c++) s += (v[i * d + c] - W[u * d + c]) ** 2
    if (s < d1) {
      b2 = b1
      d2 = d1
      b1 = u
      d1 = s
    } else if (s < d2) {
      b2 = u
      d2 = s
    }
  }
  return { b1, b2, d1 }
}

/** The batch SOM on the rows of x [n, d] as a step-through algorithm (see the module comment). */
export function selfOrganisingMapSteps(x: Tensor, params: SomParams = {}): Algorithm<void, SomState> {
  const { n, d, v } = matrix(x, 'selfOrganisingMapSteps')
  const { rows = 8, cols = 8, epochs = 30, start = 'pca' } = params
  const units = rows * cols
  const r0 = params.radius ?? Math.max(rows, cols) / 2
  const r1 = params.finalRadius ?? 0.5
  const radiusAt = (t: number) => (epochs <= 1 ? r1 : r0 * (r1 / r0) ** (t / (epochs - 1)))
  const evaluate = (W: Float64Array, t: number, radius: number): SomState => {
    const assignment = new Int32Array(n)
    let qe = 0
    let te = 0
    for (let i = 0; i < n; i++) {
      const { b1, b2, d1 } = bestTwo(v, i, W, units, d)
      assignment[i] = b1
      qe += Math.sqrt(d1) / n
      const far = Math.abs(Math.floor(b1 / cols) - Math.floor(b2 / cols)) > 1 || Math.abs((b1 % cols) - (b2 % cols)) > 1
      if (units > 1 && far) te += 1 / n
    }
    return {
      t,
      weights: mat(W, units, d),
      radius,
      quantisationError: qe,
      topographicError: te,
      assignment,
      diverged: !Number.isFinite(qe),
    }
  }
  return {
    name: 'batch-som',
    init: (_input, s) => {
      const W = new Float64Array(units * d)
      const mean = new Float64Array(d)
      for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) mean[c] += v[i * d + c] / n
      if (start === 'random') {
        const noise = values(normals(s, [units, d]))
        for (let u = 0; u < units; u++) {
          const i = integers(s, n)
          for (let c = 0; c < d; c++) W[u * d + c] = v[i * d + c] + 0.01 * noise[u * d + c]
        }
      } else {
        // Spread the grid over ±2 standard deviations of the top two principal axes.
        const S = new Float64Array(d * d)
        for (let i = 0; i < n; i++)
          for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++) S[a * d + b] += ((v[i * d + a] - mean[a]) * (v[i * d + b] - mean[b])) / n
        const e = eigh(mat(S, d, d))
        const lambda = values(e.values)
        const V = values(e.vectors)
        for (let u = 0; u < units; u++) {
          const a = rows > 1 ? (2 * Math.floor(u / cols)) / (rows - 1) - 1 : 0
          const b = cols > 1 ? (2 * (u % cols)) / (cols - 1) - 1 : 0
          for (let c = 0; c < d; c++) {
            const first = 2 * Math.sqrt(Math.max(lambda[0], 0)) * V[c * d] * b
            const second = d > 1 ? 2 * Math.sqrt(Math.max(lambda[1], 0)) * V[c * d + 1] * a : 0
            W[u * d + c] = mean[c] + first + second
          }
        }
      }
      return evaluate(W, 0, r0)
    },
    step: (st) => {
      const radius = radiusAt(st.t)
      const W = values(st.weights)
      const assignment = st.assignment
      // Rows summed per BMU, then spread over the grid by the neighbourhood kernel.
      const sums = new Float64Array(units * d)
      const counts = new Float64Array(units)
      for (let i = 0; i < n; i++) {
        const b = assignment[i]
        counts[b]++
        for (let c = 0; c < d; c++) sums[b * d + c] += v[i * d + c]
      }
      const next = new Float64Array(units * d)
      for (let u = 0; u < units; u++) {
        let total = 0
        const gu = Math.floor(u / cols)
        const hu = u % cols
        for (let b = 0; b < units; b++) {
          if (counts[b] === 0) continue
          const h = Math.exp(-((Math.floor(b / cols) - gu) ** 2 + ((b % cols) - hu) ** 2) / (2 * radius * radius))
          total += h * counts[b]
          for (let c = 0; c < d; c++) next[u * d + c] += h * sums[b * d + c]
        }
        for (let c = 0; c < d; c++) next[u * d + c] = total > 0 ? next[u * d + c] / total : W[u * d + c]
      }
      return evaluate(next, st.t + 1, radius)
    },
    done: (st) => st.t >= epochs,
  }
}

/** A fitted SOM. */
export interface SomModel extends Transforms<Tensor, Tensor>, Trained<SomState> {
  readonly kind: 'model'
  readonly name: 'self-organising-map'
  readonly rows: number
  readonly cols: number
  /** Unit weights [rows·cols, d]. */
  readonly weights: Tensor
  /** The U-matrix [rows, cols]: each unit's mean distance to its grid neighbours' weights (high on cluster borders). */
  readonly uMatrix: Tensor
  /** Grid coordinates (row, column) of each row's BMU [m, 2]. */
  transform(x: Tensor): Tensor
}

/** A batch self-organising map (see `selfOrganisingMapSteps`). */
export function selfOrganisingMap(params: SomParams = {}): Estimator<Dataset<Tensor>, SomModel> {
  const { rows = 8, cols = 8, epochs = 30 } = params
  return {
    name: 'self-organising-map',
    params: { ...params, rows, cols, epochs },
    fit({ x }, options: FitOptions = {}) {
      const { d } = matrix(x, 'selfOrganisingMap')
      const training = trace(selfOrganisingMapSteps(x, params), undefined, epochs, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { quantisationError: (s) => s.quantisationError },
      })
      const W = values(training.final.weights)
      const units = rows * cols
      const U = new Float64Array(units)
      for (let u = 0; u < units; u++) {
        let total = 0
        let count = 0
        const gr = Math.floor(u / cols)
        const gc = u % cols
        for (const [dr, dc] of [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ]) {
          const r = gr + dr
          const c = gc + dc
          if (r < 0 || r >= rows || c < 0 || c >= cols) continue
          let s = 0
          for (let k = 0; k < d; k++) s += (W[u * d + k] - W[(r * cols + c) * d + k]) ** 2
          total += Math.sqrt(s)
          count++
        }
        U[u] = count ? total / count : 0
      }
      return {
        kind: 'model',
        name: 'self-organising-map',
        rows,
        cols,
        weights: training.final.weights,
        uMatrix: mat(U, rows, cols),
        training,
        transform: (q: Tensor) => {
          const { n: m, d: dq, v: qv } = matrix(q, 'selfOrganisingMap.transform')
          if (dq !== d)
            throw new ShapeError('selfOrganisingMap', `selfOrganisingMap: fitted on ${d} features, given ${dq}`)
          const out = new Float64Array(m * 2)
          for (let i = 0; i < m; i++) {
            const { b1 } = bestTwo(qv, i, W, units, d)
            out[i * 2] = Math.floor(b1 / cols)
            out[i * 2 + 1] = b1 % cols
          }
          return mat(out, m, 2)
        },
      }
    },
  }
}

/** The grid positions (row, column) of a SOM's units [rows·cols, 2], in unit order. */
export function somGrid(rows: number, cols: number): Tensor {
  return mat(
    Float64Array.from({ length: rows * cols * 2 }, (_, k) =>
      k % 2 ? Math.floor(k / 2) % cols : Math.floor(k / 2 / cols),
    ),
    rows * cols,
    2,
  )
}

defineModel(
  {
    key: 'selfOrganisingMap',
    module: 'unsupervised/embedding/manifold',
    name: 'Self-organising map',
    summary: 'A grid of prototypes fitted by batch neighbourhood-weighted means under a shrinking radius.',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({
      rows: int(1, 40, { default: 8 }),
      cols: int(1, 40, { default: 8 }),
      epochs: int(1, 500, { default: 30 }),
      finalRadius: real(0.1, 5, { default: 0.5 }),
      start: oneOf(['pca', 'random']),
    }),
    notes: ['self-organising-maps'],
    cite: ['kohonen1982', 'kohonen2013'],
  },
  selfOrganisingMap,
)
