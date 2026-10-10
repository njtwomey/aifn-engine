/**
 * The shared layer of `aifn-methods/generative`: densities known as labelled mixtures (the `model` of a classification
 * truth from `aifn-methods/data`: a ring of Gaussians, a pinwheel, two moons), their log density and most probable
 * mode at given points, and the square grid every 2-d figure evaluates fields on.
 *
 * A labelled mixture is $p(\xvec) = \sum_{j=1}^k \pi_j \, p(\xvec \mid j)$, given by its weights $\pi_j$ and the
 * log densities $\log p(\xvec \mid j)$ of its $k$ modes; its log density and modes are computed in the log domain
 * from $\log \pi_j + \log p(\xvec \mid j)$, so far tails do not underflow. Points are rows, $[n, d]$.
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * A density known as a labelled mixture: $k$ modes (`classes`), their weights $\pi_j$ (`priors`), and `logDensity`,
 * which maps points $[n, d]$ to $\log p(\xvec \mid j)$ per row and mode, $[n, k]$.
 */
export type LabelledDensity = {
  readonly classes: number
  readonly priors: readonly number[]
  logDensity(x: Tensor): Tensor
}

/**
 * The labelled density of a dataset's truth, when it has one: a truth carrying a `model` with classes, priors and
 * class log densities (a classification truth of `aifn-methods/data`); null otherwise. Only the presence and types of
 * the three fields are checked.
 *
 * @param truth A dataset's truth, of any form (or undefined).
 * @returns Its `model` as a `LabelledDensity`, or null when it has none.
 *
 * @example A truth with a model, and one without
 * const model = { classes: 1, priors: [1], logDensity: (x) => tensor([[0]]) }
 * print('with a model:', knownDensity({ model }) === model)
 * print('without:', knownDensity({ kind: 'regression' }))
 */
export function knownDensity(truth: unknown): LabelledDensity | null {
  const model = (truth as { model?: Partial<LabelledDensity> } | undefined)?.model
  return model &&
    typeof model.classes === 'number' &&
    Array.isArray(model.priors) &&
    typeof model.logDensity === 'function'
    ? (model as LabelledDensity)
    : null
}

/**
 * $\log \pi_j + \log p(\xvec \mid j)$ per row and mode.
 *
 * @param model The labelled density.
 * @param x The points $[n, d]$.
 * @returns `a`, the $n k$ values row-major (row $i$ is entries $ik$ to $ik + k - 1$), with $n$ and $k$.
 */
function logJoint(model: LabelledDensity, x: Tensor): { a: Float64Array; n: number; k: number } {
  const k = model.classes
  const a = Float64Array.from(toFlat(model.logDensity(x)))
  const n = a.length / k
  for (let j = 0; j < k; j++) {
    const lp = Math.log(model.priors[j])
    for (let i = 0; i < n; i++) a[i * k + j] += lp
  }
  return { a, n, k }
}

/**
 * $\log p(\xvec) = \log \sum_j \pi_j \, p(\xvec \mid j)$ at each row of `x`, by log-sum-exp; $-\infty$ off the
 * support, where every mode has zero density.
 *
 * @param model The labelled density.
 * @param x The points $[n, d]$.
 * @returns The $n$ log densities.
 *
 * @example Two equal unit Gaussians at $-2$ and $2$, in 1-d
 * const logN = (v, m) => -0.5 * (v - m) ** 2 - 0.5 * Math.log(2 * Math.PI)
 * const model = {
 *   classes: 2,
 *   priors: [0.5, 0.5],
 *   logDensity: (x) => tensor(toArray(x).map(([v]) => [logN(v, -2), logN(v, 2)])),
 * }
 * print('log p at -2, 0, 2:', mixtureLogDensityOf(model, tensor([[-2], [0], [2]])))
 * print('at 0, both modes 2 away:', logN(0, 2))
 */
export function mixtureLogDensityOf(model: LabelledDensity, x: Tensor): Float64Array {
  const { a, n, k } = logJoint(model, x)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let m = -Infinity
    for (let j = 0; j < k; j++) m = Math.max(m, a[i * k + j])
    let s = 0
    if (m > -Infinity) for (let j = 0; j < k; j++) s += Math.exp(a[i * k + j] - m)
    out[i] = m > -Infinity ? m + Math.log(s) : -Infinity
  }
  return out
}

/**
 * The most probable mode of each row, $\argmax_j \pi_j \, p(\xvec \mid j)$ (the first on a tie); $-1$ where every
 * mode has zero density.
 *
 * @param model The labelled density.
 * @param x The points $[n, d]$.
 * @returns The $n$ mode indices, from 0.
 *
 * @example The mode of three points, the weights tipping the one between
 * const logN = (v, m) => -0.5 * (v - m) ** 2 - 0.5 * Math.log(2 * Math.PI)
 * const model = {
 *   classes: 2,
 *   priors: [0.3, 0.7],
 *   logDensity: (x) => tensor(toArray(x).map(([v]) => [logN(v, -2), logN(v, 2)])),
 * }
 * print('modes at -3, 0, 3:', modeOf(model, tensor([[-3], [0], [3]])))
 */
export function modeOf(model: LabelledDensity, x: Tensor): Int32Array {
  const { a, n, k } = logJoint(model, x)
  return Int32Array.from({ length: n }, (_, i) => {
    let best = -1
    let top = -Infinity
    for (let j = 0; j < k; j++)
      if (a[i * k + j] > top) {
        top = a[i * k + j]
        best = j
      }
    return best
  })
}

/**
 * A square grid of $g \times g$ cell centres over $[-b, b]^2$: the axes `x` and `y` ($g$ values each), and `points`
 * $[g^2, 2]$ row by row, so point $ig + j$ is $(x_j, y_i)$ and a field over the grid is read as $z_{ij}$ at
 * $(x_j, y_i)$.
 */
export type Grid2d = { x: Float64Array; y: Float64Array; points: Tensor }

/**
 * The grid of $g \times g$ cell centres over $[-b, b]^2$: the centres along each axis are
 * $-b + (i + \tfrac12) \, 2b / g$ for $i = 0, \dots, g - 1$.
 *
 * @param box The half-width $b$ of the square.
 * @param g The number of cells per side.
 * @returns The axes and the $g^2$ points.
 *
 * @example A $2 \times 2$ grid over $[-1, 1]^2$
 * const grid = squareGrid(1, 2)
 * print('axis:', grid.x)
 * print('points:', grid.points)
 */
export function squareGrid(box: number, g: number): Grid2d {
  const axis = Float64Array.from({ length: g }, (_, i) => -box + ((i + 0.5) * 2 * box) / g)
  const pts = new Float64Array(2 * g * g)
  for (let i = 0; i < g; i++)
    for (let j = 0; j < g; j++) {
      pts[2 * (i * g + j)] = axis[j]
      pts[2 * (i * g + j) + 1] = axis[i]
    }
  return { x: axis, y: Float64Array.from(axis), points: fromData(pts, [g * g, 2]) }
}
