/**
 * The shared layer of `aifn-methods/generative`: densities known as labelled mixtures (the `model` of a classification
 * truth from `aifn-methods/data`: a ring of Gaussians, a pinwheel, two moons), their log density and most probable
 * mode at given points, and the square grid every 2-d figure evaluates fields on.
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** A density known as a labelled mixture: mode weights πⱼ and log p(x | j) per row and mode ([n, k]). */
export type LabelledDensity = {
  readonly classes: number
  readonly priors: readonly number[]
  logDensity(x: Tensor): Tensor
}

/**
 * The labelled density of a dataset's truth, when it has one: a truth carrying a `model` with classes, priors and
 * class log densities (a classification truth of `aifn-methods/data`); null otherwise.
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

/** log πⱼ + log p(x | j) per row and mode, row-major [n × k]. */
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

/** log p(x) = log Σⱼ πⱼ p(x | j) at each row of x ([n, d]); −∞ off the support. */
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

/** The most probable mode of each row (int32 [n]); −1 where every mode has zero density. */
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

/** A square grid of g × g cell centres over [−box, box]²: axes and the points row by row (z[i][j] at (x[j], y[i])). */
export type Grid2d = { x: Float64Array; y: Float64Array; points: Tensor }

/** The grid of g × g points over [−box, box]². */
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
