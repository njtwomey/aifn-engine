/**
 * Evaluating explanations: whether an attribution reflects what the model computes.
 *
 * - `deletionCurve` (Petsiuk, Das and Saenko, 2018): remove features from x in order of decreasing attribution, setting
 *   them to a baseline, and follow the output; a faithful attribution makes it fall fast (a small area under the
 *   curve). `insertion` starts from the baseline and restores the most important features first; the output should rise
 *   fast (a large area). Compare with a random order.
 * - `faithfulnessCorrelation` (Bhatt, Weller and Moura, 2020): over random subsets S of features, the Pearson
 *   correlation between Σ_{i∈S} φᵢ and the drop f(x) − f(x with S at the baseline).
 * - `randomiseNetwork` with `explanationSimilarity` (Adebayo et al., 2018, sanity checks): re-initialise a network's
 *   layers from the top down; an explanation that depends on the model must change, so its rank correlation with the
 *   original should fall. (The data-randomisation check retrains on permuted labels and compares the same way.)
 * - `relevanceMass` (Arras, Osman and Samek, 2022): against a ground-truth mask, the share of the attribution's absolute
 *   mass that falls on the mask.
 * - `localLipschitz` (Alvarez-Melis and Jaakkola, 2018): the stability of an explanation, max ‖φ(x) − φ(x′)‖/‖x − x′‖
 *   over points sampled in a ball around x.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { correlation, spearman } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { denseLayers, type DenseNetwork } from './network'
import type { ScalarModel } from './shapley'

const evaluate = (model: ScalarModel, rows: Float64Array, m: Size, d: Size): Float64Array => {
  const o = model(fromData(rows, [m, d]))
  return 'shape' in o ? Float64Array.from(dense.data(o as Tensor)) : Float64Array.from(o)
}

/** Feature indices by decreasing attribution (ties by index). */
export function attributionOrder(attribution: ArrayLike<number>): number[] {
  return Array.from({ length: attribution.length }, (_, i) => i).sort(
    (a, b) => attribution[b] - attribution[a] || a - b,
  )
}

/**
 * The deletion (default) or insertion curve of `model` at x [d] for an attribution [d]: the output after each step of
 * `step` features (default 1) [steps + 1], the fraction of features changed at each, and the area under the curve
 * by the trapezoid rule over that fraction. `baseline` is one number or a vector [d] (default 0).
 */
export function deletionCurve(
  model: ScalarModel,
  x: VectorLike,
  attribution: ArrayLike<number>,
  options: { mode?: 'deletion' | 'insertion'; baseline?: number | VectorLike; step?: Size } = {},
): { fraction: Float64Array; output: Float64Array; area: number } {
  const xv = dense.toF64(x, 'deletionCurve')
  const d = xv.length
  if (attribution.length !== d) throw new ShapeError('deletionCurve', 'deletionCurve: one attribution per feature')
  const { mode = 'deletion', step = 1 } = options
  const base =
    typeof options.baseline === 'number' || options.baseline === undefined
      ? new Float64Array(d).fill(options.baseline ?? 0)
      : Float64Array.from(dense.toF64(options.baseline, 'deletionCurve'))
  const order = attributionOrder(attribution)
  const steps = Math.ceil(d / step)
  const rows = new Float64Array((steps + 1) * d)
  const current = Float64Array.from(mode === 'deletion' ? xv : base)
  rows.set(current, 0)
  for (let s = 1; s <= steps; s++) {
    for (let k = (s - 1) * step; k < Math.min(d, s * step); k++) {
      const i = order[k]
      current[i] = mode === 'deletion' ? base[i] : xv[i]
    }
    rows.set(current, s * d)
  }
  const output = evaluate(model, rows, steps + 1, d)
  const fraction = Float64Array.from({ length: steps + 1 }, (_, s) => Math.min(d, s * step) / d)
  let area = 0
  for (let s = 1; s <= steps; s++) area += ((output[s] + output[s - 1]) / 2) * (fraction[s] - fraction[s - 1])
  return { fraction, output, area }
}

/**
 * Faithfulness correlation of an attribution [d] at x [d]: over `samples` (default 100) random subsets of
 * `subsetSize` features (default ⌈d/4⌉), the Pearson correlation between the subset's summed attribution and the
 * output's drop when the subset is set to the baseline (default 0).
 */
export function faithfulnessCorrelation(
  model: ScalarModel,
  x: VectorLike,
  attribution: ArrayLike<number>,
  stream: Stream,
  options: { subsetSize?: Size; samples?: Size; baseline?: number | VectorLike } = {},
): { correlation: number; attributed: Float64Array; drops: Float64Array } {
  const xv = dense.toF64(x, 'faithfulnessCorrelation')
  const d = xv.length
  const { samples = 100 } = options
  const size = options.subsetSize ?? Math.max(1, Math.ceil(d / 4))
  if (!(size >= 1 && size <= d))
    throw new DomainError('faithfulnessCorrelation', 'faithfulnessCorrelation: subsetSize must lie in 1 … d')
  const base =
    typeof options.baseline === 'number' || options.baseline === undefined
      ? new Float64Array(d).fill(options.baseline ?? 0)
      : Float64Array.from(dense.toF64(options.baseline, 'faithfulnessCorrelation'))
  const rows = new Float64Array((samples + 1) * d)
  rows.set(xv, 0)
  const attributed = new Float64Array(samples)
  for (let k = 0; k < samples; k++) {
    const s = child(stream, 'subset', k)
    const order = Array.from({ length: d }, (_, i) => i)
    for (let i = 0; i < size; i++) {
      const j = i + Math.min(d - i - 1, Math.floor((uniform(s) as number) * (d - i)))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    const row = Float64Array.from(xv)
    for (let i = 0; i < size; i++) {
      row[order[i]] = base[order[i]]
      attributed[k] += attribution[order[i]]
    }
    rows.set(row, (k + 1) * d)
  }
  const out = evaluate(model, rows, samples + 1, d)
  const drops = Float64Array.from({ length: samples }, (_, k) => out[0] - out[k + 1])
  return { correlation: correlation(attributed, drops), attributed, drops }
}

/**
 * Cascading randomisation of a `DenseNetwork`: copies with the top `k` layers' weights and biases redrawn from
 * N(0, σ²) with σ each layer's own weight standard deviation, for k = 0 (the original) … L (every layer).
 */
export function randomiseNetwork(net: DenseNetwork, stream: Stream): DenseNetwork[] {
  const layers = denseLayers(net)
  const L = layers.length
  const fresh = layers.map((layer, l) => {
    const n = layer.W.length
    const m = layer.W.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(layer.W.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1
    const s = child(stream, 'layer', l)
    return {
      W: Float64Array.from(toFlat(normal(child(s, 'weight'), 0, sd, { shape: [n] }))),
      b: Float64Array.from(toFlat(normal(child(s, 'bias'), 0, sd, { shape: [layer.b.length] }))),
    }
  })
  return Array.from({ length: L + 1 }, (_, k) => ({
    activation: net.activation,
    weights: layers.map((layer, l) => fromData(l >= L - k ? fresh[l].W : layer.W, [layer.inputs, layer.outputs])),
    biases: layers.map((layer, l) => (l >= L - k ? fresh[l].b : layer.b)),
  }))
}

/**
 * How alike two attributions are: Spearman's rank correlation of the values and of their absolute values, and the
 * fraction of shared features among each one's top k by magnitude (k default ⌈d/10⌉).
 */
export function explanationSimilarity(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  options: { top?: Size } = {},
): { spearman: number; spearmanAbsolute: number; topIntersection: number } {
  if (a.length !== b.length) throw new ShapeError('explanationSimilarity', 'explanationSimilarity: lengths differ')
  const d = a.length
  const k = options.top ?? Math.max(1, Math.ceil(d / 10))
  const absA = Array.from(a, Math.abs)
  const absB = Array.from(b, Math.abs)
  const top = (v: number[]) => new Set(attributionOrder(v).slice(0, k))
  const ta = top(absA)
  const tb = top(absB)
  let shared = 0
  for (const i of ta) if (tb.has(i)) shared++
  const safe = (r: number) => (Number.isFinite(r) ? r : 0)
  return {
    spearman: safe(spearman(Array.from(a), Array.from(b))),
    spearmanAbsolute: safe(spearman(absA, absB)),
    topIntersection: shared / k,
  }
}

/**
 * Local Lipschitz estimate of an explanation map φ at x [d]: the largest and mean ‖φ(x) − φ(x′)‖/‖x − x′‖ over
 * `samples` (default 50) points x′ drawn uniformly in the ball of `radius` (default 0.1) about x.
 */
export function localLipschitz(
  explain: (x: Float64Array) => ArrayLike<number>,
  x: VectorLike,
  stream: Stream,
  options: { radius?: number; samples?: Size } = {},
): { max: number; mean: number; ratios: Float64Array } {
  const xv = Float64Array.from(dense.toF64(x, 'localLipschitz'))
  const d = xv.length
  const { radius = 0.1, samples = 50 } = options
  const phi = explain(xv)
  const ratios = new Float64Array(samples)
  for (let k = 0; k < samples; k++) {
    const s = child(stream, 'point', k)
    const dir = toFlat(normal(child(s, 'direction'), 0, 1, { shape: [d] }))
    const norm = Math.hypot(...dir) || 1
    const r = radius * (uniform(child(s, 'radius')) as number) ** (1 / d)
    const z = Float64Array.from(xv, (v, i) => v + (r * dir[i]) / norm)
    const pz = explain(z)
    let num = 0
    for (let i = 0; i < phi.length; i++) num += (phi[i] - pz[i]) ** 2
    ratios[k] = r > 0 ? Math.sqrt(num) / r : 0
  }
  return { max: Math.max(...ratios), mean: ratios.reduce((a, b) => a + b, 0) / samples, ratios }
}

/**
 * The share of an attribution's absolute mass on a ground-truth mask [d] (weights ≥ 0; a cell counts where its weight
 * is positive): Σ_{mask} |φᵢ| / Σ |φᵢ| (0 when the attribution is all zero).
 */
export function relevanceMass(attribution: ArrayLike<number>, mask: ArrayLike<number>): number {
  if (attribution.length !== mask.length) throw new ShapeError('relevanceMass', 'relevanceMass: lengths differ')
  let on = 0
  let all = 0
  for (let i = 0; i < mask.length; i++) {
    const a = Math.abs(attribution[i])
    all += a
    if (mask[i] > 0) on += a
  }
  return all > 0 ? on / all : 0
}
