/**
 * Evaluating explanations: whether an attribution reflects what the model computes.
 *
 * - `deletionCurve` (Petsiuk, Das and Saenko, 2018): remove features from $\xvec$ in order of decreasing attribution,
 *   setting them to a baseline, and follow the output; a faithful attribution makes it fall fast (a small area under
 *   the curve). `insertion` starts from the baseline and restores the most important features first; the output
 *   should rise fast (a large area). Compare with a random order.
 * - `faithfulnessCorrelation` (Bhatt, Weller and Moura, 2020): over random subsets $S$ of features, the Pearson
 *   correlation between $\sum_{i \in S} \phi_i$ and the drop $f(\xvec) - f(\xvec \text{ with } S \text{ at the
 *   baseline})$.
 * - `randomiseNetwork` with `explanationSimilarity` (Adebayo et al., 2018, sanity checks): re-initialise a network's
 *   layers from the top down; an explanation that depends on the model must change, so its rank correlation with the
 *   original should fall. (The data-randomisation check retrains on permuted labels and compares the same way.)
 * - `relevanceMass` (Arras, Osman and Samek, 2022): against a ground-truth mask, the share of the attribution's
 *   absolute mass that falls on the mask.
 * - `localLipschitz` (Alvarez-Melis and Jaakkola, 2018): the stability of an explanation,
 *   $\max \lVert \phi(\xvec) - \phi(\xvec') \rVert / \lVert \xvec - \xvec' \rVert$ over points $\xvec'$ sampled in
 *   a ball around $\xvec$.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { correlation, spearman } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { denseLayers, type DenseNetwork } from './network'
import type { ScalarModel } from './shapley'

/**
 * A model's outputs on a batch of rows, as a fresh array.
 *
 * @param model The model, called once on the $m \times d$ batch.
 * @param rows The rows, row-major ($m \times d$ values).
 * @param m The number of rows.
 * @param d The number of features.
 * @returns The $m$ outputs.
 */
const evaluate = (model: ScalarModel, rows: Float64Array, m: Size, d: Size): Float64Array => {
  const o = model(fromData(rows, [m, d]))
  return 'shape' in o ? Float64Array.from(dense.data(o as Tensor)) : Float64Array.from(o)
}

/**
 * Feature indices by decreasing (signed) attribution, ties by index: the most-relevant-first (MoRF) order.
 *
 * @param attribution One value per feature.
 * @returns The feature indices, largest attribution first.
 *
 * @example Signed, not by magnitude
 * print(attributionOrder([0.1, -2, 3, 0.1]))
 */
export function attributionOrder(attribution: ArrayLike<number>): number[] {
  return Array.from({ length: attribution.length }, (_, i) => i).sort(
    (a, b) => attribution[b] - attribution[a] || a - b,
  )
}

/**
 * The deletion or insertion curve of `model` at $\xvec$ for an attribution, in one model call. Throws `ShapeError`
 * when the attribution does not have one value per feature.
 *
 * @param model The model, called once on the $\lceil d/\text{step} \rceil + 1$ rows of the curve.
 * @param x The instance $\xvec$ ($d$ values).
 * @param attribution One value per feature; features are changed in `attributionOrder`.
 * @param options The direction, baseline and step.
 * @param options.mode `'deletion'` (default) starts from $\xvec$ and sets features to the baseline; `'insertion'`
 *   starts from the baseline and restores features of $\xvec$.
 * @param options.baseline The value of a removed feature: one number or a vector of $d$ (default 0).
 * @param options.step The number of features changed per step (default 1).
 * @returns `fraction`, the share of features changed at each point; `output`, the model's output there; and `area`,
 *   the area under the curve by the trapezoid rule over `fraction`.
 *
 * @example A faithful order falls faster than the reverse
 * const model = (X) => matmul(X, tensor([3, 1, 2]))
 * const good = deletionCurve(model, [1, 1, 1], [3, 1, 2])
 * print('faithful:', good.output, ' area =', good.area)
 * print('reversed:', deletionCurve(model, [1, 1, 1], [-3, -1, -2]).area)
 * print('insertion:', deletionCurve(model, [1, 1, 1], [3, 1, 2], { mode: 'insertion' }).output)
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
 * Faithfulness correlation of an attribution at $\xvec$: over random subsets of features, the Pearson correlation
 * between the subset's summed attribution and the output's drop when the subset is set to the baseline. One model
 * call. Throws `DomainError` when `subsetSize` is not in $1, \dots, d$.
 *
 * @param model The model, called once on $\xvec$ and every perturbed copy.
 * @param x The instance $\xvec$ ($d$ values).
 * @param attribution One value per feature.
 * @param stream The random stream: subset $k$ is drawn from `child(stream, 'subset', k)`.
 * @param options The subsets and the baseline.
 * @param options.subsetSize The features per subset, drawn without replacement (default $\lceil d/4 \rceil$).
 * @param options.samples The number of subsets (default 100).
 * @param options.baseline The value of a removed feature: one number or a vector of $d$ (default 0).
 * @returns `correlation`, the Pearson correlation (NaN when either side does not vary), and its inputs: `attributed`,
 *   each subset's summed attribution, and `drops`, each subset's drop in output.
 *
 * @example The true attribution of a linear model against a shuffled one
 * const model = (X) => matmul(X, tensor([3, -1, 2, 0.5]))
 * const x = [1, 1, 1, 1]
 * const exact = faithfulnessCorrelation(model, x, [3, -1, 2, 0.5], stream(0), { samples: 30, subsetSize: 2 })
 * const shuffled = faithfulnessCorrelation(model, x, [0.5, 2, -1, 3], stream(0), { samples: 30, subsetSize: 2 })
 * print('exact:', exact.correlation, ' shuffled:', shuffled.correlation)
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
 * Cascading randomisation of a `DenseNetwork`: copies with the top $k$ layers' weights and biases redrawn from
 * $\Gauss(0, \sigma^2)$, $\sigma$ each layer's own weight standard deviation (1 when its weights are all equal), for
 * $k = 0$ (the original) to $L$ (every layer). A layer redrawn in one copy is the same in every later copy.
 *
 * @param net The network; not modified.
 * @param stream The random stream: layer $l$ draws from `child(stream, 'layer', l)`.
 * @returns $L + 1$ networks, the $k$-th with its top $k$ layers redrawn.
 *
 * @example DeepLIFT's attributions drift as the layers are redrawn
 * const W1 = [[1, 0, 0.5], [0, 1, 0], [1, 1, -1], [-1, 0, 0.5]]
 * const net = { weights: [W1, [[1], [2], [-1]]], biases: [[0, 0, 0], [0]], activation: 'tanh' }
 * const x = [1, 2, 3, 0.5]
 * const original = deepLift(net, x).values
 * print('original:', original)
 * for (const [k, copy] of randomiseNetwork(net, stream(0)).entries()) {
 *   const values = deepLift(copy, x).values
 *   print(`top ${k} redrawn:`, values, ' spearman =', explanationSimilarity(original, values).spearman)
 * }
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
 * How alike two attributions are. Throws `ShapeError` when their lengths differ.
 *
 * @param a One attribution ($d$ values).
 * @param b The other ($d$ values).
 * @param options The size of the top sets compared.
 * @param options.top The $k$ of the top-$k$ comparison (default $\lceil d/10 \rceil$).
 * @returns `spearman`, Spearman's rank correlation of the values, and `spearmanAbsolute`, of their absolute values
 *   (each 0 when undefined, as for a constant attribution); `topIntersection`, the fraction of `a`'s top $k$ features
 *   by magnitude that are also in `b`'s.
 *
 * @example Same ranking by magnitude, different signs
 * print(explanationSimilarity([3, -2, 1, 0], [3, 2, 1, 0], { top: 2 }))
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
 * Local Lipschitz estimate of an explanation map $\phi$ at $\xvec$: the largest and the mean of
 * $\lVert \phi(\xvec) - \phi(\xvec') \rVert / \lVert \xvec - \xvec' \rVert$ over points $\xvec'$ drawn uniformly
 * in a ball about $\xvec$.
 *
 * @param explain The explanation map $\phi$: from a point ($d$ values) to its attribution; called once at $\xvec$ and
 *   once per sample.
 * @param x The instance $\xvec$ ($d$ values).
 * @param stream The random stream: point $k$ is drawn from `child(stream, 'point', k)`.
 * @param options The ball and the number of points.
 * @param options.radius The ball's radius (default 0.1).
 * @param options.samples The number of points $\xvec'$ (default 50).
 * @returns `max` and `mean` of the ratios, and every ratio in `ratios`.
 *
 * @example A smooth explanation and a jumping one
 * // The gradient 2x of |x|^2 is 2-Lipschitz; a step at 0 is not Lipschitz there.
 * print('gradient:', localLipschitz((x) => x.map((v) => 2 * v), [1, 1], stream(0), { samples: 20 }).max)
 * print('step:', localLipschitz((x) => [x[0] > 0 ? 1 : 0], [0], stream(0), { samples: 20 }).max)
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
 * The share of an attribution's absolute mass on a ground-truth mask:
 * $\sum_{i : m_i > 0} \lvert \phi_i \rvert / \sum_i \lvert \phi_i \rvert$. Throws `ShapeError` when the lengths
 * differ.
 *
 * @param attribution One value $\phi_i$ per feature.
 * @param mask The ground truth, one weight $m_i \ge 0$ per feature; a feature is on the mask where its weight is
 *   positive.
 * @returns The share, in $[0, 1]$ (0 when the attribution is all zero).
 *
 * @example Four fifths of the mass on the mask
 * print(relevanceMass([3, -1, 0, 1], [1, 1, 0, 0]))
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
