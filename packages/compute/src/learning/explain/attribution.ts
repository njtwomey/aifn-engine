/**
 * More local attributions:
 *
 * - `occlusion` (Zeiler and Fergus, 2014): slide a window over the input (a vector, or an image or sequence given by
 *   `shape`), set the features under it to a baseline, and credit each feature with the output's drop averaged over
 *   the windows covering it, as Captum's `Occlusion`.
 * - `deepLift` (Shrikumar, Greenside and Kundaje, 2017), rescale rule: through a `DenseNetwork`, each nonlinearity's
 *   multiplier is $\Delta a/\Delta z$ between the input and a reference (its derivative where
 *   $\Delta z \approx 0$), and linear layers pass multipliers back by $\Wmat^\top$. The attributions
 *   $(\xvec - \xvec') \odot \mvec$ sum to $f(\xvec) - f(\xvec')$ exactly (summation-to-delta). `deepShap`
 *   (Lundberg and Lee, 2017) averages DeepLIFT over background references.
 * - `expectedGradients` (Erion et al., 2021): integrated gradients averaged over baselines drawn from the data, with a
 *   random point on each path,
 *   $\expect_{\xvec', \alpha}[(\xvec - \xvec') \odot \nabla f(\xvec' + \alpha(\xvec - \xvec'))]$.
 * - `shapleyInteractions` (Lundberg, Erion and Lee, 2018, after Grabisch and Roubens, 1999): the Shapley interaction
 *   values $\Phi_{ij}$ of a set function, exact by enumeration; each row sums to the Shapley value $\phi_i$.
 * - `kernelShapVariance`: KernelSHAP's sampling spread, by repeating the sampled estimate on independent streams.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { activate, activateDerivative, denseForward, denseLayers, type DenseNetwork } from './network'
import { exactShapley, kernelShap, type KernelShapOptions, type ScalarModel } from './shapley'
import type { Differentiable } from './gradients'

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

// ── Occlusion ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `occlusion`. */
export type OcclusionOptions = {
  /**
   * The input's shape (default $[d]$); its product must be $d$. A 2-D shape $[h, w]$ is an image in row-major order.
   */
  shape?: readonly Size[]
  /** The window's extent per axis (default 1 per axis). */
  window?: readonly Size[]
  /** The step between window positions per axis (default 1). */
  strides?: readonly Size[]
  /** The value the occluded features take: one number or a vector of $d$ (default 0). */
  baseline?: number | VectorLike
}

/**
 * Occlusion attributions of `model` at $\xvec$: for each window position (along each axis, starts
 * $0, s, 2s, \dots$ up to and including the first whose window reaches the end, cropped there) the drop
 * $f(\xvec) - f(\xvec \text{ with the window at the baseline})$, credited in full to every feature in the window and
 * averaged over the windows covering each feature. One model call over all positions. Throws `ShapeError` when `shape`
 * does not hold $d$ features or `window` and `strides` do not have one entry per axis, and `DomainError` when a window
 * is not in $1, \dots, \text{size}$ or a stride is below 1.
 *
 * @param model The model, called once on the instance and all its occluded copies.
 * @param x The instance $\xvec$ ($d$ values, row-major in `shape`).
 * @param options The input's shape, the window, its strides and the baseline.
 * @returns `values`, the attributions ($d$; 0 for a feature no window covers); `output`, $f(\xvec)$; and `windows`, the
 *   number of window positions.
 *
 * @example A linear model with one feature per window: weight times (x minus the baseline)
 * const model = (X) => matmul(X, tensor([2, -1, 0.5]))
 * print(occlusion(model, [1, 2, 3]))
 *
 * @example Windows on a 2 x 3 image, the last one cropped
 * const model = (X) => sum(X, 1)
 * print(occlusion(model, [1, 2, 3, 4, 5, 6], { shape: [2, 3], window: [1, 2], strides: [1, 2] }))
 */
export function occlusion(
  model: ScalarModel,
  x: VectorLike,
  options: OcclusionOptions = {},
): { values: Float64Array; output: number; windows: Size } {
  const xv = dense.toF64(x, 'occlusion')
  const d = xv.length
  const shape = options.shape ?? [d]
  if (shape.reduce((a, b) => a * b, 1) !== d)
    throw new ShapeError('occlusion', `occlusion: shape [${shape.join(', ')}] does not hold ${d} features`)
  const window = options.window ?? shape.map(() => 1)
  const strides = options.strides ?? shape.map(() => 1)
  if (window.length !== shape.length || strides.length !== shape.length)
    throw new ShapeError('occlusion', 'occlusion: window and strides need one entry per axis')
  const base =
    typeof options.baseline === 'number' || options.baseline === undefined
      ? new Float64Array(d).fill(options.baseline ?? 0)
      : Float64Array.from(dense.toF64(options.baseline, 'occlusion'))
  // Window starts per axis: k·stride for k = 0 … ⌈(size − window)/stride⌉.
  const starts = shape.map((size, a) => {
    if (!(window[a] >= 1 && window[a] <= size && strides[a] >= 1))
      throw new DomainError('occlusion', 'occlusion: each window must lie in 1 … size and each stride be ≥ 1')
    const count = Math.ceil((size - window[a]) / strides[a]) + 1
    return Array.from({ length: count }, (_, k) => k * strides[a])
  })
  const positions: number[][] = [[]]
  for (const s of starts) positions.splice(0, positions.length, ...positions.flatMap((p) => s.map((v) => [...p, v])))
  const rowStride = shape.map((_, a) => shape.slice(a + 1).reduce((u, v) => u * v, 1))
  const masks = positions.map((start) => {
    const cells: number[] = []
    const visit = (a: number, offset: number) => {
      if (a === shape.length) return void cells.push(offset)
      for (let k = start[a]; k < Math.min(shape[a], start[a] + window[a]); k++) visit(a + 1, offset + k * rowStride[a])
    }
    visit(0, 0)
    return cells
  })
  const w = masks.length
  const rows = new Float64Array((w + 1) * d)
  rows.set(xv, 0)
  masks.forEach((cells, k) => {
    rows.set(xv, (k + 1) * d)
    for (const c of cells) rows[(k + 1) * d + c] = base[c]
  })
  const out = evaluate(model, rows, w + 1, d)
  const total = new Float64Array(d)
  const count = new Float64Array(d)
  masks.forEach((cells, k) => {
    for (const c of cells) {
      total[c] += out[0] - out[k + 1]
      count[c] += 1
    }
  })
  return { values: Float64Array.from(total, (t, i) => (count[i] > 0 ? t / count[i] : 0)), output: out[0], windows: w }
}

// ── DeepLIFT and DeepSHAP ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * DeepLIFT (rescale rule) attributions of one output of a `DenseNetwork` at $\xvec$ against a reference $\xvec'$ (see
 * the file comment): one forward pass on the instance and the reference, then one backward pass of multipliers.
 * Throws `ShapeError` when the baseline's length differs from $\xvec$'s (and as `denseForward`).
 *
 * @param net The network.
 * @param x The instance $\xvec$ ($d$ values).
 * @param options The reference and the output explained.
 * @param options.baseline The reference $\xvec'$ ($d$ values; default all zeros).
 * @param options.output The output unit explained (default 0).
 * @returns `values`, the attributions $(\xvec - \xvec') \odot \mvec$; `multipliers`, $\mvec$, the contribution per unit
 *   of input difference; `output` and `baselineOutput`, $f(\xvec)$ and $f(\xvec')$; and `delta`,
 *   $\sum_i \text{values}_i - (f(\xvec) - f(\xvec'))$, zero up to rounding.
 *
 * @example A 2-2-1 ReLU network: the attributions add up to the change in output
 * const net = { weights: [[[1, -1], [1, 1]], [[1], [2]]], biases: [[0, 0], [0]], activation: 'relu' }
 * print(deepLift(net, [1, 2]))
 *
 * @example Across a ReLU's kink DeepLIFT keeps summation-to-delta, gradient times input does not
 * // f(x) = relu(x - 1): f(2) - f(0) = 1.
 * const net = { weights: [[[1]], [[1]]], biases: [[-1], [0]], activation: 'relu' }
 * print('deepLift =', deepLift(net, [2]).values)
 * print('gradient times input =', inputGradient(denseFunction(net), [2]).timesInput)
 */
export function deepLift(
  net: DenseNetwork,
  x: VectorLike,
  options: { baseline?: VectorLike; output?: Size } = {},
): { values: Float64Array; multipliers: Float64Array; output: number; baselineOutput: number; delta: number } {
  const xv = dense.toF64(x, 'deepLift')
  const d = xv.length
  const ref = options.baseline ? dense.toF64(options.baseline, 'deepLift') : new Float64Array(d)
  if (ref.length !== d) throw new ShapeError('deepLift', 'deepLift: the baseline must match x')
  const layers = denseLayers(net)
  const both = new Float64Array(2 * d)
  both.set(xv, 0)
  both.set(ref, d)
  const f = denseForward(net, fromData(both, [2, d]))
  const L = layers.length
  const k = options.output ?? 0
  // Multipliers with respect to the last layer's output: 1 on the explained unit.
  let m = new Float64Array(layers[L - 1].outputs)
  m[k] = 1
  for (let l = L - 1; l >= 0; l--) {
    const { W, inputs, outputs } = layers[l]
    // m holds ∂out/∂z_l (z_l the layer's pre-activation); pass back through W to the layer's input a_{l−1}.
    const back = new Float64Array(inputs)
    for (let i = 0; i < inputs; i++) {
      let s = 0
      for (let j = 0; j < outputs; j++) s += W[i * outputs + j] * m[j]
      back[i] = s
    }
    if (l === 0) {
      m = back
      break
    }
    // Rescale through layer l − 1's nonlinearity: Δa/Δz, or the derivative where Δz is tiny.
    const z = f.pre[l - 1]
    const h = inputs
    m = Float64Array.from(back, (b, i) => {
      const zx = z[i]
      const zr = z[h + i]
      const dz = zx - zr
      const r =
        Math.abs(dz) > 1e-10
          ? (activate(net.activation, zx) - activate(net.activation, zr)) / dz
          : activateDerivative(net.activation, zx)
      return b * r
    })
  }
  const out = f.post[L]
  const K = layers[L - 1].outputs
  const values = Float64Array.from(m, (mi, i) => mi * (xv[i] - ref[i]))
  const output = out[k]
  const baselineOutput = out[K + k]
  const delta = values.reduce((a, b) => a + b, 0) - (output - baselineOutput)
  return { values, multipliers: m, output, baselineOutput, delta }
}

/**
 * DeepSHAP: DeepLIFT attributions of $\xvec$ averaged over the background rows as references. One `deepLift` call
 * per background row.
 *
 * @param net The network.
 * @param x The instance $\xvec$ ($d$ values).
 * @param background The reference rows ($b \times d$).
 * @param options The output explained.
 * @param options.output The output unit explained (default 0).
 * @returns `values`, the mean attributions ($d$); `base`, the mean output over the background; and `output`,
 *   $f(\xvec)$; $\text{base} + \sum_i \phi_i = \text{output}$ up to rounding.
 *
 * @example A linear network: weight times (x minus the background mean)
 * const net = { weights: [[[2], [-1], [0.5]]], biases: [[0]], activation: 'relu' }
 * const r = deepShap(net, [1, 2, 3], [[0, 0, 0], [2, 0, 2]])
 * print(r)
 * print('w (x - mean b) =', [2 * (1 - 1), -1 * (2 - 0), 0.5 * (3 - 1)])
 */
export function deepShap(
  net: DenseNetwork,
  x: VectorLike,
  background: MatrixLike,
  options: { output?: Size } = {},
): { values: Float64Array; base: number; output: number } {
  const bg = dense.toMatrixF64(background, 'deepShap')
  const xv = dense.toF64(x, 'deepShap')
  const values = new Float64Array(xv.length)
  let base = 0
  let output = 0
  for (let r = 0; r < bg.m; r++) {
    const res = deepLift(net, xv, { baseline: bg.data.subarray(r * bg.n, (r + 1) * bg.n), output: options.output })
    for (let i = 0; i < values.length; i++) values[i] += res.values[i] / bg.m
    base += res.baselineOutput / bg.m
    output = res.output
  }
  return { values, base, output }
}

// ── Expected gradients ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Expected gradients of $f$ at $\xvec$: the mean over draws of a background row $\xvec'$ (uniformly) and
 * $\alpha \sim \Unif(0, 1)$ of $(\xvec - \xvec') \odot \nabla f(\xvec' + \alpha(\xvec - \xvec'))$, all gradients in
 * one `vmap(grad(f))` batch. Its attributions sum, in expectation, to $f(\xvec) - \expect f(\xvec')$. Throws
 * `ShapeError` when the background does not have $d$ columns.
 *
 * @param f The function explained.
 * @param x The instance $\xvec$ ($d$ values).
 * @param background The rows the baselines are drawn from ($b \times d$).
 * @param stream The random stream: rows are drawn from `child(stream, 'rows')` and $\alpha$ from
 *   `child(stream, 'alpha')`, so `stream` itself is not advanced.
 * @param options The number of draws.
 * @param options.samples The number of (row, $\alpha$) draws averaged (default 200).
 * @returns `values`, the attributions ($d$), and `samples`, the number of draws.
 *
 * @example A linear function: close to weight times (x minus the background mean)
 * const f = (x) => sum(mul(tensor([2, -1, 0.5]), x))
 * const r = expectedGradients(f, [1, 2, 3], [[0, 0, 0], [2, 0, 2]], stream(0), { samples: 400 })
 * print('values =', r.values)
 * print('w (x - mean b) =', [2 * (1 - 1), -1 * (2 - 0), 0.5 * (3 - 1)])
 */
export function expectedGradients(
  f: Differentiable,
  x: VectorLike,
  background: MatrixLike,
  stream: Stream,
  options: { samples?: Size } = {},
): { values: Float64Array; samples: Size } {
  const xv = dense.toF64(x, 'expectedGradients')
  const d = xv.length
  const bg = dense.toMatrixF64(background, 'expectedGradients')
  if (bg.n !== d) throw new ShapeError('expectedGradients', 'expectedGradients: the background must have d columns')
  const m = options.samples ?? 200
  const pick = child(stream, 'rows')
  const at = child(stream, 'alpha')
  const rows = new Float64Array(m * d)
  const diff = new Float64Array(m * d)
  for (let k = 0; k < m; k++) {
    const r = Math.min(bg.m - 1, Math.floor((uniform(pick) as number) * bg.m))
    const a = uniform(at) as number
    for (let i = 0; i < d; i++) {
      const b = bg.data[r * d + i]
      diff[k * d + i] = xv[i] - b
      rows[k * d + i] = b + a * (xv[i] - b)
    }
  }
  const G = toFlat(vmap(grad(f))(fromData(rows, [m, d])) as Tensor)
  const values = new Float64Array(d)
  for (let k = 0; k < m; k++) for (let i = 0; i < d; i++) values[i] += (diff[k * d + i] * G[k * d + i]) / m
  return { values, samples: m }
}

// ── Shapley interaction values ───────────────────────────────────────────────────────────────────────────────────────

/**
 * Exact Shapley interaction values of a set function over $2 \le d \le 16$ players, by enumerating all $2^d$
 * coalitions: for $i \ne j$,
 * $\Phi_{ij} = \sum_{S \subseteq N \setminus \{i, j\}} \frac{\lvert S \rvert!\,(d - \lvert S \rvert - 2)!}{2(d - 1)!}
 * [v(S \cup \{i, j\}) - v(S \cup \{i\}) - v(S \cup \{j\}) + v(S)]$, and the main effect
 * $\Phi_{ii} = \phi_i - \sum_{j \ne i} \Phi_{ij}$, so each row sums to the Shapley value $\phi_i$ and the whole
 * matrix to $v(N) - v(\emptyset)$. Throws `DomainError` when $d$ is not an integer in $2, \dots, 16$.
 *
 * @param value The set function $v$: receives a coalition as a membership mask of $d$ booleans and returns its worth.
 *   Each coalition is evaluated once.
 * @param d The number of players.
 * @returns `values`, the $d \times d$ matrix $\Phi$ row-major; `shapley`, the Shapley values; `base`,
 *   $v(\emptyset)$; and `output`, $v(N)$.
 *
 * @example An AND of two players plus a third that acts alone
 * const v = (mask) => (mask[0] && mask[1] ? 1 : 0) + (mask[2] ? 1 : 0)
 * const r = shapleyInteractions(v, 3)
 * print('Phi =', [0, 1, 2].map((i) => Array.from(r.values.slice(3 * i, 3 * i + 3))))
 * print('shapley =', r.shapley)
 */
export function shapleyInteractions(
  value: (mask: readonly boolean[]) => number,
  d: Size,
): { values: Float64Array; shapley: Float64Array; base: number; output: number } {
  if (!(Number.isInteger(d) && d >= 2 && d <= 16))
    throw new DomainError('shapleyInteractions', `shapleyInteractions: d must lie in 2 … 16, got ${d}`)
  const total = 1 << d
  const v = new Float64Array(total)
  for (let c = 0; c < total; c++) v[c] = value(Array.from({ length: d }, (_, i) => ((c >> i) & 1) === 1))
  const fact = [1]
  for (let k = 1; k <= d; k++) fact.push(fact[k - 1] * k)
  const weight = Float64Array.from({ length: d - 1 }, (_, s) => (fact[s] * fact[d - s - 2]) / (2 * fact[d - 1]))
  const phi = new Float64Array(d * d)
  for (let c = 0; c < total; c++) {
    let size = 0
    for (let i = 0; i < d; i++) size += (c >> i) & 1
    if (size > d - 2) continue
    for (let i = 0; i < d; i++) {
      if ((c >> i) & 1) continue
      for (let j = i + 1; j < d; j++) {
        if ((c >> j) & 1) continue
        const delta = v[c | (1 << i) | (1 << j)] - v[c | (1 << i)] - v[c | (1 << j)] + v[c]
        phi[i * d + j] += weight[size] * delta
      }
    }
  }
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) phi[j * d + i] = phi[i * d + j]
  const shap = exactShapley((mask) => v[mask.reduce((a, b, i) => a | ((b ? 1 : 0) << i), 0)], d)
  for (let i = 0; i < d; i++) {
    let off = 0
    for (let j = 0; j < d; j++) if (j !== i) off += phi[i * d + j]
    phi[i * d + i] = shap.values[i] - off
  }
  return { values: phi, shapley: shap.values, base: shap.base, output: shap.output }
}

// ── KernelSHAP's sampling spread ─────────────────────────────────────────────────────────────────────────────────────

/**
 * KernelSHAP's sampling spread: `kernelShap` repeated on independent child streams, with the mean and standard
 * deviation of its estimates per feature (denominator $\text{repeats} - 1$). When the coalition budget enumerates
 * every coalition, the estimates agree and the deviation is 0. Throws `DomainError` for fewer than two repeats.
 *
 * @param model The model, as for `kernelShap`.
 * @param x The instance $\xvec$ ($d$ values).
 * @param background The background rows ($b \times d$).
 * @param stream The random stream: repeat $r$ samples from `child(stream, 'repeat', r)`.
 * @param options The coalitions per estimate and the number of estimates.
 * @param options.samples The coalitions per estimate, as `kernelShap`'s `samples` (default $2d + 2048$).
 * @param options.repeats The number of estimates (default 20).
 * @returns `estimates`, every estimate ($\text{repeats} \times d$), and their `mean` and `sd` per feature.
 *
 * @example How far 30 sampled coalitions stray
 * // Features 0 to 2 act only together, features 3 to 7 add up: exact values 1/3, 1/3, 1/3, then 1 each.
 * const model = (X) => toArray(X).map((r) => r[0] * r[1] * r[2] + r.slice(3).reduce((a, b) => a + b))
 * const background = [[0, 0, 0, 0, 0, 0, 0, 0]]
 * const r = kernelShapVariance(model, [1, 1, 1, 1, 1, 1, 1, 1], background, stream(0), { samples: 30, repeats: 10 })
 * print('mean =', r.mean)
 * print('sd =', r.sd)
 */
export function kernelShapVariance(
  model: ScalarModel,
  x: VectorLike,
  background: MatrixLike,
  stream: Stream,
  options: Omit<KernelShapOptions, 'stream'> & { repeats?: Size } = {},
): { estimates: Tensor; mean: Float64Array; sd: Float64Array } {
  const { repeats = 20, samples } = options
  if (!(repeats >= 2)) throw new DomainError('kernelShapVariance', 'kernelShapVariance: needs at least two repeats')
  const runs = Array.from({ length: repeats }, (_, r) =>
    kernelShap(model, x, background, { samples, stream: child(stream, 'repeat', r) }),
  )
  const d = runs[0].values.length
  const mean = new Float64Array(d)
  for (const r of runs) for (let i = 0; i < d; i++) mean[i] += r.values[i] / repeats
  const sd = new Float64Array(d)
  for (const r of runs) for (let i = 0; i < d; i++) sd[i] += (r.values[i] - mean[i]) ** 2 / (repeats - 1)
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i])
  const flat = new Float64Array(repeats * d)
  runs.forEach((r, k) => flat.set(r.values, k * d))
  return { estimates: fromData(flat, [repeats, d]), mean, sd }
}
