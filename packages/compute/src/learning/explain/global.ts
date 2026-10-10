/**
 * Global effects and interactions of a model $f$ over data:
 *
 * - `accumulatedLocalEffects` (Apley and Zhu, 2020): split feature $j$ at its quantiles; in each bin average, over the
 *   rows that fall in it, the change of $f$ as $x_j$ moves from the bin's lower to upper edge with the other features
 *   kept; accumulate the averages and centre them (as the ALEPlot package). Unlike partial dependence it never
 *   evaluates $f$ at combinations the data do not contain, so it stays faithful when features are correlated.
 * - `hStatistic` (Friedman and Popescu, 2008): the share of the variance of the joint partial dependence of features
 *   $j$ and $k$ that their separate partial dependences do not explain,
 *   $H^2_{jk} = \sum_i [\text{PD}_{jk} - \text{PD}_j - \text{PD}_k]^2 / \sum_i \text{PD}_{jk}^2$ at the data rows
 *   (each centred), and the overall statistic
 *   $H^2_j = \sum_i [f - \text{PD}_j - \text{PD}_{-j}]^2 / \sum_i f^2$ (also centred), the share of $f$'s variance
 *   from interactions involving $j$. 0 for an additive effect; 1 when the effect is pure interaction.
 * - `functionalAnova` (Hoeffding, 1948; Sobol, 1993; Hooker, 2004):
 *   $f = f_0 + \sum_j f_j + \sum_{j<k} f_{jk} + \dots$ with each term averaging to zero over each of its features,
 *   under a product measure given by a grid and weights per feature. Evaluates $f$ on the full grid, so it is exact
 *   for that measure; the variances of the terms over their total are Sobol indices. For a GAM with pairwise terms it
 *   recovers ("purifies") the main effects and interactions.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
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
 * The accumulated local effect of a feature over the rows of $\Xmat$ (see the file comment), in one model call on
 * $2n$ rows. The first bin is $[z_0, z_1]$ and the others $(z_{k-1}, z_k]$. Throws `DomainError` when `feature` is not
 * a column of $\Xmat$ or the feature is constant.
 *
 * @param model The model, called on the rows with the feature set to their bin's lower and upper edges.
 * @param X The data ($n \times d$); not modified.
 * @param feature The column whose effect is measured.
 * @param options The binning.
 * @param options.bins The number of quantile bins asked for (default 20); fewer remain when quantiles coincide.
 * @returns `edges`, the $K + 1$ bin edges $z_k$ (the minimum, then the $k/K$ quantiles by inverting the empirical
 *   distribution, duplicates removed); `effect`, the centred accumulated effect at each edge; `counts`, the rows per
 *   bin; and `local`, each bin's mean change in output ($K$ each).
 *
 * @example The effect of a square, centred
 * const model = (X) => toArray(X).map(([a, b]) => a * a + b)
 * const r = accumulatedLocalEffects(model, [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], 0, { bins: 4 })
 * print('edges =', r.edges)
 * print('local =', r.local, ' counts =', r.counts)
 * print('effect =', r.effect)
 */
export function accumulatedLocalEffects(
  model: ScalarModel,
  X: MatrixLike,
  feature: Size,
  options: { bins?: Size } = {},
): { edges: Float64Array; effect: Float64Array; counts: Float64Array; local: Float64Array } {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'accumulatedLocalEffects')
  if (!(feature >= 0 && feature < d))
    throw new DomainError('accumulatedLocalEffects', `accumulatedLocalEffects: feature ${feature} out of range`)
  const K0 = options.bins ?? 20
  const col = Array.from({ length: n }, (_, i) => data[i * d + feature])
  const sorted = [...col].sort((a, b) => a - b)
  // Type-1 quantile: the smallest order statistic whose ECDF reaches p.
  const q1 = (p: number) => sorted[Math.max(0, Math.ceil(p * n - 1e-12) - 1)]
  const edges = Float64Array.from(new Set([sorted[0], ...Array.from({ length: K0 }, (_, k) => q1((k + 1) / K0))]))
  const K = edges.length - 1
  if (K < 1) throw new DomainError('accumulatedLocalEffects', 'accumulatedLocalEffects: the feature is constant')
  // Bin of each row: [z₀, z₁] for the first, (z_{k−1}, z_k] after.
  const bin = Int32Array.from(col, (v) => {
    let k = 0
    while (k < K - 1 && v > edges[k + 1]) k++
    return k
  })
  const rows = new Float64Array(2 * n * d)
  for (let i = 0; i < n; i++) {
    rows.set(data.subarray(i * d, (i + 1) * d), i * d)
    rows.set(data.subarray(i * d, (i + 1) * d), (n + i) * d)
    rows[i * d + feature] = edges[bin[i]]
    rows[(n + i) * d + feature] = edges[bin[i] + 1]
  }
  const out = evaluate(model, rows, 2 * n, d)
  const local = new Float64Array(K)
  const counts = new Float64Array(K)
  for (let i = 0; i < n; i++) {
    local[bin[i]] += out[n + i] - out[i]
    counts[bin[i]]++
  }
  for (let k = 0; k < K; k++) local[k] = counts[k] > 0 ? local[k] / counts[k] : 0
  const effect = new Float64Array(K + 1)
  for (let k = 0; k < K; k++) effect[k + 1] = effect[k] + local[k]
  let centre = 0
  for (let k = 0; k < K; k++) centre += ((effect[k] + effect[k + 1]) / 2) * counts[k]
  centre /= n
  for (let k = 0; k <= K; k++) effect[k] -= centre
  return { edges, effect, counts, local }
}

/**
 * Centred partial dependence on the features in `set`, evaluated at each of the rows: one model call on $n^2$ rows.
 *
 * @param model The model.
 * @param data The rows, row-major ($n \times d$ values); not modified.
 * @param n The number of rows.
 * @param d The number of features.
 * @param set The features held at each row's values while the others range over the data.
 * @returns The partial dependence at each row minus its mean ($n$ values).
 */
function centredDependence(model: ScalarModel, data: Float64Array, n: Size, d: Size, set: readonly number[]) {
  const rows = new Float64Array(n * n * d)
  for (let i = 0; i < n; i++)
    for (let l = 0; l < n; l++) {
      const o = (i * n + l) * d
      rows.set(data.subarray(l * d, (l + 1) * d), o)
      for (const j of set) rows[o + j] = data[i * d + j]
    }
  const out = evaluate(model, rows, n * n, d)
  const pd = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let l = 0; l < n; l++) s += out[i * n + l]
    pd[i] = s / n
  }
  const mean = pd.reduce((a, b) => a + b, 0) / n
  return pd.map((v) => v - mean)
}

/**
 * Friedman's H-statistics on the rows of $\Xmat$: pairwise and overall, the square roots of the $H^2$ of the file
 * comment. Each partial dependence sweep costs $n^2$ model evaluations, so pass a subsample.
 *
 * @param model The model.
 * @param X The rows ($n \times d$) the partial dependences are computed over.
 * @param options The features examined.
 * @param options.features The features whose statistics are computed (default all); the others' entries stay 0.
 * @returns `pairwise`, $H_{jk}$ as a $d \times d$ matrix row-major (symmetric, the diagonal 0); `overall`, $H_j$ ($d$
 *   values); and `features`, the features examined.
 *
 * @example Features 0 and 1 interact; feature 2 adds on
 * const model = (X) => toArray(X).map(([a, b, c]) => a * b + c)
 * const r = hStatistic(model, [[-1, -1, 0], [-1, 1, 1], [1, -1, 2], [1, 1, 3]])
 * print('H01 =', r.pairwise[1], ' H02 =', r.pairwise[2], ' H12 =', r.pairwise[5])
 * print('overall =', r.overall)
 */
export function hStatistic(
  model: ScalarModel,
  X: MatrixLike,
  options: { features?: readonly Size[] } = {},
): { pairwise: Float64Array; overall: Float64Array; features: Size[] } {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'hStatistic')
  const features = [...(options.features ?? Array.from({ length: d }, (_, j) => j))]
  const single = new Map<number, Float64Array>()
  for (const j of features) single.set(j, centredDependence(model, data, n, d, [j]))
  const f = evaluate(model, Float64Array.from(data), n, d)
  const fm = f.reduce((a, b) => a + b, 0) / n
  const fc = f.map((v) => v - fm)
  const pairwise = new Float64Array(d * d)
  for (let a = 0; a < features.length; a++)
    for (let b = a + 1; b < features.length; b++) {
      const [j, k] = [features[a], features[b]]
      const both = centredDependence(model, data, n, d, [j, k])
      const pj = single.get(j) as Float64Array
      const pk = single.get(k) as Float64Array
      let num = 0
      let den = 0
      for (let i = 0; i < n; i++) {
        num += (both[i] - pj[i] - pk[i]) ** 2
        den += both[i] ** 2
      }
      const h = den > 0 ? Math.sqrt(num / den) : 0
      pairwise[j * d + k] = h
      pairwise[k * d + j] = h
    }
  let den = 0
  for (let i = 0; i < n; i++) den += fc[i] ** 2
  const overall = new Float64Array(d)
  for (const j of features) {
    const rest = centredDependence(
      model,
      data,
      n,
      d,
      Array.from({ length: d }, (_, c) => c).filter((c) => c !== j),
    )
    const pj = single.get(j) as Float64Array
    let num = 0
    for (let i = 0; i < n; i++) num += (fc[i] - pj[i] - rest[i]) ** 2
    overall[j] = den > 0 ? Math.sqrt(num / den) : 0
  }
  return { pairwise, overall, features }
}

/** The functional ANOVA decomposition of `functionalAnova`. */
export type FunctionalAnova = {
  /** $f_0 = \expect f$. */
  mean: number
  /** The grid of each feature. */
  grids: Float64Array[]
  /** The weights of each feature's grid points, normalised to sum to 1. */
  weights: Float64Array[]
  /** The main effect $f_j$ of each feature, on its grid. */
  main: Float64Array[]
  /**
   * The pairwise terms $f_{jk}$, $j < k$, in the order $(0, 1), (0, 2), \dots, (1, 2), \dots$: each with its
   * `features` and its `values` on grid $j$ times grid $k$, row-major ($g_j \times g_k$).
   */
  pairs: { features: [Size, Size]; values: Float64Array }[]
  /** $\var f$. */
  variance: number
  /** $\var f_j$ for each feature. */
  mainVariance: Float64Array
  /** $\var f_{jk}$ for each pair, in the order of `pairs`. */
  pairVariance: Float64Array
  /** The remainder, $\var f$ minus the main and pairwise variances: the variance of the higher-order terms. */
  higherVariance: number
}

/**
 * The functional ANOVA of `model` over the product of per-feature grids with weights: main effects and pairwise terms,
 * and their variances. Evaluates $f$ on all $\prod_j g_j$ grid points, in calls of at most 20000 rows. Throws
 * `DomainError` when there are more than $2 \times 10^6$ grid points.
 *
 * @param model The model.
 * @param grids The grid of each feature ($g_j$ values for feature $j$); their number is the model's $d$.
 * @param options The measure on each grid.
 * @param options.weights The weight of each grid point, per feature (default equal); normalised to sum to 1.
 * @returns The decomposition.
 *
 * @example Main effects, an interaction and a three-way term, each with its variance
 * const model = (X) => toArray(X).map(([a, b, c]) => a + 2 * b + a * b + a * b * c)
 * const r = functionalAnova(model, [[-1, 1], [-1, 1], [-1, 1]])
 * print('mean =', r.mean, ' main =', r.main)
 * print('pairs =', r.pairs.map((p) => [p.features, Array.from(p.values)]))
 * print('variance =', r.variance, ' main =', r.mainVariance, ' pairs =', r.pairVariance, ' higher =', r.higherVariance)
 */
export function functionalAnova(
  model: ScalarModel,
  grids: readonly VectorLike[],
  options: { weights?: readonly VectorLike[] } = {},
): FunctionalAnova {
  const d = grids.length
  const G = grids.map((g) => Float64Array.from(dense.toF64(g, 'functionalAnova')))
  const W = G.map((g, j) => {
    const w = options.weights
      ? Float64Array.from(dense.toF64(options.weights[j], 'functionalAnova'))
      : new Float64Array(g.length).fill(1)
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })
  const sizes = G.map((g) => g.length)
  const cells = sizes.reduce((a, b) => a * b, 1)
  if (!(cells <= 2e6)) throw new DomainError('functionalAnova', `functionalAnova: ${cells} grid points exceed 2 × 10⁶`)
  // Evaluate f over the grid in chunks; cell c has mixed-radix indices (last feature fastest).
  const f = new Float64Array(cells)
  const index = (c: number, out: Int32Array) => {
    for (let j = d - 1; j >= 0; j--) {
      out[j] = c % sizes[j]
      c = Math.floor(c / sizes[j])
    }
  }
  const idx = new Int32Array(d)
  const chunk = 20000
  for (let start = 0; start < cells; start += chunk) {
    const m = Math.min(chunk, cells - start)
    const rows = new Float64Array(m * d)
    for (let r = 0; r < m; r++) {
      index(start + r, idx)
      for (let j = 0; j < d; j++) rows[r * d + j] = G[j][idx[j]]
    }
    f.set(evaluate(model, rows, m, d), start)
  }
  const pairsList: [Size, Size][] = []
  for (let j = 0; j < d; j++) for (let k = j + 1; k < d; k++) pairsList.push([j, k])
  const condMain = G.map((g) => new Float64Array(g.length))
  const condPair = pairsList.map(([j, k]) => new Float64Array(sizes[j] * sizes[k]))
  let f0 = 0
  for (let c = 0; c < cells; c++) {
    index(c, idx)
    let w = 1
    for (let j = 0; j < d; j++) w *= W[j][idx[j]]
    const wf = w * f[c]
    f0 += wf
    for (let j = 0; j < d; j++) condMain[j][idx[j]] += wf
    pairsList.forEach(([j, k], p) => (condPair[p][idx[j] * sizes[k] + idx[k]] += wf))
  }
  const main = condMain.map((s, j) => Float64Array.from(s, (v, a) => v / W[j][a] - f0))
  const pairs = pairsList.map(([j, k], p) => ({
    features: [j, k] as [Size, Size],
    values: Float64Array.from(condPair[p], (v, ab) => {
      const a = Math.floor(ab / sizes[k])
      const b = ab % sizes[k]
      return v / (W[j][a] * W[k][b]) - main[j][a] - main[k][b] - f0
    }),
  }))
  let variance = 0
  for (let c = 0; c < cells; c++) {
    index(c, idx)
    let w = 1
    for (let j = 0; j < d; j++) w *= W[j][idx[j]]
    variance += w * (f[c] - f0) ** 2
  }
  const mainVariance = Float64Array.from(main, (m, j) => m.reduce((a, v, i) => a + W[j][i] * v * v, 0))
  const pairVariance = Float64Array.from(pairs, ({ features: [j, k], values }) =>
    values.reduce((a, v, ab) => a + W[j][Math.floor(ab / sizes[k])] * W[k][ab % sizes[k]] * v * v, 0),
  )
  const higherVariance = variance - mainVariance.reduce((a, b) => a + b, 0) - pairVariance.reduce((a, b) => a + b, 0)
  return { mean: f0, grids: G, weights: W, main, pairs, variance, mainVariance, pairVariance, higherVariance }
}
