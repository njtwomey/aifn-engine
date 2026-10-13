/**
 * Functional ANOVA of a surface tabulated on a full grid over $d$ hyperparameters, the decomposition behind the
 * hyperparameter importance of Hutter, Hoos and Leyton-Brown (2014), computed exactly on the grid rather than from a
 * random forest fitted to it.
 *
 * With a product measure $w(\lambda) = \prod_j w_j(\lambda_j)$ over the grid, the table $f$ is the sum of orthogonal
 * terms, one for each subset $U$ of the axes,
 *
 * $$ f(\lambda) = \sum_{U \subseteq \{1, \dots, d\}} f_U(\lambda_U), \qquad f_\emptyset = \expect f, $$
 *
 * where each $f_U$ is found from the marginals $m_V(\lambda_V) = \expect[f \mid \lambda_V]$ (the weighted mean of $f$
 * over the axes outside $V$) by inclusion and exclusion,
 * $f_U = \sum_{V \subseteq U} (-1)^{\lvert U \rvert - \lvert V \rvert} m_V$. The terms are uncorrelated, so the
 * variance splits as $\var f = \sum_{U \ne \emptyset} \var f_U$, and the importance of $U$ is its fraction
 * $\var f_U / \var f$ (Hutter et al., 2014, equations 2 to 4). The marginal $m_{\{i\}}$ is the curve their paper
 * plots for a single hyperparameter.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, tensor, toFlat, type NestedArray, type Tensor } from 'aifn-compute/foundation/tensor'

/** Options of `fanova`. */
export interface FanovaOptions {
  /** A name for each axis, for labelling the terms (default `'x0'`, `'x1'`, ...). */
  axes?: readonly string[]
  /**
   * The highest order of interaction returned (default 2: the main effects and the pairs). With $d$ every term is
   * returned and they sum back to the table exactly; the variance of the terms left out is `higher`.
   */
  order?: number
  /**
   * The weight of each grid value, per axis (default equal): the product measure the decomposition is taken under.
   * Each axis's weights are non-negative with a positive sum, and are normalised to sum to 1.
   */
  weights?: readonly (readonly number[])[]
}

/** One term $f_U$ of the decomposition. */
export interface FanovaTerm {
  /** The axes $U$ of the term, in increasing order. */
  axes: number[]
  /** The names of those axes. */
  names: string[]
  /** $f_U$ on the grid of its axes: a tensor with the sizes of those axes as its shape. */
  values: Tensor
  /**
   * The marginal $m_U = \expect[f \mid \lambda_U]$ on the same grid: the mean of the table over the other axes, which
   * is $f_\emptyset$ plus every term on a subset of $U$.
   */
  marginal: Tensor
  /** $\var f_U$, under the weights. */
  variance: number
  /** The fraction $\var f_U / \var f$ of the total variance (0 when the table is constant). */
  fraction: number
}

/** The decomposition `fanova` returns. */
export interface Fanova {
  /** $f_\emptyset = \expect f$, the weighted mean of the table. */
  mean: number
  /** $\var f$, the weighted variance of the table. */
  variance: number
  /** The shape of the table, $(n_1, \dots, n_d)$. */
  shape: number[]
  /** The names of the axes. */
  names: string[]
  /** The normalised weights of each axis. */
  weights: Float64Array[]
  /** Every term up to the requested order, by order and then in lexicographic order of the axes. */
  terms: FanovaTerm[]
  /** The terms of order 1, one per axis: the main effects $f_i$. */
  main: FanovaTerm[]
  /** The terms of order 2, $f_{ij}$ for $i < j$, in the order $(0, 1), (0, 2), \dots, (1, 2), \dots$. */
  pairs: FanovaTerm[]
  /** The fraction of the variance the returned terms explain. */
  explained: number
  /** The fraction left to interactions above the requested order, $1 - \text{explained}$ (0 at order $d$). */
  higher: number
}

/**
 * The set bits of a mask, in increasing order: the axes of a subset.
 *
 * @param mask The subset, bit $j$ set for axis $j$.
 * @param d The number of axes.
 * @returns The axes in the subset.
 */
const bitsOf = (mask: number, d: number): number[] =>
  Array.from({ length: d }, (_, j) => j).filter((j) => mask & (1 << j))

/**
 * The exact functional ANOVA of a table (see the file comment): every term $f_U$ with $1 \le \lvert U \rvert \le$
 * `order`, its marginal, its variance and its fraction of the total. Throws `ShapeError` for a table of rank 0 or
 * weights of the wrong length, and `DomainError` for a table with more than $2 \times 10^6$ cells or more than 16
 * axes, a value that is not finite, negative weights or an order outside $1, \dots, d$.
 *
 * @param table The surface: a tensor or nested arrays of shape $(n_1, \dots, n_d)$, one axis per hyperparameter, so
 *   that `table[i][j]` is the value at the $i$-th value of the first and the $j$-th of the second.
 * @param options The axis names, the order and the weights, as `FanovaOptions`.
 * @returns The decomposition, as `Fanova`.
 *
 * @example An additive surface plus an interaction: the importance of each term
 * // f(a, b) = a + 2 b + a b on a and b in {-1, 0, 1}.
 * const v = [-1, 0, 1]
 * const table = v.map((a) => v.map((b) => a + 2 * b + a * b))
 * const r = fanova(table, { axes: ['a', 'b'] })
 * print('mean =', r.mean, ' variance =', r.variance)
 * for (const term of r.terms) print(term.names.join(' x '), 'fraction =', term.fraction.toFixed(4))
 *
 * @example The marginal of one hyperparameter, as Hutter et al. plot it
 * const v = [0, 1, 2, 3]
 * const table = v.map((a) => v.map((b) => (a - 2) ** 2 + 0.1 * b))
 * const r = fanova(table, { axes: ['C', 'gamma'] })
 * print('marginal over C:', r.main[0].marginal)
 * print('fractions:', r.main.map((t) => t.fraction))
 */
export function fanova(table: Tensor | NestedArray, options: FanovaOptions = {}): Fanova {
  const t = isTensor(table) ? table : tensor(table as NestedArray)
  const shape = [...t.shape]
  const d = shape.length
  if (d === 0) throw new ShapeError('fanova', 'fanova: the table needs at least one axis')
  if (d > 16) throw new DomainError('fanova', `fanova: ${d} axes; at most 16`)
  const cells = shape.reduce((a, b) => a * b, 1)
  if (cells > 2e6) throw new DomainError('fanova', `fanova: ${cells} cells exceed 2 × 10⁶`)
  const f = Float64Array.from(toFlat(t) as ArrayLike<number>)
  if (!f.every(Number.isFinite)) throw new DomainError('fanova', 'fanova: every value of the table must be finite')
  const order = options.order ?? Math.min(2, d)
  if (!(Number.isInteger(order) && order >= 1 && order <= d))
    throw new DomainError('fanova', `fanova: order must be an integer from 1 to ${d}`)
  const names = shape.map((_, j) => options.axes?.[j] ?? `x${j}`)
  if (options.axes && options.axes.length !== d)
    throw new ShapeError('fanova', `fanova: ${options.axes.length} names for ${d} axes`)
  if (options.weights && options.weights.length !== d)
    throw new ShapeError('fanova', `fanova: weights for ${options.weights.length} axes, the table has ${d}`)
  const W = shape.map((n, j) => {
    const raw = options.weights ? Float64Array.from(options.weights[j]) : new Float64Array(n).fill(1)
    if (raw.length !== n) throw new ShapeError('fanova', `fanova: ${raw.length} weights for axis ${j} of size ${n}`)
    const s = raw.reduce((a, b) => a + b, 0)
    if (!raw.every((w) => w >= 0) || !(s > 0))
      throw new DomainError('fanova', `fanova: the weights of axis ${j} must be non-negative with a positive sum`)
    return raw.map((w) => w / s)
  })

  // The subsets of size at most `order`, by size then lexicographically; the marginals of all of them (and of the
  // empty set) are needed for inclusion and exclusion.
  const masks: number[] = []
  for (let mask = 1; mask < 1 << d; mask++) if (bitsOf(mask, d).length <= order) masks.push(mask)
  masks.sort((a, b) => {
    const [x, y] = [bitsOf(a, d), bitsOf(b, d)]
    if (x.length !== y.length) return x.length - y.length
    for (let k = 0; k < x.length; k++) if (x[k] !== y[k]) return x[k] - y[k]
    return 0
  })
  const sizeOf = (axes: number[]) => axes.reduce((a, j) => a * shape[j], 1)
  // The flat index into a term's grid of the cell with multi-index `idx`, row-major over the term's axes.
  const termIndex = (axes: number[], idx: Int32Array) => axes.reduce((a, j) => a * shape[j] + idx[j], 0)

  // Accumulate every marginal in one pass: cell c adds w(c) f(c) / w_U(c_U) to m_U(c_U), the weighted mean over the
  // other axes, computed as sums so that zero weights are safe.
  const marginals = new Map<number, Float64Array>([[0, new Float64Array(1)]])
  for (const mask of masks) marginals.set(mask, new Float64Array(sizeOf(bitsOf(mask, d))))
  const axesOf = new Map<number, number[]>([[0, []]])
  for (const mask of masks) axesOf.set(mask, bitsOf(mask, d))
  const idx = new Int32Array(d)
  let variance = 0
  for (let c = 0; c < cells; c++) {
    let r = c
    for (let j = d - 1; j >= 0; j--) {
      idx[j] = r % shape[j]
      r = Math.floor(r / shape[j])
    }
    for (const [mask, m] of marginals) {
      let w = 1
      for (let j = 0; j < d; j++) if (!(mask & (1 << j))) w *= W[j][idx[j]]
      m[termIndex(axesOf.get(mask) as number[], idx)] += w * f[c]
    }
  }
  const mean = (marginals.get(0) as Float64Array)[0]
  for (let c = 0; c < cells; c++) {
    let r = c
    let w = 1
    for (let j = d - 1; j >= 0; j--) {
      w *= W[j][r % shape[j]]
      r = Math.floor(r / shape[j])
    }
    variance += w * (f[c] - mean) ** 2
  }

  // f_U = sum over V ⊆ U of (−1)^{|U|−|V|} m_V(λ_V), evaluated on U's grid.
  const terms: FanovaTerm[] = masks.map((mask) => {
    const axes = axesOf.get(mask) as number[]
    const size = sizeOf(axes)
    const values = new Float64Array(size)
    const local = new Int32Array(d)
    for (let k = 0; k < size; k++) {
      let r = k
      for (let a = axes.length - 1; a >= 0; a--) {
        local[axes[a]] = r % shape[axes[a]]
        r = Math.floor(r / shape[axes[a]])
      }
      let v = 0
      // Every subset V of U, as a sub-mask.
      for (let sub = mask; ; sub = (sub - 1) & mask) {
        const vAxes = axesOf.get(sub) as number[]
        const sign = (axes.length - vAxes.length) % 2 === 0 ? 1 : -1
        v += sign * (marginals.get(sub) as Float64Array)[termIndex(vAxes, local)]
        if (sub === 0) break
      }
      values[k] = v
    }
    let termVariance = 0
    for (let k = 0; k < size; k++) {
      let r = k
      let w = 1
      for (let a = axes.length - 1; a >= 0; a--) {
        w *= W[axes[a]][r % shape[axes[a]]]
        r = Math.floor(r / shape[axes[a]])
      }
      termVariance += w * values[k] ** 2
    }
    const termShape = axes.map((j) => shape[j])
    return {
      axes,
      names: axes.map((j) => names[j]),
      values: fromData(values, termShape),
      marginal: fromData(Float64Array.from(marginals.get(mask) as Float64Array), termShape),
      variance: termVariance,
      fraction: variance > 0 ? termVariance / variance : 0,
    }
  })
  const explained = variance > 0 ? terms.reduce((a, term) => a + term.fraction, 0) : 0
  return {
    mean,
    variance,
    shape,
    names,
    weights: W,
    terms,
    main: terms.filter((term) => term.axes.length === 1),
    pairs: terms.filter((term) => term.axes.length === 2),
    explained,
    higher: variance > 0 ? Math.max(0, 1 - explained) : 0,
  }
}

/**
 * The table rebuilt from a decomposition: $f_\emptyset$ plus every returned term of order at most `order`, broadcast
 * over the full grid. With every term (a decomposition taken at order $d$) it gives back the table; with the main
 * effects alone it is the best additive approximation under the weights.
 *
 * @param result A decomposition from `fanova`.
 * @param order The highest order of the terms added (default: every term in `result`).
 * @returns A tensor of the table's shape.
 *
 * @example Main effects alone, then every term
 * const v = [-1, 0, 1]
 * const table = v.map((a) => v.map((b) => a + 2 * b + a * b))
 * const r = fanova(table)
 * print('additive part:', fanovaReconstruct(r, 1))
 * print('all terms:', fanovaReconstruct(r))
 */
export function fanovaReconstruct(result: Fanova, order = Infinity): Tensor {
  const { shape } = result
  const d = shape.length
  const cells = shape.reduce((a, b) => a * b, 1)
  const out = new Float64Array(cells).fill(result.mean)
  const idx = new Int32Array(d)
  for (const term of result.terms) {
    if (term.axes.length > order) continue
    const values = toFlat(term.values) as ArrayLike<number>
    for (let c = 0; c < cells; c++) {
      let r = c
      for (let j = d - 1; j >= 0; j--) {
        idx[j] = r % shape[j]
        r = Math.floor(r / shape[j])
      }
      out[c] += values[term.axes.reduce((a, j) => a * shape[j] + idx[j], 0)]
    }
  }
  return fromData(out, shape)
}
