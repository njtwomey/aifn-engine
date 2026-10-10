/**
 * Exceptional model mining (Leman, Feelders and Knobbe, 2008; Duivesteijn, Feelders and Knobbe, 2016): subgroup
 * discovery where the target is a model over several columns, and a subgroup is interesting when the model fitted on
 * its rows differs from the model fitted on the rest (its complement). Each model class here is a quality measure over
 * covers, so the same refinement search runs it; none has a known optimistic estimate, so search is by beam or
 * exhaustive to a depth. Below, $G$ is the subgroup ($n$ rows), $\bar G$ its complement, and $N$ the rows in all.
 *
 * - **Correlation** of two numeric targets $x$ and $y$: `absolute` $\lvert \rho_G - \rho_{\bar G} \rvert$; `entropy`
 *   $H(n/N) \lvert \rho_G - \rho_{\bar G} \rvert$, which favours balanced splits ($H$ is the binary entropy in bits);
 *   `fisher-z` $\lvert z_G - z_{\bar G} \rvert / \sqrt{1/(n - 3) + 1/(N - n - 3)}$ with
 *   $z = \operatorname{atanh} \rho$, the test statistic of equal correlations (Leman et al., 2008, §4.1).
 * - **Regression** of $y$ on $x$: `slope-difference`
 *   $\lvert b_G - b_{\bar G} \rvert / \sqrt{\mathrm{se}_G^2 + \mathrm{se}_{\bar G}^2}$, the $t$ statistic of equal
 *   slopes (Leman et al., 2008, §4.2); `entropy` $H(n/N) \lvert b_G - b_{\bar G} \rvert$; `cook` Cook's distance of
 *   deleting the subgroup, $(\betavec - \betavec_{(G)})^\top \Xmat^\top \Xmat (\betavec - \betavec_{(G)}) / (p s^2)$
 *   with $\betavec$ fitted on all rows, $\betavec_{(G)}$ on the complement, $p = 2$ coefficients and $s^2$ the full
 *   fit's residual variance (Duivesteijn, Feelders and Knobbe, 2012, "Different slopes for different folks").
 * - **Classification**: a logistic regression of a 0/1 label on $x$. Fitted with an indicator of the subgroup and its
 *   interaction with $x$, the model separates into one logistic fit inside and one outside, so the interaction's Wald
 *   statistic is `wald` $\lvert b_G - b_{\bar G} \rvert / \sqrt{\var b_G + \var b_{\bar G}}$ (the significance of the
 *   effect difference; Leman et al., 2008, §4.3); `entropy` $H(n/N) \lvert b_G - b_{\bar G} \rvert$. The two fits
 *   are computed separately.
 * - **Association** of two 0/1 targets: the difference of Yule's
 *   $Q = (n_{11} n_{00} - n_{10} n_{01}) / (n_{11} n_{00} + n_{10} n_{01})$ (each cell $+1/2$) inside and outside,
 *   `absolute` or `entropy` weighted: the cheapest Bayesian-network model class, a single edge's strength
 *   (Duivesteijn, Knobbe, Feelders and van Leeuwen, 2010).
 *
 * A subgroup or complement too small to fit its model, or a fit that is not finite, has quality $-\infty$. Target
 * columns of different lengths, or empty ones, throw `DomainError`.
 */

import type { Tensor } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { binaryEntropy } from 'aifn-compute/numerics/special'
import { bitsetCount, bitsetIndices, type Bitset } from './cover'
import type { QualityMeasure } from './quality'

type Column = ArrayLike<number> | Tensor

/**
 * A column's values as a fresh `Float64Array`: a tensor flattened, an array copied.
 *
 * @param t The column.
 * @returns Its values in row order.
 */
const flat = (t: Column): Float64Array => Float64Array.from(isTensor(t) ? toFlat(t) : t)

/** A model fitted inside the subgroup, outside it (its complement), and on every row. */
export interface ModelFits<F> {
  /** The model fitted on the subgroup's rows. */
  readonly inside: F
  /** The model fitted on the other rows. */
  readonly outside: F
  /** The model fitted on every row. */
  readonly all: F
}

/** An exceptional-model measure: a quality over covers, and the fits it compares. */
export interface ModelMeasure<F> extends QualityMeasure {
  /** The model class: `correlation`, `regression`, `classification` or `association`. */
  readonly model: string
  /** The model fitted inside a cover, outside it and on every row. */
  fit(cover: Bitset): ModelFits<F>
}

/**
 * Binary entropy in bits, $H(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$ (0 at $p = 0$ and 1, and outside $(0, 1)$).
 *
 * @param p The share $n/N$ of rows in the subgroup.
 * @returns $H(p)$, at most 1, at $p = 1/2$.
 */
const entropyBits = (p: number): number => (p > 0 && p < 1 ? binaryEntropy(p, 2) : 0)

/**
 * Target columns as arrays, checked: throws `DomainError` when they differ in length or are empty.
 *
 * @param where The caller's name, for error messages.
 * @param cols The columns.
 * @returns Their values, in the same order.
 */
function columns(where: string, ...cols: Column[]): Float64Array[] {
  const out = cols.map(flat)
  if (out.some((c) => c.length !== out[0].length))
    throw new DomainError(where, `${where}: the target columns must have equal lengths`)
  if (!out[0].length) throw new DomainError(where, `${where}: the targets are empty`)
  return out
}

// ── Moments of a bivariate sample ───────────────────────────────────────────────────────────────────────────────────

/** First and second moments of (x, y) over a set of rows. */
export interface BivariateFit {
  /** The number of rows. */
  readonly n: number
  /** The mean $\bar x$. */
  readonly meanX: number
  /** The mean $\bar y$. */
  readonly meanY: number
  /** The population variance of $x$ (divided by $n$). */
  readonly varX: number
  /** The population variance of $y$ (divided by $n$). */
  readonly varY: number
  /** The population covariance of $x$ and $y$ (divided by $n$). */
  readonly cov: number
  /** Pearson correlation (NaN when a variance is 0 or $n < 2$). */
  readonly rho: number
  /** The intercept of the least-squares line $y = \text{intercept} + \text{slope} \cdot x$. */
  readonly intercept: number
  /** The slope of the least-squares line (NaN when $x$ is constant). */
  readonly slope: number
  /** Residual variance $\mathrm{RSS}/(n - 2)$ (NaN for $n \le 2$). */
  readonly residualVariance: number
  /** $\sum_i (x_i - \bar x)^2$. */
  readonly ssx: number
}

type Sums = { n: number; x: number; y: number; xx: number; yy: number; xy: number }

/**
 * The count and the sums of $x$, $y$, $x^2$, $y^2$ and $xy$ over a set of rows.
 *
 * @param x The first column.
 * @param y The second column, as long as `x`.
 * @param rows The row indices summed over.
 * @returns The sums.
 */
function sumsOf(x: Float64Array, y: Float64Array, rows: Iterable<number>): Sums {
  const s = { n: 0, x: 0, y: 0, xx: 0, yy: 0, xy: 0 }
  for (const i of rows) {
    s.n++
    s.x += x[i]
    s.y += y[i]
    s.xx += x[i] * x[i]
    s.yy += y[i] * y[i]
    s.xy += x[i] * y[i]
  }
  return s
}

/**
 * The sums over the rows of `a` not in `b`, when `b`'s rows are a subset of `a`'s: the complement's sums.
 *
 * @param a The sums over a set of rows (all of them).
 * @param b The sums over a subset (the subgroup).
 * @returns The difference, field by field.
 */
const minus = (a: Sums, b: Sums): Sums => ({
  n: a.n - b.n,
  x: a.x - b.x,
  y: a.y - b.y,
  xx: a.xx - b.xx,
  yy: a.yy - b.yy,
  xy: a.xy - b.xy,
})

/**
 * The moments, correlation and least-squares line from sums (every field NaN for no rows).
 *
 * @param s The sums over a set of rows.
 * @returns The fit.
 */
function fitOf(s: Sums): BivariateFit {
  const n = s.n
  const meanX = s.x / n
  const meanY = s.y / n
  const ssx = Math.max(0, s.xx - n * meanX * meanX)
  const ssy = Math.max(0, s.yy - n * meanY * meanY)
  const sxy = s.xy - n * meanX * meanY
  const slope = ssx > 0 ? sxy / ssx : NaN
  const intercept = meanY - slope * meanX
  const rss = Math.max(0, ssy - slope * sxy)
  return {
    n,
    meanX,
    meanY,
    varX: ssx / n,
    varY: ssy / n,
    cov: sxy / n,
    rho: ssx > 0 && ssy > 0 ? sxy / Math.sqrt(ssx * ssy) : NaN,
    intercept,
    slope,
    residualVariance: n > 2 ? rss / (n - 2) : NaN,
    ssx,
  }
}

/**
 * The `fit` of the bivariate model classes: the sums over every row are computed once, and those of the complement as
 * the difference, so a cover costs one pass over its own rows.
 *
 * @param x The first column.
 * @param y The second column, as long as `x`.
 * @returns The fits inside a cover, outside it, and on every row.
 */
function bivariate(x: Float64Array, y: Float64Array) {
  const all = sumsOf(x, y, x.keys())
  return (cover: Bitset): ModelFits<BivariateFit> => {
    const inside = sumsOf(x, y, bitsetIndices(cover))
    return { inside: fitOf(inside), outside: fitOf(minus(all, inside)), all: fitOf(all) }
  }
}

/**
 * A quality that is not finite (from a fit that failed) as $-\infty$, so the search ranks it last.
 *
 * @param v The quality.
 * @returns `v`, or $-\infty$ when it is NaN or infinite.
 */
const finiteOr = (v: number) => (Number.isFinite(v) ? v : -Infinity)

/**
 * The correlation model class over two numeric targets: how far the Pearson correlation inside a subgroup is from the
 * one outside. A subgroup or complement of fewer than 4 rows has quality $-\infty$; for `fisher-z`, correlations are
 * clamped to $\pm 0.999999$ before the $\operatorname{atanh}$.
 *
 * @param x The first numeric target, one value per row.
 * @param y The second numeric target, as long as `x`.
 * @param options `measure`: `absolute`, `entropy` or `fisher-z` (the default).
 * @returns The measure, whose `fit` gives the correlations inside and outside.
 *
 * @example Correlated one way in the first half, the other way in the second
 * const x = [1, 2, 3, 4, 5, 6, 7, 8]
 * const y = [1, 3, 2, 4, 8, 6, 7, 5]
 * const cover = bitset(8, (i) => i < 4)
 * const model = correlationModel(x, y, { measure: 'absolute' })
 * const { inside, outside } = model.fit(cover)
 * print('rho inside =', inside.rho, 'outside =', outside.rho)
 * print('absolute =', model.quality(cover))
 * print('fisher-z =', correlationModel(x, y).quality(cover))
 */
export function correlationModel(
  x: Column,
  y: Column,
  options: { measure?: 'absolute' | 'entropy' | 'fisher-z' } = {},
): ModelMeasure<BivariateFit> {
  const [xs, ys] = columns('correlationModel', x, y)
  const measure = options.measure ?? 'fisher-z'
  const fit = bivariate(xs, ys)
  const N = xs.length
  return {
    key: `correlation/${measure}`,
    name: `Correlation (${measure})`,
    model: 'correlation',
    rows: N,
    fit,
    quality(cover) {
      const f = fit(cover)
      const n = f.inside.n
      if (n < 4 || N - n < 4) return -Infinity
      const d = f.inside.rho - f.outside.rho
      if (measure === 'absolute') return finiteOr(Math.abs(d))
      if (measure === 'entropy') return finiteOr(entropyBits(n / N) * Math.abs(d))
      const clamp = (r: number) => Math.atanh(Math.max(-0.999999, Math.min(0.999999, r)))
      return finiteOr(Math.abs(clamp(f.inside.rho) - clamp(f.outside.rho)) / Math.sqrt(1 / (n - 3) + 1 / (N - n - 3)))
    },
  }
}

/**
 * The simple-regression model class: the least-squares line of $y$ on $x$ inside a subgroup against outside it (or,
 * for `cook`, the full fit against the complement's). A subgroup or complement of fewer than 3 rows has quality
 * $-\infty$.
 *
 * @param x The regressor, one value per row.
 * @param y The response, as long as `x`.
 * @param options `measure`: `slope-difference` (the default), `entropy` or `cook`.
 * @returns The measure, whose `fit` gives the lines inside and outside.
 *
 * @example A slope of 2 inside, about 0 outside
 * const x = [1, 2, 3, 4, 5, 6]
 * const y = [2.1, 3.9, 6, 5, 5.2, 4.9]
 * const cover = bitset(6, (i) => i < 3)
 * const model = regressionModel(x, y)
 * const { inside, outside } = model.fit(cover)
 * print('slope inside =', inside.slope, 'outside =', outside.slope)
 * print('slope-difference t =', model.quality(cover))
 * print('cook =', regressionModel(x, y, { measure: 'cook' }).quality(cover))
 */
export function regressionModel(
  x: Column,
  y: Column,
  options: { measure?: 'slope-difference' | 'entropy' | 'cook' } = {},
): ModelMeasure<BivariateFit> {
  const [xs, ys] = columns('regressionModel', x, y)
  const measure = options.measure ?? 'slope-difference'
  const fit = bivariate(xs, ys)
  const N = xs.length
  return {
    key: `regression/${measure}`,
    name: `Regression (${measure})`,
    model: 'regression',
    rows: N,
    fit,
    quality(cover) {
      const f = fit(cover)
      const n = f.inside.n
      if (n < 3 || N - n < 3) return -Infinity
      const { inside: g, outside: c, all } = f
      if (measure === 'entropy') return finiteOr(entropyBits(n / N) * Math.abs(g.slope - c.slope))
      if (measure === 'slope-difference') {
        const se2 = g.residualVariance / g.ssx + c.residualVariance / c.ssx
        return finiteOr(Math.abs(g.slope - c.slope) / Math.sqrt(se2))
      }
      // Cook's distance of deleting the subgroup: XᵀX = [[N, Σx], [Σx, Σx²]] over all rows, p = 2.
      const d0 = all.intercept - c.intercept
      const d1 = all.slope - c.slope
      const sx = N * all.meanX
      const sxx = all.ssx + N * all.meanX * all.meanX
      const quad = N * d0 * d0 + 2 * sx * d0 * d1 + sxx * d1 * d1
      return finiteOr(quad / (2 * all.residualVariance))
    },
  }
}

/** A logistic regression of a 0/1 label on one feature: coefficients, the slope's variance and the rows. */
export interface LogisticFit {
  /** The number of rows fitted. */
  readonly n: number
  /** The intercept $b_0$ of $\Pr(y = 1) = \sigma(b_0 + b_1 x)$ (NaN for no rows). */
  readonly intercept: number
  /** The slope $b_1$ (NaN for no rows). */
  readonly slope: number
  /** The slope's variance from the inverse Fisher information ($+\infty$ when not identified). */
  readonly slopeVariance: number
  /** Positive rate. */
  readonly rate: number
}

/**
 * Newton's method on the logistic log-likelihood with two coefficients (a tiny ridge, $10^{-6} n$, keeps separation
 * finite), from 0 for at most 50 steps or until a step is below $10^{-10}$.
 *
 * @param x The feature, one value per row of the table.
 * @param y The 0/1 label, one value per row of the table.
 * @param rows The rows to fit on.
 * @returns The fit.
 */
function logisticFit(x: Float64Array, y: Float64Array, rows: ArrayLike<number>): LogisticFit {
  const n = rows.length
  let b0 = 0
  let b1 = 0
  let pos = 0
  for (let k = 0; k < n; k++) pos += y[rows[k]]
  if (n === 0) return { n, intercept: NaN, slope: NaN, slopeVariance: Infinity, rate: NaN }
  const ridge = 1e-6 * n
  let h00 = 0
  let h01 = 0
  let h11 = 0
  for (let it = 0; it < 50; it++) {
    let g0 = -ridge * b0
    let g1 = -ridge * b1
    h00 = ridge
    h01 = 0
    h11 = ridge
    for (let k = 0; k < n; k++) {
      const i = rows[k]
      const p = 1 / (1 + Math.exp(-(b0 + b1 * x[i])))
      const w = p * (1 - p)
      const r = y[i] - p
      g0 += r
      g1 += r * x[i]
      h00 += w
      h01 += w * x[i]
      h11 += w * x[i] * x[i]
    }
    const det = h00 * h11 - h01 * h01
    if (!(det > 0)) break
    const s0 = (h11 * g0 - h01 * g1) / det
    const s1 = (h00 * g1 - h01 * g0) / det
    b0 += s0
    b1 += s1
    if (Math.abs(s0) + Math.abs(s1) < 1e-10) break
  }
  const det = h00 * h11 - h01 * h01
  return { n, intercept: b0, slope: b1, slopeVariance: det > 0 ? h00 / det : Infinity, rate: pos / n }
}

/**
 * The classification model class: a logistic regression of a 0/1 label on $x$, inside against outside. A subgroup or
 * complement of fewer than 5 rows has quality $-\infty$.
 *
 * @param x The feature, one value per row.
 * @param label The 0/1 label, as long as `x`.
 * @param options `measure`: `wald` (the default) or `entropy`.
 * @returns The measure, whose `fit` gives the logistic fits inside and outside.
 *
 * @example The label rises with x in the first six rows and falls in the last six
 * const x = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
 * const label = [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0]
 * const cover = bitset(12, (i) => i < 6)
 * const model = logisticModel(x, label)
 * const { inside, outside } = model.fit(cover)
 * print('slope inside =', inside.slope, 'outside =', outside.slope)
 * print('wald =', model.quality(cover))
 */
export function logisticModel(
  x: Column,
  label: Column,
  options: { measure?: 'wald' | 'entropy' } = {},
): ModelMeasure<LogisticFit> {
  const [xs, ys] = columns('logisticModel', x, label)
  const measure = options.measure ?? 'wald'
  const N = xs.length
  const everyRow = Int32Array.from(xs.keys())
  const all = logisticFit(xs, ys, everyRow)
  const fit = (cover: Bitset): ModelFits<LogisticFit> => {
    const inside = bitsetIndices(cover)
    const mask = new Uint8Array(N)
    for (const i of inside) mask[i] = 1
    const outside = everyRow.filter((i) => mask[i] === 0)
    return { inside: logisticFit(xs, ys, inside), outside: logisticFit(xs, ys, outside), all }
  }
  return {
    key: `classification/${measure}`,
    name: `Logistic regression (${measure})`,
    model: 'classification',
    rows: N,
    fit,
    quality(cover) {
      const n = bitsetCount(cover)
      if (n < 5 || N - n < 5) return -Infinity
      const f = fit(cover)
      const d = Math.abs(f.inside.slope - f.outside.slope)
      if (measure === 'entropy') return finiteOr(entropyBits(n / N) * d)
      return finiteOr(d / Math.sqrt(f.inside.slopeVariance + f.outside.slopeVariance))
    },
  }
}

/** A $2 \times 2$ table of two 0/1 targets and its Yule's $Q$ (cells $+1/2$). */
export interface AssociationFit {
  /** The number of rows. */
  readonly n: number
  /** The cells $n_{00}, n_{01}, n_{10}, n_{11}$, the first index for `a` and the second for `b`. */
  readonly table: readonly [number, number, number, number]
  /** Yule's $Q$, in $(-1, 1)$. */
  readonly q: number
}

/**
 * Yule's $Q$ of a $2 \times 2$ table, each cell plus $1/2$ so it is defined for empty cells.
 *
 * @param t The cells $n_{00}, n_{01}, n_{10}, n_{11}$.
 * @returns The table with its size and $Q$.
 */
function associationOf(t: [number, number, number, number]): AssociationFit {
  const [n00, n01, n10, n11] = t.map((v) => v + 0.5)
  return { n: t[0] + t[1] + t[2] + t[3], table: t, q: (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01) }
}

/**
 * The association model class over two 0/1 targets: Yule's $Q$ inside against outside. A target is 1 where it is not 0.
 * The empty subgroup and the whole table have quality $-\infty$.
 *
 * @param a The first binary target, one value per row.
 * @param b The second binary target, as long as `a`.
 * @param options `measure`: `absolute` or `entropy` (the default).
 * @returns The measure, whose `fit` gives the tables inside and outside.
 *
 * @example Two targets that agree in the first half and disagree in the second
 * const a = [1, 1, 0, 0, 1, 1, 0, 0]
 * const b = [1, 1, 0, 0, 0, 0, 1, 1]
 * const cover = bitset(8, (i) => i < 4)
 * const model = associationModel(a, b)
 * const { inside, outside } = model.fit(cover)
 * print('Q inside =', inside.q, 'outside =', outside.q)
 * print('entropy-weighted difference =', model.quality(cover))
 */
export function associationModel(
  a: Column,
  b: Column,
  options: { measure?: 'absolute' | 'entropy' } = {},
): ModelMeasure<AssociationFit> {
  const [as, bs] = columns('associationModel', a, b)
  const measure = options.measure ?? 'entropy'
  const N = as.length
  const cell = (i: number) => (as[i] !== 0 ? 2 : 0) + (bs[i] !== 0 ? 1 : 0)
  const total: [number, number, number, number] = [0, 0, 0, 0]
  for (let i = 0; i < N; i++) total[cell(i)]++
  const fit = (cover: Bitset): ModelFits<AssociationFit> => {
    const inside: [number, number, number, number] = [0, 0, 0, 0]
    for (const i of bitsetIndices(cover)) inside[cell(i)]++
    const outside = total.map((v, j) => v - inside[j]) as [number, number, number, number]
    return { inside: associationOf(inside), outside: associationOf(outside), all: associationOf(total) }
  }
  return {
    key: `association/${measure}`,
    name: `Association (${measure})`,
    model: 'association',
    rows: N,
    fit,
    quality(cover) {
      const f = fit(cover)
      const n = f.inside.n
      if (n < 1 || n === N) return -Infinity
      const d = Math.abs(f.inside.q - f.outside.q)
      return measure === 'entropy' ? entropyBits(n / N) * d : d
    },
  }
}
