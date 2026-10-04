/**
 * Exceptional model mining (Leman, Feelders and Knobbe, 2008; Duivesteijn, Feelders and Knobbe, 2016): subgroup
 * discovery where the target is a model over several columns, and a subgroup is interesting when the model fitted on
 * its rows differs from the model fitted on the rest (its complement). Each model class here is a quality measure over
 * covers, so the same refinement search runs it; none has a known optimistic estimate, so search is by beam or
 * exhaustive to a depth.
 *
 * - **Correlation** of two numeric targets x and y: `absolute` |ρ_G − ρ_Ḡ|; `entropy` H(n/N)|ρ_G − ρ_Ḡ|, which
 *   favours balanced splits (H is the binary entropy in bits); `fisher-z` |z_G − z_Ḡ|/√(1/(n − 3) + 1/(N − n − 3))
 *   with z = atanh ρ, the test statistic of equal correlations (Leman et al., 2008, §4.1).
 * - **Regression** of y on x: `slope-difference` |b_G − b_Ḡ|/√(se_G² + se_Ḡ²), the t statistic of equal slopes
 *   (Leman et al., 2008, §4.2); `entropy` H(n/N)|b_G − b_Ḡ|; `cook` Cook's distance of deleting the subgroup,
 *   (β − β₍G₎)ᵀXᵀX(β − β₍G₎)/(p s²) with β fitted on all rows, β₍G₎ on the complement and s² the full fit's residual
 *   variance (Duivesteijn, Feelders and Knobbe, 2012, "Different slopes for different folks").
 * - **Classification**: a logistic regression of a 0/1 label on x. Fitted with an indicator of the subgroup and its
 *   interaction with x, the model separates into one logistic fit inside and one outside, so the interaction's Wald
 *   statistic is `wald` |b_G − b_Ḡ|/√(var b_G + var b_Ḡ) (the significance of the effect difference; Leman et al.,
 *   2008, §4.3); `entropy` H(n/N)|b_G − b_Ḡ|.
 * - **Association** of two 0/1 targets: the difference of Yule's Q = (n₁₁n₀₀ − n₁₀n₀₁)/(n₁₁n₀₀ + n₁₀n₀₁) (each cell
 *   + ½) inside and outside, `absolute` or `entropy` weighted: the cheapest Bayesian-network model class, a single
 *   edge's strength (Duivesteijn, Knobbe, Feelders and van Leeuwen, 2010).
 */

import type { Tensor } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { binaryEntropy } from 'aifn-compute/numerics/special'
import { bitsetCount, bitsetIndices, type Bitset } from './cover'
import type { QualityMeasure } from './quality'

type Column = ArrayLike<number> | Tensor

const flat = (t: Column): Float64Array => Float64Array.from(isTensor(t) ? toFlat(t) : t)

/** A model fitted inside the subgroup, outside it (its complement), and on every row. */
export interface ModelFits<F> {
  readonly inside: F
  readonly outside: F
  readonly all: F
}

/** An exceptional-model measure: a quality over covers, and the fits it compares. */
export interface ModelMeasure<F> extends QualityMeasure {
  /** The model class: `correlation`, `regression`, `classification` or `association`. */
  readonly model: string
  fit(cover: Bitset): ModelFits<F>
}

/** Binary entropy in bits (0 at p = 0 and 1). */
const entropyBits = (p: number): number => (p > 0 && p < 1 ? binaryEntropy(p, 2) : 0)

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
  readonly n: number
  readonly meanX: number
  readonly meanY: number
  /** Population variances and covariance (divided by n). */
  readonly varX: number
  readonly varY: number
  readonly cov: number
  /** Pearson correlation (NaN when a variance is 0 or n < 2). */
  readonly rho: number
  /** Least-squares line y = intercept + slope · x. */
  readonly intercept: number
  readonly slope: number
  /** Residual variance RSS/(n − 2) (NaN for n ≤ 2). */
  readonly residualVariance: number
  /** Σ(x − x̄)². */
  readonly ssx: number
}

type Sums = { n: number; x: number; y: number; xx: number; yy: number; xy: number }

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

const minus = (a: Sums, b: Sums): Sums => ({
  n: a.n - b.n,
  x: a.x - b.x,
  y: a.y - b.y,
  xx: a.xx - b.xx,
  yy: a.yy - b.yy,
  xy: a.xy - b.xy,
})

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

function bivariate(x: Float64Array, y: Float64Array) {
  const all = sumsOf(x, y, x.keys())
  return (cover: Bitset): ModelFits<BivariateFit> => {
    const inside = sumsOf(x, y, bitsetIndices(cover))
    return { inside: fitOf(inside), outside: fitOf(minus(all, inside)), all: fitOf(all) }
  }
}

const finiteOr = (v: number) => (Number.isFinite(v) ? v : -Infinity)

/** The correlation model class over two numeric targets. */
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

/** The simple-regression model class: y on x. */
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

/** A logistic regression of a 0/1 label on one feature: coefficients, their variances and the rows. */
export interface LogisticFit {
  readonly n: number
  readonly intercept: number
  readonly slope: number
  /** The slope's variance from the inverse Fisher information (∞ when not identified). */
  readonly slopeVariance: number
  /** Positive rate. */
  readonly rate: number
}

/** Newton's method on the logistic log-likelihood with two coefficients (a tiny ridge keeps separation finite). */
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

/** The classification model class: a logistic regression of a 0/1 label on x, inside against outside. */
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

/** A 2 × 2 table of two 0/1 targets (cells n₀₀, n₀₁, n₁₀, n₁₁ for a, b) and its Yule's Q (cells + ½). */
export interface AssociationFit {
  readonly n: number
  readonly table: readonly [number, number, number, number]
  readonly q: number
}

function associationOf(t: [number, number, number, number]): AssociationFit {
  const [n00, n01, n10, n11] = t.map((v) => v + 0.5)
  return { n: t[0] + t[1] + t[2] + t[3], table: t, q: (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01) }
}

/** The association model class over two 0/1 targets: Yule's Q inside against outside. */
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
