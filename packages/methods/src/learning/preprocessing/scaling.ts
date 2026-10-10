/**
 * Per-column affine scalers, $z_{ij} = (x_{ij} - c_j) / s_j$ with a centre $c_j$ and scale $s_j$ per column, as
 * scikit-learn's `StandardScaler`, `MinMaxScaler`, `RobustScaler` and `MaxAbsScaler` (Pedregosa et al., 2011).
 *
 * Each scaler is an estimator: `fit({ x })` on an $n \times d$ matrix returns an `AffineScaler`, whose `transform` and
 * `inverseTransform` apply the map and its inverse $x_{ij} = z_{ij} s_j + c_j$ to any matrix of $d$ columns. A column
 * whose spread is zero keeps scale 1, as scikit-learn does, and is reported in `constant`.
 */

import { quantile } from 'aifn-compute/probability/stats'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { column, mapColumns, matrix, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A fitted per-column affine map $z_{ij} = (x_{ij} - c_j) / s_j$, with $c_j$ `center` and $s_j$ `scale`. */
export interface AffineScaler extends FittedTransform, Invertible {
  /** The centre $c_j$ subtracted from each column, $d$ values. */
  readonly center: Tensor
  /** The scale $s_j$ each centred column is divided by, $d$ values (never 0). */
  readonly scale: Tensor
  /** True for columns whose spread was zero (their scale was set to 1). */
  readonly constant: readonly boolean[]
}

/**
 * Build the fitted scaler from a centre and a spread per column: a spread of 0 becomes a scale of 1 and is flagged in
 * `constant`.
 *
 * @param name The scaler's name, as its `name` field and in error messages.
 * @param center The centre $c_j$ of each column, $d$ values; captured by the transforms, so not to be written after.
 * @param spread The spread of each column, $d$ values: the scale $s_j$, except that 0 becomes 1.
 * @param extra The scaler's own fitted fields (its mean, minimum, median, ...), spread into the result.
 * @returns The fitted scaler, with `center`, `scale`, `constant`, `transform` and `inverseTransform`.
 */
function affine<K extends string, E extends object>(
  name: K,
  center: Float64Array,
  spread: Float64Array,
  extra: E,
): AffineScaler & E & { readonly name: K } {
  const d = center.length
  const constant = Array.from(spread, (s) => s === 0)
  const scale = Float64Array.from(spread, (s) => (s === 0 ? 1 : s))
  return {
    kind: 'model',
    name,
    ...extra,
    center: fromData(center, [d]),
    scale: fromData(scale, [d]),
    constant,
    transform: (x) => mapColumns(x, d, name, (v, j) => (v - center[j]) / scale[j]),
    inverseTransform: (z) => mapColumns(z, d, name, (v, j) => v * scale[j] + center[j]),
  }
}

/** A fitted standard scaler. */
export type StandardScaler = AffineScaler & {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'standard-scaler'
  /** The column means, $d$ values (0 when `withMean` is false). */
  readonly mean: Tensor
  /** The population column variances (divided by $n$), $d$ values, computed whether or not `withStd` is set. */
  readonly variance: Tensor
}

/**
 * Standardise each column to zero mean and unit population standard deviation: $z_{ij} = (x_{ij} - m_j) / s_j$, with
 * $m_j$ the column mean and $s_j^2 = \frac{1}{n} \sum_i (x_{ij} - m_j)^2$ (divided by $n$, as scikit-learn's
 * `StandardScaler`). A column of zero variance keeps $s_j = 1$ and is flagged in `constant`.
 *
 * @param options What to remove from each column.
 * @param options.withMean Subtract the column mean; when false the centre is 0 (and so is the reported `mean`).
 * @param options.withStd Divide by the column standard deviation; when false the scale is 1 (and no column is flagged
 *   `constant`).
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `StandardScaler`.
 *
 * @example Fitted mean and standard deviation of normal columns
 * const x = normal(stream(1), tensor([5, -2]), tensor([2, 0.5]), { shape: [500, 2] })
 * const model = standardScaler().fit({ x })
 * print('mean =', model.mean)
 * print('sd =', model.scale)
 * const z = model.transform(x)
 * print('mean of z =', mean(z, 0))
 * print('sd of z =', std(z, 0))
 *
 * @example The inverse recovers the inputs
 * const x = tensor([[1, 4], [2, 4], [3, 4]])
 * const model = standardScaler().fit({ x })
 * print('z =', model.transform(x))
 * print('constant columns:', model.constant)
 * print('back =', model.inverseTransform(model.transform(x)))
 */
export function standardScaler({
  withMean = true,
  withStd = true,
}: { withMean?: boolean; withStd?: boolean } = {}): Transformer<Tensor, StandardScaler> {
  return {
    name: 'standard-scaler',
    params: { withMean, withStd },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'standardScaler')
      const mean = new Float64Array(d)
      const variance = new Float64Array(d)
      for (let j = 0; j < d; j++) {
        const c = column(v, n, d, j)
        let m = 0
        for (const a of c) m += a / n
        let s = 0
        for (const a of c) s += (a - m) ** 2
        mean[j] = m
        variance[j] = s / n
      }
      const center = withMean ? mean : new Float64Array(d)
      const spread = withStd ? variance.map(Math.sqrt) : new Float64Array(d).fill(1)
      return affine('standard-scaler' as const, center, spread, {
        mean: fromData(center.slice(), [d]),
        variance: fromData(variance, [d]),
      })
    },
  }
}

/** A fitted min–max scaler. */
export type MinMaxScaler = AffineScaler & {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'min-max-scaler'
  /** The column minima, $d$ values. */
  readonly dataMin: Tensor
  /** The column maxima, $d$ values. */
  readonly dataMax: Tensor
  /** The range $[a, b]$ the columns were mapped onto. */
  readonly featureRange: readonly [number, number]
}

/**
 * Map each column's $[\min_i x_{ij}, \max_i x_{ij}]$ linearly onto `range` $[a, b]$:
 * $z_{ij} = a + (x_{ij} - \min_i x_{ij})(b - a) / (\max_i x_{ij} - \min_i x_{ij})$, as scikit-learn's `MinMaxScaler`.
 * A constant column maps to $a$. Values outside the training range map outside $[a, b]$ (nothing is clipped). Throws
 * `DomainError` unless $b > a$.
 *
 * @param options The target range.
 * @param options.range The interval $[a, b]$ each column is mapped onto, with $b > a$.
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `MinMaxScaler`.
 *
 * @example Two columns onto $[0, 1]$, and one onto $[-1, 1]$
 * const x = tensor([[1, 10], [2, 30], [5, 20]])
 * print('[0, 1]:', minMaxScaler().fit({ x }).transform(x))
 * print('[-1, 1]:', minMaxScaler({ range: [-1, 1] }).fit({ x }).transform(x))
 */
export function minMaxScaler({ range = [0, 1] }: { range?: readonly [number, number] } = {}): Transformer<
  Tensor,
  MinMaxScaler
> {
  const [a, b] = range
  if (!(b > a)) throw new DomainError('minMaxScaler', 'minMaxScaler: range must be increasing')
  return {
    name: 'min-max-scaler',
    params: { range },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'minMaxScaler')
      const lo = new Float64Array(d).fill(Infinity)
      const hi = new Float64Array(d).fill(-Infinity)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          lo[j] = Math.min(lo[j], v[i * d + j])
          hi[j] = Math.max(hi[j], v[i * d + j])
        }
      }
      // z = a + (x − min)(b − a)/(max − min) = (x − center)/scale with scale = (max − min)/(b − a).
      const spread = Float64Array.from(lo, (l, j) => (hi[j] - l) / (b - a))
      const center = Float64Array.from(lo, (l, j) => l - a * (spread[j] === 0 ? 1 : spread[j]))
      return affine('min-max-scaler' as const, center, spread, {
        dataMin: fromData(lo, [d]),
        dataMax: fromData(hi, [d]),
        featureRange: range,
      })
    },
  }
}

/** A fitted robust scaler. */
export type RobustScaler = AffineScaler & {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'robust-scaler'
  /** The column medians, $d$ values (computed whether or not `withCentering` is set). */
  readonly median: Tensor
  /**
   * The column interquantile ranges, $d$ values: the difference between the two `quantileRange` percentiles (computed
   * whether or not `withScaling` is set).
   */
  readonly interquantileRange: Tensor
}

/**
 * Centre each column on its median and divide by its interquartile range (or another `quantileRange`, in percent),
 * both robust to outliers, as scikit-learn's `RobustScaler`. Quantiles interpolate linearly (numpy's default). A
 * column whose range is 0 keeps scale 1 and is flagged in `constant`.
 *
 * @param options What to remove from each column, and which quantiles measure its spread.
 * @param options.withCentering Subtract the column median; when false the centre is 0.
 * @param options.withScaling Divide by the interquantile range; when false the scale is 1.
 * @param options.quantileRange The lower and upper percentiles (between 0 and 100) whose difference is the scale.
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `RobustScaler`.
 *
 * @example An outlier moves the standard scaler's centre but not the median
 * const x = tensor([[1], [2], [3], [4], [100]])
 * const robust = robustScaler().fit({ x })
 * print('median =', robust.median, ' IQR =', robust.interquantileRange)
 * print('robust z =', robust.transform(x))
 * print('standard scaler mean =', standardScaler().fit({ x }).mean)
 */
export function robustScaler({
  withCentering = true,
  withScaling = true,
  quantileRange = [25, 75],
}: { withCentering?: boolean; withScaling?: boolean; quantileRange?: readonly [number, number] } = {}): Transformer<
  Tensor,
  RobustScaler
> {
  return {
    name: 'robust-scaler',
    params: { withCentering, withScaling, quantileRange },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'robustScaler')
      const median = new Float64Array(d)
      const iqr = new Float64Array(d)
      for (let j = 0; j < d; j++) {
        const c = column(v, n, d, j)
        const [q1, q2, q3] = toFlat(quantile(c, [quantileRange[0] / 100, 0.5, quantileRange[1] / 100]))
        median[j] = q2
        iqr[j] = q3 - q1
      }
      return affine(
        'robust-scaler' as const,
        withCentering ? median : new Float64Array(d),
        withScaling ? iqr : new Float64Array(d).fill(1),
        { median: fromData(median.slice(), [d]), interquantileRange: fromData(iqr.slice(), [d]) },
      )
    },
  }
}

/** A fitted max-abs scaler: its `name`, and `maxAbs`, the largest absolute value of each column ($d$ values). */
export type MaxAbsScaler = AffineScaler & { readonly name: 'max-abs-scaler'; readonly maxAbs: Tensor }

/**
 * Divide each column by its largest absolute value, so training values lie in $[-1, 1]$; there is no centring, so
 * zeros stay zero (sparsity is preserved), as scikit-learn's `MaxAbsScaler`. An all-zero column keeps scale 1.
 *
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `MaxAbsScaler`.
 *
 * @example Each column divided by its largest magnitude
 * const x = tensor([[1, -8, 0], [-2, 4, 0], [0, 2, 0]])
 * const model = maxAbsScaler().fit({ x })
 * print('max |x| =', model.maxAbs)
 * print('z =', model.transform(x))
 */
export function maxAbsScaler(): Transformer<Tensor, MaxAbsScaler> {
  return {
    name: 'max-abs-scaler',
    fit({ x }) {
      const { n, d, v } = matrix(x, 'maxAbsScaler')
      const m = new Float64Array(d)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] = Math.max(m[j], Math.abs(v[i * d + j]))
      return affine('max-abs-scaler' as const, new Float64Array(d), m, { maxAbs: fromData(m.slice(), [d]) })
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'standardScaler',
    module: 'learning/preprocessing',
    name: 'Standard scaler',
    summary: 'Centre each feature and divide by its standard deviation.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ withMean: bool({ default: true }), withStd: bool({ default: true }) }),
    notes: ['feature-scaling'],
    cite: ['pedregosa2011'],
  },
  standardScaler,
)

defineModel(
  {
    key: 'minMaxScaler',
    module: 'learning/preprocessing',
    name: 'Min–max scaler',
    summary: 'Map each feature linearly onto a range, [0, 1] by default.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({}),
    notes: ['feature-scaling'],
    cite: ['pedregosa2011'],
  },
  minMaxScaler,
)

defineModel(
  {
    key: 'robustScaler',
    module: 'learning/preprocessing',
    name: 'Robust scaler',
    summary: 'Centre each feature on its median and scale by its interquartile range.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ withCentering: bool({ default: true }), withScaling: bool({ default: true }) }),
    notes: ['feature-scaling'],
    cite: ['pedregosa2011'],
  },
  robustScaler,
)

defineModel(
  {
    key: 'maxAbsScaler',
    module: 'learning/preprocessing',
    name: 'Max-abs scaler',
    summary: 'Divide each feature by its largest absolute value.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({}),
    notes: ['feature-scaling'],
    cite: ['pedregosa2011'],
  },
  maxAbsScaler,
)
