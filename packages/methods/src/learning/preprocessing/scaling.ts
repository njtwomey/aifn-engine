/**
 * Per-column affine scalers, z = (x − center) / scale, as scikit-learn's `StandardScaler`, `MinMaxScaler`,
 * `RobustScaler` and `MaxAbsScaler` (Pedregosa et al., 2011). A column whose spread is zero keeps scale 1, as
 * scikit-learn does, and is reported in `constant`.
 */

import { quantile } from 'aifn-compute/probability/stats'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { column, mapColumns, matrix, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A fitted per-column affine map z = (x − center) / scale. */
export interface AffineScaler extends FittedTransform, Invertible {
  /** Subtracted from each column, [d]. */
  readonly center: Tensor
  /** Each column is divided by this, [d]. */
  readonly scale: Tensor
  /** True for columns whose spread was zero (their scale was set to 1). */
  readonly constant: readonly boolean[]
}

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
  /** Column means [d] (0 when `withMean` is false). */
  readonly mean: Tensor
  /** Population column variances [d]. */
  readonly variance: Tensor
}

/**
 * Standardise each column to zero mean and unit (population, ÷ n) standard deviation.
 *
 * @param withMean subtract the mean (default true); `withStd` divide by the standard deviation (default true)
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
  readonly dataMin: Tensor
  readonly dataMax: Tensor
  readonly featureRange: readonly [number, number]
}

/** Map each column's [min, max] onto `range` (default [0, 1]). */
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
  /** Column medians [d]. */
  readonly median: Tensor
  /** Column interquantile ranges [d] (between the `quantileRange` percentiles). */
  readonly interquantileRange: Tensor
}

/**
 * Centre each column on its median and divide by its interquartile range (or another `quantileRange`, in percent),
 * both robust to outliers. Quantiles interpolate linearly (numpy's default).
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

/** A fitted max-abs scaler. */
export type MaxAbsScaler = AffineScaler & { readonly name: 'max-abs-scaler'; readonly maxAbs: Tensor }

/** Divide each column by its largest absolute value, so values lie in [−1, 1]; sparsity (zeros) is preserved. */
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
