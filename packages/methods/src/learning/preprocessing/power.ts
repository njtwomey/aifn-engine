/**
 * scikit-learn's `PowerTransformer` over the Box–Cox and Yeo–Johnson transforms and $\lambda$ searches of
 * `aifn-compute/probability/stats`, which this file also re-exports.
 *
 * Box–Cox (Box and Cox, 1964) is $(x^\lambda - 1)/\lambda$ ($\log x$ at $\lambda = 0$) and needs $x > 0$; Yeo–Johnson
 * (Yeo and Johnson, 2000) extends it to every real $x$. Each column gets its own $\lambda$, by maximum likelihood under
 * a normal model of the transformed column, so that skewed columns come out closer to normal.
 */

import {
  boxCox,
  boxCoxInverse,
  boxCoxLambda,
  yeoJohnson,
  yeoJohnsonInverse,
  yeoJohnsonLambda,
  type PowerLambda,
} from 'aifn-compute/probability/stats'
import { fromData, variance, type Tensor } from 'aifn-compute/foundation/tensor'
import { column, mapColumns, matrix, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, oneOf, space } from 'aifn-compute/foundation/space'

export { boxCox, boxCoxInverse, boxCoxLambda, yeoJohnson, yeoJohnsonInverse, yeoJohnsonLambda, type PowerLambda }

// ── The transformer ──────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted power transform. */
export interface PowerTransform extends FittedTransform, Invertible {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'power-transform'
  /** The transform: `'box-cox'` (positive data) or `'yeo-johnson'` (any real data). */
  readonly method: 'box-cox' | 'yeo-johnson'
  /** The $\lambda$ of each column, $d$ values. */
  readonly lambdas: readonly number[]
  /** The $\lambda$ searches, per column, with their log-likelihoods and convergence. */
  readonly searches: readonly PowerLambda[]
  /** The means of the transformed training columns, subtracted when `standardize` (all 0 otherwise). */
  readonly mean: readonly number[]
  /**
   * The population standard deviations of the transformed training columns, divided by when `standardize` (all 1
   * otherwise; 1 for a constant column).
   */
  readonly scale: readonly number[]
  /** Whether the transformed columns are standardised. */
  readonly standardize: boolean
}

/**
 * A power transform per column with $\lambda$ by maximum likelihood (`boxCoxLambda` or `yeoJohnsonLambda`), then (by
 * default) standardisation to zero mean and unit population variance, as scikit-learn's `PowerTransformer`. Box–Cox
 * needs positive data (`fit` throws `DomainError` otherwise); Yeo–Johnson takes any real values.
 *
 * @param options The transform and whether to standardise.
 * @param options.method `'yeo-johnson'` (any real data) or `'box-cox'` (positive data).
 * @param options.standardize Standardise each transformed column with its training mean and population standard
 *   deviation.
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `PowerTransform`.
 *
 * @example Log-normal data: $\lambda$ near 0, a log transform
 * const x = exp(normal(stream(1), 0, 1, { shape: [200, 1] }))
 * const model = powerTransform({ method: 'box-cox' }).fit({ x })
 * print('lambda =', model.lambdas)
 * const z = model.transform(x)
 * print('mean of z =', mean(z), ' sd of z =', std(z))
 * print('round trip of the first rows:', slice(model.inverseTransform(z), [0, 3]), slice(x, [0, 3]))
 */
export function powerTransform({
  method = 'yeo-johnson',
  standardize = true,
}: { method?: 'box-cox' | 'yeo-johnson'; standardize?: boolean } = {}): Transformer<Tensor, PowerTransform> {
  const forward = (a: number, lambda: number): number => (method === 'box-cox' ? boxCox : yeoJohnson)(a, lambda)
  const backward = (z: number, lambda: number): number =>
    (method === 'box-cox' ? boxCoxInverse : yeoJohnsonInverse)(z, lambda)
  return {
    name: 'power-transform',
    params: { method, standardize },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'powerTransform')
      const searches: PowerLambda[] = []
      const mean: number[] = []
      const scale: number[] = []
      for (let j = 0; j < d; j++) {
        const c = column(v, n, d, j)
        const s = method === 'box-cox' ? boxCoxLambda(c) : yeoJohnsonLambda(c)
        searches.push(s)
        const t = Float64Array.from(c, (a) => forward(a, s.lambda))
        let m = 0
        for (const a of t) m += a / n
        mean.push(standardize ? m : 0)
        const sd = Math.sqrt(variance(fromData(t)) as number)
        scale.push(standardize ? (sd === 0 ? 1 : sd) : 1)
      }
      const lambdas = searches.map((s) => s.lambda)
      return {
        kind: 'model',
        name: 'power-transform',
        method,
        lambdas,
        searches,
        mean,
        scale,
        standardize,
        transform: (input) =>
          mapColumns(input, d, 'powerTransform', (a, j) => (forward(a, lambdas[j]) - mean[j]) / scale[j]),
        inverseTransform: (z) =>
          mapColumns(z, d, 'powerTransform', (a, j) => backward(a * scale[j] + mean[j], lambdas[j])),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'powerTransform',
    module: 'learning/preprocessing',
    name: 'Power transform',
    summary: 'Box–Cox or Yeo–Johnson transform per feature with λ by maximum likelihood.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ method: oneOf(['yeo-johnson', 'box-cox']), standardize: bool({ default: true }) }),
    notes: ['feature-scaling'],
  },
  powerTransform,
)
