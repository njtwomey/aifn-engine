/** scikit-learn's `PowerTransformer` over the Box–Cox and Yeo–Johnson transforms and λ searches of `aifn-compute/probability/stats`. */

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
  readonly method: 'box-cox' | 'yeo-johnson'
  /** λ of each column, [d]. */
  readonly lambdas: readonly number[]
  /** The λ searches, per column. */
  readonly searches: readonly PowerLambda[]
  /** Means and (population) standard deviations of the transformed columns, used when `standardize`. */
  readonly mean: readonly number[]
  readonly scale: readonly number[]
  readonly standardize: boolean
}

/**
 * A power transform per column with λ by maximum likelihood, then (by default) standardisation to zero mean and unit
 * variance, as scikit-learn's `PowerTransformer`. Box–Cox needs positive data; Yeo–Johnson takes any real values.
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
