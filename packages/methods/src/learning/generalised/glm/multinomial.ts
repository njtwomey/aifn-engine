/**
 * Multinomial logistic regression with standard errors: `logisticRegression` (softmax, fitted by Newton's method; one
 * definition of the fit) plus the inverse Fisher information and baseline-category contrasts
 * $\betavec_k - \betavec_r$, the parameterisation of R's `nnet::multinom` and statsmodels' `MNLogit` (Agresti, 2013,
 * "Categorical Data Analysis", §8.1).
 */

import { logisticRegression, type LogisticRegressionModel } from './logistic'
import { type Estimator, type Supervised } from 'aifn-compute/learning/estimators'
import { pinv } from 'aifn-compute/numerics/linalg'
import { normalCdf, softmax } from 'aifn-compute/numerics/special'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'

/** Hyperparameters of `multinomialLogisticRegression`. */
export type MultinomialParams = {
  /** L2 penalty on the weights, as `logisticRegression` (default 0: maximum likelihood). */
  l2?: number
  /** Fit intercepts (default true). */
  intercept?: boolean
  /** The baseline class $r$ of the contrasts (default 0). */
  reference?: number
  /** Newton convergence tolerance (see `logisticRegression`). */
  tolerance?: number
  /** Most Newton steps (see `logisticRegression`). */
  maxSteps?: number
}

/** Coefficients of each class $k$ against the reference class $r$, $\betavec_k - \betavec_r$, with Wald inference. */
export type Contrasts = {
  /**
   * The contrasts, $p \times (K - 1)$: row $j$ is design column $j$ (the intercept last), column $c$ the $c$-th
   * non-reference class.
   */
  coefficients: Tensor
  /** Their standard errors, $p \times (K - 1)$, from the covariance of the weights. */
  standardErrors: Tensor
  /** Wald statistics, coefficient over standard error, $p \times (K - 1)$. */
  statistics: Tensor
  /** Two-sided normal p-values of the Wald statistics, $p \times (K - 1)$. */
  pValues: Tensor
  /** The non-reference classes, in column order. */
  classes: number[]
}

/** A fitted multinomial logistic regression with inference. */
export type MultinomialModel = LogisticRegressionModel & {
  /**
   * The covariance of the softmax weights $\Wmat$, $pK \times pK$ (row-major over $p \times K$, the intercept row
   * last): the Moore–Penrose inverse of the Fisher information (plus the penalty), whose null space is the direction
   * that adds a constant to every class. Estimable contrasts $\betavec_k - \betavec_r$ get their exact variances from
   * it.
   */
  covariance: Tensor
  /** Each class against the reference class, with Wald inference. */
  contrasts: Contrasts
}

/**
 * Softmax regression for labels $0, \dots, K - 1$ by `logisticRegression({ multinomial: true })`, with the covariance
 * of the weights and Wald ($z$) tests for each class against a reference class. The Fisher information is
 * $\sum_i (\diag(\boldsymbol{\pi}_i) - \boldsymbol{\pi}_i\boldsymbol{\pi}_i^\top) \otimes \xvec_i\xvec_i^\top$
 * at the fitted class probabilities $\boldsymbol{\pi}_i$.
 *
 * @param params The penalty (default 0), intercept, reference class and Newton controls.
 * @returns An estimator whose `fit` takes `{ x, y }` ($n \times d$ inputs, $n$ integer labels) and returns the model.
 *
 * @example Three classes against the first: the contrasts recover known slopes and intercepts
 * const s = stream(3)
 * const x = normals(s, [500, 1])
 * const y = categorical(s, exp(add(matmul(x, tensor([[0, 1.5, -1.5]])), tensor([0, 0.5, 0]))))
 * const { contrasts } = multinomialLogisticRegression().fit({ x, y })
 * print('classes =', contrasts.classes)
 * print('contrasts (rows x, intercept; true [[1.5, -1.5], [0.5, 0]]) =', contrasts.coefficients)
 * print('standard errors =', contrasts.standardErrors)
 * print('p-values =', contrasts.pValues)
 */
export function multinomialLogisticRegression(
  params: MultinomialParams = {},
): Estimator<Supervised<Tensor, Tensor>, MultinomialModel> {
  const { l2 = 0, intercept = true, reference = 0, tolerance, maxSteps } = params
  const base = logisticRegression({ l2, intercept, multinomial: true, tolerance, maxSteps })
  return {
    name: 'multinomial-logistic-regression',
    params,
    fit(data, options) {
      const model = base.fit(data, options)
      const [n, d] = data.x.shape
      const K = model.classes
      const p = d + (intercept ? 1 : 0)
      const X = toFlat(data.x)
      const P = p * K
      // Fisher information: Σᵢ (diag(πᵢ) − πᵢπᵢᵀ) ⊗ xᵢxᵢᵀ, in [p, K] order, plus the penalty on the weights.
      const info = new Float64Array(P * P)
      const row = new Float64Array(p)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) row[j] = X[i * d + j]
        if (intercept) row[d] = 1
        const eta = toFlat(model.forward(fromData(Float64Array.from(row.subarray(0, d)), [1, d])))
        const pi = toFlat(softmax(fromData(Float64Array.from(eta), [eta.length])))
        for (let a = 0; a < p; a++)
          for (let k = 0; k < K; k++)
            for (let b = 0; b < p; b++)
              for (let l = 0; l < K; l++)
                info[(a * K + k) * P + b * K + l] += row[a] * row[b] * pi[k] * ((k === l ? 1 : 0) - pi[l])
      }
      for (let a = 0; a < d; a++) for (let k = 0; k < K; k++) info[(a * K + k) * P + a * K + k] += l2
      const cov = toFlat(pinv(fromData(info, [P, P])))
      const W = toFlat(model.weights)
      const b = toFlat(model.intercept)
      const coef = (a: number, k: number) => (a < d ? W[a * K + k] : b[k])
      const classes = Array.from({ length: K }, (_, k) => k).filter((k) => k !== reference)
      const C = classes.length
      const est = new Float64Array(p * C)
      const se = new Float64Array(p * C)
      for (let a = 0; a < p; a++)
        classes.forEach((k, c) => {
          const i = a * K + k
          const r = a * K + reference
          est[a * C + c] = coef(a, k) - coef(a, reference)
          se[a * C + c] = Math.sqrt(cov[i * P + i] + cov[r * P + r] - 2 * cov[i * P + r])
        })
      const stat = Float64Array.from(est, (v, i) => v / se[i])
      return {
        ...model,
        covariance: fromData(Float64Array.from(cov), [P, P]),
        contrasts: {
          coefficients: fromData(est, [p, C]),
          standardErrors: fromData(se, [p, C]),
          statistics: fromData(stat, [p, C]),
          pValues: fromData(
            Float64Array.from(stat, (t) => 2 * (normalCdf(-Math.abs(t)) as number)),
            [p, C],
          ),
          classes,
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'multinomialLogisticRegression',
    module: 'learning/generalised/glm',
    name: 'Multinomial logistic regression',
    summary: 'Softmax regression with a reference class and Wald standard errors.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({
      l2: real(0, 100, { default: 0, label: 'L2 penalty' }),
      intercept: bool({ default: true }),
      reference: int(0, 20, { default: 0 }),
    }),
    notes: ['multinomial-logistic-regression'],
    cite: ['hastie2009'],
  },
  multinomialLogisticRegression,
)
