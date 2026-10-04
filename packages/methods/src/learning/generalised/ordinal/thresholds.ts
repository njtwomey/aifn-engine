/**
 * Threshold losses for ordinal regression (Rennie & Srebro, 2005, "Loss functions for preference levels: regression
 * with discrete ordered labels", IJCAI workshop): a score s and K − 1 thresholds θ₀ < … < θ_{K−2} predict the class
 * ŷ = #{k : θ_k < s} ∈ {0, …, K − 1}, and a margin penalty h(z), a convex surrogate of the step 1[z ≤ 0], charges the
 * thresholds on the wrong side of s. For a label y:
 *
 * - **all-threshold**: Σ_{k<y} h(s − θ_k) + Σ_{k≥y} h(θ_k − s), every threshold (it bounds the absolute error
 *   |ŷ − y| when h ≥ the step, e.g. the hinge);
 * - **immediate-threshold**: h(s − θ_{y−1}) + h(θ_y − s), only the two thresholds beside the class (it bounds the
 *   zero-one error).
 *
 * Penalties: `hinge` max(0, 1 − z), `smooth-hinge` (0 for z ≥ 1, ½(1 − z)² on (0, 1), ½ − z for z ≤ 0; Rennie &
 * Srebro's choice), `logistic` log(1 + e^{−z}) and `modified-least-squares` max(0, 1 − z)². `thresholdOrdinalRegression`
 * fits a linear score s = xᵀw with ½λ‖w‖² by L-BFGS. The all-threshold optimum has ordered thresholds (Chu & Keerthi,
 * 2005); the immediate-threshold one need not, when a class is sparse, so thresholds are kept increasing through
 * `orderedBijector` unless `ordered: false`.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import {
  add,
  dense,
  expandDims,
  fromData,
  matmul,
  maximum,
  minimum,
  mul,
  neg,
  shapeOfValue,
  reshape,
  slice,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import {
  defineModel,
  matrixShape,
  targetValues,
  type Decides,
  type Estimator,
  type FitOptions,
  type Fitted,
  type Scores,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { defineLoss, flatValues, reduce, type ReductionOptions, type Target } from 'aifn-compute/learning/losses'
import { softplus } from 'aifn-compute/numerics/special'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { orderedBijector } from 'aifn-compute/probability/bijectors'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A margin penalty h(z) of the threshold losses. */
export type ThresholdPenalty = 'hinge' | 'smooth-hinge' | 'logistic' | 'modified-least-squares'

/** Which thresholds a label charges: all of them, or the two beside its class. */
export type ThresholdConstruction = 'all' | 'immediate'

/** h(z) elementwise, written with primitives so that it differentiates. */
export function thresholdPenalty(z: Value, penalty: ThresholdPenalty = 'logistic'): Value {
  switch (penalty) {
    case 'hinge':
      return maximum(sub(1, z), 0)
    case 'smooth-hinge': {
      // ½ min(max(1 − z, 0), 1)² + max(−z, 0): 0 for z ≥ 1, ½(1 − z)² on (0, 1), ½ − z for z ≤ 0.
      const clipped = minimum(maximum(sub(1, z), 0), 1)
      return add(mul(0.5, square(clipped)), maximum(neg(z), 0))
    }
    case 'logistic':
      return softplus(neg(z))
    case 'modified-least-squares':
      return square(maximum(sub(1, z), 0))
  }
  throw new DomainError('thresholdPenalty', `thresholdPenalty: unknown penalty "${penalty as string}"`)
}

/** Options of the threshold losses. */
export type ThresholdLossOptions = ReductionOptions & {
  /** The margin penalty h (default `logistic`). */
  penalty?: ThresholdPenalty
}

/** Constant sign and mask matrices [n, K − 1]: the sign is +1 where θ_k should lie below s (k < y), −1 above. */
function signsAndMask(labels: Float64Array, m: number, construction: ThresholdConstruction) {
  const n = labels.length
  const sign = new Float64Array(n * m)
  const mask = new Float64Array(n * m)
  for (let i = 0; i < n; i++) {
    const y = labels[i]
    if (!(Number.isInteger(y) && y >= 0 && y <= m))
      throw new DomainError('threshold loss', `threshold loss: class ${y} outside 0 … ${m} for ${m} thresholds`)
    for (let k = 0; k < m; k++) {
      sign[i * m + k] = k < y ? 1 : -1
      mask[i * m + k] = construction === 'all' || k === y - 1 || k === y ? 1 : 0
    }
  }
  return { sign: fromData(sign, [n, m]), mask: fromData(mask, [n, m]) }
}

/** Σ_k mask · h(sign · (s − θ_k)) per example, for scores [n] (or a number) and thresholds [K − 1]. */
function thresholdLoss(
  construction: ThresholdConstruction,
  scores: Value,
  labels: Target,
  thresholds: Value,
  { penalty = 'logistic', reduction }: ThresholdLossOptions,
): Value {
  const shape = shapeOfValue(scores)
  if (shape.length > 1)
    throw new DomainError('threshold loss', 'threshold loss: scores must be a number or a vector [n]')
  const m = shapeOfValue(thresholds)[0]
  const y = flatValues(labels)
  const n = shape.length === 0 ? 1 : shape[0]
  if (y.length !== n) throw new ShapeError('threshold loss', `threshold loss: ${y.length} labels for ${n} scores`)
  const { sign, mask } = signsAndMask(y, m, construction)
  const s = expandDims(shape.length === 0 ? reshape(scores, [1]) : scores, -1) // [n, 1]
  const z = mul(sign, sub(s, expandDims(thresholds, 0))) // [n, K − 1]
  const perExample = sum(mul(mask, thresholdPenalty(z, penalty)), -1)
  return reduce(shape.length === 0 ? sum(perExample) : perExample, reduction)
}

const thresholdInfo = (key: string, name: string, target: string) =>
  ({
    key,
    name,
    module: 'learning/generalised/ordinal',
    family: 'classification',
    inputs: 'scores',
    target,
    notes: ['threshold-losses-for-ordinal-regression'],
    cite: ['rennie2005'],
  }) as const

/**
 * The all-threshold loss (Rennie & Srebro, 2005): Σ_{k<y} h(s − θ_k) + Σ_{k≥y} h(θ_k − s) for scores s [n],
 * class labels y ∈ {0, …, K − 1} and thresholds θ [K − 1]. With the hinge it bounds the absolute error |ŷ − y|.
 */
export const allThresholdLoss = defineLoss(
  {
    ...thresholdInfo('allThresholdLoss', 'All-threshold loss', 'the absolute error |ŷ − y|'),
    pairedMetric: 'ordinalMeanAbsoluteError',
  },
  (scores: Value, labels: Target, thresholds: Value, options: ThresholdLossOptions = {}): Value =>
    thresholdLoss('all', scores, labels, thresholds, options),
)

/**
 * The immediate-threshold loss (Rennie & Srebro, 2005): h(s − θ_{y−1}) + h(θ_y − s), the two thresholds beside the
 * true class only (one for the first and last classes). With the hinge it bounds the zero-one error.
 */
export const immediateThresholdLoss = defineLoss(
  thresholdInfo('immediateThresholdLoss', 'Immediate-threshold loss', 'the zero-one error'),
  (scores: Value, labels: Target, thresholds: Value, options: ThresholdLossOptions = {}): Value =>
    thresholdLoss('immediate', scores, labels, thresholds, options),
)

/** The class #{k : θ_k < s} of each score, given increasing thresholds. */
export function thresholdClasses(scores: ArrayLike<number>, thresholds: ArrayLike<number>): Int32Array {
  return Int32Array.from(scores as ArrayLike<number>, (s) => {
    let c = 0
    for (let k = 0; k < thresholds.length; k++) if (thresholds[k] < s) c++
    return c
  })
}

// ── The estimator ────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `thresholdOrdinalRegression`. */
export type ThresholdOrdinalRegressionParams = {
  /** `all` (default) or `immediate` thresholds. */
  construction?: ThresholdConstruction
  /** The margin penalty (default `logistic`). */
  penalty?: ThresholdPenalty
  /** L2 penalty ½λ‖w‖² on the weights (thresholds are not penalised). Default 1. */
  l2?: number
  /** Keep the thresholds increasing (default true); false leaves them free, Rennie and Srebro's formulation. */
  ordered?: boolean
  /** Number of classes K (default: the largest label + 1). */
  classes?: number
  /** Most L-BFGS steps (default 500). */
  maxSteps?: number
  /** Gradient-norm tolerance of L-BFGS (default 1e-6). */
  tolerance?: number
}

/** A fitted threshold model. */
export interface ThresholdOrdinalRegressionModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<LbfgsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'threshold-ordinal-regression'
  readonly construction: ThresholdConstruction
  readonly penalty: ThresholdPenalty
  readonly classes: number
  /** w [d]: the score is s = xᵀw. */
  readonly coefficients: Tensor
  /** The thresholds θ [K − 1] (increasing unless fitted with `ordered: false`). */
  readonly thresholds: Tensor
  /** The minimised objective: the summed threshold loss plus ½λ‖w‖². */
  readonly objective: number
  readonly converged: boolean
}

/**
 * Ordinal regression with a threshold loss (see the module comment): a linear score s = xᵀw and increasing thresholds
 * minimise Σᵢ loss(sᵢ, yᵢ, θ) + ½λ‖w‖² by L-BFGS (the support vector ordinal regression of Chu & Keerthi, 2005, in
 * the primal, with the hinge). Labels are class indices 0 … K − 1. Capabilities: `forward` and `score` (s), `decide`
 * (#{k : θ_k < s}, defined for any thresholds). The model is not probabilistic: there is no predictive distribution.
 */
export function thresholdOrdinalRegression(
  params: ThresholdOrdinalRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, ThresholdOrdinalRegressionModel> {
  const { construction = 'all', penalty = 'logistic', l2 = 1, maxSteps = 500, tolerance = 1e-6 } = params
  const loss = construction === 'all' ? allThresholdLoss : immediateThresholdLoss
  const bijector = orderedBijector()
  const ordered = params.ordered === false ? { forward: (u: Value) => u, inverse: (t: Value) => t } : bijector
  return {
    name: 'threshold-ordinal-regression',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'thresholdOrdinalRegression')
      const t = targetValues(y, 'thresholdOrdinalRegression')
      if (t.length !== n)
        throw new ShapeError(
          'thresholdOrdinalRegression',
          `thresholdOrdinalRegression: ${n} inputs but ${t.length} labels`,
        )
      const labels = Float64Array.from(t, (v) => {
        if (!(Number.isInteger(v) && v >= 0))
          throw new DomainError(
            'thresholdOrdinalRegression',
            'thresholdOrdinalRegression: labels must be class indices 0, 1, …',
          )
        return v
      })
      const K = params.classes ?? Math.max(...labels) + 1
      if (K < 2)
        throw new DomainError('thresholdOrdinalRegression', 'thresholdOrdinalRegression: needs at least two classes')
      const m = K - 1
      const X = fromData(Float64Array.from(dense.data(x)), [n, d])
      const split = (w: Value) => ({ weights: slice(w, [0, d]), theta: ordered.forward(slice(w, [d, d + m])) })
      const objectiveOf = (w: Value): Value => {
        const { weights, theta } = split(w)
        const s = d > 0 ? matmul(X, weights) : fromData(new Float64Array(n), [n])
        let f = loss(s, labels, theta, { penalty, reduction: 'sum' })
        if (l2 > 0 && d > 0) f = add(f, mul(0.5 * l2, sum(square(weights))))
        return f
      }
      const vg = valueAndGrad(objectiveOf)
      const objective = (w: Tensor) => {
        const { value, grad } = vg(w)
        return { value: typeof value === 'number' ? value : toFlat(unwrap(value) as Tensor)[0], grad: grad as Tensor }
      }
      // Start with w = 0 and thresholds evenly spread on [−1, 1].
      const theta0 = Float64Array.from({ length: m }, (_, k) => (m === 1 ? 0 : -1 + (2 * k) / (m - 1)))
      const w0 = new Float64Array(d + m)
      w0.set(toFlat(ordered.inverse(fromData(theta0, [m])) as Tensor), d)
      const training: Trace<LbfgsState> = trace(lbfgs(objective, { tolerance }), { x0: w0 }, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { loss: (st) => st.value, gradNorm: (st) => st.gradNorm },
      })
      const final = training.final
      const w = fromData(Float64Array.from(toFlat(final.x)), [d + m])
      const parts = split(w)
      const coefficients = fromData(Float64Array.from(toFlat(parts.weights as Tensor)), [d])
      const thresholds = fromData(Float64Array.from(toFlat(parts.theta as Tensor)), [m])
      const forward = (input: Tensor): Tensor => {
        const [rows, cols] = matrixShape(input, 'thresholdOrdinalRegression.forward')
        if (cols !== d)
          throw new ShapeError(
            'thresholdOrdinalRegression',
            `thresholdOrdinalRegression: fitted on ${d} features, given ${cols}`,
          )
        return d > 0 ? (matmul(input, coefficients) as Tensor) : fromData(new Float64Array(rows), [rows])
      }
      const theta = toFlat(thresholds)
      return {
        kind: 'model' as const,
        name: 'threshold-ordinal-regression' as const,
        construction,
        penalty,
        classes: K,
        coefficients,
        thresholds,
        objective: final.value,
        converged: final.converged === true,
        training,
        forward,
        score: forward,
        decide: (input: Tensor): Tensor => {
          const c = thresholdClasses(toFlat(forward(input)), theta)
          return fromData(c, [c.length])
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'thresholdOrdinalRegression',
    module: 'learning/generalised/ordinal',
    name: 'Threshold ordinal regression',
    summary: 'A linear score and ordered thresholds fitted by the all-threshold or immediate-threshold loss.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({
      construction: oneOf(['all', 'immediate']),
      ordered: bool({ default: true }),
      penalty: oneOf(['logistic', 'hinge', 'smooth-hinge', 'modified-least-squares']),
      l2: real(0, 100, { default: 1, label: 'L2 penalty' }),
      maxSteps: int(1, 5000, { default: 500 }),
      tolerance: real(1e-14, 1e-2, { default: 1e-6, scale: 'log' }),
    }),
    notes: ['threshold-losses-for-ordinal-regression', 'ordinal-regression'],
    cite: ['rennie2005', 'chu2005svor'],
  },
  thresholdOrdinalRegression,
)
