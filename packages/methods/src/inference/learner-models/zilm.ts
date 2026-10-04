/**
 * Zero-inflated learner models (Twomey, McMullan, Elhalal, Poyiadzi and Vaquero, 2022, "Equitable Ability Estimation
 * in Neurodivergent Student Populations with Zero-Inflated Learner Models"). A learner model (LM) explains a zero (an
 * incorrect or unanswered item) by low ability. A zero-inflated learner model (ZILM) admits a second explanation: a low
 * learning quality factor (LQF), the delivery and response type (DRT) of the item being unsuitable for the student. With
 * p the base model's probability of a correct answer and π the zero-inflation probability (LQF = 1 − π), the paper's
 * Eqn (1) is
 *
 *   Pr(Y = 0 | x) = π(x_π) + (1 − π(x_π))(1 − p(x_p)),   Pr(Y = 1 | x) = (1 − π(x_π)) p(x_p),
 *
 * the zero-inflated Bernoulli `ZeroInflated({ logits: η }, Bernoulli({ logits: z }))` of `aifn-compute/probability/distributions`.
 * In IRT-ZILM the base is the two-parameter IRT model, z = aᵢ(θₚ − bᵢ), and π = σ(η) is logistic in the student's
 * neurodivergent conditions (NDCs) crossed with the item's DRT and content features:
 *
 *   η_{pi} = w₀ + Σ_k z_{pk} Σ_f x_{if} W_{fk},
 *
 * so the model learns which DRTs depress which conditions' answers. When π does not depend on the student, (1 − π) is
 * the upper asymptote of Barton and Lord's (1981) four-parameter model. Two baselines share the base model and the
 * fitter: `irt` (π ≡ 0, so IRT-ZILM with the zero inflation switched off is exactly this model) and `ktm`, a linear
 * knowledge-tracing machine in the sense of Vie and Kashima (2019), which adds the same context features to the logit,
 * logit Pr(Y = 1) = aᵢ(θₚ − bᵢ) + Σ_k z_{pk} Σ_f x_{if} W_{fk}, instead of to a separate cause of zeros.
 *
 * Fitting follows the paper: the negative log-likelihood of the training responses is minimised over every parameter at
 * once (abilities, difficulties, discriminations and the π weights), with gradients by autodiff, here by L-BFGS, and
 * weak Gaussian priors θ ~ N(0, 1), b ~ N(0, 2²), log a ~ N(0, 1), W ~ N(0, 2²) that fix the scale of θ and keep the
 * estimates finite. IRT-ZILM starts from the IRT fit.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  exp,
  fromData,
  gather,
  matmul,
  mul,
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
import { Bernoulli, ZeroInflated } from 'aifn-compute/probability/distributions'
import { minimize } from 'aifn-compute/optim/minimize'

/** The learner models: two-parameter IRT, a linear knowledge-tracing machine and IRT-ZILM. */
export type LearnerModelKind = 'irt' | 'ktm' | 'zilm'

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

/** Pr(Y = 1) = (1 − π) σ(a(θ − b)): IRT-ZILM's probability of a correct answer (π = 0 gives the 2PL model). */
export function zilmProbability(theta: number, a: number, b: number, pi = 0): number {
  return (1 - pi) * sigmoid(a * (theta - b))
}

/**
 * The posterior probability that an observed zero is structural (caused by the context, probability π) rather than an
 * ability-driven incorrect answer (probability (1 − π)(1 − p)): π / (π + (1 − π)(1 − p)), Bayes' rule on Eqn (1).
 */
export function structuralZeroPosterior(pi: number, p: number): number {
  const structural = pi
  const driven = (1 - pi) * (1 - p)
  return structural + driven === 0 ? 0 : structural / (structural + driven)
}

/** Options of `fitLearnerModel`. */
export interface LearnerModelOptions {
  /** `irt`, `ktm` or `zilm` (default `zilm`). */
  model?: LearnerModelKind
  /** Student conditions [students, K] (0 or 1) and item features [items, F]; required by `ktm` and `zilm`. */
  conditions?: Tensor
  itemFeatures?: Tensor
  /** Prior standard deviations of θ, b, log a and the context weights W (1, 2, 1, 2). */
  abilitySd?: number
  difficultySd?: number
  logDiscriminationSd?: number
  weightSd?: number
  /**
   * Most L-BFGS steps (default 1000) and the gradient-norm tolerance that ends the fit (default 10⁻⁴ per observed
   * response: the objective is a sum over responses, so its gradient grows with them).
   */
  maxSteps?: number
  tolerance?: number
  /** A fit to start from (its θ, b and a): IRT-ZILM otherwise starts from an IRT fit of its own. */
  init?: LearnerModelFit
}

/** A fitted learner model. */
export interface LearnerModelFit {
  readonly model: LearnerModelKind
  /** Abilities θ (one per student; MAP), difficulties b and discriminations a (one per item). */
  readonly ability: Float64Array
  readonly difficulty: Float64Array
  readonly discrimination: Float64Array
  /** w₀, the logit of π for a student without conditions (`zilm`; −∞ otherwise). */
  readonly intercept: number
  /** Context weights W [F × K] row-major: π's logit (`zilm`) or the answer's logit (`ktm`); empty for `irt`. */
  readonly weights: Float64Array
  readonly features: number
  readonly conditions: number
  /** The log-likelihood of the training responses at the fit (without the priors). */
  readonly logLikelihood: number
  readonly converged: boolean
  readonly steps: number
}

/**
 * The observed (training) responses as a list: student and item of each, and the answer. The likelihood is evaluated
 * on these pairs only, not on the whole [students, items] matrix.
 */
function observed(responses: Tensor, train?: Uint8Array) {
  const raw = toFlat(responses)
  const I = responses.shape[1]
  const student: number[] = []
  const item: number[] = []
  const y: number[] = []
  for (let k = 0; k < raw.length; k++)
    if (Number.isFinite(raw[k]) && (!train || train[k])) {
      if (raw[k] !== 0 && raw[k] !== 1)
        throw new DomainError('fitLearnerModel', 'fitLearnerModel: responses are 0 or 1')
      student.push(Math.floor(k / I))
      item.push(k % I)
      y.push(raw[k])
    }
  return {
    student: Int32Array.from(student),
    item: Int32Array.from(item),
    y: fromData(Float64Array.from(y), [y.length]),
  }
}

/** The log-likelihood of each observed response [N], from its z = a(θ − b), context and (IRT-ZILM) intercept. */
function pointwise(
  model: LearnerModelKind,
  y: Tensor,
  z: Value,
  context: Value | null,
  intercept: Value | null,
): Value {
  if (model === 'irt') return Bernoulli({ logits: z }).logProb(y)
  if (model === 'ktm') return Bernoulli({ logits: add(z, context!) }).logProb(y)
  return ZeroInflated({ logits: add(intercept!, context!) }, Bernoulli({ logits: z })).logProb(y)
}

/** A learner model's objective over a flat parameter vector x = [θ (P), b (I), log a (I), w₀ (zilm), W (F × K)]. */
export interface LearnerObjective {
  readonly model: LearnerModelKind
  /** The length of x, and the offsets of w₀ (zilm; −1 otherwise) and W. */
  readonly dim: number
  readonly interceptAt: number
  readonly weightsAt: number
  /** Observed (training) responses the likelihood sums over. */
  readonly n: number
  /** The log-likelihood of the observed responses, differentiable in x. */
  logLikelihood(x: Value): Value
  /** The negative log-posterior: the negative log-likelihood plus the Gaussian prior penalties. */
  objective(x: Value): Value
}

/**
 * The objective of `irt`, `ktm` or `zilm` on responses [students, items] of 0 and 1 (NaN: not attempted), over the
 * responses selected by `train` (a row-major mask; all observed ones by default). The log-likelihood is a sum over the
 * observed pairs only, with the answer's log-probability from `Bernoulli` (irt, ktm) or `ZeroInflated` (zilm).
 */
export function learnerObjective(
  responses: Tensor,
  options: LearnerModelOptions = {},
  train?: Uint8Array,
): LearnerObjective {
  const { model = 'zilm', abilitySd = 1, difficultySd = 2, logDiscriminationSd = 1, weightSd = 2 } = options
  const [P, I] = responses.shape
  const contextual = model !== 'irt'
  if (contextual && (!options.conditions || !options.itemFeatures))
    throw new DomainError('learnerObjective', `learnerObjective: ${model} needs conditions and itemFeatures`)
  const Z = options.conditions
  const X = options.itemFeatures
  if (Z && Z.shape[0] !== P)
    throw new ShapeError('learnerObjective', 'learnerObjective: one row of conditions per student')
  if (X && X.shape[0] !== I) throw new ShapeError('learnerObjective', 'learnerObjective: one row of features per item')
  const K = contextual ? Z!.shape[1] : 0
  const F = contextual ? X!.shape[1] : 0
  const { student, item, y } = observed(responses, train)
  const N = y.shape[0]
  // Each response's conditions [N, K], and the flat indices that gather its item's row of U = X W [I, K].
  const zFlat = Z ? toFlat(Z) : null
  const zObs = fromData(
    Float64Array.from({ length: N * K }, (_, m) => zFlat![student[Math.floor(m / K)] * K + (m % K)]),
    [N, K],
  )
  const uIndex = Int32Array.from({ length: N * K }, (_, m) => item[Math.floor(m / K)] * K + (m % K))
  const weightsAt = P + 2 * I + (model === 'zilm' ? 1 : 0)
  const dim = weightsAt + F * K
  const parts = (x: Value) => ({
    theta: slice(x, [0, P]),
    b: slice(x, [P, P + I]),
    logA: slice(x, [P + I, P + 2 * I]),
    intercept: model === 'zilm' ? slice(x, [P + 2 * I, P + 2 * I + 1]) : null,
    W: contextual ? reshape(slice(x, [weightsAt, dim]), [F, K]) : null,
  })
  const logLikelihood = (x: Value): Value => {
    const q = parts(x)
    // z = a_i (θ_s − b_i) for each observed (s, i).
    const z = mul(exp(gather(q.logA, item, [N])), sub(gather(q.theta, student, [N]), gather(q.b, item, [N])))
    // The context Σ_k z_sk Σ_f x_if W_fk of each response.
    const context = q.W ? sum(mul(zObs, gather(matmul(X!, q.W), uIndex, [N, K])), 1) : null
    return sum(pointwise(model, y, z, context, q.intercept))
  }
  const objective = (x: Value): Value => {
    const q = parts(x)
    let penalty: Value = add(
      mul(0.5 / (abilitySd * abilitySd), sum(square(q.theta))),
      mul(0.5 / (difficultySd * difficultySd), sum(square(q.b))),
    )
    penalty = add(penalty, mul(0.5 / (logDiscriminationSd * logDiscriminationSd), sum(square(q.logA))))
    if (q.W) penalty = add(penalty, mul(0.5 / (weightSd * weightSd), sum(square(q.W))))
    return sub(penalty, logLikelihood(x))
  }
  return {
    model,
    dim,
    interceptAt: model === 'zilm' ? P + 2 * I : -1,
    weightsAt,
    n: N,
    logLikelihood,
    objective,
  }
}

/**
 * Fit `irt`, `ktm` or `zilm` to responses [students, items] of 0 and 1 (NaN: not attempted) by penalised joint maximum
 * likelihood (`learnerObjective`, L-BFGS). `train`, a mask over the response matrix (row-major), restricts the fit to
 * some responses.
 */
export function fitLearnerModel(
  responses: Tensor,
  options: LearnerModelOptions = {},
  train?: Uint8Array,
): LearnerModelFit {
  const { model = 'zilm', maxSteps = 1000 } = options
  const [P, I] = responses.shape
  const f = learnerObjective(responses, options, train)
  const tolerance = options.tolerance ?? 1e-4 * f.n
  const { dim, weightsAt: offW } = f
  const K = model === 'irt' ? 0 : options.conditions!.shape[1]
  const F = model === 'irt' ? 0 : options.itemFeatures!.shape[1]
  const x0 = new Float64Array(dim)
  if (model === 'zilm' || options.init) {
    // IRT-ZILM starts from an IRT fit, with a small zero-inflation rate everywhere.
    const base = options.init ?? fitLearnerModel(responses, { ...options, model: 'irt' }, train)
    x0.set(base.ability, 0)
    x0.set(base.difficulty, P)
    x0.set(Float64Array.from(base.discrimination, Math.log), P + I)
    if (model === 'zilm') x0[P + 2 * I] = Math.log(0.05 / 0.95)
  }
  const vg = valueAndGrad(f.objective)
  const scalar = (v: Value) => {
    const u = unwrap(v)
    return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
  }
  const result = minimize(
    (x) => {
      const { value, grad } = vg(x)
      return { value: scalar(value), grad: unwrap(grad as Value) as Tensor }
    },
    x0,
    { method: 'lbfgs', maxSteps, tolerance },
  )
  const x = toFlat(result.x)
  return {
    model,
    ability: Float64Array.from(x.slice(0, P)),
    difficulty: Float64Array.from(x.slice(P, P + I)),
    discrimination: Float64Array.from(x.slice(P + I, P + 2 * I), Math.exp),
    intercept: model === 'zilm' ? x[P + 2 * I] : -Infinity,
    weights: Float64Array.from(x.slice(offW, dim)),
    features: F,
    conditions: K,
    logLikelihood: scalar(f.logLikelihood(result.x)),
    converged: result.converged,
    steps: result.steps,
  }
}

/** A fitted model's predictions for every student and item. */
export interface LearnerPredictions {
  /** Pr(Y = 1) [students × items] row-major. */
  correct: Float64Array
  /** The base model's p (the answer's probability were the context suitable) and π (0 for `irt` and `ktm`). */
  base: Float64Array
  pi: Float64Array
}

/** Predictions of a fitted learner model for every student and item, given the same conditions and item features. */
export function predictLearner(fit: LearnerModelFit, conditions?: Tensor, itemFeatures?: Tensor): LearnerPredictions {
  const P = fit.ability.length
  const I = fit.difficulty.length
  const { features: F, conditions: K } = fit
  const zc = conditions ? toFlat(conditions) : null
  const xc = itemFeatures ? toFlat(itemFeatures) : null
  if (fit.model !== 'irt' && (!zc || !xc))
    throw new DomainError('predictLearner', `predictLearner: ${fit.model} needs conditions and itemFeatures`)
  // u_ik = Σ_f x_if W_fk, the context of item i for condition k.
  const u = new Float64Array(I * K)
  for (let i = 0; i < I; i++)
    for (let k = 0; k < K; k++) {
      let v = 0
      for (let f = 0; f < F; f++) v += xc![i * F + f] * fit.weights[f * K + k]
      u[i * K + k] = v
    }
  const correct = new Float64Array(P * I)
  const base = new Float64Array(P * I)
  const pi = new Float64Array(P * I)
  for (let p = 0; p < P; p++)
    for (let i = 0; i < I; i++) {
      let c = 0
      for (let k = 0; k < K; k++) c += zc![p * K + k] * u[i * K + k]
      const z = fit.discrimination[i] * (fit.ability[p] - fit.difficulty[i])
      const at = p * I + i
      if (fit.model === 'zilm') {
        base[at] = sigmoid(z)
        pi[at] = sigmoid(fit.intercept + c)
        correct[at] = (1 - pi[at]) * base[at]
      } else {
        base[at] = sigmoid(z)
        correct[at] = fit.model === 'ktm' ? sigmoid(z + c) : base[at]
      }
    }
  return { correct, base, pi }
}
