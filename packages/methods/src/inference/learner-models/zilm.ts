/**
 * Zero-inflated learner models: IRT-ZILM and its two baselines, their objective, fit and predictions, and the
 * posterior that a zero is structural.
 *
 * The models are those of Twomey, McMullan, Elhalal, Poyiadzi and Vaquero (2022), "Equitable Ability Estimation in
 * Neurodivergent Student Populations with Zero-Inflated Learner Models". A learner model (LM) explains a zero (an
 * incorrect or unanswered item) by low ability. A zero-inflated learner model (ZILM) admits a second explanation: a low
 * learning quality factor (LQF), the delivery and response type (DRT) of the item being unsuitable for the student.
 * With $p$ the base model's probability of a correct answer and $\pi$ the zero-inflation probability (LQF $= 1 - \pi$),
 * the paper's Eqn (1) is
 *
 * $\pr(Y = 0 \mid \xvec) = \pi(\xvec_\pi) + (1 - \pi(\xvec_\pi))(1 - p(\xvec_p))$ and
 * $\pr(Y = 1 \mid \xvec) = (1 - \pi(\xvec_\pi))\, p(\xvec_p)$,
 *
 * the zero-inflated Bernoulli `ZeroInflated({ logits: η }, Bernoulli({ logits: z }))` of
 * `aifn-compute/probability/distributions`. In IRT-ZILM the base is the two-parameter IRT model, with logit
 * $z_{pi} = a_i(\theta_p - b_i)$ for student $p$ and item $i$, and $\pi = \sigma(\eta)$ is logistic in the student's
 * neurodivergent conditions (NDCs, the 0 or 1 entries $Z_{pk}$ of a $P \times K$ matrix $\Zmat$) crossed with the
 * item's DRT and content features (the $I \times F$ matrix $\Xmat$):
 *
 * $\eta_{pi} = w_0 + \sum_k Z_{pk} \sum_f X_{if} W_{fk}$,
 *
 * so the model learns, through the $F \times K$ weights $\Wmat$, which DRTs depress which conditions' answers. When
 * $\pi$ does not depend on the student, $1 - \pi$ is the upper asymptote of Barton and Lord's (1981) four-parameter
 * model. Two baselines share the base model and the fitter: `irt` ($\pi \equiv 0$, so IRT-ZILM with the zero
 * inflation switched off is exactly this model) and `ktm`, a linear knowledge-tracing machine in the sense of Vie and
 * Kashima (2019), which adds the same context to the logit,
 * $\operatorname{logit} \pr(Y = 1) = a_i(\theta_p - b_i) + \sum_k Z_{pk} \sum_f X_{if} W_{fk}$, instead of to a
 * separate cause of zeros.
 *
 * Fitting follows the paper: the negative log-likelihood of the training responses is minimised over every parameter
 * at once (abilities, difficulties, discriminations and the $\pi$ weights), with gradients by autodiff, here by
 * L-BFGS, and weak Gaussian priors $\theta \sim \Gauss(0, 1)$, $b \sim \Gauss(0, 2^2)$, $\log a \sim \Gauss(0, 1)$,
 * $W \sim \Gauss(0, 2^2)$ (none on $w_0$) that fix the scale of $\theta$ and keep the estimates finite. IRT-ZILM
 * starts from the IRT fit. Responses are a $P \times I$ matrix of 0 and 1, with NaN for an item not attempted.
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

/**
 * The logistic function.
 *
 * @param v The logit.
 * @returns $\sigma(v) = 1 / (1 + e^{-v})$.
 */
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

/**
 * $\pr(Y = 1) = (1 - \pi)\, \sigma(a(\theta - b))$: IRT-ZILM's probability of a correct answer ($\pi = 0$ gives the
 * 2PL model).
 *
 * @param theta The student's ability $\theta$.
 * @param a The item's discrimination $a$.
 * @param b The item's difficulty $b$.
 * @param pi The zero-inflation probability $\pi$ (0 for plain IRT).
 * @returns The probability of a correct answer.
 *
 * @example An even match, then the context taking 20%
 * print('2PL:', zilmProbability(0, 1, 0))
 * print('π = 0.2:', zilmProbability(0, 1, 0, 0.2))
 * print('able student, steep item:', zilmProbability(2, 1.5, 0.5))
 */
export function zilmProbability(theta: number, a: number, b: number, pi = 0): number {
  return (1 - pi) * sigmoid(a * (theta - b))
}

/**
 * The posterior probability that an observed zero is structural (caused by the context, probability $\pi$) rather
 * than an ability-driven incorrect answer (probability $(1 - \pi)(1 - p)$): $\pi / (\pi + (1 - \pi)(1 - p))$, Bayes'
 * rule on Eqn (1). Returns 0 when both are 0 (a zero that cannot occur).
 *
 * @param pi The zero-inflation probability $\pi$.
 * @param p The base model's probability $p$ of a correct answer.
 * @returns The posterior that the zero is structural.
 *
 * @example A zero from an able student is more likely the context's
 * print('p = 0.9:', structuralZeroPosterior(0.2, 0.9))
 * print('p = 0.1:', structuralZeroPosterior(0.2, 0.1))
 */
export function structuralZeroPosterior(pi: number, p: number): number {
  const structural = pi
  const driven = (1 - pi) * (1 - p)
  return structural + driven === 0 ? 0 : structural / (structural + driven)
}

/** Options of `fitLearnerModel` and `learnerObjective`. */
export interface LearnerModelOptions {
  /** `irt`, `ktm` or `zilm` (default `zilm`). */
  model?: LearnerModelKind
  /** Student conditions $\Zmat$, $P \times K$ (0 or 1); required by `ktm` and `zilm`. */
  conditions?: Tensor
  /** Item features $\Xmat$, $I \times F$; required by `ktm` and `zilm`. */
  itemFeatures?: Tensor
  /** Prior standard deviation of the abilities $\theta$ (default 1). */
  abilitySd?: number
  /** Prior standard deviation of the difficulties $b$ (default 2). */
  difficultySd?: number
  /** Prior standard deviation of the log discriminations $\log a$ (default 1). */
  logDiscriminationSd?: number
  /** Prior standard deviation of the context weights $\Wmat$ (default 2). */
  weightSd?: number
  /** The most L-BFGS steps (default 1000). */
  maxSteps?: number
  /**
   * The gradient-norm tolerance that ends the fit (default $10^{-4}$ per observed response: the objective is a sum
   * over responses, so its gradient grows with them).
   */
  tolerance?: number
  /** A fit to start from (its $\theta$, $b$ and $a$): IRT-ZILM otherwise starts from an IRT fit of its own. */
  init?: LearnerModelFit
}

/** A fitted learner model (plain data). */
export interface LearnerModelFit {
  /** Which model was fitted. */
  readonly model: LearnerModelKind
  /** The abilities $\theta_p$, one per student (MAP estimates). */
  readonly ability: Float64Array
  /** The difficulties $b_i$, one per item. */
  readonly difficulty: Float64Array
  /** The discriminations $a_i$, one per item. */
  readonly discrimination: Float64Array
  /** $w_0$, the logit of $\pi$ for a student without conditions (`zilm`; $-\infty$ otherwise). */
  readonly intercept: number
  /**
   * The context weights $\Wmat$ ($F \times K$, row-major): in $\pi$'s logit (`zilm`) or the answer's logit (`ktm`);
   * empty for `irt`.
   */
  readonly weights: Float64Array
  /** $F$, the number of item features (0 for `irt`). */
  readonly features: number
  /** $K$, the number of conditions (0 for `irt`). */
  readonly conditions: number
  /** The log-likelihood of the training responses at the fit (without the priors). */
  readonly logLikelihood: number
  /** Whether L-BFGS met the tolerance. */
  readonly converged: boolean
  /** The L-BFGS steps taken. */
  readonly steps: number
}

/**
 * The observed (training) responses as a list: student and item of each, and the answer. The likelihood is evaluated
 * on these pairs only, not on the whole $P \times I$ matrix. Throws `DomainError` for a finite response other than 0
 * or 1, naming `learnerObjective`, its only caller.
 *
 * @param responses The $P \times I$ responses: 0, 1, or NaN for not attempted.
 * @param train A row-major mask over the responses: only those with a non-zero entry are kept. All observed ones when
 *   left out.
 * @returns `student` and `item` (int32) and the answers `y` (a tensor), one entry per kept response, in row-major
 *   order.
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
        throw new DomainError('learnerObjective', `learnerObjective: responses are 0, 1 or NaN, got ${raw[k]}`)
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

/**
 * The log-likelihood of each observed response (length $N$), from its base logit $z = a(\theta - b)$, its context
 * and (IRT-ZILM) the intercept: `Bernoulli` on $z$ (`irt`), on $z$ plus the context (`ktm`), or `ZeroInflated` with
 * $\pi$'s logit $w_0$ plus the context (`zilm`).
 *
 * @param model The learner model.
 * @param y The observed answers, 0 or 1 (length $N$).
 * @param z The base logits $a_i(\theta_p - b_i)$ of the responses (length $N$).
 * @param context The context $\sum_k Z_{pk} \sum_f X_{if} W_{fk}$ of each response; null for `irt`, and required
 *   otherwise.
 * @param intercept The intercept $w_0$ (length 1); used by `zilm` only.
 * @returns The log-probabilities of the answers (length $N$), differentiable in the parameters.
 */
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

/**
 * A learner model's objective over a flat parameter vector $\xvec$: $\thetavec$ ($P$ values), $\bvec$ ($I$),
 * $\log \avec$ ($I$), $w_0$ (`zilm` only) and $\Wmat$ ($F \times K$, row-major), in that order.
 */
export interface LearnerObjective {
  /** Which model the objective is of. */
  readonly model: LearnerModelKind
  /** The length of $\xvec$. */
  readonly dim: number
  /** The offset of $w_0$ in $\xvec$ (`zilm`; $-1$ otherwise). */
  readonly interceptAt: number
  /** The offset of $\Wmat$ in $\xvec$ (equal to `dim` for `irt`, which has none). */
  readonly weightsAt: number
  /** Observed (training) responses the likelihood sums over. */
  readonly n: number
  /** The log-likelihood of the observed responses, differentiable in x. */
  logLikelihood(x: Value): Value
  /** The negative log-posterior: the negative log-likelihood plus the Gaussian prior penalties. */
  objective(x: Value): Value
}

/**
 * The objective of `irt`, `ktm` or `zilm` on responses ($P \times I$, of 0 and 1, NaN for not attempted), over the
 * responses selected by `train`. The log-likelihood is a sum over the observed pairs only, with the answer's
 * log-probability from `Bernoulli` (`irt`, `ktm`) or `ZeroInflated` (`zilm`); the objective adds the Gaussian prior
 * penalties. Both are differentiable in $\xvec$. Throws `DomainError` when `ktm` or `zilm` lacks `conditions` or
 * `itemFeatures` (or a response is not 0, 1 or NaN), and `ShapeError` when they do not have one row per student and
 * per item.
 *
 * @param responses The $P \times I$ responses.
 * @param options The model, its context and its priors; `maxSteps`, `tolerance` and `init` are not read here.
 * @param train A row-major mask over the responses selecting the training ones (non-zero: used). All observed ones
 *   when left out.
 * @returns The objective, with the layout of $\xvec$ and the number of responses it sums over.
 *
 * @example At zero parameters every answer has probability one half
 * const f = learnerObjective(tensor([[1, 0, NaN], [1, 1, 0]]), { model: 'irt' })
 * print('dim:', f.dim, 'responses:', f.n)
 * print('objective:', f.objective(zeros([f.dim])), '= 5 log 2:', 5 * Math.log(2))
 * print('gradient:', grad(f.objective)(zeros([f.dim])))
 *
 * @example IRT-ZILM's parameter layout
 * const conditions = tensor([[0], [1]])
 * const itemFeatures = tensor([[1, 0], [0, 1], [1, 1]])
 * const f = learnerObjective(tensor([[1, 0, NaN], [1, 1, 0]]), { model: 'zilm', conditions, itemFeatures })
 * print('dim:', f.dim, 'w0 at', f.interceptAt, 'W at', f.weightsAt)
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
 * Fit `irt`, `ktm` or `zilm` to responses ($P \times I$, of 0 and 1, NaN for not attempted) by penalised joint
 * maximum likelihood (`learnerObjective`, L-BFGS from `aifn-compute/optim/minimize`). IRT starts from zeros; IRT-ZILM
 * starts from `init`, or from an IRT fit of its own, with $\pi = 0.05$ everywhere. Throws as `learnerObjective` does.
 *
 * @param responses The $P \times I$ responses.
 * @param options The model, its context and priors, the L-BFGS limits and a fit to start from.
 * @param train A row-major mask over the responses that restricts the fit to some of them (non-zero: used). All
 *   observed ones when left out.
 * @returns The fitted parameters, the log-likelihood of the training responses, and whether L-BFGS converged.
 *
 * @example Two-parameter IRT on six students and four items
 * const responses = tensor([
 *   [1, 1, 1, 0],
 *   [1, 1, 0, 0],
 *   [1, 0, 0, 0],
 *   [1, 1, 1, 1],
 *   [0, 1, 0, 0],
 *   [1, 1, 0, NaN],
 * ])
 * const fit = fitLearnerModel(responses, { model: 'irt' })
 * print('ability:', fit.ability)
 * print('difficulty:', fit.difficulty)
 * print('converged:', fit.converged, 'in', fit.steps, 'steps')
 *
 * @example Zeros caused by the context: IRT lowers the ability, IRT-ZILM raises pi
 * // 60 students, a third with a condition; the odd items are timed, and a timed item gives a student with the
 * // condition a zero with probability 0.9 whatever their ability. On data this small the fit can also settle on
 * // pi near 0 (a large negative intercept) and reproduce IRT; the paper's experiments have hundreds of students.
 * const s = stream(1)
 * const ability = toFlat(normal(s, 0, 1, { shape: [60] }))
 * const timed = [0, 1, 0, 1, 0, 1, 0, 1]
 * const difficulty = timed.map((_, i) => -1 + (2 * i) / 7)
 * const group = Array.from(ability, (_, p) => (p % 3 === 0 ? 1 : 0))
 * const y = (p, b) => bernoulli(s, zilmProbability(ability[p], 1, b))
 * const rows = group.map((g, p) => difficulty.map((b, i) => (g && timed[i] && uniform(s) < 0.9 ? 0 : y(p, b))))
 * const context = { conditions: tensor(group.map((g) => [g])), itemFeatures: tensor(timed.map((t) => [t])) }
 * const irt = fitLearnerModel(tensor(rows), { model: 'irt' })
 * const zilm = fitLearnerModel(tensor(rows), { model: 'zilm', ...context, init: irt })
 * const bias = (fit) => fit.ability.reduce((a, v, p) => a + (group[p] ? v - ability[p] : 0), 0) / 20
 * print('mean ability error with the condition: IRT', bias(irt), 'IRT-ZILM', bias(zilm))
 * const sigmoid = (v) => 1 / (1 + Math.exp(-v))
 * print('IRT-ZILM pi: without the condition', sigmoid(zilm.intercept))
 * print('with it, on a timed item', sigmoid(zilm.intercept + zilm.weights[0]))
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

/** A fitted model's predictions for every student and item, each $P \times I$ row-major. */
export interface LearnerPredictions {
  /** $\pr(Y = 1)$. */
  correct: Float64Array
  /** The base model's $p = \sigma(a_i(\theta_p - b_i))$: the answer's probability were the context suitable. */
  base: Float64Array
  /** The zero-inflation probability $\pi$ (0 for `irt` and `ktm`). */
  pi: Float64Array
}

/**
 * Predictions of a fitted learner model for every student and item, given the same conditions and item features it
 * was fitted with. Throws `DomainError` when a `ktm` or `zilm` fit is given no conditions or item features.
 *
 * @param fit The fitted model (only its fields are read, so a fit written by hand works too).
 * @param conditions The student conditions $\Zmat$, $P \times K$; not needed for `irt`.
 * @param itemFeatures The item features $\Xmat$, $I \times F$; not needed for `irt`.
 * @returns $\pr(Y = 1)$, $p$ and $\pi$ for every student and item.
 *
 * @example One timed item, a student without and one with the condition
 * const fit = {
 *   model: 'zilm',
 *   ability: [0, 1],
 *   difficulty: [0],
 *   discrimination: [1],
 *   intercept: -2,
 *   weights: [3],
 *   features: 1,
 *   conditions: 1,
 * }
 * const pred = predictLearner(fit, tensor([[0], [1]]), tensor([[1]]))
 * print('p = σ(θ):', pred.base)
 * print('π = σ(-2), σ(1):', pred.pi)
 * print('Pr(correct):', pred.correct)
 */
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
