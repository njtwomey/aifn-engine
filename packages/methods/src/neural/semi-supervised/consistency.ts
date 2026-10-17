/**
 * Consistency regularisation for a small classifier network as one traceable algorithm, and the target and threshold
 * rules the methods share.
 *
 * Every step takes one update of the network on the supervised cross-entropy of all labelled points plus a weighted
 * unsupervised term on a batch of unlabelled points (all of them by default), with the engine's autodiff and a pytree
 * update rule (Adam by default). The unsupervised term is chosen by `method`:
 *
 * - `supervised`: none (the baseline).
 * - `pi-model`: the squared distance between the softmax outputs of two independently perturbed copies of each input,
 *   gradients through both (Laine and Aila, 2017, §2.1; Sajjadi et al., 2016).
 * - `temporal-ensembling`: the squared distance between the output on one perturbed copy and the bias-corrected
 *   running average $Z_i / (1 - \alpha^{c_i})$ of the point's past outputs, $Z_i \leftarrow \alpha Z_i + (1 - \alpha)
 *   z_i$, where $c_i$ counts the updates of row $i$ (Laine and Aila, 2017, §2.2, who update once per epoch). A row
 *   enters the loss once it has an average.
 * - `mean-teacher`: the squared distance between the student's output and that of a teacher whose weights are the
 *   exponential moving average of the student's, $\thetavec' \leftarrow \alpha \thetavec' + (1 - \alpha) \thetavec$
 *   (`parameterEma`), each on its own perturbed copy (Tarvainen and Valpola, 2017).
 * - `vat`: the KL divergence between the prediction at the clean input and at the input moved by the virtual
 *   adversarial perturbation of length $\epsilon$, found by `iterations` power-iteration steps from a random direction
 *   with finite-difference length $\xi$ (Miyato et al., 2019, Algorithm 1); optionally plus the entropy of the clean
 *   predictions (VAT with entropy minimisation).
 * - `uda`: the cross-entropy between the sharpened prediction at the clean input (temperature $\tau$, kept only where
 *   the unsharpened confidence exceeds $\beta$) and the prediction on a strongly perturbed copy, normalised by the
 *   batch size; training-signal annealing drops labelled points whose true-class probability exceeds
 *   $\eta_t = \alpha_t (1 - 1/K) + 1/K$ under the log, linear or exp schedule (Xie et al., 2020, §2.4 and
 *   appendix A.1).
 * - `mixmatch`: label guessing by averaging the softmax over `augmentations` weakly perturbed copies, sharpening with
 *   temperature $T$, then mixup of the labelled and unlabelled batches with one $\lambda' = \max(\lambda, 1 -
 *   \lambda)$, $\lambda \sim \operatorname{Beta}(\alpha, \alpha)$, per step; the cross-entropy on the mixed labelled
 *   batch and the squared distance (divided by $K$) on the mixed unlabelled batch (Berthelot et al., 2019,
 *   Algorithm 1).
 * - `fixmatch`: the cross-entropy between the arg max of the prediction on a weakly perturbed copy and the prediction
 *   on a strongly perturbed copy, for the points whose weak-view confidence reaches the threshold, normalised by the
 *   batch size (Sohn et al., 2020, eq. 4). `thresholds` picks the threshold rule: `fixed` ($\tau$), `flexmatch`
 *   (per-class $\mathcal{M}(\beta_t(c))\,\tau$, $\mathcal{M}(x) = x / (2 - x)$, from each point's last confident
 *   prediction, with the warm-up denominator; Zhang et al., 2021, eqs. 5, 11 and 12), `freematch` (the EMA of the
 *   batch's mean confidence times the max-normalised EMA of its mean class probabilities; Wang et al., 2023, eqs. 5
 *   to 7) or `softmatch` (weight 1 at or above the EMA mean confidence $\mu_t$,
 *   $\exp(-(c - \mu_t)^2 / (2 \sigma_t^2 / 4))$ below it, with $\sigma_t^2$ the EMA of the confidence variance; Chen
 *   et al., 2023, eq. 7 and §4.2).
 *
 * Differences from the papers: perturbations are the caller's `augment` (default: isotropic Gaussian jitter of
 * standard deviation `weak` or `strong`), not image augmentations; the labelled set enters every step whole; the
 * unlabelled batch is a fresh uniform subset each step (no epochs); FreeMatch's fairness term and SoftMatch's uniform
 * alignment are left out. The unsupervised weight follows the Gaussian ramp-up $\lambda_u \exp(-5 (1 - t/R)^2)$ of
 * Laine and Aila for $t < R$ when `rampUp` $= R > 0$. States are plain data; a step is a pure function of its state
 * and stream. Probabilities are row-major $n \times K$ `Float64Array`s.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { grad, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, permutation, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  float64Data,
  fromData,
  mean,
  mul,
  shapeOf,
  square,
  sub,
  sum,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import type { Layer } from 'aifn-compute/nn/layers'
import { parameterEma } from 'aifn-compute/nn/training'
import { logSoftmax, softmax } from 'aifn-compute/numerics/special'
import { adamRule, applyUpdates, type UpdateRule } from 'aifn-compute/optim/first-order'
import { beta } from 'aifn-compute/probability/samplers'
import { rowArgmax } from 'aifn-methods/learning/semi-supervised'

/** Threshold rules of `fixmatch`: a fixed $\tau$, or FlexMatch's, FreeMatch's or SoftMatch's adaptive ones. */
export type ConfidenceThresholdRule = 'fixed' | 'flexmatch' | 'freematch' | 'softmatch'

/** Training-signal annealing schedules of UDA: none, or $\alpha_t$ rising as a log, linear or exp curve. */
export type AnnealingSchedule = 'none' | 'log' | 'linear' | 'exp'

/** The unsupervised term; see the file comment. */
export type ConsistencyMethod =
  | { kind: 'supervised' }
  | { kind: 'pi-model' }
  /** `momentum`: the ensemble momentum $\alpha$ (default 0.6, Laine and Aila's value). */
  | { kind: 'temporal-ensembling'; momentum?: number }
  /** `decay`: the EMA decay $\alpha$ of the teacher's weights (default 0.99). */
  | { kind: 'mean-teacher'; decay?: number }
  /**
   * `radius` $\epsilon$ (default 0.3), `xi` $\xi$ (default 1e-6), `iterations` of the power method (default 1),
   * `entropy`: add the entropy of the clean predictions (default false).
   */
  | { kind: 'vat'; radius?: number; xi?: number; iterations?: number; entropy?: boolean }
  /**
   * `temperature` $\tau$ of the sharpened target (default 0.4), `confidence` mask $\beta$ (default 0.8, 0 turns it
   * off), `annealing` schedule of the labelled loss (default `none`).
   */
  | { kind: 'uda'; temperature?: number; confidence?: number; annealing?: AnnealingSchedule }
  /**
   * `augmentations` $K'$ (default 2), sharpening `temperature` $T$ (default 0.5), mixup `alpha` (default 0.75; 0 turns
   * mixup off).
   */
  | { kind: 'mixmatch'; augmentations?: number; temperature?: number; alpha?: number }
  /**
   * `threshold` $\tau$ (default 0.95), the `thresholds` rule (default `fixed`), the `momentum` of FreeMatch's and
   * SoftMatch's moving averages (default 0.999, the papers' value), FlexMatch's `warmup` denominator (default true).
   */
  | { kind: 'fixmatch'; threshold?: number; thresholds?: ConfidenceThresholdRule; momentum?: number; warmup?: boolean }

/** Perturbation strength asked of `augment`. */
export type Strength = 'weak' | 'strong'

/** The data of a run. */
export interface ConsistencyProblem {
  /** Labelled inputs, $n_L \times d$. */
  labelledX: Tensor
  /** Their classes in $0, \dots, K - 1$. */
  labelledY: ArrayLike<number>
  /** Unlabelled inputs, $n_U \times d$. */
  unlabelledX: Tensor
  /** The number of classes $K \ge 2$. */
  classes: number
  /** True classes of the unlabelled inputs, for diagnostics only (never used to fit). */
  truth?: ArrayLike<number>
}

/** Controls of `consistencyTraining`. */
export interface ConsistencyOptions {
  /** The network: inputs $n \times d$ to logits $n \times K$ (e.g. the engine's `Mlp([d, ..., K])`). */
  model: Layer<Params[]>
  /** The unsupervised term. */
  method: ConsistencyMethod
  /** The unsupervised weight $\lambda_u$ (default 1). */
  weight?: number
  /** Steps $R$ of the Gaussian ramp-up of $\lambda_u$ (default 0: none). */
  rampUp?: number
  /** Planned steps $T$, for UDA's annealing schedule (default 500). */
  steps?: number
  /** Unlabelled points per step (default all of them). */
  batch?: number
  /** Standard deviation of the default weak jitter (default 0.05). */
  weak?: number
  /** Standard deviation of the default strong jitter (default 0.25). */
  strong?: number
  /** A perturbation of a batch of inputs; default isotropic Gaussian jitter of sd `weak` or `strong`. */
  augment?: (x: Tensor, strength: Strength, s: Stream) => Tensor
  /** The update rule (default `adamRule({ stepSize: 0.01 })`). */
  optimizer?: UpdateRule<unknown>
}

/**
 * The state after $t$ updates. The per-point diagnostics (`mask`, `targets`, `pseudo`, `perturbation`) hold, for each
 * unlabelled point, the values of the last step whose batch contained it; the scalar diagnostics describe the step
 * that produced the state (NaN at $t = 0$).
 */
export interface ConsistencyState extends Status {
  /** Updates made. */
  t: number
  /** The network's parameters (the student's, for mean teacher). */
  params: Params[]
  /** The update rule's state. */
  optimizer: unknown
  /** Mean teacher's EMA weights; null for the other methods. */
  teacher: Params[] | null
  /** Temporal ensembling's accumulated outputs $\Zmat$, row-major $n_U \times K$; empty for the other methods. */
  ensemble: Float64Array
  /** Temporal ensembling's update count of each row (empty otherwise). */
  ensembleCounts: Int32Array
  /** FlexMatch's mark of each unlabelled point: its class when last predicted above $\tau$, else $-1$. */
  marks: Int32Array
  /** FreeMatch's global threshold $\tau_t$, or SoftMatch's mean confidence $\mu_t$ (NaN otherwise). */
  confidenceMean: number
  /** SoftMatch's confidence variance $\sigma_t^2$ (NaN otherwise). */
  confidenceVariance: number
  /** FreeMatch's EMA of the mean class probabilities $\tilde p_t$ (empty otherwise). */
  classMean: number[]
  /** The objective of the step: supervised plus weighted unsupervised loss. */
  loss: number
  /** The supervised part. */
  supervisedLoss: number
  /** The unsupervised part, before weighting. */
  unsupervisedLoss: number
  /** The unsupervised weight used in the step. */
  weight: number
  /** The unlabelled rows of the step's batch. */
  batch: Int32Array
  /** Each unlabelled point's weight in the unsupervised term, in $[0, 1]$. */
  mask: Float64Array
  /** The target distribution of each unlabelled point (teacher output, guess, sharpened prediction), $n_U \times K$. */
  targets: Float64Array
  /** The arg max of each target (the pseudo-label), $-1$ before the point has one. */
  pseudo: Int32Array
  /** Mean of the mask over the step's batch: the fraction of the batch used. */
  maskRate: number
  /** Mask-weighted accuracy of the batch's pseudo-labels against `truth` (NaN without truth or with no weight). */
  pseudoAccuracy: number
  /** The per-class thresholds of the step (`fixmatch`; empty otherwise; SoftMatch reports $\mu_t$ for every class). */
  thresholds: number[]
  /** Fraction of labelled points kept by UDA's annealing (1 otherwise). */
  labelledKept: number
  /** VAT's adversarial perturbation of each unlabelled point, $n_U \times d$ (empty otherwise). */
  perturbation: Float64Array
}

/** The empty array a method's unused per-point state holds. */
const EMPTY: Float64Array = new Float64Array(0)

/**
 * The class probabilities of a network on `x`: the softmax of its logits, row-major.
 *
 * @param model The network, inputs to $K$ logits per row.
 * @param params Its parameters.
 * @param x The inputs, $n \times d$.
 * @returns $nK$ probabilities, row $i$ at entries $iK$ to $iK + K - 1$.
 *
 * @example The probabilities of an untrained network
 * import { Mlp } from 'aifn-compute/nn/layers'
 * const net = Mlp([2, 4, 3])
 * const p = predictProbabilities(net, net.init(stream(1)), tensor([[0, 0], [1, -1]]))
 * print('two rows of three:', p)
 */
export function predictProbabilities(model: Layer<Params[]>, params: Params[], x: Tensor): Float64Array {
  return Float64Array.from(float64Data(softmax(model.apply(params, x) as Tensor)))
}

/**
 * Sharpening $p_k^{1/T} / \sum_j p_j^{1/T}$ of each row (MixMatch, eq. 7): $T = 1$ leaves a row unchanged and
 * $T \to 0$ gives the one-hot arg max. Each row is first scaled by its largest entry, so a small $T$ does not
 * underflow. Throws `DomainError` for a row with no positive entry, which has no sharpened form, and for $T \le 0$.
 *
 * @param p Probability rows, row-major $n \times K$; not modified.
 * @param K The number of classes (columns).
 * @param T The temperature $T > 0$.
 * @returns The sharpened rows, a new array of the same length.
 *
 * @example Sharper and sharper
 * print('T = 1:', sharpen([0.6, 0.4], 2, 1))
 * print('T = 0.5:', sharpen([0.6, 0.4], 2, 0.5))
 * print('T = 0.01:', sharpen([0.6, 0.4], 2, 0.01))
 */
export function sharpen(p: ArrayLike<number>, K: number, T: number): Float64Array {
  if (!(T > 0)) throw new DomainError('sharpen', 'sharpen: the temperature T must be positive')
  const out = new Float64Array(p.length)
  for (let i = 0; i < p.length / K; i++) {
    let top = 0
    for (let k = 0; k < K; k++) top = Math.max(top, p[i * K + k])
    if (!(top > 0)) throw new DomainError('sharpen', `sharpen: row ${i} has no positive entry`)
    let z = 0
    for (let k = 0; k < K; k++) {
      // Scaled by the row's largest entry so that a small T does not underflow.
      const v = Math.pow(p[i * K + k] / top, 1 / T)
      out[i * K + k] = v
      z += v
    }
    for (let k = 0; k < K; k++) out[i * K + k] /= z
  }
  return out
}

/**
 * UDA's training-signal annealing threshold $\eta_t = \alpha_t (1 - 1/K) + 1/K$ at step $t$ of $T$ (Xie et al., 2020,
 * appendix A.1), with $\alpha_t$ equal to $1 - e^{-5r}$ (log), $r$ (linear) or $e^{5(r - 1)}$ (exp) for
 * $r = \min(1, t/T)$; 1 for `none`. Labelled points whose true-class probability exceeds it leave the loss.
 *
 * @param schedule The schedule.
 * @param t The step.
 * @param T The planned number of steps.
 * @param K The number of classes.
 * @returns The threshold $\eta_t \in [1/K, 1]$.
 *
 * @example The three schedules half-way through
 * for (const s of ['log', 'linear', 'exp']) print(s, annealingThreshold(s, 50, 100, 2))
 */
export function annealingThreshold(schedule: AnnealingSchedule, t: number, T: number, K: number): number {
  if (schedule === 'none') return 1
  const r = Math.min(1, t / T)
  const a = schedule === 'log' ? 1 - Math.exp(-5 * r) : schedule === 'linear' ? r : Math.exp(5 * (r - 1))
  return a * (1 - 1 / K) + 1 / K
}

/**
 * FlexMatch's per-class thresholds $\mathcal{M}(\beta(c))\,\tau$ with $\mathcal{M}(x) = x / (2 - x)$ and
 * $\beta(c) = \sigma(c) / \max(\max_c \sigma(c),\ N - \sum_c \sigma(c))$ from the class counts $\sigma(c)$ of the $N$
 * unlabelled points (Zhang et al., 2021, eqs. 11 and 12; without warm-up the denominator is $\max_c \sigma(c)$). A
 * class no point has been confidently predicted as gets threshold 0.
 *
 * @param counts $\sigma(c)$: how many unlabelled points were last predicted as class $c$ above $\tau$.
 * @param N The number of unlabelled points.
 * @param tau The fixed threshold $\tau$ the rule scales.
 * @param warmup Whether the points not yet predicted confidently count in the denominator.
 * @returns One threshold per class.
 *
 * @example A well-learnt class and a lagging one
 * print('thresholds:', flexMatchThresholds([50, 25], 100, 0.95))
 */
export function flexMatchThresholds(counts: readonly number[], N: number, tau: number, warmup = true): number[] {
  const top = Math.max(...counts)
  const unused = N - counts.reduce((a, b) => a + b, 0)
  const denominator = warmup ? Math.max(top, unused) : top
  return counts.map((c) => {
    const b = denominator > 0 ? c / denominator : 0
    return (b / (2 - b)) * tau
  })
}

/**
 * The Gaussian ramp-up weight $\lambda \exp(-5 (1 - t / R)^2)$ for $t < R$, else $\lambda$ (Laine and Aila, 2017,
 * appendix A); $R \le 0$ turns it off.
 *
 * @param weight The full weight $\lambda$.
 * @param t The step.
 * @param R The length of the ramp-up in steps.
 * @returns The weight at step $t$.
 *
 * @example From $\lambda e^{-5}$ to $\lambda$
 * print([0, 5, 9, 10].map((t) => rampUpWeight(1, t, 10)))
 */
export function rampUpWeight(weight: number, t: number, R: number): number {
  if (!(R > 0) || t >= R) return weight
  const r = t / R
  return weight * Math.exp(-5 * (1 - r) * (1 - r))
}

/**
 * A float64 tensor of the given shape over the data (a constant in the traced objective).
 *
 * @param data The values, row-major.
 * @param shape The shape.
 * @returns The tensor.
 */
const constant = (data: Float64Array, shape: readonly number[]) => fromData(data, shape)

/** `valueAndGrad` with the parameter type erased, which keeps the checker from expanding the pytree types. */
const gradientOf = valueAndGrad as unknown as (
  f: (p: Params[]) => Value,
) => (p: Params[]) => { value: Value; grad: unknown }

/**
 * The mean over rows of the squared distance between probability rows, divided by $K$ (Laine and Aila's
 * normalisation).
 *
 * @param p The first rows, $n \times K$ (traced or not).
 * @param q The second rows, of the same shape.
 * @param K The number of classes.
 * @returns $\frac{1}{nK} \sum_i \lVert \pvec_i - \qvec_i \rVert^2$.
 */
const squaredDistance = (p: Value, q: Value, K: number): Value => mul(1 / K, mean(sum(square(sub(p, q)), -1)))

/**
 * The mean over rows of $\mathrm{KL}(\text{target} \,\Vert\, \operatorname{softmax}(\text{logits}))$, the targets
 * constant.
 *
 * @param targets The target rows, row-major $n \times K$.
 * @param logits The logits, $n \times K$ (traced or not).
 * @param n The number of rows.
 * @param K The number of classes.
 * @returns The mean KL divergence.
 */
function klFromTargets(targets: Float64Array, logits: Value, n: number, K: number): Value {
  let negativeEntropy = 0
  for (let i = 0; i < targets.length; i++) if (targets[i] > 0) negativeEntropy += targets[i] * Math.log(targets[i])
  const cross = sum(mul(constant(targets, [n, K]), logSoftmax(logits)))
  return mul(1 / n, sub(negativeEntropy, cross))
}

/**
 * The rows of a row-major matrix normalised to unit Euclidean length (a zero row stays zero).
 *
 * @param v The rows, row-major; not modified.
 * @param d The row length.
 * @returns The normalised rows.
 */
function unitRows(v: ArrayLike<number>, d: number): Float64Array {
  const out = new Float64Array(v.length)
  for (let i = 0; i < v.length / d; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += v[i * d + j] ** 2
    const r = Math.sqrt(s) || 1
    for (let j = 0; j < d; j++) out[i * d + j] = v[i * d + j] / r
  }
  return out
}

/**
 * The listed rows of a row-major matrix.
 *
 * @param data The matrix, row-major; not modified.
 * @param rows The row indices to copy, in order.
 * @param width The number of columns.
 * @returns The rows, row-major.
 */
function pickRows(data: ArrayLike<number>, rows: ArrayLike<number>, width: number): Float64Array {
  const out = new Float64Array(rows.length * width)
  for (let r = 0; r < rows.length; r++) for (let j = 0; j < width; j++) out[r * width + j] = data[rows[r] * width + j]
  return out
}

/**
 * Consistency-regularised training of a classifier network (see the file comment for the methods). Step $t$ draws
 * the unlabelled batch from `child(stream, 'batch')`, the perturbed views from `child(stream, 'view', k)` (MixMatch's
 * guesses from `'guess'`, its labelled view from `'labelled'`, shuffle from `'shuffle'` and mixup from `'mixup'`;
 * VAT's start direction from `'direction'`). Throws `ShapeError` when the labelled inputs and classes, or the
 * unlabelled inputs and `truth`, differ in rows. The state flags `diverged` when the loss is not finite. Options hold
 * a network and an optional function: for a worker, `consistencyRun` takes plain data.
 *
 * @param problem The labelled and unlabelled inputs, the classes, and optional truth for diagnostics.
 * @param options The network, the method, the unsupervised weight and ramp-up, the batch, the perturbations and the
 *   update rule.
 * @returns The algorithm; start it from `{ params }` (e.g. `model.init(stream)`), or from `undefined` to initialise
 *   from the init stream's child `params`.
 *
 * @example FixMatch on four points
 * import { Mlp } from 'aifn-compute/nn/layers'
 * const problem = {
 *   labelledX: tensor([[-1, 0], [1, 0]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.3], [-0.8, -0.2], [0.9, 0.4], [1.1, -0.3]]),
 *   classes: 2,
 *   truth: [0, 0, 1, 1],
 * }
 * const net = Mlp([2, 8, 2], { activation: 'tanh' })
 * const alg = consistencyTraining(problem, { model: net, method: { kind: 'fixmatch', threshold: 0.7 } })
 * const s = run(alg, { params: net.init(stream(1)) }, 30)
 * print('loss:', s.loss, ' share of the batch used:', s.maskRate)
 * print('pseudo-labels:', s.pseudo)
 */
export function consistencyTraining(
  problem: ConsistencyProblem,
  options: ConsistencyOptions,
): Algorithm<{ params: Params[] } | undefined, ConsistencyState> {
  const { labelledX, labelledY, unlabelledX, classes: K, truth } = problem
  const { model, method } = options
  const lambda = options.weight ?? 1
  const R = options.rampUp ?? 0
  const T = options.steps ?? 500
  const sdWeak = options.weak ?? 0.05
  const sdStrong = options.strong ?? 0.25
  const rule = options.optimizer ?? adamRule({ stepSize: 0.01 })
  const nL = shapeOf(labelledX)[0]
  const [nU, d] = shapeOf(unlabelledX)
  const B = Math.min(nU, Math.max(1, Math.round(options.batch ?? nU)))
  if (labelledY.length !== nL)
    throw new ShapeError('consistencyTraining', 'consistencyTraining: labelledX and labelledY differ in rows')
  if (truth && truth.length !== nU)
    throw new ShapeError('consistencyTraining', 'consistencyTraining: truth and unlabelledX differ in rows')
  const augment =
    options.augment ??
    ((x: Tensor, strength: Strength, s: Stream) =>
      add(x, normals(s, shapeOf(x) as number[], 0, strength === 'weak' ? sdWeak : sdStrong)) as Tensor)
  const yL = Array.from(labelledY)
  const oneHotL = new Float64Array(nL * K)
  yL.forEach((c, i) => (oneHotL[i * K + c] = 1))
  const allX = float64Data(unlabelledX)
  const allRows = Int32Array.from({ length: nU }, (_, i) => i)
  const adaptive = method.kind === 'fixmatch' ? (method.thresholds ?? 'fixed') : 'fixed'

  return {
    name: `consistencyTraining(${method.kind})`,
    init(start, s) {
      const params = start?.params ?? model.init(child(s, 'params'))
      return {
        t: 0,
        params,
        optimizer: rule.init(params),
        teacher: method.kind === 'mean-teacher' ? params : null,
        ensemble: method.kind === 'temporal-ensembling' ? new Float64Array(nU * K) : EMPTY,
        ensembleCounts: new Int32Array(method.kind === 'temporal-ensembling' ? nU : 0),
        marks: new Int32Array(adaptive === 'flexmatch' ? nU : 0).fill(-1),
        confidenceMean: adaptive === 'freematch' || adaptive === 'softmatch' ? 1 / K : NaN,
        confidenceVariance: adaptive === 'softmatch' ? 1 : NaN,
        classMean: adaptive === 'freematch' ? new Array(K).fill(1 / K) : [],
        loss: NaN,
        supervisedLoss: NaN,
        unsupervisedLoss: NaN,
        weight: NaN,
        batch: new Int32Array(0),
        mask: new Float64Array(nU),
        targets: new Float64Array(nU * K),
        pseudo: new Int32Array(nU).fill(-1),
        maskRate: NaN,
        pseudoAccuracy: NaN,
        thresholds: [],
        labelledKept: 1,
        perturbation: method.kind === 'vat' ? new Float64Array(nU * d) : EMPTY,
      }
    },
    step(state, ctx) {
      const t = state.t
      const s = ctx.stream
      const params = state.params
      const w = rampUpWeight(lambda, t, R)
      const rows = B === nU ? allRows : Int32Array.from(float64Data(permutation(child(s, 'batch'), nU)).subarray(0, B))
      const nB = rows.length
      const xU = B === nU ? unlabelledX : constant(pickRows(allX, rows, d), [nB, d])
      const next: Partial<ConsistencyState> = {}
      // Per-batch-row diagnostics, written back into the full arrays at the end.
      // Declared as plain Float64Array so that views and engine results (any buffer type) can be assigned.
      let mask: Float64Array = new Float64Array(nB).fill(1)
      let targets: Float64Array = new Float64Array(nB * K)
      let hasTarget = true
      let thresholds: number[] = []
      let labelledKept = 1
      let supervised: (p: Params[]) => Value = (p) => softmaxCrossEntropy(model.apply(p, labelledX), yL)
      let unsupervised: ((p: Params[]) => Value) | null = null

      switch (method.kind) {
        case 'supervised':
          mask = new Float64Array(nB)
          hasTarget = false
          break
        case 'pi-model': {
          const a = augment(xU, 'weak', child(s, 'view', 0))
          const b = augment(xU, 'weak', child(s, 'view', 1))
          targets = predictProbabilities(model, params, b)
          unsupervised = (p) => squaredDistance(softmax(model.apply(p, a)), softmax(model.apply(p, b)), K)
          break
        }
        case 'temporal-ensembling': {
          const alpha = method.momentum ?? 0.6
          const a = augment(xU, 'weak', child(s, 'view', 0))
          const z = predictProbabilities(model, params, a)
          const Z = Float64Array.from(state.ensemble)
          const counts = Int32Array.from(state.ensembleCounts)
          for (let r = 0; r < nB; r++) {
            const i = rows[r]
            const c = counts[i]
            if (c > 0) for (let k = 0; k < K; k++) targets[r * K + k] = Z[i * K + k] / (1 - Math.pow(alpha, c))
            else mask[r] = 0
          }
          const target = constant(Float64Array.from(targets), [nB, K])
          const m = constant(Float64Array.from(mask), [nB, 1])
          unsupervised = (p) => mul(1 / K, mean(sum(mul(m, square(sub(softmax(model.apply(p, a)), target))), -1)))
          for (let r = 0; r < nB; r++) {
            const i = rows[r]
            for (let k = 0; k < K; k++) Z[i * K + k] = alpha * Z[i * K + k] + (1 - alpha) * z[r * K + k]
            counts[i]++
          }
          next.ensemble = Z
          next.ensembleCounts = counts
          break
        }
        case 'mean-teacher': {
          const a = augment(xU, 'weak', child(s, 'view', 0))
          const b = augment(xU, 'weak', child(s, 'view', 1))
          targets = predictProbabilities(model, state.teacher ?? params, b)
          const target = constant(Float64Array.from(targets), [nB, K])
          unsupervised = (p) => squaredDistance(softmax(model.apply(p, a)), target, K)
          break
        }
        case 'vat': {
          const eps = method.radius ?? 0.3
          const xi = method.xi ?? 1e-6
          targets = predictProbabilities(model, params, xU)
          const kl = (r: Value) => klFromTargets(targets, model.apply(params, add(xU, r)), nB, K)
          let dir = unitRows(float64Data(normals(child(s, 'direction'), [nB, d])), d)
          for (let k = 0; k < (method.iterations ?? 1); k++) {
            const g = grad(kl)(
              constant(
                Float64Array.from(dir, (v) => xi * v),
                [nB, d],
              ),
            ) as Tensor
            dir = unitRows(float64Data(g), d)
          }
          const r = Float64Array.from(dir, (v) => eps * v)
          const moved = add(xU, constant(r, [nB, d]))
          const P = Float64Array.from(state.perturbation)
          for (let q = 0; q < nB; q++) for (let j = 0; j < d; j++) P[rows[q] * d + j] = r[q * d + j]
          next.perturbation = P
          const entropy = method.entropy ?? false
          unsupervised = (p) => {
            const vat = klFromTargets(targets, model.apply(p, moved), nB, K)
            if (!entropy) return vat
            const logits = model.apply(p, xU)
            return add(vat, mul(-1 / nB, sum(mul(softmax(logits), logSoftmax(logits)))))
          }
          break
        }
        case 'uda': {
          const tau = method.temperature ?? 0.4
          const beta0 = method.confidence ?? 0.8
          const logits = float64Data(model.apply(params, xU) as Tensor)
          const p = float64Data(softmax(fromData(logits, [nB, K])))
          targets = Float64Array.from(
            float64Data(
              softmax(
                fromData(
                  Float64Array.from(logits, (v) => v / tau),
                  [nB, K],
                ),
              ),
            ),
          )
          const { conf } = rowArgmax(p, K)
          mask = Float64Array.from(conf, (c) => (c > beta0 ? 1 : 0))
          const strong = augment(xU, 'strong', child(s, 'view', 0))
          const target = constant(
            Float64Array.from(targets, (v, j) => v * mask[Math.floor(j / K)]),
            [nB, K],
          )
          unsupervised = (q) => mul(-1 / nB, sum(mul(target, logSoftmax(model.apply(q, strong)))))
          // Training-signal annealing: keep the labelled points whose true-class probability is at most eta_t.
          const eta = annealingThreshold(method.annealing ?? 'none', t, T, K)
          const pL = predictProbabilities(model, params, labelledX)
          const keep = yL.map((c, i) => (pL[i * K + c] > eta ? 0 : 1))
          const kept = keep.reduce((a: number, b) => a + b, 0)
          labelledKept = kept / nL
          const kt = constant(
            Float64Array.from(oneHotL, (v, j) => v * keep[Math.floor(j / K)]),
            [nL, K],
          )
          supervised = (q) => (kept === 0 ? 0 : mul(-1 / kept, sum(mul(kt, logSoftmax(model.apply(q, labelledX))))))
          break
        }
        case 'mixmatch': {
          const Kp = method.augmentations ?? 2
          const Tm = method.temperature ?? 0.5
          const alpha = method.alpha ?? 0.75
          const views = Array.from({ length: Kp }, (_, k) => augment(xU, 'weak', child(s, 'guess', k)))
          const avg = new Float64Array(nB * K)
          for (const v of views) predictProbabilities(model, params, v).forEach((x, j) => (avg[j] += x / Kp))
          targets = sharpen(avg, K, Tm)
          // The labelled batch and the K' unlabelled copies with their targets, mixed with one shuffle of all of them.
          const total = nL + Kp * nB
          const X = new Float64Array(total * d)
          const Y = new Float64Array(total * K)
          X.set(float64Data(augment(labelledX, 'weak', child(s, 'labelled'))), 0)
          Y.set(oneHotL, 0)
          views.forEach((v, k) => {
            X.set(float64Data(v), (nL + k * nB) * d)
            Y.set(targets, (nL + k * nB) * K)
          })
          const order = float64Data(permutation(child(s, 'shuffle'), total))
          // alpha = 0 turns mixup off (lambda' = 1), leaving label guessing and sharpening.
          const draw = alpha > 0 ? (beta(child(s, 'mixup'), alpha, alpha) as number) : 1
          const lam = Math.max(draw, 1 - draw)
          const MX = new Float64Array(total * d)
          const MY = new Float64Array(total * K)
          for (let i = 0; i < total; i++) {
            const j = order[i]
            for (let c = 0; c < d; c++) MX[i * d + c] = lam * X[i * d + c] + (1 - lam) * X[j * d + c]
            for (let c = 0; c < K; c++) MY[i * K + c] = lam * Y[i * K + c] + (1 - lam) * Y[j * K + c]
          }
          const mxL = constant(MX.slice(0, nL * d), [nL, d])
          const myL = constant(MY.slice(0, nL * K), [nL, K])
          const mxU = constant(MX.slice(nL * d), [Kp * nB, d])
          const myU = constant(MY.slice(nL * K), [Kp * nB, K])
          supervised = (q) => softmaxCrossEntropy(model.apply(q, mxL), myL)
          unsupervised = (q) => squaredDistance(softmax(model.apply(q, mxU)), myU, K)
          break
        }
        case 'fixmatch': {
          const tau = method.threshold ?? 0.95
          const m = method.momentum ?? 0.999
          const weakX = augment(xU, 'weak', child(s, 'view', 0))
          const strongX = augment(xU, 'strong', child(s, 'view', 1))
          const q = predictProbabilities(model, params, weakX)
          const { cls, conf } = rowArgmax(q, K)
          targets = new Float64Array(nB * K)
          cls.forEach((c, i) => (targets[i * K + c] = 1))
          if (adaptive === 'fixed') {
            thresholds = new Array(K).fill(tau)
            mask = Float64Array.from(conf, (c) => (c >= tau ? 1 : 0))
          } else if (adaptive === 'flexmatch') {
            const counts = new Array(K).fill(0)
            for (const c of state.marks) if (c >= 0) counts[c]++
            thresholds = flexMatchThresholds(counts, nU, tau, method.warmup ?? true)
            mask = Float64Array.from(conf, (c, i) => (c > thresholds[cls[i]] ? 1 : 0))
            const marks = Int32Array.from(state.marks)
            for (let r = 0; r < nB; r++) if (conf[r] > tau) marks[rows[r]] = cls[r]
            next.marks = marks
          } else if (adaptive === 'freematch') {
            let meanConf = 0
            const classMean = new Array(K).fill(0)
            for (let i = 0; i < nB; i++) {
              meanConf += conf[i] / nB
              for (let k = 0; k < K; k++) classMean[k] += q[i * K + k] / nB
            }
            const global = m * state.confidenceMean + (1 - m) * meanConf
            const local = state.classMean.map((v, k) => m * v + (1 - m) * classMean[k])
            const top = Math.max(...local)
            thresholds = local.map((v) => (v / top) * global)
            mask = Float64Array.from(conf, (c, i) => (c > thresholds[cls[i]] ? 1 : 0))
            next.confidenceMean = global
            next.classMean = local
          } else {
            let mu = 0
            for (let i = 0; i < nB; i++) mu += conf[i] / nB
            let v = 0
            for (let i = 0; i < nB; i++) v += (conf[i] - mu) ** 2 / Math.max(1, nB - 1)
            const muT = m * state.confidenceMean + (1 - m) * mu
            const varT = m * state.confidenceVariance + (1 - m) * v
            thresholds = new Array(K).fill(muT)
            mask = Float64Array.from(conf, (c) => (c >= muT ? 1 : Math.exp(-((c - muT) ** 2) / (2 * (varT / 4)))))
            next.confidenceMean = muT
            next.confidenceVariance = varT
          }
          const target = constant(
            Float64Array.from(targets, (v, j) => v * mask[Math.floor(j / K)]),
            [nB, K],
          )
          unsupervised = (p) => mul(-1 / nB, sum(mul(target, logSoftmax(model.apply(p, strongX)))))
          break
        }
      }

      const sup = supervised
      const uns = unsupervised
      const objective = (p: Params[]): Value => (uns && w > 0 ? add(sup(p), mul(w, uns(p))) : sup(p))
      const { value, grad: g } = gradientOf(objective)(params)
      const loss = Number(value)
      // The two parts at the step's parameters, for readouts (one more forward pass each).
      const supervisedLoss = Number(sup(params))
      const unsupervisedLoss = uns ? Number(uns(params)) : 0
      const { updates, state: opt } = rule.update(g as Params, state.optimizer, params)
      const newParams = applyUpdates(params, updates)
      let teacher = state.teacher
      if (method.kind === 'mean-teacher' && teacher) teacher = parameterEma(teacher, newParams, method.decay ?? 0.99)

      // Write the batch's diagnostics back into the full per-point arrays.
      const fullMask = Float64Array.from(state.mask)
      const fullTargets = Float64Array.from(state.targets)
      const fullPseudo = Int32Array.from(state.pseudo)
      const { cls } = rowArgmax(targets, K)
      let used = 0
      let right = 0
      for (let r = 0; r < nB; r++) {
        const i = rows[r]
        fullMask[i] = mask[r]
        for (let k = 0; k < K; k++) fullTargets[i * K + k] = targets[r * K + k]
        // A temporal-ensembling row has no target before its first average.
        const target = hasTarget && (method.kind !== 'temporal-ensembling' || mask[r] > 0)
        fullPseudo[i] = target ? cls[r] : -1
        used += mask[r]
        if (truth && cls[r] === truth[i]) right += mask[r]
      }
      return {
        ...state,
        ...next,
        t: t + 1,
        params: newParams,
        optimizer: opt,
        teacher,
        loss,
        supervisedLoss,
        unsupervisedLoss,
        weight: w,
        batch: rows,
        mask: fullMask,
        targets: fullTargets,
        pseudo: fullPseudo,
        maskRate: used / nB,
        pseudoAccuracy: truth && used > 0 ? right / used : NaN,
        thresholds,
        labelledKept,
        diverged: !Number.isFinite(loss),
      }
    },
  }
}
