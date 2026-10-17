/**
 * Self-distillation with no labels (DINO; Caron et al., 2021, "Emerging Properties in Self-Supervised Vision
 * Transformers", Algorithm 1 and §3.1) for a small network, as one traceable algorithm.
 *
 * A student network $g_\theta$ and a teacher $g_\xi$ of the same architecture map each input to $K$ logits. Every step
 * draws two augmented views of a batch, forms the teacher's targets
 * $P_t(\xvec) = \operatorname{softmax}((g_\xi(\xvec) - \cvec) / \tau_t)$ and the student's predictions
 * $P_s(\xvec) = \operatorname{softmax}(g_\theta(\xvec) / \tau_s)$, and takes one optimiser step on the cross-entropy of
 * each view's student prediction against the other view's teacher target,
 * $\tfrac12 [H(P_t(\xvec_1), P_s(\xvec_2)) + H(P_t(\xvec_2), P_s(\xvec_1))]$, with no gradient into the teacher. The
 * teacher then follows the student as an exponential moving average, $\xi \leftarrow \lambda \xi + (1 - \lambda)
 * \theta$ (`parameterEma`), and the centre follows the batch mean of the teacher's logits,
 * $\cvec \leftarrow m \cvec + (1 - m) \bar g_\xi$.
 *
 * Differences from the paper: two global views only (no multi-crop), a caller-supplied augmentation (default isotropic
 * Gaussian jitter), fixed temperatures and momenta (no warm-up schedules), and the network is one MLP with the logits
 * as its output (no separate backbone, projection head or weight-normalised last layer). Diagnostics in the state
 * follow the paper's collapse analysis (§5.3): the mean entropy of the teacher's output per input, the entropy of its
 * batch-mean output, and the KL divergence from the teacher to the student, all on the clean data. Without centring
 * and with $\tau_t \ll \tau_s$ the teacher collapses to one class (marginal entropy 0); with centring alone it
 * collapses to the uniform output (conditional entropy $\log K$).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, permutation, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  float64Data,
  fromData,
  mul,
  shapeOf,
  sub,
  sum,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Layer } from 'aifn-compute/nn/layers'
import { parameterEma } from 'aifn-compute/nn/training'
import { logSoftmax, softmax } from 'aifn-compute/numerics/special'
import { adamRule, applyUpdates, type UpdateRule } from 'aifn-compute/optim/first-order'
import { entropy } from 'aifn-compute/probability/information'

/** Controls of `selfDistillationTraining`. */
export interface SelfDistillationOptions {
  /** The network: inputs $n \times d$ to logits $n \times K$ (e.g. the engine's `Mlp([d, ..., K])`). */
  model: Layer<Params[]>
  /** The student temperature $\tau_s$ (default 0.1, the paper's). */
  studentTemperature?: number
  /** The teacher temperature $\tau_t$ (default 0.04, the paper's final value); below $\tau_s$ it sharpens. */
  teacherTemperature?: number
  /** Subtract the running centre from the teacher's logits (default true). */
  centring?: boolean
  /** The centre's momentum $m$ (default 0.9, the paper's). */
  centreMomentum?: number
  /** The teacher's EMA momentum $\lambda$ (default 0.99; the paper raises 0.996 towards 1 over long runs). */
  teacherMomentum?: number
  /** Inputs per step (default all of them). */
  batch?: number
  /** Standard deviation of the default Gaussian jitter (default 0.3). */
  jitter?: number
  /** One augmented view of a batch; default isotropic Gaussian jitter of sd `jitter`. */
  augment?: (x: Tensor, s: Stream) => Tensor
  /** The update rule (default `adamRule({ stepSize: 0.01 })`). */
  optimizer?: UpdateRule<unknown>
}

/** The state after $t$ updates. Diagnostics are computed on the clean data at the state's parameters. */
export interface SelfDistillationState extends Status {
  /** Updates made. */
  t: number
  /** The student's parameters. */
  params: Params[]
  /** The teacher's parameters (an EMA of the student's). */
  teacher: Params[]
  /** The centre $\cvec$ ($K$ values). */
  centre: Float64Array
  /** The update rule's state. */
  optimizer: unknown
  /** The step's loss (NaN at $t = 0$). */
  loss: number
  /** The teacher's output distributions on the clean data, row-major $n \times K$. */
  teacherProbabilities: Float64Array
  /** The teacher's arg max per input. */
  assignment: Int32Array
  /** Mean over inputs of the entropy of the teacher's output, in nats: $\log K$ when outputs are uniform. */
  conditionalEntropy: number
  /** Entropy of the teacher's mean output over inputs, in nats: 0 when every input gets the same one-hot output. */
  marginalEntropy: number
  /** Mean over inputs of $\KL(P_t \,\|\, P_s)$ on the clean data, in nats. */
  teacherStudentKl: number
}

/** `valueAndGrad` with the parameter type erased, which keeps the checker from expanding the pytree types. */
const gradientOf = valueAndGrad as unknown as (
  f: (p: Params[]) => Value,
) => (p: Params[]) => { value: Value; grad: unknown }

/**
 * Row-wise $\operatorname{softmax}((\zvec - \cvec) / \tau)$ of a row-major logit matrix, as a flat array.
 *
 * @param logits The logits, row-major $n \times K$.
 * @param K The number of classes.
 * @param tau The temperature $\tau$.
 * @param centre The centre $\cvec$ subtracted from every row ($K$ values), or none.
 * @returns Row-major $n \times K$ probabilities.
 */
function tempered(logits: Float64Array, K: number, tau: number, centre?: Float64Array): Float64Array {
  const n = logits.length / K
  const z = fromData(logits, [n, K])
  const shifted = centre ? sub(z, fromData(centre, [1, K])) : z
  return Float64Array.from(float64Data(softmax(shifted as Tensor, { temperature: tau })))
}

/**
 * The rows `rows` of a row-major matrix.
 *
 * @param data The matrix, row-major; not modified.
 * @param rows The rows to copy, in order.
 * @param width The number of columns.
 * @returns The rows, row-major.
 */
function pickRows(data: ArrayLike<number>, rows: ArrayLike<number>, width: number): Float64Array {
  const out = new Float64Array(rows.length * width)
  for (let r = 0; r < rows.length; r++) for (let j = 0; j < width; j++) out[r * width + j] = data[rows[r] * width + j]
  return out
}

/**
 * DINO-style self-distillation on the rows of `x` (see the file comment). Step $t$ draws its batch from
 * `child(stream, 'batch')` and the two views from `child(stream, 'view', 0)` and `child(stream, 'view', 1)`. The
 * teacher starts as a copy of the student and the centre at zero. Options hold a network and an optional function, so
 * the algorithm runs on the page's thread.
 *
 * @param x The unlabelled inputs, $n \times d$.
 * @param options The network, the temperatures, centring and its momentum, the teacher momentum, the batch, the
 *   augmentation and the update rule.
 * @returns The algorithm; start it from `{ params }`, or from `undefined` to initialise from the init stream's child
 *   `'params'`.
 *
 * @example Collapse without centring
 * import { Mlp } from 'aifn-compute/nn/layers'
 * const x = normal(stream(1), 0, 1, { shape: [20, 2] })
 * const net = Mlp([2, 8, 4], { activation: 'tanh' })
 * const alg = selfDistillationTraining(x, { model: net, centring: false, teacherTemperature: 0.02 })
 * const s = run(alg, undefined, 20)
 * print('conditional entropy:', s.conditionalEntropy, ' marginal entropy:', s.marginalEntropy)
 * print('teacher assignments:', s.assignment)
 */
export function selfDistillationTraining(
  x: Tensor,
  options: SelfDistillationOptions,
): Algorithm<{ params: Params[] } | undefined, SelfDistillationState> {
  const { model } = options
  const tauS = options.studentTemperature ?? 0.1
  const tauT = options.teacherTemperature ?? 0.04
  const centring = options.centring ?? true
  const m = options.centreMomentum ?? 0.9
  const lambda = options.teacherMomentum ?? 0.99
  const sd = options.jitter ?? 0.3
  const rule = options.optimizer ?? adamRule({ stepSize: 0.01 })
  const [n, d] = shapeOf(x)
  const B = Math.min(n, Math.max(1, Math.round(options.batch ?? n)))
  const augment =
    options.augment ?? ((v: Tensor, s: Stream) => add(v, normals(s, shapeOf(v) as number[], 0, sd)) as Tensor)
  const allX = float64Data(x)

  // The collapse diagnostics on the clean data.
  const diagnostics = (params: Params[], teacher: Params[], centre: Float64Array) => {
    const tLogits = Float64Array.from(float64Data(model.apply(teacher, x) as Tensor))
    const K = tLogits.length / n
    const pt = tempered(tLogits, K, tauT, centring ? centre : undefined)
    const ps = tempered(Float64Array.from(float64Data(model.apply(params, x) as Tensor)), K, tauS)
    const assignment = new Int32Array(n)
    let kl = 0
    for (let i = 0; i < n; i++) {
      let best = 0
      for (let k = 0; k < K; k++) {
        const a = pt[i * K + k]
        if (a > pt[i * K + best]) best = k
        if (a > 0) kl += (a * (Math.log(a) - Math.log(Math.max(ps[i * K + k], 1e-300)))) / n
      }
      assignment[i] = best
    }
    const rowEntropies = float64Data(entropy(fromData(pt, [n, K])) as Tensor)
    let conditional = 0
    for (let i = 0; i < n; i++) conditional += rowEntropies[i] / n
    const meanRow = new Float64Array(K)
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) meanRow[k] += pt[i * K + k] / n
    const marginal = entropy(meanRow)
    return {
      teacherProbabilities: pt,
      assignment,
      conditionalEntropy: conditional,
      marginalEntropy: typeof marginal === 'number' ? marginal : float64Data(marginal as Tensor)[0],
      teacherStudentKl: kl,
    }
  }

  return {
    name: 'selfDistillationTraining',
    init(start, s) {
      const params = start?.params ?? model.init(child(s, 'params'))
      const K = float64Data(model.apply(params, fromData(allX.subarray(0, d), [1, d])) as Tensor).length
      const centre = new Float64Array(K)
      return {
        t: 0,
        params,
        teacher: params,
        centre,
        optimizer: rule.init(params),
        loss: NaN,
        ...diagnostics(params, params, centre),
      }
    },
    step(state, ctx) {
      const s = ctx.stream
      const rows = B === n ? null : float64Data(permutation(child(s, 'batch'), n)).subarray(0, B)
      const xb = rows ? fromData(pickRows(allX, rows, d), [B, d]) : x
      const v1 = augment(xb, child(s, 'view', 0))
      const v2 = augment(xb, child(s, 'view', 1))
      const t1 = Float64Array.from(float64Data(model.apply(state.teacher, v1) as Tensor))
      const t2 = Float64Array.from(float64Data(model.apply(state.teacher, v2) as Tensor))
      const K = t1.length / B
      const c = centring ? state.centre : undefined
      // The teacher's targets are constants: no gradient flows into the teacher.
      const p1 = fromData(tempered(t1, K, tauT, c), [B, K])
      const p2 = fromData(tempered(t2, K, tauT, c), [B, K])
      const objective = (p: Params[]): Value => {
        const ls1 = logSoftmax(mul(1 / tauS, model.apply(p, v1)))
        const ls2 = logSoftmax(mul(1 / tauS, model.apply(p, v2)))
        return mul(-0.5 / B, add(sum(mul(p2, ls1)), sum(mul(p1, ls2))))
      }
      const { value, grad: g } = gradientOf(objective)(state.params)
      const { updates, state: opt } = rule.update(g as Params, state.optimizer, state.params)
      const params = applyUpdates(state.params, updates)
      const teacher = parameterEma(state.teacher, params, lambda)
      const centre = new Float64Array(K)
      for (let k = 0; k < K; k++) {
        let average = 0
        for (let i = 0; i < B; i++) average += (t1[i * K + k] + t2[i * K + k]) / (2 * B)
        centre[k] = m * state.centre[k] + (1 - m) * average
      }
      const loss = Number(value)
      return {
        ...state,
        t: state.t + 1,
        params,
        teacher,
        centre,
        optimizer: opt,
        loss,
        diverged: !Number.isFinite(loss),
        ...diagnostics(params, teacher, centre),
      }
    },
  }
}
