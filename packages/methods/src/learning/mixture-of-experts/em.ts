/**
 * EM for a mixture of linear experts and for the hierarchical mixture (Jordan and Jacobs, 1994, "Hierarchical mixtures
 * of experts and the EM algorithm", Neural Computation 6(2), §4), as a step-through algorithm: one step is one E-step
 * and one M-step.
 *
 * - E-step: the responsibility of expert $i$ for row $t$ is its posterior given $y$,
 *   $h_{ti} \propto g_i(\xvec_t) p_i(y_t \mid \xvec_t)$.
 * - M-step, experts: each expert is refitted to every row weighted by $h_{ti}$: weighted least squares and
 *   $\sigma_i^2 = \sum_t h_{ti} r_{ti}^2 / \sum_t h_{ti}$ (with $r_{ti}$ the residual) for regression; a few Newton
 *   (IRLS) steps of weighted logistic regression for classification. A tiny ridge keeps an expert with almost no
 *   responsibility solvable.
 * - M-step, gate: the gate is refitted as a multinomial logistic regression on the soft targets $\Hmat$, maximising
 *   $\sum_t \sum_i h_{ti} \log g_i(\xvec_t)$ by L-BFGS from the current gate (a generalised EM step: the likelihood
 *   should not decrease). For the hierarchy this one objective holds the top gate (targets $\sum_j h_{t,gj}$) and
 *   every lower gate together.
 *
 * Both M-steps use the responsibilities of the E-step before them, so the experts and the gate are refitted to the
 * same targets.
 */

import type { Algorithm, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import {
  fromData,
  mean,
  mul,
  neg,
  reshape,
  slice,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { logSoftmax } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import type { LinearParams } from 'aifn-compute/nn/layers'
import { expertLogLikelihood, moeForward, moeLoss, type MoeModel, type MoeParams } from './model'
import { DomainError, NumericalError } from 'aifn-compute/foundation/errors'

/** Training data: inputs `x` ($T \times d$) and targets `y` ($T$ floats, or 0/1 labels). */
export type MoeData = { x: Tensor; y: Tensor }

/** Options of `moeEm`. */
export type MoeEmOptions = {
  /** L-BFGS steps of each gate refit (default 20). */
  gateSteps?: Size
  /** Newton steps of each logistic expert refit (default 3). */
  newtonSteps?: Size
  /** Ridge added to each expert's normal equations (default 1e-6). */
  ridge?: number
  /**
   * The smallest $\sigma_i^2$ of a regression expert (default 1e-6), so an expert on a few exact points stays finite.
   */
  minVariance?: number
}

/** The state of `moeEm` after $t$ EM steps. */
export interface MoeEmState extends Status {
  /** EM steps done. */
  readonly t: Size
  /** The parameters after $t$ steps. */
  readonly params: MoeParams
  /** The mean negative log-likelihood at `params`. */
  readonly loss: number
  /** The responsibilities $h_{ti}$ at `params` (the next E-step's), $T$ rows of $N$. */
  readonly responsibilities: number[][]
}

/**
 * Check that EM applies: linear experts, a dense gate (softmax or hierarchical), the mixture objective.
 *
 * @param model The model; its spec is read.
 * @returns Null when EM applies, else the reason it does not.
 *
 * @example A mixture of linear experts, and one of MLP experts
 * print('linear:', emApplies(moeModel({ inputs: 1, task: 'regression' })))
 * print('MLP:', emApplies(moeModel({ inputs: 1, task: 'regression', expert: 'mlp' })))
 */
export function emApplies(model: MoeModel): string | null {
  const { expert, gate, objective } = model.spec
  if (expert !== 'linear') return 'EM needs linear experts (an MLP expert has no closed-form M-step)'
  if (gate !== 'softmax' && gate !== 'hierarchical') return 'EM needs a dense gate (softmax or hierarchical)'
  if (objective !== 'mixture') return 'EM fits the mixture likelihood, not a blend of outputs'
  return null
}

/**
 * The log gate weights $\log g_i(\xvec)$ of a dense routing: the log-softmax of the scores (flat gate), or the
 * log-probabilities the hierarchy keeps in `logits`.
 *
 * @param model The model; its gate is read.
 * @param scores The routing's `scores`, $T \times N$ (read for a flat gate).
 * @param logits The routing's `logits`, $T \times N$ (read for a hierarchy, where they are log-probabilities).
 * @returns The $T \times N$ log weights, differentiable.
 */
function logGate(model: MoeModel, scores: Value, logits: Value): Value {
  return model.spec.gate === 'hierarchical' ? logits : logSoftmax(scores)
}

/**
 * The rows of a matrix as plain arrays.
 *
 * @param t The matrix, $n \times k$.
 * @param n The number of rows.
 * @param k The number of columns.
 * @returns $n$ arrays of $k$ values.
 */
const rowsOf = (t: Tensor, n: number, k: number) => {
  const f = toFlat(t)
  return Array.from({ length: n }, (_, i) => f.slice(i * k, (i + 1) * k))
}

/**
 * The E-step: the mean negative log-likelihood and the responsibilities at the parameters, in log space.
 *
 * @param model The model.
 * @param params The parameters.
 * @param data The training data.
 * @returns `loss`, the mean negative log-likelihood, and `h`, the responsibilities, $T$ rows of $N$ summing to 1.
 */
function responsibilities(model: MoeModel, params: MoeParams, data: MoeData) {
  const { routing, outputs } = moeForward(model, params, data.x)
  const T = data.x.shape[0]
  const N = model.spec.experts
  const lg = unwrap(logGate(model, routing.scores, routing.logits)) as Tensor
  const ll = unwrap(expertLogLikelihood(model, params, outputs, data.y)) as Tensor
  const a = toFlat(lg)
  const b = toFlat(ll)
  const h: number[][] = []
  let nll = 0
  for (let t = 0; t < T; t++) {
    const row = Array.from({ length: N }, (_, i) => a[t * N + i] + b[t * N + i])
    const m = Math.max(...row)
    const lse = m + Math.log(row.reduce((s, v) => s + Math.exp(v - m), 0))
    nll -= lse / T
    h.push(row.map((v) => Math.exp(v - lse)))
  }
  return { loss: nll, h }
}

/**
 * Refit one expert by weighted least squares (regression) or weighted IRLS (classification), with the bias as the
 * weight of a constant feature and `ridge` added to the diagonal of the normal equations (bias included). Throws
 * `NumericalError` when the equations are singular.
 *
 * @param model The model; its task and input width are read.
 * @param current The expert's current parameters: IRLS starts from them; least squares does not read them.
 * @param data The training data.
 * @param h The expert's responsibility for each row, $T$ values: the weights of the fit.
 * @param options The EM options, every default filled in (`ridge`, `newtonSteps`, `minVariance`).
 * @returns The refitted parameters, and for regression the weighted residual variance $\sigma_i^2$ (at least
 *   `minVariance`).
 */
function refitExpert(
  model: MoeModel,
  current: LinearParams,
  data: MoeData,
  h: readonly number[],
  options: Required<MoeEmOptions>,
): { params: LinearParams; variance?: number } {
  const T = data.x.shape[0]
  const d = model.spec.inputs
  const D = d + 1
  const xs = rowsOf(data.x, T, d).map((r) => [...r, 1])
  const ys = toFlat(data.y).map(Number)
  const w0 = [...toFlat(current.weight), current.bias ? toFlat(current.bias)[0] : 0]
  const normal = (weights: readonly number[], rhs: readonly number[]) => {
    const A = new Float64Array(D * D)
    for (let j = 0; j < D; j++) A[j * D + j] += options.ridge
    const b = new Float64Array(D)
    for (let t = 0; t < T; t++) {
      const x = xs[t]
      for (let j = 0; j < D; j++) {
        b[j] += rhs[t] * x[j]
        for (let l = 0; l < D; l++) A[j * D + l] += weights[t] * x[j] * x[l]
      }
    }
    const sol = solveDense(A, b, D)
    if (!sol.x)
      throw new NumericalError('moeEm', 'moeEm: an expert’s weighted normal equations are singular', 'singular')
    return Array.from(sol.x)
  }
  let w = w0
  let variance: number | undefined
  if (model.spec.task === 'regression') {
    w = normal(
      h,
      ys.map((y, t) => h[t] * y),
    )
    const mass = h.reduce((a, v) => a + v, 0)
    const sse = xs.reduce((a, x, t) => a + h[t] * (ys[t] - x.reduce((s, v, j) => s + v * w[j], 0)) ** 2, 0)
    variance = Math.max(options.minVariance, mass > 0 ? sse / mass : options.minVariance)
  } else {
    // Newton on the weighted log-likelihood: w ← w + (Xᵀ H S X)⁻¹ Xᵀ H (y − p), S = diag(p(1 − p)).
    for (let it = 0; it < options.newtonSteps; it++) {
      const p = xs.map((x) => 1 / (1 + Math.exp(-x.reduce((s, v, j) => s + v * w[j], 0))))
      const step = normal(
        p.map((q, t) => h[t] * q * (1 - q)),
        p.map((q, t) => h[t] * (ys[t] - q)),
      )
      w = w.map((v, j) => v + step[j])
    }
  }
  return {
    params: {
      weight: fromData(Float64Array.from(w.slice(0, d)), [d, 1]),
      bias: fromData(Float64Array.from([w[d]]), [1]),
    },
    variance,
  }
}

/**
 * The gate's parameters as one flat vector, and back: the router's weight ($d \times N$) and bias, then for a
 * hierarchy the top gate's weight ($d \times G$) and bias.
 *
 * @param model The model.
 * @param params The parameters whose gate is flattened.
 * @returns `v`, the flat gate, and `read`, which turns a flat vector (traced or not) into `params` with that gate.
 */
function gateVector(model: MoeModel, params: MoeParams): { v: Float64Array; read: (v: Value) => MoeParams } {
  const { inputs: d, experts: N, groups: G } = model.spec
  const parts: Tensor[] = [params.moe.router.weight, params.moe.router.bias!]
  const hier = model.spec.gate === 'hierarchical'
  if (hier) parts.push(params.top!.weight, params.top!.bias!)
  const v = Float64Array.from(parts.flatMap((p) => toFlat(p)))
  const read = (u: Value): MoeParams => {
    let at = 0
    const take = (shape: number[]) => {
      const size = shape.reduce((a, b) => a * b, 1)
      const out = reshape(slice(u, [at, at + size]), shape)
      at += size
      return out as Tensor
    }
    const router = { weight: take([d, N]), bias: take([N]) }
    const top = hier ? { weight: take([d, G]), bias: take([G]) } : undefined
    return { ...params, moe: { ...params.moe, router }, ...(top ? { top } : {}) }
  }
  return { v, read }
}

/**
 * Refit the gate to the soft targets by L-BFGS on $-\frac{1}{T} \sum_t \sum_i h_{ti} \log g_i(\xvec_t)$, from the
 * current gate.
 *
 * @param model The model.
 * @param params The parameters; only the gate is refitted, the rest is kept.
 * @param data The training data; only `x` is read.
 * @param h The soft targets: the responsibilities, $T$ rows of $N$.
 * @param steps The most L-BFGS steps.
 * @returns The parameters with the refitted gate, as plain tensors.
 */
function refitGate(model: MoeModel, params: MoeParams, data: MoeData, h: number[][], steps: Size): MoeParams {
  const T = data.x.shape[0]
  const N = model.spec.experts
  const targets = fromData(Float64Array.from(h.flat()), [T, N])
  const { v, read } = gateVector(model, params)
  const objective = (u: Value) => {
    const { routing } = moeForward(model, read(u), data.x)
    return neg(mean(sum(mul(targets, logGate(model, routing.scores, routing.logits)), -1)))
  }
  const vg = valueAndGrad(objective)
  const f = (u: Tensor) => {
    const { value, grad } = vg(u)
    const raw = unwrap(value)
    return { value: typeof raw === 'number' ? raw : toFlat(raw)[0], grad: grad as Tensor }
  }
  const result = minimize(f, fromData(v, [v.length]), { method: 'lbfgs', maxSteps: steps })
  const fitted = read(result.x)
  // Plain tensors again (the reader slices a traced or raw vector).
  const raw = (t: Tensor) => unwrap(t) as Tensor
  return {
    ...fitted,
    moe: { ...fitted.moe, router: { weight: raw(fitted.moe.router.weight), bias: raw(fitted.moe.router.bias!) } },
    ...(fitted.top ? { top: { weight: raw(fitted.top.weight), bias: raw(fitted.top.bias!) } } : {}),
  }
}

/**
 * EM for a mixture (or hierarchical mixture) of linear experts on `data`: `init` takes `{ params }` (such as
 * `model.init(stream)`); each step is an M-step on the responsibilities the state holds (experts, then gate), then
 * the E-step at the new parameters. Neither uses the stream. Throws `DomainError` when `emApplies` says no, and the
 * steps throw `NumericalError` when an expert's normal equations are singular; a non-finite loss marks the state
 * `diverged`.
 *
 * @param model The model: linear experts, a softmax or hierarchical gate, the mixture objective.
 * @param data The training data.
 * @param options The M-step's settings.
 * @returns The algorithm, one EM iteration per step.
 *
 * @example The loss falls as two experts take a V of two regimes apart
 * const xs = Array.from({ length: 16 }, (_, t) => -1 + (2 * t) / 15)
 * const x = tensor(xs.map((v) => [v]))
 * const y = add(tensor(xs.map((v) => Math.abs(2 * v))), normals(stream(1), 16, 0, 0.1))
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const alg = moeEm(model, { x, y })
 * const tr = trace(alg, { params: model.init(stream(2)) }, 10, { record: { loss: (s) => s.loss } })
 * print('loss by step:', tr.series.loss)
 * print('expert sigmas:', exp(tr.final.params.logSigma))
 */
export function moeEm(
  model: MoeModel,
  data: MoeData,
  options: MoeEmOptions = {},
): Algorithm<{ params: MoeParams }, MoeEmState> {
  const why = emApplies(model)
  if (why) throw new DomainError('moeEm', `moeEm: ${why}`)
  const opts: Required<MoeEmOptions> = { gateSteps: 20, newtonSteps: 3, ridge: 1e-6, minVariance: 1e-6, ...options }
  const N = model.spec.experts
  const state = (t: Size, params: MoeParams): MoeEmState => {
    const { loss, h } = responsibilities(model, params, data)
    return { t, params, loss, responsibilities: h, diverged: !Number.isFinite(loss) }
  }
  return {
    name: 'mixture-of-experts-em',
    init: ({ params }) => state(0, params),
    step: (s: MoeEmState, _ctx: StepContext) => {
      const h = s.responsibilities
      const experts = [...s.params.moe.experts]
      const variances: number[] = []
      for (let i = 0; i < N; i++) {
        const fit = refitExpert(
          model,
          experts[i] as LinearParams,
          data,
          h.map((r) => r[i]),
          opts,
        )
        experts[i] = fit.params
        if (fit.variance !== undefined) variances.push(fit.variance)
      }
      let params: MoeParams = { ...s.params, moe: { ...s.params.moe, experts } }
      if (variances.length)
        params = {
          ...params,
          logSigma: fromData(
            Float64Array.from(variances, (v) => 0.5 * Math.log(v)),
            [N],
          ),
        }
      params = refitGate(model, params, data, h, opts.gateSteps)
      return state(s.t + 1, params)
    },
  }
}

/**
 * The mean negative log-likelihood of the parameters on the data (the EM objective, the data term of `moeLoss`), for
 * checks. Evaluation mode: no gate noise.
 *
 * @param model The model (with the mixture objective, for the value to be a likelihood).
 * @param params The parameters.
 * @param data The data.
 * @returns $-\frac{1}{T} \sum_t \log p(y_t \mid \xvec_t)$.
 *
 * @example The same value as an EM state's loss
 * const x = tensor([[-1], [0], [1]])
 * const y = tensor([2, 0, 2])
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const s = run(moeEm(model, { x, y }), { params: model.init(stream(1)) }, 2)
 * print('state loss:', s.loss, 'recomputed:', moeNegativeLogLikelihood(model, s.params, { x, y }))
 */
export function moeNegativeLogLikelihood(model: MoeModel, params: MoeParams, data: MoeData): number {
  const raw = unwrap(moeLoss(model, params, data.x, data.y).data)
  return typeof raw === 'number' ? raw : toFlat(raw)[0]
}
