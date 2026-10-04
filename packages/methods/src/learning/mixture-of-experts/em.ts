/**
 * EM for a mixture of linear experts and for the hierarchical mixture (Jordan and Jacobs, 1994, "Hierarchical mixtures
 * of experts and the EM algorithm", Neural Computation 6(2), §4), as a step-through algorithm: one step is one E-step
 * and one M-step.
 *
 * - E-step: the responsibility of expert i for row t is its posterior given y, hₜᵢ ∝ gᵢ(xₜ) pᵢ(yₜ | xₜ).
 * - M-step, experts: each expert is refitted to every row weighted by hₜᵢ: weighted least squares and
 *   σᵢ² = Σₜ hₜᵢ rₜᵢ² / Σₜ hₜᵢ for regression; a few Newton (IRLS) steps of weighted logistic regression for
 *   classification. A tiny ridge keeps an expert with almost no responsibility solvable.
 * - M-step, gate: the gate is refitted as a multinomial logistic regression on the soft targets h, maximising
 *   Σₜ Σᵢ hₜᵢ log gᵢ(xₜ) by L-BFGS from the current gate (a generalised EM step: the likelihood never decreases). For
 *   the hierarchy this one objective holds the top gate (targets Σⱼ hₜ,gⱼ) and every lower gate together.
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

/** Training data: inputs [T, d] and targets [T] (floats, or 0/1 labels). */
export type MoeData = { x: Tensor; y: Tensor }

/** Options of `moeEm`. */
export type MoeEmOptions = {
  /** L-BFGS steps of each gate refit (default 20). */
  gateSteps?: Size
  /** Newton steps of each logistic expert refit (default 3). */
  newtonSteps?: Size
  /** Ridge added to each expert's normal equations (default 1e-6). */
  ridge?: number
  /** The smallest σᵢ² of a regression expert (default 1e-6), so an expert on a few exact points stays finite. */
  minVariance?: number
}

/** The state of `moeEm` after t EM steps. */
export interface MoeEmState extends Status {
  readonly t: Size
  readonly params: MoeParams
  /** The mean negative log-likelihood at `params`. */
  readonly loss: number
  /** The responsibilities hₜᵢ at `params` (the next E-step's), [T][N]. */
  readonly responsibilities: number[][]
}

/** Check that EM applies: linear experts, a dense gate, the mixture objective. */
export function emApplies(model: MoeModel): string | null {
  const { expert, gate, objective } = model.spec
  if (expert !== 'linear') return 'EM needs linear experts (an MLP expert has no closed-form M-step)'
  if (gate !== 'softmax' && gate !== 'hierarchical') return 'EM needs a dense gate (softmax or hierarchical)'
  if (objective !== 'mixture') return 'EM fits the mixture likelihood, not a blend of outputs'
  return null
}

/** log gᵢ(x) [T, N] of a dense routing: logSoftmax of the scores (flat gate) or the log-probabilities (hierarchy). */
function logGate(model: MoeModel, scores: Value, logits: Value): Value {
  return model.spec.gate === 'hierarchical' ? logits : logSoftmax(scores)
}

const rowsOf = (t: Tensor, n: number, k: number) => {
  const f = toFlat(t)
  return Array.from({ length: n }, (_, i) => f.slice(i * k, (i + 1) * k))
}

/** Mean negative log-likelihood and responsibilities [T][N] at params. */
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

/** Refit expert i by weighted least squares (regression) or weighted IRLS (classification). */
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

/** The gate's parameters as one flat vector, and back. */
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

/** Refit the gate to the soft targets h by L-BFGS on −mean Σᵢ hₜᵢ log gᵢ(xₜ). */
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
 * EM for a mixture (or hierarchical mixture) of linear experts on `data`: `init` takes `{ params }` (e.g.
 * `model.init(stream)`); each step is an E-step and an M-step (experts, then gate). Throws when `emApplies` says no.
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

/** The mean negative log-likelihood of params on data (the EM objective), for checks. */
export function moeNegativeLogLikelihood(model: MoeModel, params: MoeParams, data: MoeData): number {
  const raw = unwrap(moeLoss(model, params, data.x, data.y).data)
  return typeof raw === 'number' ? raw : toFlat(raw)[0]
}
