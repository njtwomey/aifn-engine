/**
 * The mixture of experts as a statistical model (Jacobs, Jordan, Nowlan and Hinton, 1991, "Adaptive mixtures of local
 * experts", Neural Computation 3(1); Jordan and Jacobs, 1994, "Hierarchical mixtures of experts and the EM algorithm",
 * Neural Computation 6(2)), built on the compute layer `MixtureOfExperts` of `aifn-compute/nn/experts`.
 *
 * N experts (linear, or a small MLP) each map x to a prediction: a mean of y for regression, the log-odds of y = 1 for
 * binary classification. A gate turns x into weights over the experts: any gate of `route` (softmax, top-k, noisy
 * top-k, Switch, expert choice), or a two-level hierarchy, a softmax over G groups times a softmax over the experts
 * within each group.
 *
 * Two objectives:
 * - `mixture`: the classic model, a conditional mixture p(y | x) = Σᵢ gᵢ(x) pᵢ(y | x) with a normal (σᵢ per expert)
 *   or Bernoulli law per expert, fitted by maximum likelihood. Each expert specialises in the inputs it is
 *   responsible for, and where the gate is mixed p(y | x) is multimodal.
 * - `blend`: the modern sparse layer, which blends outputs, ŷ = Σᵢ gᵢ(x) fᵢ(x), with a squared error or a logistic
 *   loss on the blend. The experts cooperate rather than compete (Jacobs et al., 1991, §1 contrasts the two).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  exp,
  expandDims,
  fromData,
  log,
  logsumexp,
  matmul,
  mean,
  mul,
  neg,
  reshape,
  shapeOfValue,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  where,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { logSoftmax, softplus } from 'aifn-compute/numerics/special'
import {
  denseRouting,
  importanceLoss,
  loadBalancingLoss,
  MixtureOfExperts,
  routerZLoss,
  type GateKind,
  type MixtureOfExpertsLayer,
  type MixtureOfExpertsParams,
  type Routing,
} from 'aifn-compute/nn/experts'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Linear, linear, Mlp, type Context, type Layer, type LinearParams } from 'aifn-compute/nn/layers'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The experts' form. */
export type ExpertKind = 'linear' | 'mlp'
/** A gate of `route`, or the two-level gate of a hierarchical mixture of experts. */
export type GateChoice = GateKind | 'hierarchical'
/** What the experts' outputs are combined into: a conditional mixture, or a blend of outputs. */
export type MoeObjective = 'mixture' | 'blend'

/** A mixture of experts' structure. */
export type MoeConfig = {
  /** Input width d. */
  inputs: Size
  task: 'regression' | 'classification'
  /** Number of experts N (default 4). */
  experts?: Size
  /** Default `linear`. */
  expert?: ExpertKind
  /** Hidden width of an MLP expert (default 16; tanh). */
  hidden?: Size
  /** Default `softmax`. */
  gate?: GateChoice
  /** Experts per token of the top-k gates (default 2). */
  k?: Size
  /** Gate temperature τ (default 1): the gate is a softmax of logits/τ. */
  temperature?: number
  /** Capacity factor of the sparse gates (default ∞: no dropping). */
  capacityFactor?: number
  /** Groups of the hierarchical gate (default 2; N must be a multiple). */
  groups?: Size
  /** Default `mixture`. */
  objective?: MoeObjective
  /**
   * The router's initial weights: `default` (LeCun uniform) or `zero`, so every expert starts equally likely and, under
   * top-1 routing, every tie goes to the first expert.
   */
  routerInit?: 'default' | 'zero'
}

/** A config with every default filled in. */
export type MoeSpec = Required<MoeConfig>

/** Fill in the defaults of a config, checking the hierarchical gate's grouping. */
export function moeSpec(config: MoeConfig): MoeSpec {
  const spec: MoeSpec = {
    experts: 4,
    expert: 'linear',
    hidden: 16,
    gate: 'softmax',
    k: 2,
    temperature: 1,
    capacityFactor: Infinity,
    groups: 2,
    objective: 'mixture',
    routerInit: 'default',
    ...config,
  }
  if (spec.gate === 'hierarchical' && spec.experts % spec.groups !== 0)
    throw new DomainError('moeSpec', `moeSpec: ${spec.experts} experts do not split into ${spec.groups} equal groups`)
  return spec
}

/** Parameters: the compute layer's (router, noise map, experts), the top gate of a hierarchy, and log σ per expert. */
export type MoeParams = {
  moe: MixtureOfExpertsParams
  /** The top-level gate over groups (hierarchical gate only). */
  top?: LinearParams
  /** log σᵢ of each expert's normal law (regression with the mixture objective). */
  logSigma?: Tensor
}

/** A built mixture of experts: the compute layer and the spec. */
export type MoeModel = {
  readonly spec: MoeSpec
  readonly layer: MixtureOfExpertsLayer
  readonly top: Layer<LinearParams> | null
  init(s: Stream): MoeParams
}

function expertLayer(spec: MoeSpec): Layer {
  return spec.expert === 'linear'
    ? (Linear(spec.inputs, 1) as Layer)
    : (Mlp([spec.inputs, spec.hidden, 1], { activation: 'tanh', init: xavierUniform() }) as Layer)
}

/** Build the model of a config. */
export function moeModel(config: MoeConfig): MoeModel {
  const spec = moeSpec(config)
  const experts = Array.from({ length: spec.experts }, () => expertLayer(spec))
  const flatGate: GateKind = spec.gate === 'hierarchical' ? 'softmax' : spec.gate
  const layer = MixtureOfExperts(spec.inputs, experts, {
    gate: flatGate,
    k: spec.k,
    temperature: spec.temperature,
    capacityFactor: spec.capacityFactor,
    zeroRouter: spec.routerInit === 'zero',
  })
  const top = spec.gate === 'hierarchical' ? Linear(spec.inputs, spec.groups) : null
  return {
    spec,
    layer,
    top,
    init: (s) => ({
      moe: layer.init(child(s, 'moe')),
      ...(top ? { top: top.init(child(s, 'top')) } : {}),
      ...(spec.task === 'regression' && spec.objective === 'mixture' ? { logSigma: zeros([spec.experts]) } : {}),
    }),
  }
}

/** A forward pass: the routing and each expert's raw output ([T, N]: means, or log-odds). */
export type MoeForward = { routing: Routing; outputs: Value }

/**
 * The hierarchical gate (Jordan and Jacobs, 1994): g_{gj}(x) = softmax_g(top(x)/τ) · softmax_j(z_{g·}/τ), the router's
 * N logits read as G groups of N/G; returns a dense routing of the products (and their logs).
 */
function hierarchicalRouting(model: MoeModel, params: MoeParams, x: Value): Routing {
  const { experts: N, groups: G, temperature: tau } = model.spec
  const T = shapeOfValue(x)[0]
  const z = mul(linear(x, params.moe.router.weight, params.moe.router.bias), 1 / tau)
  const lower = logSoftmax(reshape(z, [T, G, N / G]))
  const upper = logSoftmax(mul(linear(x, params.top!.weight, params.top!.bias), 1 / tau))
  const logProbs = reshape(add(lower, expandDims(upper, -1)), [T, N])
  return denseRouting(exp(logProbs), logProbs)
}

/** Evaluate the gate and every expert on x [T, d]. */
export function moeForward(model: MoeModel, params: MoeParams, x: Value, ctx?: Context): MoeForward {
  if (model.spec.gate === 'hierarchical') {
    const routing = hierarchicalRouting(model, params, x)
    const outs = model.layer.experts.map((e, i) => e.apply(params.moe.experts[i], x, ctx))
    return { routing, outputs: concatColumns(outs) }
  }
  const { routing, expertOutputs } = model.layer.forward(params.moe, x, ctx)
  return { routing, outputs: concatColumns(expertOutputs) }
}

/** [T, 1] columns side by side as [T, N], with constant one-hot rows (differentiable). */
function concatColumns(cols: readonly Value[]): Value {
  const N = cols.length
  let out: Value | null = null
  cols.forEach((c, i) => {
    const e = fromData(
      Float64Array.from({ length: N }, (_, j) => (j === i ? 1 : 0)),
      [1, N],
    )
    const term = matmul(c, e)
    out = out === null ? term : add(out, term)
  })
  return out!
}

const NEG = -1e30
const LOG_2PI = Math.log(2 * Math.PI)

/** log of the combine weights, with −∞ (a large negative number) where a weight is exactly zero. */
function logWeights(routing: Routing): Value {
  const positive = routing.dispatch
  return where(positive, log(where(positive, routing.combine, 1)), NEG)
}

/**
 * Each expert's log-likelihood of y, [T, N]: log N(y; μᵢ, σᵢ²) for regression, log Bernoulli(y; σ(ηᵢ)) for
 * classification.
 */
export function expertLogLikelihood(model: MoeModel, params: MoeParams, outputs: Value, y: Tensor): Value {
  const T = y.shape[0]
  const yc = reshape(y, [T, 1])
  if (model.spec.task === 'regression') {
    const logSigma = params.logSigma ?? zeros([model.spec.experts])
    const z = mul(sub(yc, outputs), exp(neg(logSigma)))
    return sub(mul(-0.5, square(z)), add(logSigma, 0.5 * LOG_2PI))
  }
  // log σ(η) = −softplus(−η), log(1 − σ(η)) = −softplus(η).
  const yf = fromData(Float64Array.from(toFlat(y), Number), [T, 1])
  return neg(add(mul(yf, softplus(neg(outputs))), mul(sub(1, yf), softplus(outputs))))
}

/** The parts of the training objective. */
export type MoeLoss = {
  /** The data term: the mixture's mean negative log-likelihood, or the blend's squared error or log-loss. */
  data: Value
  /** The Switch load-balancing loss (for sparse gates; 1 for dense ones). */
  balance: Value
  /** The importance loss CV² (Shazeer et al., 2017). */
  importance: Value
  /** The router z-loss. */
  z: Value
  total: Value
  forward: MoeForward
}

/** Weights of the auxiliary losses. */
export type AuxWeights = { balance?: number; importance?: number; z?: number }

/** The objective at params on (x, y): the data term plus the weighted auxiliary losses. */
export function moeLoss(
  model: MoeModel,
  params: MoeParams,
  x: Value,
  y: Tensor,
  weights: AuxWeights = {},
  ctx?: Context,
): MoeLoss {
  const forward = moeForward(model, params, x, ctx)
  const { routing, outputs } = forward
  const T = y.shape[0]
  let data: Value
  if (model.spec.objective === 'mixture') {
    const joint = add(logWeights(routing), expertLogLikelihood(model, params, outputs, y))
    data = neg(mean(logsumexp(joint, -1)))
  } else {
    const blend = sum(mul(routing.combine, outputs), -1)
    if (model.spec.task === 'regression') data = mean(square(sub(blend, reshape(y, [T]))))
    else {
      const yf = fromData(Float64Array.from(toFlat(y), Number), [T])
      data = mean(add(mul(yf, softplus(neg(blend))), mul(sub(1, yf), softplus(blend))))
    }
  }
  const balance = loadBalancingLoss(routing)
  const importance = importanceLoss(routing)
  const z = routerZLoss(routing)
  let total = data
  if (weights.balance) total = add(total, mul(weights.balance, balance))
  if (weights.importance) total = add(total, mul(weights.importance, importance))
  if (weights.z) total = add(total, mul(weights.z, z))
  return { data, balance, importance, z, total, forward }
}

/** Predictions on a batch, as plain numbers. */
export type MoePrediction = {
  /** The combine weights [T][N]. */
  readonly gate: number[][]
  /** Each expert's prediction [T][N]: its mean of y, or its P(y = 1). */
  readonly experts: number[][]
  /** The model's prediction [T]: E[y | x] (regression) or P(y = 1 | x) (classification). */
  readonly prediction: number[]
  /** The expert with the largest weight, per row. */
  readonly assignment: number[]
}

const rowsOf = (v: Value, T: number, N: number) => {
  const f = toFlat(unwrap(v) as Tensor)
  return Array.from({ length: T }, (_, t) => f.slice(t * N, (t + 1) * N))
}
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

/** Predict on x [T, d] (evaluation mode: no gate noise). */
export function moePredict(model: MoeModel, params: MoeParams, x: Tensor): MoePrediction {
  const { routing, outputs } = moeForward(model, params, x)
  const T = x.shape[0]
  const N = model.spec.experts
  const gate = rowsOf(routing.combine, T, N)
  const raw = rowsOf(outputs, T, N)
  const classification = model.spec.task === 'classification'
  const experts = classification ? raw.map((r) => r.map(sigmoid)) : raw
  const prediction = gate.map((g, t) => {
    if (classification && model.spec.objective === 'blend') return sigmoid(g.reduce((a, w, i) => a + w * raw[t][i], 0))
    return g.reduce((a, w, i) => a + w * experts[t][i], 0)
  })
  const assignment = gate.map((g) => g.indexOf(Math.max(...g)))
  return { gate, experts, prediction, assignment }
}
