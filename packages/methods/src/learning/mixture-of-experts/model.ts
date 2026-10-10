/**
 * The mixture of experts as a statistical model (Jacobs, Jordan, Nowlan and Hinton, 1991, "Adaptive mixtures of local
 * experts", Neural Computation 3(1); Jordan and Jacobs, 1994, "Hierarchical mixtures of experts and the EM algorithm",
 * Neural Computation 6(2)), built on the compute layer `MixtureOfExperts` of `aifn-compute/nn/experts`.
 *
 * $N$ experts (linear, or a small MLP) each map $\xvec$ to a prediction $f_i(\xvec)$: a mean of $y$ for regression,
 * the log-odds of $y = 1$ for binary classification. A gate turns $\xvec$ into weights $g_i(\xvec)$ over the experts:
 * any gate of `route` (softmax, top-k, noisy top-k, Switch, expert choice), or a two-level hierarchy, a softmax over
 * $G$ groups times a softmax over the experts within each group.
 *
 * Two objectives:
 * - `mixture`: the classic model, a conditional mixture $p(y \mid \xvec) = \sum_i g_i(\xvec) p_i(y \mid \xvec)$ with
 *   a normal ($\sigma_i$ per expert) or Bernoulli law per expert, fitted by maximum likelihood. Each expert
 *   specialises in the inputs it is responsible for, and where the gate is mixed $p(y \mid \xvec)$ is multimodal.
 * - `blend`: the modern sparse layer, which blends outputs, $\hat y = \sum_i g_i(\xvec) f_i(\xvec)$, with a squared
 *   error or a logistic loss on the blend. The experts cooperate rather than compete (Jacobs et al., 1991, §1
 *   contrasts the two).
 *
 * Inputs are $T \times d$, one row per token or example; targets are $T$ floats, or 0/1 labels.
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
  /** Input width $d$. */
  inputs: Size
  /** Real targets (a normal law per expert) or 0/1 labels (a Bernoulli law per expert). */
  task: 'regression' | 'classification'
  /** Number of experts $N$ (default 4). */
  experts?: Size
  /** The experts' form (default `linear`). */
  expert?: ExpertKind
  /** Hidden width of an MLP expert (default 16; tanh). */
  hidden?: Size
  /** The gate (default `softmax`). */
  gate?: GateChoice
  /** Experts per token of the top-k gates (default 2). */
  k?: Size
  /** Gate temperature $\tau$ (default 1): the gate is a softmax of the logits over $\tau$. */
  temperature?: number
  /** Capacity factor of the sparse gates (default $\infty$: no dropping). */
  capacityFactor?: number
  /** Groups $G$ of the hierarchical gate (default 2; $N$ must be a multiple of $G$). */
  groups?: Size
  /** What the experts' outputs are combined into (default `mixture`). */
  objective?: MoeObjective
  /**
   * The router's initial weights: `default` (LeCun uniform) or `zero`, so every expert starts equally likely and, under
   * top-1 routing, every tie goes to the first expert.
   */
  routerInit?: 'default' | 'zero'
}

/** A config with every default filled in. */
export type MoeSpec = Required<MoeConfig>

/**
 * Fill in the defaults of a config, checking the hierarchical gate's grouping. Throws `DomainError` when a
 * hierarchical gate's experts do not split into `groups` equal groups.
 *
 * @param config The structure; fields it sets override the defaults.
 * @returns The config with every field set.
 *
 * @example The defaults, and a grouping that does not divide
 * print(moeSpec({ inputs: 1, task: 'regression' }))
 * try {
 *   moeSpec({ inputs: 1, task: 'regression', experts: 3, gate: 'hierarchical' })
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
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

/**
 * Parameters: the compute layer's (router, noise map, experts), the top gate of a hierarchy, and $\log\sigma_i$ per
 * expert.
 */
export type MoeParams = {
  /**
   * The compute layer's parameters: the router (whose $N$ logits a hierarchy reads as $G$ groups), its noise map, and
   * each expert's.
   */
  moe: MixtureOfExpertsParams
  /** The top-level gate over groups (hierarchical gate only). */
  top?: LinearParams
  /** $\log\sigma_i$ of each expert's normal law, $N$ values (regression with the mixture objective). */
  logSigma?: Tensor
}

/** A built mixture of experts: the compute layer and the spec. */
export type MoeModel = {
  /** The structure, every default filled in. */
  readonly spec: MoeSpec
  /** The compute layer: the router and the experts (for a hierarchy, the router gives the lower gates' logits). */
  readonly layer: MixtureOfExpertsLayer
  /** The top-level gate over the groups, $d \to G$ (hierarchical gate only, else null). */
  readonly top: Layer<LinearParams> | null
  /**
   * Initial parameters from a stream: the layer's, the top gate's, and $\log\sigma_i = 0$ for regression with the
   * mixture objective.
   */
  init(s: Stream): MoeParams
}

/**
 * One expert's layer: linear $d \to 1$, or an MLP $d \to$ `hidden` $\to 1$ with tanh and Xavier-uniform weights.
 *
 * @param spec The structure; `expert`, `inputs` and `hidden` are read.
 * @returns The layer.
 */
function expertLayer(spec: MoeSpec): Layer {
  return spec.expert === 'linear'
    ? (Linear(spec.inputs, 1) as Layer)
    : (Mlp([spec.inputs, spec.hidden, 1], { activation: 'tanh', init: xavierUniform() }) as Layer)
}

/**
 * Build the model of a config: $N$ experts under the compute layer `MixtureOfExperts`, whose router is a softmax for
 * a hierarchy (with the top gate beside it). Throws `DomainError` as `moeSpec` does.
 *
 * @param config The structure.
 * @returns The model, whose `init` draws parameters from a stream.
 *
 * @example Two linear experts on one input, and the shapes of their parameters
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const params = model.init(stream(1))
 * print('router weight:', params.moe.router.weight.shape, 'bias:', params.moe.router.bias.shape)
 * print('experts:', params.moe.experts.length, 'log sigma:', params.logSigma)
 */
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

/**
 * A forward pass: `routing`, the gate's routing, and `outputs`, each expert's raw output ($T \times N$: means, or
 * log-odds).
 */
export type MoeForward = { routing: Routing; outputs: Value }

/**
 * The hierarchical gate (Jordan and Jacobs, 1994):
 * $g_{gj}(\xvec) = \operatorname{softmax}_g(\mathrm{top}(\xvec)/\tau) \cdot \operatorname{softmax}_j(z_{gj}/\tau)$,
 * with the router's $N$ logits $\zvec$ read as $G$ groups of $N/G$; returns a dense routing of the products (and their
 * logs). Differentiable in the parameters.
 *
 * @param model The model; its gate must be hierarchical.
 * @param params The parameters; `moe.router` and `top` are read.
 * @param x The inputs, $T \times d$.
 * @returns A dense `softmax` routing whose combine weights are the $T \times N$ products, expert $j$ of group $g$ at
 *   column $g N/G + j$, and whose `logits` are their logs.
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

/**
 * Evaluate the gate and every expert on $T$ rows. Differentiable in the parameters.
 *
 * @param model The model.
 * @param params The parameters.
 * @param x The inputs, $T \times d$.
 * @param ctx The layers' context: in training mode (`train`, with a `stream`) the noisy gates draw their noise;
 *   left out, evaluation mode.
 * @returns The routing and the experts' raw outputs, $T \times N$.
 *
 * @example Two softmax-gated experts on three rows: each row's gate weights sum to 1
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const { routing, outputs } = moeForward(model, model.init(stream(1)), tensor([[-1], [0], [1]]))
 * print('gate weights:', routing.combine)
 * print('expert outputs:', outputs)
 */
export function moeForward(model: MoeModel, params: MoeParams, x: Value, ctx?: Context): MoeForward {
  if (model.spec.gate === 'hierarchical') {
    const routing = hierarchicalRouting(model, params, x)
    const outs = model.layer.experts.map((e, i) => e.apply(params.moe.experts[i], x, ctx))
    return { routing, outputs: concatColumns(outs) }
  }
  const { routing, expertOutputs } = model.layer.forward(params.moe, x, ctx)
  return { routing, outputs: concatColumns(expertOutputs) }
}

/**
 * Columns side by side, as the sum of each column times a constant one-hot row (differentiable).
 *
 * @param cols $N$ columns, $T \times 1$ each.
 * @returns The $T \times N$ matrix whose column $i$ is `cols[i]`.
 */
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

/**
 * The log of the combine weights, with $-10^{30}$ (standing for $-\infty$) where an expert does not process a row.
 *
 * @param routing The routing; its `dispatch` mask and `combine` weights are read.
 * @returns The $T \times N$ log weights, differentiable in the combine weights.
 */
function logWeights(routing: Routing): Value {
  const positive = routing.dispatch
  return where(positive, log(where(positive, routing.combine, 1)), NEG)
}

/**
 * Each expert's log-likelihood of $y$: $\log \Gauss(y; \mu_i, \sigma_i^2)$ for regression,
 * $\log \Bern(y; \operatorname{sigmoid}(\eta_i))$ for classification. Differentiable in the outputs and $\log\sigma_i$.
 *
 * @param model The model; its task is read.
 * @param params The parameters; only `logSigma` is read (taken as 0, so $\sigma_i = 1$, when absent).
 * @param outputs The experts' raw outputs $\mu_i$ or $\eta_i$, $T \times N$, as `moeForward` returns them.
 * @param y The targets, $T$ values (0/1 labels for classification).
 * @returns The $T \times N$ log-likelihoods.
 *
 * @example At initialisation, with unit standard deviations: each expert's log-density of the targets
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const params = model.init(stream(1))
 * const { outputs } = moeForward(model, params, tensor([[-1], [1]]))
 * print('outputs:', outputs)
 * print('log-likelihoods of y = 0, 0:', expertLogLikelihood(model, params, outputs, tensor([0, 0])))
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
  /** The importance loss $\mathrm{CV}^2$ (Shazeer et al., 2017). */
  importance: Value
  /** The router z-loss. */
  z: Value
  /** The data term plus the weighted auxiliary losses: the objective to minimise. */
  total: Value
  /** The forward pass the losses were computed from. */
  forward: MoeForward
}

/**
 * Weights of the auxiliary losses: `balance` for the load-balancing loss, `importance` for the importance loss and `z`
 * for the router z-loss (each default 0: left out).
 */
export type AuxWeights = { balance?: number; importance?: number; z?: number }

/**
 * The objective at the parameters on $(\xvec, y)$: the data term plus the weighted auxiliary losses. The mixture's
 * data term is $-\frac{1}{T} \sum_t \log \sum_i g_i(\xvec_t) p_i(y_t \mid \xvec_t)$; the blend's is the mean squared
 * error or log-loss of $\hat y$. Differentiable in the parameters.
 *
 * @param model The model.
 * @param params The parameters.
 * @param x The inputs, $T \times d$.
 * @param y The targets, $T$ values (0/1 labels for classification).
 * @param weights The auxiliary losses' weights (default none).
 * @param ctx The layers' context, passed to `moeForward` (noise and training mode).
 * @returns Every part, the total and the forward pass.
 *
 * @example The parts of the objective at initialisation, with a small z-loss weight
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const parts = moeLoss(model, model.init(stream(1)), tensor([[-1], [0], [1]]), tensor([2, 0, 2]), { z: 0.01 })
 * print('data:', parts.data, 'importance:', parts.importance, 'z:', parts.z)
 * print('total = data + 0.01 z:', parts.total)
 */
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
  /** The combine weights, $T$ rows of $N$. */
  readonly gate: number[][]
  /** Each expert's prediction, $T$ rows of $N$: its mean of $y$, or its $P(y = 1)$. */
  readonly experts: number[][]
  /**
   * The model's prediction, $T$ values: $\expect[y \mid \xvec]$ (regression), $P(y = 1 \mid \xvec)$ (classification
   * with the mixture objective), or the sigmoid of the blended log-odds (classification with the blend).
   */
  readonly prediction: number[]
  /** The expert with the largest weight, per row (the first on a tie). */
  readonly assignment: number[]
}

/**
 * A $T \times N$ value as $T$ rows of plain numbers.
 *
 * @param v The value (traced or not; its primal is read).
 * @param T The number of rows.
 * @param N The number of columns.
 * @returns The rows.
 */
const rowsOf = (v: Value, T: number, N: number) => {
  const f = toFlat(unwrap(v) as Tensor)
  return Array.from({ length: T }, (_, t) => f.slice(t * N, (t + 1) * N))
}
/**
 * The logistic function $1/(1 + e^{-v})$.
 *
 * @param v The log-odds.
 * @returns The probability.
 */
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

/**
 * Predict on $T$ rows, in evaluation mode (no gate noise), as plain numbers.
 *
 * @param model The model.
 * @param params The parameters.
 * @param x The inputs, $T \times d$.
 * @returns The gate weights, each expert's prediction, the model's prediction and the assigned expert per row.
 *
 * @example Two experts fitted by EM to a V of two regimes, falling left of 0 and rising right of it
 * const xs = Array.from({ length: 16 }, (_, t) => -1 + (2 * t) / 15)
 * const x = tensor(xs.map((v) => [v]))
 * const y = add(tensor(xs.map((v) => Math.abs(2 * v))), normals(stream(1), 16, 0, 0.1))
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const { params } = run(moeEm(model, { x, y }), { params: model.init(stream(2)) }, 15)
 * const p = moePredict(model, params, tensor([[-0.5], [0.5]]))
 * print('gate at x = -0.5 and 0.5:', p.gate)
 * print('experts there:', p.experts)
 * print('predictions:', p.prediction, 'assigned experts:', p.assignment)
 */
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
