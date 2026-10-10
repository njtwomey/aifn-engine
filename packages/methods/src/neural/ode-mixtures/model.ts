/**
 * Stochastic vector field mixtures (SVFM; Twomey, Kozłowski & Santos-Rodríguez, 2020, "Neural ODEs with stochastic
 * vector field mixtures", ECAI): a neural ODE $\nabla \hvec(t) = f(\hvec(t), t; \thetavec)$ (eq. 1) whose vector
 * field (VF) is one of $K$ components $f^{(k)}$, with component membership $\pivec(t) \in \Delta^K$, and whose
 * components may be stochastic.
 *
 * - **VF unit** (fig. 2a): $f^{(k)}(\hvec, t)$ is an MLP.
 * - **SVF unit** (fig. 3, eqs. 6–7): a shared representation $f_z = g_z(\hvec, t)$ gives the mean and variance of the
 *   VF's direction $\uvec \sim \Gauss_s(\muvec^{(u)}, \tau^{(u)})$ on the sphere and of its log length,
 *   $\log v \sim \Gauss(\log \mu^{(v)}, \tau^{(v)})$; a sample of the VF is $\nabla \hvec = v \uvec$. The network
 *   outputs a VF $\avec$, decomposed into its orientation $\muvec^{(u)} = \avec / \norm{\avec}$ and length
 *   $\mu^{(v)} = \norm{\avec}$, and the two variances. $\Gauss_s$ is read as a projected normal:
 *   $\uvec = \operatorname{normalise}(\muvec^{(u)} + \sqrt{\tau^{(u)}} \Pmat \epsilonvec)$ with
 *   $\Pmat = \Imat - \muvec^{(u)} \muvec^{(u)\top}$, so $\uvec^\top \muvec^{(u)} > 0$ always (the paper's
 *   directional preservation).
 * - **Component selection** on a grid $t_0 < t_1 < \dots < t_T$: *pick and stick* holds
 *   $\pivec(t_i) = \pivec(t_0) = f_{\pi_{t_0}}(\hvec(t_0), t_0)$ (eq. 2); *forward filtering* (eqs. 3–5) has emission
 *   $\psivec(t_i) = f_\psi(\hvec(t_i), t_i) \in \Delta^K$ and transition $\Psimat(t_i) = f_\Psi(\hvec(t_i), t_i)$
 *   with rows in $\Delta^K$, and $\pivec(t_i) \propto \Psimat(t_i)^\top (\psivec(t_i) \odot \pivec(t_{i-1}))$.
 *
 * Two computations share the parameters. `propagate` carries, for each component $k$, the state reached under it and
 * the spread of its stochastic VF's samples over the interval (moments: $\mvec_k' = \expect[\nabla \hvec^{(k)}]$,
 * $r_k'$ the VF's standard deviation per coordinate, which adds linearly because a sample keeps its noise for the whole
 * solve), so the output at $t_i$ is the mixture $\sum_k \pi_k(t_i) \Gauss(\mvec_k(t_i), (r_k^2 + \sigma_0^2)\Imat)$
 * of eq. 10. Under forward filtering, $\psi_j$ and row $j$ of $\Psimat$ are evaluated on the state reached under
 * component $j$, and the component-conditioned states are re-mixed with the same joint weights (moment matching).
 * `realisedRhs` (in `sampling.ts`) evaluates one sampled path's VF (§3: a frozen uniform picks the component from
 * $\pivec$, frozen normals sample the SVF), for trajectories and the per-instance work of a solve.
 *
 * Augmentation (fig. 6, the A- models; Dupont, Doucet & Teh, 2019) pads $\hvec$ with zeros. A context $\cvec$ (e.g.
 * the time of day as $(\cos, \sin)$) is appended to the input of every network.
 */

import type { Scalar } from 'aifn-compute/foundation/contracts'
import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  div,
  exp,
  expandDims,
  eye,
  fromData,
  log,
  logsumexp,
  matmul,
  maximum,
  mul,
  ones,
  permute,
  reshape,
  shapeOfValue,
  slice,
  sqrt,
  square,
  sub,
  sum,
  tanh,
  toFlat,
  unwrap,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { logSoftmax, sigmoid, softplus } from 'aifn-compute/numerics/special'
import { relu } from 'aifn-compute/nn/functional'
import { heUniform, xavierUniform } from 'aifn-compute/nn/init'
import { odeFlow, type OdeFlowMethod, type OdeSolveInfo } from 'aifn-compute/dynamics/ode'
import { DomainError } from 'aifn-compute/foundation/errors'

/** How the component membership $\pivec(t)$ is specified over the integration interval (§2.1). */
export type ComponentSelection = 'pick-and-stick' | 'forward-filtering'

/** Field activations; the paper uses rectified units (§4.1.3). */
export type FieldActivation = 'relu' | 'tanh' | 'softplus'

/** Options of {@link svfm}; plain data. */
export type SvfmOptions = {
  /** Dimension $D$ of the data $\hvec(t_0) = \xvec$. */
  dim: number
  /** Components $K$ of the mixture (1: a single VF). Default 1. */
  components?: number
  /** SVF units (fig. 3) rather than VF units (fig. 2a). Default false. */
  stochastic?: boolean
  /** How $\pivec(t)$ evolves over the grid. Default `'pick-and-stick'`. */
  selection?: ComponentSelection
  /** Zero-padded extra state dimensions (the A- models of fig. 6). Default 0. */
  augment?: number
  /** Context inputs appended to every network's input (e.g. the time-of-day tuple). Default 0. */
  context?: number
  /** Hidden units per layer (the paper: 32 or 64). Default 32. */
  hidden?: number
  /** Hidden layers (the paper: 1 or 2). Default 2. */
  layers?: number
  /** The activation of the component VFs' hidden layers. Default `'relu'`. */
  activation?: FieldActivation
  /** Classes of a classifier readout (a linear map of $\hvec(t_T)$ to logits), or 0 to read the state. Default 0. */
  classes?: number
  /**
   * Grid intervals $T$ on $[0, 1]$: $\pivec$ is updated and the trajectory losses sum at $t_1, \dots, t_T$. Default
   * 10.
   */
  grid?: number
  /** The differentiable solver within each interval. Default RK4. */
  method?: OdeFlowMethod
  /** Step of a fixed-step solver (at most the grid spacing). Default 0.05. */
  stepSize?: number
  /** Relative tolerance of Dormand–Prince. Default $10^{-3}$. */
  rtol?: number
  /** Absolute tolerance of Dormand–Prince. Default $10^{-5}$. */
  atol?: number
  /**
   * The largest variance $\tau^{(u)}, \tau^{(v)}$ of an SVF ($\tau = \tau_{\max} \operatorname{sigmoid}(\cdot)$, with
   * $\tau_{\max}$ this value). Default 0.5.
   */
  maxVariance?: number
  /** Initial bias of the SVF variance heads (default $-5$: $\tau$ starts near 0). */
  varianceBias?: number
  /**
   * Learn $\tau^{(u)}, \tau^{(v)}$ from the state, or hold them at $\tau_{\max} \operatorname{sigmoid}(b)$, $b$ the
   * `varianceBias`. Default true.
   */
  learnVariance?: boolean
  /** The component VFs share their hidden layers (a trunk) and differ in their output layers. Default false. */
  sharedTrunk?: boolean
  /** The VFs take $t$ as an input, $f(\hvec(t), t)$ (eq. 1), or are autonomous, $f(\hvec(t))$. Default true. */
  timeDependent?: boolean
  /**
   * $\pivec(t_0)$ (eq. 2): a function $f_{\pi_{t_0}}$ of $(\hvec(t_0), t_0)$ (`'state'`) or a learned constant.
   * Default `'state'`.
   */
  prior?: 'state' | 'constant'
  /**
   * Hidden units of the $\pivec$ networks $f_{\pi_{t_0}}$, $f_\psi$ and $f_\Psi$, when they have a hidden layer.
   * $\pivec$ only chooses which field; the fields carry the dynamics, so $\pivec$ stays simple. Default 8.
   */
  piHidden?: number
  /** Hidden layers of the $\pivec$ networks (0: linear, a softmax of an affine map). Default 0. */
  piLayers?: number
  /** Activation of the $\pivec$ networks' hidden layers. Default the fields' `activation`. */
  piActivation?: FieldActivation
  /** The softmax temperature of $\pivec(t_0)$. Default 1. */
  temperature?: number
  /** Forward filtering: emissions $\psivec$ learned (eq. 4) or uniform. Default learned. */
  emissions?: 'learned' | 'uniform'
  /** Forward filtering: transitions $\Psimat$ learned (eq. 3) or fixed. Default learned. */
  transitions?: 'learned' | 'fixed'
  /** The probability of staying in a component: fixed transitions keep it; learned ones start at it. Default 0.9. */
  stickiness?: number
}

/**
 * A stack of $K$ MLPs evaluated together, one entry per layer: `weight` $[K, \text{in}, \text{out}]$ and `bias`
 * $[K, 1, \text{out}]$ (leading size 1 for a layer shared by the $K$).
 */
export type StackedMlpParams = { weight: Tensor; bias: Tensor }[]

/** Parameters of an SVFM. Networks absent from a configuration are empty. */
export type SvfmParams = {
  /** The $K$ component networks (VF: $S$ outputs; SVF: $S + 2$, the mean VF and the two variances' logits). */
  fields: StackedMlpParams
  /** $f_{\pi_{t_0}}$ (eq. 2): logits over components at $(\hvec(t_0), t_0)$; empty when $K = 1$. */
  prior: StackedMlpParams
  /** $f_\psi$ (eq. 4): emission logits $[K]$; empty unless forward filtering with learned emissions. */
  emission: StackedMlpParams
  /** $f_\Psi$ (eq. 3): transition logits $[K \times K]$; empty unless forward filtering with learned transitions. */
  transition: StackedMlpParams
  /** Classifier readout: `weight` $[S, \text{classes}]$ and `bias` $[\text{classes}]$; empty for state outputs. */
  readout: { weight?: Tensor; bias?: Tensor }
  /** $\log \sigma_0$, the output noise floor of the mixture density (eq. 10); starts at $\log 0.1$. */
  logNoise: Tensor
}

/** The moments `propagate` carries, at every grid time $t_0, \dots, t_T$. */
export type Propagation = {
  /** Grid times $t_0, \dots, t_T$. */
  times: number[]
  /**
   * Component-conditioned states $\mvec_k(t_i)$ $[K, B, S]$ after the update at $t_i$ (what interval $i + 1$ starts
   * from).
   */
  states: Value[]
  /** The same before the update at $t_i$ (where interval $i$ ended; equal to `states` under pick and stick). */
  arrived: Value[]
  /** Spread $r_k(t_i)$ $[K, B]$ after the update. */
  spread: Value[]
  /** $\log \pivec(t_i)$ $[B, K]$. */
  logWeights: Value[]
  /**
   * The mean VF $\expect[\nabla \hvec^{(k)}]$ at the arrived states, for $i = 1, \dots, T$ ($[K, B, S]$); index 0
   * holds the field at $t_0$.
   */
  fields: Value[]
}

/** An SVFM, ready to train. */
export type Svfm = {
  /** The options with their defaults filled in. */
  options: Required<SvfmOptions>
  /** State dimension $S = D + \text{augment}$. */
  stateDim: number
  /** Fresh parameters drawn from a stream (see `svfm` for the initialisation). */
  init(s: Stream): SvfmParams
  /** $\hvec(t_0)$: $\xvec$ $[B, D]$ padded with zeros to $[B, S]$. */
  lift(x: Value): Value
  /**
   * The mean VF and its spread rate per component at time $t$ on states $\zvec$ $[K, B, S]$ with context $[B, C]$ or
   * null: $[K, B, S]$ and $[K, B]$ (the spread rate is 0 for VF units).
   */
  moments(params: SvfmParams, t: Scalar, z: Value, c: Value | null): { mean: Value; sd: Value }
  /** $\log \pivec(t_0)$ $[B, K]$ (eq. 2) at $\hvec(t_0)$ $[B, S]$ (zeros $[B, 1]$ when $K = 1$). */
  prior(params: SvfmParams, h: Value, c: Value | null): Value
  /**
   * Forward filtering at $t_i$ (eq. 5) from $\log \pivec(t_{i-1})$ $[B, K]$, on component-conditioned states
   * $[K, B, S]$ with spreads $[K, B]$: the new $\log \pivec$ $[B, K]$ and the re-mixed states and spreads. Returns its
   * inputs unchanged unless forward filtering with $K > 1$.
   */
  filter(
    params: SvfmParams,
    t: Scalar,
    logWeights: Value,
    states: Value,
    spread: Value,
    c: Value | null,
  ): { logWeights: Value; states: Value; spread: Value }
  /** The moments over the grid from $\xvec$ $[B, D]$ (see the file comment); `onSolve` sees every solve. */
  propagate(params: SvfmParams, x: Value, c: Value | null, onSolve?: (info: OdeSolveInfo) => void): Propagation
  /**
   * Mixture output at grid time $i$ over the first $D$ coordinates: $\log \pivec$ $[B, K]$, means $[B, K, D]$ and
   * $\log \sigma$ $[B, K, D]$, with $\sigma^2 = r_k^2 + \sigma_0^2$.
   */
  output(params: SvfmParams, p: Propagation, i: number): { logWeights: Value; means: Value; logScales: Value }
  /**
   * Classifier log-likelihoods $\log p(y = c \mid k)$ at $t_T$: $[B, K, \text{classes}]$ (probit-scaled for SVF
   * units). Throws `DomainError` without a readout.
   */
  classLogLikelihoods(params: SvfmParams, p: Propagation): Value
}

/**
 * The prior's logits are its network's output times this factor. Under Adam the factor slows the prior's learning by
 * the same amount, so the components specialise (each target picks the component that serves it best, as in EM)
 * before $\pivec(t_0)$ commits; a prior that saturates first gates a component off everywhere, and it never recovers.
 */
const PRIOR_RATE = 0.2

/**
 * The layer sizes of an MLP.
 *
 * @param inF The number of inputs.
 * @param hidden The units of each hidden layer.
 * @param layers The number of hidden layers (0 for an affine map).
 * @param out The number of outputs.
 * @returns The sizes, inputs first.
 */
const mlpSizes = (inF: number, hidden: number, layers: number, out: number) => [
  inF,
  ...Array.from({ length: layers }, () => hidden),
  out,
]

/**
 * Initialise $K$ stacked MLPs of the given sizes: He-uniform hidden layers, a Xavier-uniform output layer and zero
 * biases.
 *
 * @param s The stream the weights are drawn from.
 * @param K The number of MLPs.
 * @param sizes The layer sizes, inputs first, outputs last.
 * @param outScale The gain of the output layer's Xavier-uniform draw (0 gives zero output weights).
 * @param sharedTrunk Draw the hidden layers once (leading size 1, shared by the $K$ MLPs); the output layer is per MLP.
 * @returns The layers' parameters.
 *
 * @example Three MLPs of sizes 2, 4, 1, with and without a shared trunk
 * const own = stackedMlpInit(stream(0), 3, [2, 4, 1])
 * print('weights:', own.map((l) => l.weight.shape), ' biases:', own.map((l) => l.bias.shape))
 * const shared = stackedMlpInit(stream(0), 3, [2, 4, 1], 1, true)
 * print('shared trunk weights:', shared.map((l) => l.weight.shape))
 */
export function stackedMlpInit(
  s: Stream,
  K: number,
  sizes: readonly number[],
  outScale = 1,
  sharedTrunk = false,
): StackedMlpParams {
  const he = heUniform()
  const xa = xavierUniform({ gain: outScale })
  return sizes.slice(0, -1).map((inF, l) => {
    const out = sizes[l + 1]
    const last = l === sizes.length - 2
    const init = last ? xa : he
    // A shared trunk: hidden layers stacked once (broadcast over the K components), output layers per component.
    const k = sharedTrunk && !last ? 1 : K
    return {
      weight: init(child(s, 'layer', l), [k, inF, out], { fanIn: inF, fanOut: out }),
      bias: zeros([k, 1, out]),
    }
  })
}

/**
 * Apply an activation.
 *
 * @param a The activation.
 * @param x The pre-activations.
 * @returns The activations, with the shape of `x`.
 */
const activate = (a: FieldActivation, x: Value): Value =>
  a === 'relu' ? relu(x) : a === 'tanh' ? tanh(x) : softplus(x)

/**
 * Evaluate $K$ stacked MLPs at once, by batched matrix products (differentiable).
 *
 * @param params The layers, as `stackedMlpInit` makes them.
 * @param x The inputs: $[K, B, \text{in}]$, one batch per MLP, or $[B, \text{in}]$, shared by all $K$.
 * @param activation The activation after every layer but the last.
 * @returns The outputs, $[K, B, \text{out}]$.
 *
 * @example Three MLPs evaluated on the same two inputs
 * const params = stackedMlpInit(stream(0), 3, [2, 4, 1])
 * const y = stackedMlp(params, tensor([[1, 0], [0, 1]]), 'tanh')
 * print('outputs:', y.shape, y)
 */
export function stackedMlp(params: StackedMlpParams, x: Value, activation: FieldActivation): Value {
  let h = x
  params.forEach((layer, l) => {
    h = add(matmul(h, layer.weight), layer.bias)
    if (l < params.length - 1) h = activate(activation, h)
  })
  return h
}

/**
 * The inputs of a network: the state, the time and the context, joined on the last axis.
 *
 * @param z The states, $[B, S]$ or $[K, B, S]$.
 * @param t The time, repeated for every row.
 * @param c The context $[B, C]$, repeated over the $K$ components when `z` has them, or null for none.
 * @returns The inputs, with the leading axes of `z` and $S + 1 + C$ columns.
 */
function inputsOf(z: Value, t: Scalar, c: Value | null): Value {
  const shape = shapeOfValue(z)
  const lead = shape.slice(0, -1)
  const parts: Value[] = [z, mul(t, ones([...lead, 1]))]
  if (c !== null) {
    const cs = shapeOfValue(c)
    parts.push(lead.length === 2 ? mul(expandDims(c, 0), ones([lead[0], 1, cs[1]])) : c)
  }
  return concat(parts, -1)
}

/**
 * The SVF unit's heads (eqs. 6–7) from its raw outputs: the VF $\avec$ whose orientation
 * $\muvec^{(u)} = \avec / \norm{\avec}$ and length $\mu^{(v)} = \norm{\avec}$ are the means (the paper's
 * decomposition of a VF into length and orientation), and the variances
 * $\tau^{(u)}, \tau^{(v)} = \tau_{\max} \operatorname{sigmoid}(\cdot)$ of the last two outputs. The length vanishes
 * with $\avec$, so the VF stays continuous where its orientation is undefined. Differentiable.
 *
 * @param raw The unit's outputs, $[\dots, S + 2]$: the VF $\avec$ in the first $S$, then the two variances' logits.
 * @param S The state dimension.
 * @param maxVariance The largest variance $\tau_{\max}$.
 * @returns `vector`, $\avec$ $[\dots, S]$; `length2`, $\norm{\avec}^2$ $[\dots]$; and `tauU` and `tauV`, $[\dots]$.
 *
 * @example The VF $(3, 4)$ has orientation $(0.6, 0.8)$ and length 5; logits of 0 give half the largest variance
 * const h = svfHeads(tensor([[3, 4, 0, 0]]), 2)
 * print('vector', h.vector, ' squared length', h.length2, ' tauU', h.tauU, ' tauV', h.tauV)
 */
export function svfHeads(raw: Value, S: number, maxVariance = 0.5) {
  const lead = Array(shapeOfValue(raw).length - 1).fill(null)
  const a = slice(raw, ...lead, [0, S])
  const at = (j: number) => {
    const r = slice(raw, ...lead, [S + j, S + j + 1])
    return reshape(r, shapeOfValue(r).slice(0, -1))
  }
  const length2 = sum(square(a), -1)
  return { vector: a, length2, tauU: mul(maxVariance, sigmoid(at(0))), tauV: mul(maxVariance, sigmoid(at(1))) }
}

/**
 * A stochastic vector field mixture (see the file comment), or with one deterministic component the plain neural ODE
 * baseline. The component networks are `layers` hidden layers of `hidden` units, initialised by `stackedMlpInit` with
 * output gain 0.5; the $\pivec$ networks are affine by default, the prior's with gain 1 (or a learned constant), the
 * emissions' and transitions' with gain 0.1, and learned transitions start at the `stickiness`. An SVF's variance
 * heads start at `varianceBias`, so the components first specialise as deterministic VFs would. Everything is
 * differentiable in the parameters, through the solver by backprop.
 *
 * @param options The data's dimension, the mixture (components, units, selection), the networks, the grid and solver,
 *   the variances and the $\pivec$ networks; see `SvfmOptions`. Throws `DomainError` when `components` or `grid` is not
 *   a positive integer.
 * @returns The model: its options, initialiser, and the moments, prior, filter, propagation and outputs.
 *
 * @example A mixture's component weights: constant under pick and stick, moving under forward filtering
 * const x = tensor([[1, 0], [0, 1]])
 * for (const selection of ['pick-and-stick', 'forward-filtering']) {
 *   const model = svfm({ dim: 2, components: 2, selection, grid: 4 })
 *   const p = model.propagate(model.init(stream(0)), x, null)
 *   print(selection, 'pi of the first point at t_0 ... t_4:', p.logWeights.map((w) => toArray(exp(w))[0]))
 * }
 *
 * @example The spread of SVF units grows from 0 over the grid
 * const model = svfm({ dim: 2, components: 3, stochastic: true, varianceBias: 0 })
 * const p = model.propagate(model.init(stream(0)), tensor([[1, 0], [0, 1]]), null)
 * print('states at t_T:', p.states.at(-1).shape)
 * print('spread r_k at t_0 and t_T:', p.spread[0], p.spread.at(-1))
 */
export function svfm(options: SvfmOptions): Svfm {
  const o: Required<SvfmOptions> = {
    components: 1,
    stochastic: false,
    selection: 'pick-and-stick',
    augment: 0,
    context: 0,
    hidden: 32,
    layers: 2,
    activation: 'relu',
    classes: 0,
    grid: 10,
    method: 'rk4',
    stepSize: 0.05,
    rtol: 1e-3,
    atol: 1e-5,
    maxVariance: 0.5,
    varianceBias: -5,
    learnVariance: true,
    sharedTrunk: false,
    timeDependent: true,
    prior: 'state',
    piHidden: 8,
    piLayers: 0,
    piActivation: options.activation ?? 'relu',
    temperature: 1,
    emissions: 'learned',
    transitions: 'learned',
    stickiness: 0.9,
    ...options,
  }
  const { dim: D, components: K, stochastic, selection, augment, context: C, hidden, layers, activation, classes } = o
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError('svfm', 'svfm: components must be a positive integer')
  if (!(Number.isInteger(o.grid) && o.grid >= 1)) throw new DomainError('svfm', 'svfm: grid must be a positive integer')
  const S = D + augment
  const inF = S + 1 + C
  const filtering = selection === 'forward-filtering' && K > 1
  const times = Array.from({ length: o.grid + 1 }, (_, i) => i / o.grid)
  const fieldOut = stochastic ? S + 2 : S

  const lift = (x: Value): Value => (augment === 0 ? x : concat([x, zeros([shapeOfValue(x)[0], augment])], 1))

  const tt = (t: Scalar): Scalar => (o.timeDependent ? t : 0)
  const fixedTau = o.maxVariance / (1 + Math.exp(-o.varianceBias))
  const moments = (params: SvfmParams, t: Scalar, z: Value, c: Value | null) => {
    const raw = stackedMlp(params.fields, inputsOf(z, tt(t), c), activation)
    const [k, b] = shapeOfValue(z)
    if (!stochastic) return { mean: raw, sd: zeros([k, b]) }
    const heads = svfHeads(raw, S, o.maxVariance)
    const { vector, length2 } = heads
    const tauU = o.learnVariance ? heads.tauU : fixedTau
    const tauV = o.learnVariance ? heads.tauV : fixedTau
    // E[L u] = μᵛ e^{τᵛ/2} μ⁽ᵘ⁾ and tr Cov[L u] = E[L²] − ‖E[L u]‖² for L = exp(ℓ), ℓ ~ N(log μᵛ, τᵛ), with u linearised
    // on the tangent plane (‖E[u]‖² ≈ 1 − (S − 1)τᵘ; D1 in the log).
    const trace = mul(mul(length2, exp(tauV)), add(sub(exp(tauV), 1), mul(S - 1, tauU)))
    return { mean: mul(vector, expandDims(exp(mul(0.5, tauV)), -1)), sd: sqrt(add(div(trace, S), 1e-10)) }
  }

  const prior = (params: SvfmParams, h: Value, c: Value | null): Value => {
    const B = shapeOfValue(h)[0]
    if (K === 1) return zeros([B, 1])
    const input = o.prior === 'constant' ? zeros([B, 1]) : inputsOf(h, 0, c)
    const logits = mul(PRIOR_RATE / o.temperature, stackedMlp(params.prior, input, o.piActivation))
    return logSoftmax(reshape(logits, [B, K]))
  }

  const eyeK = eye(K)
  const rho = Math.min(1 - 1e-9, Math.max(1e-9, o.stickiness))
  const fixedLogTransitions = fromData(
    Float64Array.from({ length: K * K }, (_, q) =>
      Math.log(Math.floor(q / K) === q % K ? rho : (1 - rho) / Math.max(1, K - 1)),
    ),
    [K, K],
  )
  const filter = (params: SvfmParams, t: Scalar, logWeights: Value, states: Value, spread: Value, c: Value | null) => {
    if (!filtering) return { logWeights, states, spread }
    const B = shapeOfValue(states)[1]
    const x = reshape(inputsOf(states, t, c), [K * B, inF])
    // log ψ_j(h | j): entry j of ψ at the state reached under component j → [K(j), B] (uniform ψ cancels: 0).
    const logPsi =
      o.emissions === 'uniform'
        ? zeros([K, B])
        : sum(
            mul(
              reshape(logSoftmax(stackedMlp(params.emission, x, o.piActivation)), [K, B, K]),
              reshape(eyeK, [K, 1, K]),
            ),
            -1,
          )
    // log Ψ_jk at the state reached under j: row j of Ψ → [K(j), B, K(k)] (fixed: ρ to stay, the rest shared).
    const logRow =
      o.transitions === 'fixed'
        ? reshape(fixedLogTransitions, [K, 1, K])
        : sum(
            mul(
              logSoftmax(reshape(stackedMlp(params.transition, x, o.piActivation), [K, B, K, K]), { axis: -1 }),
              reshape(eyeK, [K, 1, K, 1]),
            ),
            2,
          )
    // Joint log w_jk = log π_j + log ψ_j + log Ψ_jk (eq. 5 before the sum over j), [K(j), B, K(k)].
    const logPrev = expandDims(permute(logWeights, [1, 0]), -1)
    const logJoint = add(add(logPrev, expandDims(logPsi, -1)), logRow)
    const logUnnorm = logsumexp(logJoint, 0) // [B, K]
    const next = sub(logUnnorm, expandDims(logsumexp(logUnnorm, -1), -1))
    // Mixing weights ω_jk = P(j at tᵢ₋₁ | k at tᵢ), [B, K(k), K(j)], and the moment-matched states and spreads.
    const omega = exp(permute(sub(logJoint, expandDims(logUnnorm, 0)), [1, 2, 0]))
    const m = permute(states, [1, 0, 2]) // [B, K(j), S]
    const mixed = matmul(omega, m) // [B, K(k), S]
    const second = add(square(permute(spread, [1, 0])), div(sum(square(m), -1), S)) // [B, K(j)]
    const mixedSecond = reshape(matmul(omega, expandDims(second, -1)), [B, K])
    const variance = sub(mixedSecond, div(sum(square(mixed), -1), S))
    return {
      logWeights: next,
      states: permute(mixed, [1, 0, 2]),
      spread: permute(sqrt(add(maximum(variance, 0), 1e-12)), [1, 0]),
    }
  }

  const propagate = (params: SvfmParams, x: Value, c: Value | null, onSolve?: (info: OdeSolveInfo) => void) => {
    const h0 = lift(x)
    const B = shapeOfValue(h0)[0]
    const start = mul(expandDims(h0, 0), ones([K, 1, 1]))
    // The ODE state per component: [m (S) | r (1)] → [K, B, S + 1].
    const rhs = (t: Scalar, y: Value): Value => {
      const m = slice(y, null, null, [0, S])
      const { mean, sd } = moments(params, t, m, c)
      return concat([mean, expandDims(sd, -1)], -1)
    }
    const solverOptions = { method: o.method, stepSize: o.stepSize, rtol: o.rtol, atol: o.atol, onSolve }
    const split = (y: Value) => ({
      m: slice(y, null, null, [0, S]),
      r: reshape(slice(y, null, null, [S, S + 1]), [K, B]),
    })
    let logW = prior(params, h0, c)
    const out: Propagation = {
      times,
      states: [start],
      arrived: [start],
      spread: [zeros([K, B])],
      logWeights: [logW],
      fields: [moments(params, 0, start, c).mean],
    }
    const pack = (m: Value, r: Value) => concat([m, expandDims(r, -1)], -1)
    if (!filtering) {
      // One solve over the whole grid: π is constant, so nothing changes at the grid times.
      const ys = odeFlow((t, y) => rhs(t, y), times, solverOptions)(pack(start, zeros([K, B])), 0)
      for (let i = 1; i < ys.length; i++) {
        const { m, r } = split(ys[i])
        out.states.push(m)
        out.arrived.push(m)
        out.spread.push(r)
        out.logWeights.push(logW)
        out.fields.push(moments(params, times[i], m, c).mean)
      }
      return out
    }
    let y = pack(start, zeros([K, B]))
    for (let i = 1; i < times.length; i++) {
      y = odeFlow((t, z) => rhs(t, z), [times[i - 1], times[i]], solverOptions)(y, 0)[1]
      const { m, r } = split(y)
      out.arrived.push(m)
      out.fields.push(moments(params, times[i], m, c).mean)
      const f = filter(params, times[i], logW, m, r, c)
      logW = f.logWeights
      out.states.push(f.states)
      out.spread.push(f.spread)
      out.logWeights.push(logW)
      y = pack(f.states, f.spread)
    }
    return out
  }

  const output = (params: SvfmParams, p: Propagation, i: number) => {
    const m = slice(p.states[i], null, null, [0, D]) // [K, B, D]
    const B = shapeOfValue(m)[1]
    const noise = exp(mul(2, params.logNoise))
    const sd2 = add(square(p.spread[i]), noise) // [K, B]
    const logScale = mul(0.5, log(sd2))
    return {
      logWeights: p.logWeights[i],
      means: permute(m, [1, 0, 2]),
      logScales: mul(expandDims(permute(logScale, [1, 0]), -1), ones([B, K, D])),
    }
  }

  const classLogLikelihoods = (params: SvfmParams, p: Propagation): Value => {
    const { weight, bias } = params.readout
    if (!weight || !bias) throw new DomainError('svfm', 'svfm: no classifier readout (classes = 0)')
    const last = p.states.length - 1
    const m = permute(p.states[last], [1, 0, 2]) // [B, K, S]
    let logits = add(matmul(m, weight), bias) // [B, K, classes]
    if (stochastic) {
      // The probit approximation of E[softmax(w·h)] under h ~ N(m, r² I): logits / √(1 + π r² ‖w_c‖² / 8).
      const r2 = expandDims(square(permute(p.spread[last], [1, 0])), -1) // [B, K, 1]
      const w2 = sum(square(weight), 0) // [classes]
      logits = div(logits, sqrt(add(1, mul((Math.PI / 8) * 1, mul(r2, w2)))))
    }
    return logSoftmax(logits, { axis: -1 })
  }

  return {
    options: o,
    stateDim: S,
    init: (s) => {
      const p: SvfmParams = {
        fields: stackedMlpInit(child(s, 'fields'), K, mlpSizes(inF, hidden, layers, fieldOut), 0.5, o.sharedTrunk),
        prior:
          K === 1
            ? []
            : o.prior === 'constant'
              ? stackedMlpInit(child(s, 'prior'), 1, [1, K], 0)
              : stackedMlpInit(child(s, 'prior'), 1, mlpSizes(inF, o.piHidden, o.piLayers, K)),
        emission:
          filtering && o.emissions === 'learned'
            ? stackedMlpInit(child(s, 'emission'), 1, mlpSizes(inF, o.piHidden, o.piLayers, K), 0.1)
            : [],
        transition:
          filtering && o.transitions === 'learned'
            ? stackedMlpInit(child(s, 'transition'), 1, mlpSizes(inF, o.piHidden, o.piLayers, K * K), 0.1)
            : [],
        readout:
          classes > 0
            ? {
                weight: xavierUniform()(child(s, 'readout'), [S, classes], { fanIn: S, fanOut: classes }),
                bias: zeros([classes]),
              }
            : {},
        logNoise: fromData(Float64Array.of(Math.log(0.1)), []),
      }
      if (filtering && o.transitions === 'learned') {
        // Transitions start sticky: the logits of staying begin log(ρ(K − 1)/(1 − ρ)) above the others, so P(stay) = ρ.
        const last = p.transition[p.transition.length - 1]
        const b = Float64Array.from(toFlat(last.bias))
        for (let k = 0; k < K; k++) b[k * K + k] = Math.log((rho * Math.max(1, K - 1)) / (1 - rho))
        p.transition[p.transition.length - 1] = { weight: last.weight, bias: fromData(b, [1, 1, K * K]) }
      }
      if (stochastic) {
        // SVF variances start near zero (the τ heads' biases at varianceBias = −5: τ ≈ 0.007·maxVariance), so the
        // components first specialise as deterministic VFs would, and grow their variance only where targets spread.
        const last = p.fields[p.fields.length - 1]
        const b = Float64Array.from(toFlat(last.bias))
        for (let k = 0; k < K; k++) {
          b[k * fieldOut + S] = o.varianceBias
          b[k * fieldOut + S + 1] = o.varianceBias
        }
        p.fields[p.fields.length - 1] = { weight: last.weight, bias: fromData(b, [K, 1, fieldOut]) }
      }
      return p
    },
    lift,
    moments,
    prior,
    filter,
    propagate,
    output,
    classLogLikelihoods,
  }
}

/**
 * A value's entries as a `Float64Array`.
 *
 * @param v A number or a tensor (traced values are unwrapped).
 * @returns Its entries in row-major order, a copy.
 *
 * @example A matrix and a number, flattened
 * print(flatOf(tensor([[1, 2], [3, 4]])), flatOf(5))
 */
export const flatOf = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))
