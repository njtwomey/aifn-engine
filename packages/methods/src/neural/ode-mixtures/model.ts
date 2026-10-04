/**
 * Stochastic vector field mixtures (SVFM; Twomey, Kozłowski & Santos-Rodríguez, 2020, "Neural ODEs with stochastic
 * vector field mixtures", ECAI): a neural ODE ∇h(t) = f(h(t), t; θ) (eq. 1) whose vector field (VF) is one of K
 * components f⁽ᵏ⁾, with component membership π(t) ∈ △ᴷ, and whose components may be stochastic.
 *
 * - **VF unit** (fig. 2a): f⁽ᵏ⁾(h, t) is an MLP.
 * - **SVF unit** (fig. 3, eqs. 6–7): a shared representation f_z = g_z(h, t) gives the mean and variance of the VF's
 *   direction u ~ N_s(μ⁽ᵘ⁾, τ⁽ᵘ⁾) on the sphere and of its log length, log v ~ N(log μ⁽ᵛ⁾, τ⁽ᵛ⁾); a sample of the VF is
 *   ∇h = v·u. The network outputs a VF a, decomposed into its orientation μ⁽ᵘ⁾ = a/‖a‖ and length μ⁽ᵛ⁾ = ‖a‖, and the
 *   two variances. N_s is read as a projected normal: u = normalise(μ⁽ᵘ⁾ + √τ⁽ᵘ⁾ P ε) with P = I − μ⁽ᵘ⁾μ⁽ᵘ⁾ᵀ, so
 *   u·μ⁽ᵘ⁾ > 0 always (the paper's directional preservation).
 * - **Component selection** on a grid t₀ < t₁ < … < t_T: *pick and stick* holds π(tᵢ) = π(t₀) = f_{π_t0}(h(t₀), t₀)
 *   (eq. 2); *forward filtering* (eqs. 3–5) has emission ψ(tᵢ) = f_ψ(h(tᵢ), tᵢ) ∈ △ᴷ and transition Ψ(tᵢ) =
 *   f_Ψ(h(tᵢ), tᵢ) with rows in △ᴷ, and π(tᵢ) ∝ Ψ(tᵢ)ᵀ(ψ(tᵢ) ⊙ π(tᵢ₋₁)).
 *
 * Two computations share the parameters. `propagate` carries, for each component k, the state reached under it and
 * the spread of its stochastic VF's samples over the interval (moments: m_k′ = E[∇h⁽ᵏ⁾], r_k′ = the VF's standard
 * deviation per coordinate, which adds linearly because a sample keeps its noise for the whole solve), so the output at
 * tᵢ is the mixture Σₖ πₖ(tᵢ) N(m_k(tᵢ), (r_k² + σ₀²) I) of eq. 10. Under forward filtering, ψ_j and row j of Ψ are
 * evaluated on the state reached under component j, and the component-conditioned states are re-mixed with the same
 * joint weights (moment matching). `realisedField` evaluates one sampled path's VF (§3: a frozen uniform picks the
 * component from π, frozen normals sample the SVF), for trajectories and the per-instance work of a solve.
 *
 * Augmentation (fig. 6, the A- models; Dupont, Doucet & Teh, 2019) pads h with zeros. A context c (e.g. the time of
 * day as (cos, sin)) is appended to the input of every network.
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

/** How the component membership π(t) is specified over the integration interval (§2.1). */
export type ComponentSelection = 'pick-and-stick' | 'forward-filtering'

/** Field activations; the paper uses rectified units (§4.1.3). */
export type FieldActivation = 'relu' | 'tanh' | 'softplus'

/** Options of {@link svfm}; plain data. */
export type SvfmOptions = {
  /** Dimension D of the data h(t₀) = x. */
  dim: number
  /** Components K of the mixture (1: a single VF). Default 1. */
  components?: number
  /** SVF units (fig. 3) rather than VF units (fig. 2a). Default false. */
  stochastic?: boolean
  /** Default `'pick-and-stick'`. */
  selection?: ComponentSelection
  /** Zero-padded extra state dimensions (the A- models of fig. 6). Default 0. */
  augment?: number
  /** Context inputs appended to every network's input (e.g. the time-of-day tuple). Default 0. */
  context?: number
  /** Hidden units per layer (the paper: 32 or 64). Default 32. */
  hidden?: number
  /** Hidden layers (the paper: 1 or 2). Default 2. */
  layers?: number
  /** Default `'relu'`. */
  activation?: FieldActivation
  /** Classes of a classifier readout (a linear map of h(t_T) to logits), or 0 to read the state. Default 0. */
  classes?: number
  /** Grid intervals T on [0, 1]: π is updated and the trajectory losses sum at t₁ … t_T. Default 10. */
  grid?: number
  /** The differentiable solver within each interval. Default RK4. */
  method?: OdeFlowMethod
  /** Step of a fixed-step solver (≤ the grid spacing). Default 0.05. */
  stepSize?: number
  /** Tolerances of Dormand–Prince. */
  rtol?: number
  atol?: number
  /** The largest variance τ⁽ᵘ⁾, τ⁽ᵛ⁾ of an SVF (τ = maxVariance·sigmoid(·)). Default 0.5. */
  maxVariance?: number
  /** Initial bias of the SVF variance heads (default −5: τ starts near 0). */
  varianceBias?: number
  /** Learn τ⁽ᵘ⁾, τ⁽ᵛ⁾ from the state, or hold them at maxVariance·sigmoid(varianceBias). Default true. */
  learnVariance?: boolean
  /** The component VFs share their hidden layers (a trunk) and differ in their output layers. Default false. */
  sharedTrunk?: boolean
  /** The VFs take t as an input, f(h(t), t) (eq. 1), or are autonomous, f(h(t)). Default true. */
  timeDependent?: boolean
  /** π(t₀) (eq. 2): a function f_{π_t0} of (h(t₀), t₀) (`'state'`) or a learned constant. Default `'state'`. */
  prior?: 'state' | 'constant'
  /**
   * Hidden units, hidden layers (0: linear, a softmax of an affine map) and activation of the π networks f_{π_t0}, f_ψ
   * and f_Ψ. π only chooses which field; the fields carry the dynamics, so π stays simple: defaults linear (0 layers),
   * 8 units when a layer is used.
   */
  piHidden?: number
  piLayers?: number
  piActivation?: FieldActivation
  /** The softmax temperature of π(t₀). Default 1. */
  temperature?: number
  /** Forward filtering: emissions ψ learned (eq. 4) or uniform. Default learned. */
  emissions?: 'learned' | 'uniform'
  /** Forward filtering: transitions Ψ learned (eq. 3) or fixed. Default learned. */
  transitions?: 'learned' | 'fixed'
  /** The probability of staying in a component: fixed transitions keep it; learned ones start at it. Default 0.9. */
  stickiness?: number
}

/** A stack of K MLPs evaluated together: weights [K, in, out] and biases [K, 1, out] per layer. */
export type StackedMlpParams = { weight: Tensor; bias: Tensor }[]

/** Parameters of an SVFM. Networks absent from a configuration are empty. */
export type SvfmParams = {
  /** The K component networks (VF: out S; SVF: out S + 2, the mean VF and the two variances). */
  fields: StackedMlpParams
  /** f_{π_t0} (eq. 2): logits over components at (h(t₀), t₀); empty when K = 1. */
  prior: StackedMlpParams
  /** f_ψ (eq. 4) and f_Ψ (eq. 3): emission logits [K] and transition logits [K × K]; empty unless forward filtering. */
  emission: StackedMlpParams
  transition: StackedMlpParams
  /** Classifier readout [S, classes] and [classes]; empty for state outputs. */
  readout: { weight?: Tensor; bias?: Tensor }
  /** log σ₀, the output noise floor of the mixture density (eq. 10). */
  logNoise: Tensor
}

/** The moments `propagate` carries, at every grid time t₀ … t_T. */
export type Propagation = {
  /** Grid times t₀ … t_T. */
  times: number[]
  /** Component-conditioned states m_k(tᵢ) [K, B, S] after the update at tᵢ (what interval i + 1 starts from). */
  states: Value[]
  /** The same before the update at tᵢ (where interval i ended; equal to `states` under pick and stick). */
  arrived: Value[]
  /** Spread r_k(tᵢ) [K, B] after the update. */
  spread: Value[]
  /** log π(tᵢ) [B, K]. */
  logWeights: Value[]
  /** The mean VF E[∇h⁽ᵏ⁾] at the arrived states, for i = 1 … T ([K, B, S]); index 0 holds the field at t₀. */
  fields: Value[]
}

/** An SVFM, ready to train. */
export type Svfm = {
  options: Required<SvfmOptions>
  /** State dimension S = D + augment. */
  stateDim: number
  init(s: Stream): SvfmParams
  /** h(t₀): x [B, D] padded with zeros to [B, S]. */
  lift(x: Value): Value
  /** The mean VF and its spread rate per component at time t on states z [K, B, S]: [K, B, S] and [K, B]. */
  moments(params: SvfmParams, t: Scalar, z: Value, c: Value | null): { mean: Value; sd: Value }
  /** log π(t₀) [B, K] (eq. 2) at h(t₀) [B, S]. */
  prior(params: SvfmParams, h: Value, c: Value | null): Value
  /**
   * Forward filtering at tᵢ (eq. 5) on component-conditioned states [K, B, S] with spreads [K, B]: the new log π
   * [B, K] and the re-mixed states and spreads.
   */
  filter(
    params: SvfmParams,
    t: Scalar,
    logWeights: Value,
    states: Value,
    spread: Value,
    c: Value | null,
  ): { logWeights: Value; states: Value; spread: Value }
  /** The moments over the grid from x [B, D] (see the module comment). */
  propagate(params: SvfmParams, x: Value, c: Value | null, onSolve?: (info: OdeSolveInfo) => void): Propagation
  /** Mixture output at a grid time over the first D coordinates: log π [B, K], means [B, K, D], log σ [B, K, D]. */
  output(params: SvfmParams, p: Propagation, i: number): { logWeights: Value; means: Value; logScales: Value }
  /** Classifier log-likelihoods log p(y = c | component k) at t_T: [B, K, classes] (probit-scaled for SVF units). */
  classLogLikelihoods(params: SvfmParams, p: Propagation): Value
}

/**
 * The prior's logits are its network's output times this factor. Under Adam the factor slows the prior's learning by
 * the same amount, so the components specialise (each target picks the component that serves it best, as in EM)
 * before π(t₀) commits; a prior that saturates first gates a component off everywhere, and it never recovers.
 */
const PRIOR_RATE = 0.2

const mlpSizes = (inF: number, hidden: number, layers: number, out: number) => [
  inF,
  ...Array.from({ length: layers }, () => hidden),
  out,
]

/** Initialise K stacked MLPs of the given sizes (He-uniform hidden layers, Xavier-uniform output, zero biases). */
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

const activate = (a: FieldActivation, x: Value): Value =>
  a === 'relu' ? relu(x) : a === 'tanh' ? tanh(x) : softplus(x)

/** K stacked MLPs on inputs [K, B, in] → [K, B, out]; a [B, in] input is shared by all K. */
export function stackedMlp(params: StackedMlpParams, x: Value, activation: FieldActivation): Value {
  let h = x
  params.forEach((layer, l) => {
    h = add(matmul(h, layer.weight), layer.bias)
    if (l < params.length - 1) h = activate(activation, h)
  })
  return h
}

/** The inputs [.., S + 1 + C] of a network: the state, the time and the context (broadcast over leading axes). */
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
 * The SVF unit's heads (eqs. 6–7) from its raw outputs [.., S + 2]: the VF a whose orientation μ⁽ᵘ⁾ = a/‖a‖ and length
 * μ⁽ᵛ⁾ = ‖a‖ are the means (the paper's decomposition of a VF into length and orientation), and the variances τ⁽ᵘ⁾,
 * τ⁽ᵛ⁾ = maxVariance·sigmoid(·). The length vanishes with a, so the VF stays continuous where its orientation is undefined.
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

/** A stochastic vector field mixture (see the module comment). */
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

/** A value's entries as a Float64Array. */
export const flatOf = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))
