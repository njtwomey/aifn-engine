/**
 * First-order update rules over parameter pytrees, in the style of optax (Babuschkin et al., 2020, "The DeepMind JAX
 * Ecosystem"): each rule is a pure pair `init(params) → state` and `update(grads, state, params) → { updates, state }`,
 * applied leaf by leaf, and `applyUpdates(params, updates)` adds the updates. The rules are the one definition of
 * SGD, momentum, Nesterov, AdaGrad, RMSProp and Adam / AdamW: the traceable algorithms of `aifn-compute/optim/first-order`
 * run them on one vector with a fixed objective, and a training loop runs them on a parameter tree with a new
 * minibatch loss at every step.
 *
 * Sources: Polyak (1964), "Some methods of speeding up the convergence of iteration methods"; Sutskever, Martens, Dahl
 * & Hinton (2013), "On the importance of initialization and momentum in deep learning"; Duchi, Hazan & Singer (2011),
 * "Adaptive subgradient methods"; Tieleman & Hinton (2012), Coursera lecture 6.5; Kingma & Ba (2015), "Adam";
 * Loshchilov & Hutter (2019), "Decoupled weight decay regularization"; Pascanu, Mikolov & Bengio (2013), "On the
 * difficulty of training recurrent neural networks" (gradient clipping).
 */

import {
  add,
  div,
  greater,
  isTraced,
  mul,
  neg,
  norm,
  sqrt,
  square,
  stack,
  toFlat,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { treeLeaves, treeMap, treeZip, zerosLike, type LeafValue, type Params } from 'aifn-compute/foundation/pytree'
import type { Scalar, Schedule, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A step size: a constant, or a schedule t ↦ η_t read at the rule's update count t = 0, 1, 2, … A constant may be a
 * traced value, so that the result of running a rule (e.g. `unrolled` over `gradientDescent`) can be differentiated
 * with respect to it: a learning-rate hypergradient.
 */
export type StepSize = Value | Schedule

/** The state of a named rule: the number of updates applied so far, and its running quantities as trees. */
export type RuleState = {
  /** Updates applied so far (the step t at which the next update is made). */
  readonly t: Size
  /**
   * Running quantities, each shaped like the parameters: `velocity` (momentum), `sumSquares` (AdaGrad), `meanSquare`
   * (RMSProp), `firstMoment` and `secondMoment` (Adam, before bias correction). Empty for plain SGD.
   */
  readonly slots: Readonly<Record<string, Params>>
}

/**
 * A gradient transformation (optax's `GradientTransformation`): `init(params)` makes the state and
 * `update(grads, state, params)` returns the updates, the deltas to add to the parameters (already scaled by −η), and
 * the next state. Both are pure. `params` is needed only by rules that read the parameters (weight decay).
 */
export interface UpdateRule<S = RuleState> {
  readonly name: string
  init(params: Params): S
  update(grads: Params, state: S, params?: Params): { updates: Params; state: S }
}

/** The step size at update t. */
export const stepSizeAt = (stepSize: StepSize, t: Size): Value =>
  typeof stepSize === 'function' ? stepSize(t) : stepSize

/** −η, staying a number for a number step size. */
const negated = (eta: Value): Value => (typeof eta === 'number' ? -eta : neg(eta))

/** Leaf arithmetic that keeps numbers as numbers and tensors as tensors. */
const leaf = (v: unknown) => v as LeafValue & Tensor

const needParams = (name: string, params: Params | undefined): Params => {
  if (params === undefined)
    throw new DomainError(name, `${name}: weight decay needs the parameters; pass them to update`)
  return params
}

/** params + updates, leaf by leaf. */
export function applyUpdates<P extends Params>(params: P, updates: Params): P {
  return treeZip([params, updates], ([p, u]) => add(leaf(p), leaf(u)))
}

/**
 * Rules applied in sequence, each transforming the updates of the one before (optax's `chain`), e.g.
 * `chainRules(clipByGlobalNorm(1), adamRule())`. The state is the list of the rules' states.
 */
export function chainRules(...rules: readonly UpdateRule<unknown>[]): UpdateRule<readonly unknown[]> {
  return {
    name: rules.map((r) => r.name).join('+'),
    init: (params) => rules.map((r) => r.init(params)),
    update: (grads, state, params) => {
      let updates = grads
      const next = rules.map((r, k) => {
        const out = r.update(updates, state[k], params)
        updates = out.updates
        return out.state
      })
      return { updates, state: next }
    },
  }
}

/** The Euclidean norm of all the leaves of a tree taken together. */
export function globalNorm(tree: Params): Scalar {
  // Scaled by the largest entry, so that gradients near 1e200 do not overflow the sum of squares to ∞ (which would
  // make `clipByGlobalNorm` scale them to 0).
  const values: number[] = []
  for (const { value } of treeLeaves(tree)) {
    if (typeof value === 'number') values.push(value)
    else for (const v of toFlat(value)) values.push(v)
  }
  let big = 0
  for (const v of values) big = Math.max(big, Math.abs(v))
  if (big === 0 || !Number.isFinite(big)) return big
  let s = 0
  for (const v of values) s += (v / big) ** 2
  return big * Math.sqrt(s)
}

/**
 * The global norm of a tree with traced leaves, by tensor primitives (each leaf's stable Euclidean norm, then the norm
 * of those), so that it differentiates.
 */
function tracedGlobalNorm(tree: Params): Value {
  const norms = treeLeaves(tree).map(({ value }) => norm(value as Value))
  return norm(stack(norms))
}

/**
 * Clips the gradient tree to a global norm of at most `maxNorm` (Pascanu et al., 2013): g ← g · min(1, c/‖g‖). The
 * state counts the updates. With traced gradients (a hypergradient through `unrolled`) the factor is computed with
 * tensor primitives and differentiated through, as optax's clip is.
 */
export function clipByGlobalNorm(maxNorm: Scalar): UpdateRule<RuleState> {
  return {
    name: 'clip-by-global-norm',
    init: () => ({ t: 0, slots: {} }),
    update: (grads, state) => {
      if (treeLeaves(grads).some(({ value }) => isTraced(value as Value))) {
        const total = tracedGlobalNorm(grads)
        // The unused branch's divisor is replaced, so it stays finite and so does its derivative.
        const over = greater(total, maxNorm)
        const scale = where(over, div(maxNorm, where(over, total, 1)), 1)
        return { updates: treeMap(grads, (g) => mul(leaf(g), scale)), state: { t: state.t + 1, slots: {} } }
      }
      const total = globalNorm(grads)
      const factor = total > maxNorm ? maxNorm / total : 1
      return {
        updates: factor === 1 ? grads : treeMap(grads, (g) => mul(leaf(g), factor)),
        state: { t: state.t + 1, slots: {} },
      }
    },
  }
}

/** Options of `sgdRule`. */
export type SgdRuleOptions = {
  /** Step size η, or a schedule. Default 0.01. */
  stepSize?: StepSize
  /** Momentum μ in [0, 1). Default 0 (plain SGD). */
  momentum?: Scalar
  /** Nesterov momentum in the form of Sutskever et al. (2013). Default false. */
  nesterov?: boolean
  /** L2 penalty λ added to the gradient as λθ. Default 0. */
  weightDecay?: Scalar
}

/**
 * Stochastic gradient descent with momentum, in PyTorch's form: g ← ∇ + λθ, v ← μv + g, and the update
 * −η(g + μv) with Nesterov momentum, else −ηv (Polyak, 1964; Sutskever et al., 2013). With μ = 0 the update is −ηg.
 * The velocity is `slots.velocity`.
 */
export function sgdRule(options: SgdRuleOptions = {}): UpdateRule {
  const { stepSize = 0.01, momentum = 0, nesterov = false, weightDecay = 0 } = options
  const name = momentum > 0 ? (nesterov ? 'nesterov' : 'momentum') : 'sgd'
  return {
    name,
    init: (params) => ({ t: 0, slots: momentum > 0 ? { velocity: zerosLike(params) } : {} }),
    update: (grads, state, params) => {
      const minusEta = negated(stepSizeAt(stepSize, state.t))
      const g =
        weightDecay === 0
          ? grads
          : treeZip([grads, needParams(name, params)], ([gi, p]) => add(leaf(gi), mul(weightDecay, leaf(p))))
      if (momentum === 0)
        return { updates: treeMap(g, (gi) => mul(minusEta, leaf(gi))), state: { t: state.t + 1, slots: {} } }
      const velocity = treeZip([state.slots.velocity, g], ([v, gi]) => add(mul(momentum, leaf(v)), leaf(gi)))
      const direction = nesterov ? treeZip([g, velocity], ([gi, v]) => add(leaf(gi), mul(momentum, leaf(v)))) : velocity
      return {
        updates: treeMap(direction, (d) => mul(minusEta, leaf(d))),
        state: { t: state.t + 1, slots: { velocity } },
      }
    },
  }
}

/** Options of the adaptive rules. */
export type AdaptiveRuleOptions = {
  /** Step size η, or a schedule. */
  stepSize?: StepSize
  /** Added to the root in the denominator to avoid division by zero. Default 1e-8. */
  epsilon?: Scalar
}

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): G ← G + g², update −ηg / (√G + ε), elementwise. The accumulated squares
 * are `slots.sumSquares`. Default η = 0.1.
 */
export function adagradRule(options: AdaptiveRuleOptions = {}): UpdateRule {
  const { stepSize = 0.1, epsilon = 1e-8 } = options
  return {
    name: 'adagrad',
    init: (params) => ({ t: 0, slots: { sumSquares: zerosLike(params) } }),
    update: (grads, state) => {
      const minusEta = negated(stepSizeAt(stepSize, state.t))
      const sumSquares = treeZip([state.slots.sumSquares, grads], ([G, g]) => add(leaf(G), square(leaf(g))))
      const updates = treeZip([grads, sumSquares], ([g, G]) => div(mul(minusEta, leaf(g)), add(sqrt(leaf(G)), epsilon)))
      return { updates, state: { t: state.t + 1, slots: { sumSquares } } }
    },
  }
}

/** Options of `rmspropRule`. */
export type RmspropRuleOptions = AdaptiveRuleOptions & {
  /** Decay ρ of the mean of squared gradients. Default 0.9. */
  decay?: Scalar
}

/**
 * RMSProp (Tieleman & Hinton, 2012): s ← ρs + (1 − ρ)g², update −ηg / (√s + ε). The mean square is
 * `slots.meanSquare`. Default η = 0.01, ρ = 0.9.
 */
export function rmspropRule(options: RmspropRuleOptions = {}): UpdateRule {
  const { stepSize = 0.01, epsilon = 1e-8, decay = 0.9 } = options
  return {
    name: 'rmsprop',
    init: (params) => ({ t: 0, slots: { meanSquare: zerosLike(params) } }),
    update: (grads, state) => {
      const minusEta = negated(stepSizeAt(stepSize, state.t))
      const meanSquare = treeZip([state.slots.meanSquare, grads], ([s, g]) =>
        add(mul(decay, leaf(s)), mul(1 - decay, square(leaf(g)))),
      )
      const updates = treeZip([grads, meanSquare], ([g, s]) => div(mul(minusEta, leaf(g)), add(sqrt(leaf(s)), epsilon)))
      return { updates, state: { t: state.t + 1, slots: { meanSquare } } }
    },
  }
}

/** Options of `adamRule` and `adamwRule`. */
export type AdamRuleOptions = AdaptiveRuleOptions & {
  /** Decay of the first-moment estimate. Default 0.9. */
  beta1?: Scalar
  /** Decay of the second-moment estimate. Default 0.999. */
  beta2?: Scalar
  /** Weight decay λ. Default 0 for `adamRule`, 0.01 for `adamwRule`. */
  weightDecay?: Scalar
  /**
   * Decoupled weight decay (AdamW): the update is −η(m̂/(√v̂ + ε) + λθ). When false, λθ is added to the gradient
   * before the moment updates (L2 regularisation as Adam implements it). Default false for `adamRule`, true for
   * `adamwRule`.
   */
  decoupled?: boolean
}

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², m̂ = m/(1 − β₁^{t+1}),
 * v̂ = v/(1 − β₂^{t+1}), update −ηm̂/(√v̂ + ε). The raw moments are `slots.firstMoment` and `slots.secondMoment`.
 * Default η = 0.001. With `weightDecay` and `decoupled` it is AdamW (Loshchilov & Hutter, 2019).
 */
export function adamRule(options: AdamRuleOptions = {}): UpdateRule {
  const { stepSize = 1e-3, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.999, weightDecay = 0, decoupled = false } = options
  const name = decoupled ? 'adamw' : 'adam'
  return {
    name,
    init: (params) => ({ t: 0, slots: { firstMoment: zerosLike(params), secondMoment: zerosLike(params) } }),
    update: (grads, state, params) => {
      const minusEta = negated(stepSizeAt(stepSize, state.t))
      const decay = weightDecay !== 0
      const g =
        decay && !decoupled
          ? treeZip([grads, needParams(name, params)], ([gi, p]) => add(leaf(gi), mul(weightDecay, leaf(p))))
          : grads
      const m = treeZip([state.slots.firstMoment, g], ([mi, gi]) => add(mul(beta1, leaf(mi)), mul(1 - beta1, leaf(gi))))
      const v = treeZip([state.slots.secondMoment, g], ([vi, gi]) =>
        add(mul(beta2, leaf(vi)), mul(1 - beta2, square(leaf(gi)))),
      )
      const c1 = 1 - beta1 ** (state.t + 1)
      const c2 = 1 - beta2 ** (state.t + 1)
      const step = treeZip([m, v], ([mi, vi]) => div(div(leaf(mi), c1), add(sqrt(div(leaf(vi), c2)), epsilon)))
      const updates =
        decay && decoupled
          ? treeZip([step, needParams(name, params)], ([s, p]) =>
              mul(minusEta, add(leaf(s), mul(weightDecay, leaf(p)))),
            )
          : treeMap(step, (s) => mul(minusEta, leaf(s)))
      return { updates, state: { t: state.t + 1, slots: { firstMoment: m, secondMoment: v } } }
    },
  }
}

/** AdamW (Loshchilov & Hutter, 2019): `adamRule` with decoupled weight decay, default λ = 0.01. */
export function adamwRule(options: AdamRuleOptions = {}): UpdateRule {
  return adamRule({ weightDecay: 0.01, ...options, decoupled: options.decoupled ?? true })
}
