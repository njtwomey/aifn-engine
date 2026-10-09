/**
 * First-order update rules over parameter pytrees, in the style of optax (Babuschkin et al., 2020, "The DeepMind JAX
 * Ecosystem"): each rule is a pure pair `init(params) → state` and `update(grads, state, params) → { updates, state }`,
 * applied leaf by leaf, and `applyUpdates(params, updates)` adds the updates. The rules are the one definition of SGD,
 * momentum, Nesterov, AdaGrad, RMSProp and Adam / AdamW: the traceable algorithms of `aifn-compute/optim/first-order`
 * run them on one vector with a fixed objective, and a training loop runs them on a parameter tree with a new minibatch
 * loss at every step.
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
 * A step size: a constant, or a schedule $t \mapsto \eta_t$ read at the rule's update count $t = 0, 1, 2, \dots$ A
 * constant may be a traced value, so that the result of running a rule (e.g. `unrolled` over `gradientDescent`) can be
 * differentiated with respect to it: a learning-rate hypergradient.
 */
export type StepSize = Value | Schedule

/** The state of a named rule: the number of updates applied so far, and its running quantities as trees. */
export type RuleState = {
  /** Updates applied so far (the step $t$ at which the next update is made). */
  readonly t: Size
  /**
   * Running quantities, each shaped like the parameters: `velocity` (momentum), `sumSquares` (AdaGrad), `meanSquare`
   * (RMSProp), `firstMoment` and `secondMoment` (Adam, before bias correction). Empty for plain SGD.
   */
  readonly slots: Readonly<Record<string, Params>>
}

/**
 * A gradient transformation (optax's `GradientTransformation`): `init(params)` makes the state and
 * `update(grads, state, params)` returns the updates, the deltas to add to the parameters (already scaled by
 * $-\eta$), and the next state. Both are pure. `params` is needed only by rules that read the parameters (weight
 * decay).
 */
export interface UpdateRule<S = RuleState> {
  /** The rule's name (`'sgd'`, `'momentum'`, `'adam'`, …; chained rules join theirs with `+`). */
  readonly name: string
  /** The initial state for parameters shaped like `params` (their values are not read). */
  init(params: Params): S
  /** The updates for the gradient tree `grads` (shaped like the parameters), and the state after them. */
  update(grads: Params, state: S, params?: Params): { updates: Params; state: S }
}

/**
 * The step size at update $t$: the constant itself, or the schedule read at $t$.
 *
 * @param stepSize A constant step size (a number or a traced value) or a schedule.
 * @param t The update count, from 0.
 * @returns The step size $\eta_t$.
 *
 * @example A constant and a schedule
 * print('constant at t = 5:', stepSizeAt(0.1, 5))
 * print('0.1 / (1 + t) at t = 4:', stepSizeAt((t) => 0.1 / (1 + t), 4))
 */
export const stepSizeAt = (stepSize: StepSize, t: Size): Value =>
  typeof stepSize === 'function' ? stepSize(t) : stepSize

/**
 * $-\eta$, staying a number for a number step size.
 *
 * @param eta The step size, a number or a traced value.
 * @returns Its negation, a number for a number and a traced value for a traced one.
 */
const negated = (eta: Value): Value => (typeof eta === 'number' ? -eta : neg(eta))

/**
 * Leaf arithmetic that keeps numbers as numbers and tensors as tensors: a type cast only, so a leaf can be passed to
 * the tensor primitives.
 *
 * @param v A leaf of a parameter tree.
 * @returns The same leaf, typed for the primitives.
 */
const leaf = (v: unknown) => v as LeafValue & Tensor

/**
 * The parameters a weight-decay rule needs, or a `DomainError` when the caller did not pass them to `update`.
 *
 * @param name The rule's name, for the error message.
 * @param params The parameters passed to `update`, possibly undefined.
 * @returns `params`, when given.
 */
const needParams = (name: string, params: Params | undefined): Params => {
  if (params === undefined)
    throw new DomainError(name, `${name}: weight decay needs the parameters; pass them to update`)
  return params
}

/**
 * `params` + `updates`, leaf by leaf (optax's `apply_updates`).
 *
 * @param params The parameter tree; not modified.
 * @param updates The updates a rule returned, shaped like `params`.
 * @returns A new tree with the updated parameters.
 *
 * @example One SGD step on a small parameter tree
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0.2, -0.4]), b: 1 }
 * const rule = sgdRule({ stepSize: 0.1 })
 * const { updates } = rule.update(grads, rule.init(params), params)
 * print('updates =', updates)
 * print('new params =', applyUpdates(params, updates))
 */
export function applyUpdates<P extends Params>(params: P, updates: Params): P {
  return treeZip([params, updates], ([p, u]) => add(leaf(p), leaf(u)))
}

/**
 * Rules applied in sequence, each transforming the updates of the one before (optax's `chain`), e.g.
 * `chainRules(clipByGlobalNorm(1), adamRule())`. The state is the list of the rules' states.
 *
 * @param rules The rules, in the order they are applied; the first receives the gradients. Each is given the same
 *   `params`.
 * @returns One rule whose name joins the rules' names with `+`.
 *
 * @example Clip, then take an SGD step
 * // The gradient has global norm 50; clipping scales it to norm 1 before the step of size 0.1.
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([30, 0]), b: 40 }
 * const rule = chainRules(clipByGlobalNorm(1), sgdRule({ stepSize: 0.1 }))
 * const { updates, state } = rule.update(grads, rule.init(params), params)
 * print('rule =', rule.name)
 * print('updates =', updates)
 * print('update counts =', state.map((s) => s.t))
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

/**
 * The Euclidean norm of all the leaves of a tree taken together, $\sqrt{\sum_i g_i^2}$ over every entry of every
 * leaf. Computed on plain values (not differentiable), scaled by the largest entry so that it does not overflow.
 *
 * @param tree A tree of numbers and tensors, such as a gradient tree.
 * @returns The norm; 0 for an all-zero tree, and the largest magnitude itself when that is not finite.
 *
 * @example The norm of a tree, and one that would overflow a sum of squares
 * print('norm of { w: [3, 0], b: 4 } =', globalNorm({ w: tensor([3, 0]), b: 4 }))
 * print('norm of [3e200, 4e200] =', globalNorm({ w: tensor([3e200, 4e200]) }))
 */
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
 *
 * @param tree A tree whose leaves may be traced.
 * @returns The global norm, a traced value.
 */
function tracedGlobalNorm(tree: Params): Value {
  const norms = treeLeaves(tree).map(({ value }) => norm(value as Value))
  return norm(stack(norms))
}

/**
 * Clips the gradient tree to a global norm of at most `maxNorm` (Pascanu et al., 2013):
 * $\gvec \leftarrow \gvec \cdot \min(1, c/\lVert \gvec \rVert)$. The state counts the updates. With traced gradients
 * (a hypergradient through `unrolled`) the factor is computed with tensor primitives and differentiated through, as
 * optax's clip is. It takes no step itself: chain it before a rule with `chainRules`.
 *
 * @param maxNorm The largest global norm $c$ the clipped tree may have.
 * @returns A rule whose updates are the gradients, scaled down when their global norm exceeds $c$.
 *
 * @example A large gradient is scaled down, a small one passes unchanged
 * const clip = clipByGlobalNorm(1)
 * const big = clip.update({ w: tensor([3, 0]), b: 4 }, clip.init({})).updates
 * print('clipped =', big, ' norm =', globalNorm(big))
 * print('small =', clip.update({ w: tensor([0.3, 0]), b: 0.4 }, clip.init({})).updates)
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
  /** Step size $\eta$, or a schedule. Default 0.01. */
  stepSize?: StepSize
  /** Momentum $\mu$ in $[0, 1)$. Default 0 (plain SGD). */
  momentum?: Scalar
  /** Nesterov momentum in the form of Sutskever et al. (2013); used only when `momentum` > 0. Default false. */
  nesterov?: boolean
  /** L2 penalty $\lambda$ added to the gradient as $\lambda\thetavec$. Default 0. */
  weightDecay?: Scalar
}

/**
 * Stochastic gradient descent with momentum, in PyTorch's form: $\gvec \leftarrow \nabla + \lambda\thetavec$,
 * $\vvec \leftarrow \mu\vvec + \gvec$, and the update $-\eta(\gvec + \mu\vvec)$ with Nesterov momentum, else
 * $-\eta\vvec$ (Polyak, 1964; Sutskever et al., 2013). With $\mu = 0$ the update is $-\eta\gvec$. The velocity is
 * `slots.velocity`. With weight decay, `update` must be given the parameters $\thetavec$ (else `DomainError`).
 *
 * @param options The step size $\eta$ or schedule, the momentum $\mu$, Nesterov or not, and the weight decay
 *   $\lambda$.
 * @returns The rule, named `'sgd'`, `'momentum'` or `'nesterov'`.
 *
 * @example The velocity builds up over three updates with the same gradient
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0.2, -0.4]), b: 1 }
 * const rule = sgdRule({ stepSize: 0.1, momentum: 0.9 })
 * let state = rule.init(params)
 * for (let k = 0; k < 3; k++) {
 *   const out = rule.update(grads, state, params)
 *   state = out.state
 *   print(`update ${k}: b changes by`, out.updates.b, ' velocity of b:', state.slots.velocity.b)
 * }
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
  /** Step size $\eta$, or a schedule. The default depends on the rule. */
  stepSize?: StepSize
  /** $\varepsilon$, added to the root in the denominator to avoid division by zero. Default 1e-8. */
  epsilon?: Scalar
}

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): $\mathbf{G} \leftarrow \mathbf{G} + \gvec^2$, update
 * $-\eta\gvec / (\sqrt{\mathbf{G}} + \varepsilon)$, elementwise. The accumulated squares are `slots.sumSquares`.
 * Default $\eta = 0.1$.
 *
 * @param options The step size $\eta$ or schedule, and $\varepsilon$.
 * @returns The rule, named `'adagrad'`; it does not read the parameters.
 *
 * @example The steps shrink as the squares accumulate
 * // The same gradient three times: the updates are η/√1, η/√2, η/√3 in size.
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0.2, -0.4]), b: 1 }
 * const rule = adagradRule({ stepSize: 0.1 })
 * let state = rule.init(params)
 * for (let k = 0; k < 3; k++) {
 *   const out = rule.update(grads, state)
 *   state = out.state
 *   print(`update ${k}:`, out.updates)
 * }
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
  /** Decay $\rho$ of the mean of squared gradients. Default 0.9. */
  decay?: Scalar
}

/**
 * RMSProp (Tieleman & Hinton, 2012): $\svec \leftarrow \rho\svec + (1 - \rho)\gvec^2$, update
 * $-\eta\gvec / (\sqrt{\svec} + \varepsilon)$, elementwise. The mean square is `slots.meanSquare`. Default
 * $\eta = 0.01$, $\rho = 0.9$.
 *
 * @param options The step size $\eta$ or schedule, $\varepsilon$ and the decay $\rho$.
 * @returns The rule, named `'rmsprop'`; it does not read the parameters.
 *
 * @example The mean square starts at zero, so the first steps are large
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0.2, -0.4]), b: 1 }
 * const rule = rmspropRule({ stepSize: 0.01 })
 * let state = rule.init(params)
 * for (let k = 0; k < 3; k++) {
 *   const out = rule.update(grads, state)
 *   state = out.state
 *   print(`update ${k}: b changes by`, out.updates.b, ' mean square of b:', state.slots.meanSquare.b)
 * }
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
  /** Decay $\beta_1$ of the first-moment estimate. Default 0.9. */
  beta1?: Scalar
  /** Decay $\beta_2$ of the second-moment estimate. Default 0.999. */
  beta2?: Scalar
  /** Weight decay $\lambda$. Default 0 for `adamRule`, 0.01 for `adamwRule`. */
  weightDecay?: Scalar
  /**
   * Decoupled weight decay (AdamW): the update is
   * $-\eta(\hat{\mvec}/(\sqrt{\hat{\vvec}} + \varepsilon) + \lambda\thetavec)$. When false, $\lambda\thetavec$ is
   * added to the gradient before the moment updates (L2 regularisation as Adam implements it). Default false for
   * `adamRule`, true for `adamwRule`.
   */
  decoupled?: boolean
}

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): $\mvec \leftarrow \beta_1\mvec + (1 - \beta_1)\gvec$,
 * $\vvec \leftarrow \beta_2\vvec + (1 - \beta_2)\gvec^2$, $\hat{\mvec} = \mvec/(1 - \beta_1^{t+1})$,
 * $\hat{\vvec} = \vvec/(1 - \beta_2^{t+1})$, update $-\eta\hat{\mvec}/(\sqrt{\hat{\vvec}} + \varepsilon)$, with
 * $t$ the updates applied before this one. The raw moments are `slots.firstMoment` and `slots.secondMoment`. Default
 * $\eta = 0.001$. With `weightDecay` and `decoupled` it is AdamW (Loshchilov & Hutter, 2019). With weight decay,
 * `update` must be given the parameters $\thetavec$ (else `DomainError`).
 *
 * @param options The step size $\eta$ or schedule, $\beta_1$, $\beta_2$, $\varepsilon$, the weight decay $\lambda$
 *   and whether it is decoupled.
 * @returns The rule, named `'adam'`, or `'adamw'` when the decay is decoupled.
 *
 * @example The first update has the size of the step whatever the gradient
 * // After bias correction m̂ = g and v̂ = g², so the step is η g / |g| whatever the gradient's size.
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0.2, -0.4]), b: 1 }
 * const rule = adamRule({ stepSize: 0.1 })
 * const { updates, state } = rule.update(grads, rule.init(params), params)
 * print('updates =', updates)
 * print('raw first moment =', state.slots.firstMoment)
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

/**
 * AdamW (Loshchilov & Hutter, 2019): `adamRule` with decoupled weight decay, default $\lambda = 0.01$. The decay
 * reads the parameters, so `update` must be given them.
 *
 * @param options As for `adamRule`; `weightDecay` defaults to 0.01 and `decoupled` to true.
 * @returns The rule, named `'adamw'` (or `'adam'` if `decoupled` is set to false).
 *
 * @example With a zero gradient only the decay moves the parameters
 * // The update is −ηλθ = −0.1 · 0.01 · θ.
 * const params = { w: tensor([1, -2]), b: 0.5 }
 * const grads = { w: tensor([0, 0]), b: 0 }
 * const rule = adamwRule({ stepSize: 0.1 })
 * const { updates } = rule.update(grads, rule.init(params), params)
 * print('updates =', updates)
 */
export function adamwRule(options: AdamRuleOptions = {}): UpdateRule {
  return adamRule({ weightDecay: 0.01, ...options, decoupled: options.decoupled ?? true })
}
