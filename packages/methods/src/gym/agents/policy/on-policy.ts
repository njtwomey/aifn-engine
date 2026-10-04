/**
 * On-policy gradient agents for a box observation and discrete actions, on the gym protocol. Each holds a categorical
 * policy network π_θ(a|o) (logits from a multilayer perceptron) and a value network V_φ(o), both trained by Adam:
 *
 * - **REINFORCE with a baseline** (Williams, 1992): at the end of every episode, the returns Gₜ = Σ γᵏ rₜ₊ₖ weigh the
 *   score: the policy loss is −mean[log π(aₜ|oₜ) (Gₜ − V(oₜ))] with the advantages standardised, and V regresses on Gₜ.
 * - **Advantage actor–critic (A2C)** (Mnih et al., 2016, synchronous form): every n steps (or at an episode's end) the
 *   n-step returns Rₜ = rₜ + γ rₜ₊₁ + … + γⁿ V(oₜ₊ₙ), bootstrapped unless the episode terminated, give advantages
 *   Rₜ − V(oₜ); one step on −mean[log π · A] + c_v mean[(R − V)²] − β mean[entropy].
 * - **Proximal policy optimisation (PPO, clipped)** (Schulman et al., 2017): collect `horizon` steps across episodes,
 *   compute generalised advantage estimates (GAE(λ); Schulman et al., 2016) from the values recorded when the steps
 *   were taken, then run `epochs` passes of minibatch Adam on −mean[min(ρA, clip(ρ, 1 ± ε)A)] + c_v mean[(V − R)²]
 *   − β entropy, where ρ = π_θ(a|o)/π_old(a|o) and π_old is the policy that collected the steps.
 *
 * A truncated episode (a time limit) is not terminal: its last step bootstraps from V of the next observation. The
 * agents' states are plain data; minibatch orders are drawn from a seed fixed in `init`, so `learn` stays pure.
 */

import type { Agent, AgentInfo, EnvironmentShape, Transition } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { categorical, child, integers, permutation, stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { domainDimension, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import {
  clip,
  exp,
  fromData,
  mean,
  minimum,
  mul,
  square,
  sub,
  toFlat,
  unwrap,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { oneHot } from 'aifn-compute/learning/losses'
import type { ActivationName } from 'aifn-compute/nn/functional'
import { standardise as zScores } from 'aifn-compute/probability/stats'
import {
  actionProbabilities,
  adamWithClip,
  argmax,
  entropies,
  gradientStep,
  logProbabilities,
  networkCache,
  rows,
  scalarOf,
} from './shared'

/** Options shared by the on-policy agents. */
export interface OnPolicyOptions {
  /** Hidden widths of the policy and value networks (default [64, 64]). */
  hidden?: readonly number[]
  activation?: ActivationName
  /** Adam's step size for both networks (default 3e-4 for PPO, 1e-3 for A2C, 3e-3 for REINFORCE). */
  learningRate?: number
  /** The discount (default: the environment's). */
  gamma?: number
  /** Entropy bonus β (default 0 for PPO and REINFORCE, 0.01 for A2C). */
  entropy?: number
  /** Value-loss weight c_v (default 0.5). */
  valueCoef?: number
  /** Gradient clipping by global norm (default 0.5 for PPO and A2C, Infinity for REINFORCE). */
  clipNorm?: number
}

/** The stored steps of the current batch, as flat columns. */
interface Steps {
  obs: number[]
  actions: number[]
  rewards: number[]
  /** 1 when the step ended the episode by termination. */
  terminated: number[]
  /** 1 when the step ended the episode (terminated or truncated). */
  ends: number[]
  /** V(o) and V(o′) under the networks that collected the step (PPO and A2C), and log π_old(a|o) (PPO). */
  values: number[]
  nextValues: number[]
  logp: number[]
}

/** An on-policy agent's state. */
export interface OnPolicyState {
  seed: number
  dim: number
  actions: number
  gamma: number
  policy: Params[]
  value: Params[]
  policyOptimizer: unknown
  valueOptimizer: unknown
  steps: Steps
  /** Environment steps and gradient updates so far. */
  t: number
  updates: number
  /** Losses and diagnostics of the last update (NaN before the first). */
  last: { policyLoss: number; valueLoss: number; entropy: number; clipFraction: number; approxKl: number }
}

const emptySteps = (): Steps => ({
  obs: [],
  actions: [],
  rewards: [],
  terminated: [],
  ends: [],
  values: [],
  nextValues: [],
  logp: [],
})

const NAN_LAST = { policyLoss: NaN, valueLoss: NaN, entropy: NaN, clipFraction: NaN, approxKl: NaN }

type Kind = 'reinforce' | 'a2c' | 'ppo'

/** The protocol shared by the three agents; `update` decides when and how to learn from the stored steps. */
function onPolicyAgent(
  name: string,
  kind: Kind,
  options: OnPolicyOptions,
  defaults: { learningRate: number; entropy: number; clipNorm: number },
  update: (
    g: OnPolicyState,
    nets: { policy: ReturnType<ReturnType<typeof networkCache>>; value: ReturnType<ReturnType<typeof networkCache>> },
    rule: ReturnType<typeof adamWithClip>,
    coef: { entropy: number; valueCoef: number },
  ) => OnPolicyState,
  due: (g: OnPolicyState, t: Transition<Float64Array, number>) => boolean,
): Agent<OnPolicyState, Float64Array, number> {
  const { hidden = [64, 64], activation = 'tanh', valueCoef = 0.5 } = options
  const learningRate = options.learningRate ?? defaults.learningRate
  const entropy = options.entropy ?? defaults.entropy
  const clipNorm = options.clipNorm ?? defaults.clipNorm
  const net = networkCache(activation)
  const policyNet = (g: Pick<OnPolicyState, 'dim' | 'actions'>) => net([g.dim, ...hidden, g.actions])
  const valueNet = (g: Pick<OnPolicyState, 'dim'>) => net([g.dim, ...hidden, 1])
  const rule = adamWithClip(learningRate, clipNorm)
  const valueAt = (g: OnPolicyState, o: ArrayLike<number>) =>
    toFlat(unwrap(valueNet(g).apply(g.value, rows(o, 1, g.dim))) as Tensor)[0]
  return {
    name,
    init: (env: EnvironmentShape, s) => {
      if (env.observation.kind !== 'box') throw new TypeError(`${name}: ${env.name} needs a box observation`)
      if (env.action.kind !== 'discrete') throw new TypeError(`${name}: ${env.name} needs discrete actions`)
      const dim = domainDimension(env.observation)
      const actions = env.action.n
      const shape = { dim, actions }
      const policy = policyNet(shape).init(child(s, 'policy'))
      const value = valueNet(shape).init(child(s, 'value'))
      return {
        seed: integers(child(s, 'seed'), 2 ** 31),
        dim,
        actions,
        gamma: options.gamma ?? env.gamma,
        policy,
        value,
        policyOptimizer: rule.init(policy),
        valueOptimizer: rule.init(value),
        steps: emptySteps(),
        t: 0,
        updates: 0,
        last: NAN_LAST,
      }
    },
    act(g, obs, s) {
      const p = actionProbabilities(policyNet(g), g.policy, obs)
      return { action: categorical(child(s, 'action'), p), probabilities: p }
    },
    greedy: (g, obs) => argmax(actionProbabilities(policyNet(g), g.policy, obs)),
    learn(g, tr) {
      const steps = g.steps
      const end = tr.terminated || tr.truncated
      const next: Steps = {
        obs: [...steps.obs, ...tr.observation],
        actions: [...steps.actions, tr.action],
        rewards: [...steps.rewards, tr.reward],
        terminated: [...steps.terminated, tr.terminated ? 1 : 0],
        ends: [...steps.ends, end ? 1 : 0],
        values: kind === 'reinforce' ? steps.values : [...steps.values, valueAt(g, tr.observation)],
        nextValues:
          kind === 'reinforce' || tr.terminated ? [...steps.nextValues, 0] : [...steps.nextValues, valueAt(g, tr.next)],
        logp:
          kind === 'ppo'
            ? [...steps.logp, Math.log(actionProbabilities(policyNet(g), g.policy, tr.observation)[tr.action])]
            : steps.logp,
      }
      const state = { ...g, steps: next, t: g.t + 1 }
      if (!due(state, tr)) return state
      const updated = update(state, { policy: policyNet(g), value: valueNet(g) }, rule, { entropy, valueCoef })
      return { ...updated, steps: emptySteps() }
    },
    scalars: (g) => ({
      'policy loss': g.last.policyLoss,
      'value loss': g.last.valueLoss,
      entropy: g.last.entropy,
      ...(kind === 'ppo' ? { 'clip fraction': g.last.clipFraction, 'approx KL': g.last.approxKl } : {}),
    }),
  }
}

/** Policy and value losses on a batch with fixed advantages and value targets; returns both updated networks. */
function actorCriticStep(
  g: OnPolicyState,
  nets: { policy: ReturnType<ReturnType<typeof networkCache>>; value: ReturnType<ReturnType<typeof networkCache>> },
  rule: ReturnType<typeof adamWithClip>,
  batch: { obs: number[]; actions: number[]; advantages: Float64Array; targets: Float64Array; logpOld?: Float64Array },
  coef: { entropy: number; valueCoef: number; clip?: number },
): {
  state: OnPolicyState
  policyLoss: number
  valueLoss: number
  entropy: number
  clipFraction: number
  approxKl: number
} {
  const n = batch.actions.length
  const x = rows(batch.obs, n, g.dim)
  const mask = oneHot(batch.actions, g.actions)
  const A = fromData(batch.advantages, [n])
  let entropyValue = NaN
  const policyLoss = (p: Params[]) => {
    const logits = nets.policy.apply(p, x)
    const logp = logProbabilities(logits, mask)
    const ent = mean(entropies(logits))
    entropyValue = scalarOf(ent)
    let surrogate
    if (batch.logpOld && coef.clip !== undefined) {
      const ratio = exp(sub(logp, fromData(batch.logpOld, [n])))
      surrogate = mean(minimum(mul(ratio, A), mul(clip(ratio, 1 - coef.clip, 1 + coef.clip), A)))
    } else surrogate = mean(mul(logp, A))
    return sub(mul(-1, surrogate), mul(coef.entropy, ent))
  }
  const vLoss = (p: Params[]) => {
    const v = nets.value.apply(p, x)
    return mul(coef.valueCoef, mean(square(sub(v, fromData(batch.targets, [n, 1])))))
  }
  const pStep = gradientStep(rule, g.policy, g.policyOptimizer, policyLoss)
  const vStep = gradientStep(rule, g.value, g.valueOptimizer, vLoss)
  let clipFraction = NaN
  let approxKl = NaN
  if (batch.logpOld && coef.clip !== undefined) {
    const logits = nets.policy.apply(g.policy, x)
    const logp = toFlat(unwrap(logProbabilities(logits, mask)) as Tensor)
    let clipped = 0
    let kl = 0
    for (let i = 0; i < n; i++) {
      const ratio = Math.exp(logp[i] - batch.logpOld[i])
      if (Math.abs(ratio - 1) > coef.clip) clipped++
      kl += (ratio - 1 - Math.log(ratio)) / n
    }
    clipFraction = clipped / n
    approxKl = kl
  }
  return {
    state: {
      ...g,
      policy: pStep.params,
      policyOptimizer: pStep.optimizer,
      value: vStep.params,
      valueOptimizer: vStep.optimizer,
      updates: g.updates + 1,
    },
    policyLoss: pStep.loss,
    valueLoss: vStep.loss,
    entropy: entropyValue,
    clipFraction,
    approxKl,
  }
}

/** Advantages standardised with SB3's 1e-8 guard, so a constant batch (one step) gives zeros rather than NaN. */
const standardise = (a: Float64Array): Float64Array => {
  const { mean: m, scale } = zScores(a)
  return a.map((v) => (v - m) / (scale + 1e-8))
}

/** REINFORCE with a learned value baseline (module docs). */
export function reinforceBaselineAgent(
  options: OnPolicyOptions & { normalise?: boolean } = {},
): Agent<OnPolicyState, Float64Array, number> {
  const { normalise = true } = options
  return onPolicyAgent(
    'REINFORCE with baseline',
    'reinforce',
    options,
    { learningRate: 3e-3, entropy: 0, clipNorm: Infinity },
    (g, nets, rule, coef) => {
      const { rewards, obs, actions } = g.steps
      const n = rewards.length
      const G = new Float64Array(n)
      let running = 0
      for (let t = n - 1; t >= 0; t--) G[t] = running = rewards[t] + g.gamma * running
      const V = toFlat(unwrap(nets.value.apply(g.value, rows(obs, n, g.dim))) as Tensor)
      const adv = Float64Array.from(G, (v, t) => v - V[t])
      const r = actorCriticStep(
        g,
        nets,
        rule,
        {
          obs,
          actions,
          advantages: normalise ? standardise(adv) : adv,
          targets: G,
        },
        coef,
      )
      return { ...r.state, last: { ...NAN_LAST, policyLoss: r.policyLoss, valueLoss: r.valueLoss, entropy: r.entropy } }
    },
    (_, t) => t.terminated || t.truncated,
  )
}

/** Synchronous advantage actor–critic with n-step returns (module docs). */
export function a2cAgent(
  options: OnPolicyOptions & { nSteps?: number } = {},
): Agent<OnPolicyState, Float64Array, number> {
  const { nSteps = 16 } = options
  return onPolicyAgent(
    'A2C',
    'a2c',
    options,
    { learningRate: 1e-3, entropy: 0.01, clipNorm: 0.5 },
    (g, nets, rule, coef) => {
      const { rewards, obs, actions, values, nextValues, ends } = g.steps
      const n = rewards.length
      const R = new Float64Array(n)
      // Backwards: at an episode's end or the batch's last step, bootstrap from V(o′) (0 if terminated).
      let running = 0
      for (let t = n - 1; t >= 0; t--) {
        if (ends[t] || t === n - 1) running = nextValues[t]
        R[t] = running = rewards[t] + g.gamma * running
      }
      const adv = Float64Array.from(R, (v, t) => v - values[t])
      const r = actorCriticStep(g, nets, rule, { obs, actions, advantages: adv, targets: R }, coef)
      return { ...r.state, last: { ...NAN_LAST, policyLoss: r.policyLoss, valueLoss: r.valueLoss, entropy: r.entropy } }
    },
    (g, t) => g.steps.rewards.length >= nSteps || t.terminated || t.truncated,
  )
}

/** Options of `ppoAgent`. */
export interface PpoOptions extends OnPolicyOptions {
  /** Steps collected per update (SB3's `n_steps`; default 512). */
  horizon?: number
  /** Passes over the collected steps (default 10) and rows per minibatch (default 64). */
  epochs?: number
  batchSize?: number
  /** The clip range ε (default 0.2) and GAE's λ (default 0.95). */
  clipRange?: number
  lambda?: number
}

/** PPO with the clipped surrogate and GAE(λ) (module docs). */
export function ppoAgent(options: PpoOptions = {}): Agent<OnPolicyState, Float64Array, number> {
  const { horizon = 512, epochs = 10, batchSize = 64, clipRange = 0.2, lambda = 0.95 } = options
  return onPolicyAgent(
    'PPO',
    'ppo',
    options,
    { learningRate: 3e-4, entropy: 0, clipNorm: 0.5 },
    (g, nets, rule, coef) => {
      const { rewards, obs, actions, values, nextValues, ends, logp } = g.steps
      const n = rewards.length
      // GAE: δₜ = rₜ + γ V(o′ₜ) − V(oₜ) (V(o′) = 0 at a termination), Aₜ = δₜ + γλ Aₜ₊₁ within an episode.
      const adv = new Float64Array(n)
      let running = 0
      for (let t = n - 1; t >= 0; t--) {
        const delta = rewards[t] + g.gamma * nextValues[t] - values[t]
        running = delta + (ends[t] ? 0 : g.gamma * lambda * running)
        adv[t] = running
      }
      const targets = Float64Array.from(adv, (a, t) => a + values[t])
      let state = g
      const size = Math.min(batchSize, n)
      const per = Math.floor(n / size)
      const sums = { p: 0, v: 0, e: 0, c: 0, k: 0, m: 0 }
      for (let epoch = 0; epoch < epochs; epoch++) {
        const order = toFlat(permutation(child(stream(g.seed), 'ppo', g.updates, epoch), n))
        for (let b = 0; b < per; b++) {
          const ids = Array.from(order.slice(b * size, (b + 1) * size))
          const mb = standardise(Float64Array.from(ids, (i) => adv[i]))
          const r = actorCriticStep(
            state,
            nets,
            rule,
            {
              obs: ids.flatMap((i) => obs.slice(i * g.dim, (i + 1) * g.dim)),
              actions: ids.map((i) => actions[i]),
              advantages: mb,
              targets: Float64Array.from(ids, (i) => targets[i]),
              logpOld: Float64Array.from(ids, (i) => logp[i]),
            },
            { ...coef, clip: clipRange },
          )
          state = { ...r.state, updates: g.updates }
          sums.p += r.policyLoss
          sums.v += r.valueLoss
          sums.e += r.entropy
          sums.c += r.clipFraction
          sums.k += r.approxKl
          sums.m++
        }
      }
      const last = {
        policyLoss: sums.p / sums.m,
        valueLoss: sums.v / sums.m,
        entropy: sums.e / sums.m,
        clipFraction: sums.c / sums.m,
        approxKl: sums.k / sums.m,
      }
      return { ...state, updates: g.updates + 1, last }
    },
    (g) => g.steps.rewards.length >= horizon,
  )
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const agent = definer<AgentInfo>('agent', 'gym/agents/policy')
const ACTIVATIONS = ['tanh', 'relu', 'gelu', 'elu', 'silu'] as const
const requires = { observation: 'box', action: 'discrete', families: ['control'] } as const

agent(
  {
    key: 'reinforceBaselineAgent',
    name: 'REINFORCE with baseline',
    summary: 'A neural softmax policy updated after each episode by returns minus a learned value baseline.',
    params: space({
      learningRate: real(1e-5, 1e-1, { default: 3e-3, scale: 'log', label: 'learning rate' }),
      activation: oneOf(ACTIVATIONS, { default: 'tanh', label: 'activation' }),
    }),
    requires,
    notes: ['policy-gradient-theorem', 'reinforcement-learning'],
    cite: ['williams1992', 'sutton2000'],
    random: true,
  },
  reinforceBaselineAgent,
)
agent(
  {
    key: 'a2cAgent',
    name: 'Advantage actor–critic (A2C)',
    summary: 'Actor and critic updated every n steps on n-step advantages, with an entropy bonus.',
    params: space({
      learningRate: real(1e-5, 1e-1, { default: 1e-3, scale: 'log', label: 'learning rate' }),
      nSteps: int(1, 2048, { default: 16, label: 'steps per update' }),
      entropy: real(0, 0.1, { default: 0.01, label: 'entropy bonus' }),
    }),
    requires,
    notes: ['actor-critic', 'policy-gradient-theorem'],
    cite: ['mnih2016'],
    random: true,
  },
  a2cAgent,
)
agent(
  {
    key: 'ppoAgent',
    name: 'Proximal policy optimisation (PPO)',
    summary: 'Clipped-ratio surrogate over several epochs of minibatches from each rollout, with GAE(λ) advantages.',
    params: space({
      learningRate: real(1e-5, 1e-1, { default: 3e-4, scale: 'log', label: 'learning rate' }),
      horizon: int(16, 8192, { default: 512, label: 'steps per rollout' }),
      epochs: int(1, 50, { default: 10, label: 'epochs per rollout' }),
      clipRange: real(0.01, 1, { default: 0.2, label: 'clip range ε' }),
    }),
    requires,
    notes: ['trust-region-and-proximal-policy-optimisation', 'actor-critic'],
    cite: ['schulman2017', 'schulman2016'],
    random: true,
  },
  ppoAgent,
)
