/**
 * On-policy gradient agents for a box observation and discrete actions, on the gym protocol. Each holds a categorical
 * policy network $\pi_\theta(a \mid o)$ (logits from a multilayer perceptron) and a value network $V_\phi(o)$, each
 * trained by its own Adam step on its part of the loss:
 *
 * - **REINFORCE with a baseline** (Williams, 1992): at the end of every episode, the returns
 *   $G_t = \sum_k \gamma^k r_{t+k}$ weigh the score: the policy loss is
 *   $-\operatorname{mean}[\log \pi(a_t \mid o_t) (G_t - V(o_t))]$ with the advantages standardised, and $V$ regresses
 *   on $G_t$.
 * - **Advantage actor–critic (A2C)** (Mnih et al., 2016, synchronous form): every $n$ steps (or at an episode's end)
 *   the $n$-step returns $R_t = r_t + \gamma r_{t+1} + \dots + \gamma^n V(o_{t+n})$, bootstrapped unless the episode
 *   terminated, give advantages $R_t - V(o_t)$; one step on $-\operatorname{mean}[\log \pi \cdot A]$ plus
 *   $c_v \operatorname{mean}[(R - V)^2] - \beta \operatorname{mean}[\text{entropy}]$.
 * - **Proximal policy optimisation (PPO, clipped)** (Schulman et al., 2017): collect `horizon` steps across episodes,
 *   compute generalised advantage estimates (GAE($\lambda$); Schulman et al., 2016) from the values recorded when the
 *   steps were taken, then run `epochs` passes of minibatch Adam on the clipped surrogate
 *   $-\operatorname{mean}[\min(\rho A, \operatorname{clip}(\rho, 1 \pm \varepsilon) A)]$ plus
 *   $c_v \operatorname{mean}[(V - R)^2] - \beta\,\text{entropy}$, where
 *   $\rho = \pi_\theta(a \mid o) / \pi_{\text{old}}(a \mid o)$ and $\pi_{\text{old}}$ is the policy that collected the
 *   steps; each minibatch's advantages are standardised.
 *
 * For A2C and PPO a truncated episode (a time limit) is not terminal: its last step bootstraps from $V$ of the next
 * observation. REINFORCE's returns stop at the episode's last reward, truncated or not. The agents' states are plain
 * data; minibatch orders are drawn from a seed fixed in `init`, so `learn` stays pure. `act` draws from a child of its
 * stream's key, so it needs a fresh stream for every step, as the gym's rollouts pass.
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
  /** The hidden activation of both networks (default tanh). */
  activation?: ActivationName
  /** Adam's step size for both networks (default 3e-4 for PPO, 1e-3 for A2C, 3e-3 for REINFORCE). */
  learningRate?: number
  /** The discount (default: the environment's). */
  gamma?: number
  /** Entropy bonus $\beta$ (default 0 for PPO and REINFORCE, 0.01 for A2C). */
  entropy?: number
  /** Value-loss weight $c_v$ (default 0.5). */
  valueCoef?: number
  /** Gradient clipping by global norm (default 0.5 for PPO and A2C, Infinity for REINFORCE). */
  clipNorm?: number
}

/** The stored steps of the current batch, as flat columns. */
interface Steps {
  /** The observations, $d$ numbers per step, row after row. */
  obs: number[]
  /** The actions taken. */
  actions: number[]
  /** The rewards received. */
  rewards: number[]
  /** 1 when the step ended the episode by termination. */
  terminated: number[]
  /** 1 when the step ended the episode (terminated or truncated). */
  ends: number[]
  /** $V(o)$ under the value network that collected the step (PPO and A2C; empty for REINFORCE). */
  values: number[]
  /** $V(o')$ under the same network, 0 after a termination (PPO and A2C; 0 for REINFORCE). */
  nextValues: number[]
  /** $\log \pi_{\text{old}}(a \mid o)$ of the policy that acted (PPO; empty otherwise). */
  logp: number[]
}

/** An on-policy agent's state. */
export interface OnPolicyState {
  /** The seed of the minibatch orders, fixed in `init`. */
  seed: number
  /** The observation length $d$. */
  dim: number
  /** The number of actions. */
  actions: number
  /** The discount $\gamma$. */
  gamma: number
  /** The policy network's parameters $\theta$. */
  policy: Params[]
  /** The value network's parameters $\phi$. */
  value: Params[]
  /** The policy's Adam state. */
  policyOptimizer: unknown
  /** The value network's Adam state. */
  valueOptimizer: unknown
  /** The steps stored since the last update. */
  steps: Steps
  /** Environment steps so far. */
  t: number
  /** Updates so far: one per batch of steps learnt from (PPO's epochs of minibatches count as one). */
  updates: number
  /** Losses and diagnostics of the last update (NaN before the first). */
  last: { policyLoss: number; valueLoss: number; entropy: number; clipFraction: number; approxKl: number }
}

/**
 * No stored steps: every column empty.
 *
 * @returns A fresh `Steps`.
 */
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

/** Which of the three agents: it decides what each step records. */
type Kind = 'reinforce' | 'a2c' | 'ppo'

/**
 * The protocol shared by the three agents: `act` samples the policy, `learn` stores the step (with its values for A2C
 * and PPO, and its log-probability for PPO) and, when `due`, runs `update` on the stored steps and clears them.
 * `greedy` is the most probable action. `init` throws `TypeError` unless the observation is a box and the actions are
 * discrete.
 *
 * @param name The agent's readable name, also used in its errors.
 * @param kind Which agent, for what each step records.
 * @param options The caller's options; the unset ones fall back to `defaults`.
 * @param defaults The agent's own defaults for the step size, the entropy bonus and the clipping norm.
 * @param update How to learn from the stored steps: the new state from the state, the two networks, the update rule
 *   and the loss weights.
 * @param due Whether to update after storing a transition: from the state with the step stored, and the transition.
 * @returns The agent.
 */
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

/**
 * Policy and value losses on a batch with fixed advantages and value targets; returns both updated networks. The
 * policy loss is $-\operatorname{mean}[\log \pi \cdot A]$, or PPO's clipped surrogate when `logpOld` and a clip range
 * are given, minus $\beta$ times the mean entropy; the value loss is $c_v \operatorname{mean}[(V - R)^2]$.
 *
 * @param g The agent's state, whose networks and optimisers are updated.
 * @param nets The policy and value networks.
 * @param rule The update rule both networks step with.
 * @param batch The rows: observations ($n \times d$, flat), actions, advantages $A$, value targets $R$ and, for PPO,
 *   the log-probabilities $\log \pi_{\text{old}}$ of the policy that acted.
 * @param coef The entropy bonus $\beta$, the value weight $c_v$ and, for PPO, the clip range $\varepsilon$.
 * @returns The new state (one more update), the two losses before the step, the mean entropy, and for PPO the share
 *   of rows whose ratio left $1 \pm \varepsilon$ and the approximate KL divergence, both under the parameters before
 *   the step (NaN otherwise).
 */
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

/**
 * Advantages standardised with SB3's $10^{-8}$ guard, so a constant batch (one step) gives zeros rather than NaN.
 *
 * @param a The advantages.
 * @returns $(a - \bar{a}) / (s + 10^{-8})$, with $\bar{a}$ their mean and $s$ their standard deviation.
 */
const standardise = (a: Float64Array): Float64Array => {
  const { mean: m, scale } = zScores(a)
  return a.map((v) => (v - m) / (scale + 1e-8))
}

/**
 * REINFORCE with a learned value baseline (module docs): one update at the end of every episode, on that episode's
 * steps. Defaults: step size $3 \times 10^{-3}$, no entropy bonus, no gradient clipping.
 *
 * @param options The networks and the optimiser (`OnPolicyOptions`), and `normalise`, whether to standardise the
 *   advantages $G_t - V(o_t)$ (default true).
 * @returns The agent, named `'REINFORCE with baseline'`.
 *
 * @example REINFORCE learns which action each sign asks for
 * // Each step the observation is -1 or 1, and the action matching its sign (0 or 1) pays 1; episodes last 10 steps.
 * const env = { name: 'sign', observation: { kind: 'box', shape: [1], low: [-1], high: [1] } }
 * const agent = reinforceBaselineAgent({ hidden: [8], learningRate: 0.02 })
 * let g = agent.init({ ...env, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }, stream(0))
 * const s = stream(1)
 * for (let episode = 0; episode < 20; episode++)
 *   for (let t = 0; t < 10; t++) {
 *     const o = Float64Array.of(uniform(s) < 0.5 ? -1 : 1)
 *     const action = agent.act(g, o, stream(`${episode} ${t}`)).action
 *     const reward = action === (o[0] > 0 ? 1 : 0) ? 1 : 0
 *     g = agent.learn(g, { observation: o, action, reward, next: o, terminated: false, truncated: t === 9 })
 *   }
 * print('updates:', g.updates)
 * print('pi at -1:', agent.act(g, [-1], stream('a')).probabilities)
 * print('pi at 1:', agent.act(g, [1], stream('b')).probabilities)
 */
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

/**
 * Synchronous advantage actor–critic with $n$-step returns (module docs): one update every `nSteps` steps (default
 * 16) or at an episode's end, whichever comes first. Defaults: step size $10^{-3}$, entropy bonus 0.01, gradients
 * clipped to global norm 0.5.
 *
 * @param options The networks and the optimiser (`OnPolicyOptions`), and `nSteps`, the most steps per update.
 * @returns The agent, named `'A2C'`.
 *
 * @example A2C on a ten-step task
 * // Each step the observation is -1 or 1, and the action matching its sign (0 or 1) pays 1; episodes last 10 steps.
 * const env = { name: 'sign', observation: { kind: 'box', shape: [1], low: [-1], high: [1] } }
 * const agent = a2cAgent({ hidden: [8], learningRate: 0.05, nSteps: 10 })
 * let g = agent.init({ ...env, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }, stream(0))
 * const s = stream(1)
 * for (let episode = 0; episode < 20; episode++)
 *   for (let t = 0; t < 10; t++) {
 *     const o = Float64Array.of(uniform(s) < 0.5 ? -1 : 1)
 *     const action = agent.act(g, o, stream(`${episode} ${t}`)).action
 *     const reward = action === (o[0] > 0 ? 1 : 0) ? 1 : 0
 *     g = agent.learn(g, { observation: o, action, reward, next: o, terminated: false, truncated: t === 9 })
 *   }
 * print('updates:', g.updates)
 * print('pi at -1:', agent.act(g, [-1], stream('a')).probabilities)
 * print('pi at 1:', agent.act(g, [1], stream('b')).probabilities)
 */
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
  /** Passes over the collected steps (default 10). */
  epochs?: number
  /** Rows per minibatch (default 64); the rows left over after whole minibatches sit out that epoch. */
  batchSize?: number
  /** The clip range $\varepsilon$ (default 0.2). */
  clipRange?: number
  /** GAE's $\lambda$ (default 0.95). */
  lambda?: number
}

/**
 * PPO with the clipped surrogate and GAE($\lambda$) (module docs): one update every `horizon` steps, of `epochs`
 * passes over the steps in shuffled minibatches. Defaults: step size $3 \times 10^{-4}$, no entropy bonus, gradients
 * clipped to global norm 0.5. `scalars` add the clip fraction and the approximate KL divergence.
 *
 * @param options The networks, the optimiser and the rollout (see `PpoOptions` for each default).
 * @returns The agent, named `'PPO'`.
 *
 * @example PPO on a ten-step task
 * // Each step the observation is -1 or 1, and the action matching its sign (0 or 1) pays 1; episodes last 10 steps.
 * const env = { name: 'sign', observation: { kind: 'box', shape: [1], low: [-1], high: [1] } }
 * const agent = ppoAgent({ hidden: [8], learningRate: 0.01, horizon: 50, epochs: 4, batchSize: 25 })
 * let g = agent.init({ ...env, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }, stream(0))
 * const s = stream(1)
 * for (let episode = 0; episode < 20; episode++)
 *   for (let t = 0; t < 10; t++) {
 *     const o = Float64Array.of(uniform(s) < 0.5 ? -1 : 1)
 *     const action = agent.act(g, o, stream(`${episode} ${t}`)).action
 *     const reward = action === (o[0] > 0 ? 1 : 0) ? 1 : 0
 *     g = agent.learn(g, { observation: o, action, reward, next: o, terminated: false, truncated: t === 9 })
 *   }
 * print('updates:', g.updates)
 * print('pi at -1:', agent.act(g, [-1], stream('a')).probabilities)
 * print('pi at 1:', agent.act(g, [1], stream('b')).probabilities)
 * print('last update:', g.last)
 */
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
