/**
 * Tabular learning agents (Sutton and Barto, 2018, chs. 5–7 and 13) on discrete observations and actions: Q-learning,
 * SARSA and expected SARSA (`tdControlAgent`), $n$-step SARSA, Monte Carlo control, TD(0) prediction of a fixed policy
 * and REINFORCE with a tabular softmax policy, and `greedyPath` to read a route off a learnt policy. Each agent is an
 * `Agent`: `act` draws from its stream, `learn` is a pure update from one transition. They need no model of the
 * environment, only its observation and action domains (both discrete, or `init` throws `DomainError`); tables are
 * $S \times A$ tensors, row $s$ for observation $s$.
 *
 * Conventions. The environment folds a terminal state's value into the reward of arriving there, so a `terminated`
 * transition never bootstraps; a `truncated` one does, from the $\varepsilon$-greedy expectation at `next` for the
 * on-policy methods (the next action is never taken). When the environment masks actions, `act` chooses among the
 * legal ones and maxima and expectations at `next` run over `nextLegal`. Learners that need what comes after a
 * transition buffer it in their state: SARSA one transition (it needs the next action), $n$-step SARSA $n$ of them,
 * Monte Carlo control and REINFORCE the whole episode, updated when the episode ends.
 */

import type { Agent, AgentInfo, Decision, EnvironmentShape, Transition } from 'aifn-compute/foundation/contracts'
import { integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, domainSize, int, real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { isActive, type MdpTables, type PolicyInput } from '../mdp'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Shared pieces ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The number of observations and actions of an environment with discrete domains. Throws `DomainError` when either
 * domain is not discrete.
 *
 * @param env The environment the agent is initialised for.
 * @returns `S`, the number of observations, and `A`, the number of actions.
 */
function sizes(env: EnvironmentShape): { S: number; A: number } {
  if (env.observation.kind !== 'discrete' || env.action.kind !== 'discrete')
    throw new DomainError('sizes', 'tabular agents need discrete observations and actions')
  return { S: domainSize(env.observation), A: domainSize(env.action) }
}

/**
 * The actions to choose among: the legal ones, or all $A$.
 *
 * @param A The number of actions.
 * @param legal The legal actions, when the environment masks some.
 * @returns `legal` itself when given, else $0, \dots, A - 1$.
 */
const choices = (A: number, legal?: readonly number[]) => legal ?? Array.from({ length: A }, (_, a) => a)

/**
 * $\varepsilon$-greedy on $Q(s, \cdot)$ over the allowed actions: with probability $\varepsilon$ a uniform action, else
 * a greedy one with ties broken uniformly at random (so an untrained agent does not always go one way). Reports the
 * action values and probabilities.
 *
 * @param Q The action values, $S \times A$ row-major.
 * @param s The current observation (row of `Q`).
 * @param A The number of actions.
 * @param eps The exploration rate $\varepsilon$.
 * @param r The stream the choice draws from.
 * @param legal The legal actions, when the environment masks some; others get probability 0.
 * @returns The action, `scores` (row $s$ of `Q`, every action) and `probabilities`
 *   ($\varepsilon$ over the allowed count, plus $1 - \varepsilon$ shared by the tied greedy actions).
 */
function epsilonGreedyDecision(
  Q: ArrayLike<number>,
  s: number,
  A: number,
  eps: number,
  r: Stream,
  legal?: readonly number[],
): Decision<number> {
  const allowed = choices(A, legal)
  const scores = Float64Array.from({ length: A }, (_, a) => Q[s * A + a])
  let best = -Infinity
  let ties: number[] = []
  for (const a of allowed) {
    if (scores[a] > best) {
      best = scores[a]
      ties = [a]
    } else if (scores[a] === best) ties.push(a)
  }
  const probabilities = new Float64Array(A)
  for (const a of allowed) probabilities[a] = eps / allowed.length + (scores[a] === best ? (1 - eps) / ties.length : 0)
  const action =
    uniform(r) < eps
      ? allowed[integers(r, allowed.length)]
      : ties.length === 1
        ? ties[0]
        : ties[integers(r, ties.length)]
  return { action, scores, probabilities }
}

/**
 * The first action with the largest $Q(s, a)$ among the allowed ones: the deterministic greedy action.
 *
 * @param Q The table, $S \times A$ row-major (action values, or any per-action scores).
 * @param s The observation (row of `Q`).
 * @param A The number of actions.
 * @param legal The legal actions, when the environment masks some.
 * @returns The action, or $-1$ when no action is allowed.
 */
function argmaxQ(Q: ArrayLike<number>, s: number, A: number, legal?: readonly number[]): number {
  let best = -Infinity
  let arg = -1
  for (const a of choices(A, legal))
    if (Q[s * A + a] > best || arg < 0) {
      best = Q[s * A + a]
      arg = a
    }
  return arg
}

/**
 * The mean over observations of $\max_a Q(o, a)$ (over every action): the agent's value estimate, a training-curve
 * scalar.
 *
 * @param Q The action values, $S \times A$.
 * @returns The mean of the row maxima.
 */
function meanMaxQ(Q: Tensor): number {
  const [S, A] = Q.shape
  let sum = 0
  for (let s = 0; s < S; s++) sum += maxQ(Q.data, s, A)
  return sum / S
}

/** What every value-based agent adds for evaluation and training curves. */
const valueBased = {
  greedy: (g: TabularAgentState, o: number, legal?: readonly number[]) => argmaxQ(g.Q.data, o, g.Q.shape[1], legal),
  scalars: (g: TabularAgentState) => ({ 'mean max Q': meanMaxQ(g.Q) }),
}

/**
 * $\max_a Q(s, a)$ over the allowed actions.
 *
 * @param Q The action values, $S \times A$ row-major.
 * @param s The observation (row of `Q`).
 * @param A The number of actions.
 * @param legal The legal actions, when the environment masks some.
 * @returns The largest value ($-\infty$ when no action is allowed).
 */
function maxQ(Q: ArrayLike<number>, s: number, A: number, legal?: readonly number[]): number {
  let best = -Infinity
  for (const a of choices(A, legal)) best = Math.max(best, Q[s * A + a])
  return best
}

/**
 * $\expect_\pi[Q(s, \cdot)]$ under the $\varepsilon$-greedy policy over the allowed actions: $\varepsilon$ times the
 * mean plus $(1 - \varepsilon)$ times the maximum.
 *
 * @param Q The action values, $S \times A$ row-major.
 * @param s The observation (row of `Q`).
 * @param A The number of actions.
 * @param eps The exploration rate $\varepsilon$.
 * @param legal The legal actions, when the environment masks some.
 * @returns The expected action value.
 */
function expectedQ(Q: ArrayLike<number>, s: number, A: number, eps: number, legal?: readonly number[]): number {
  const allowed = choices(A, legal)
  let mean = 0
  for (const a of allowed) mean += Q[s * A + a] / allowed.length
  return eps * mean + (1 - eps) * maxQ(Q, s, A, legal)
}

/** One step of a buffered episode: the observation $s$, the action $a$ and the reward $r$ that followed. */
interface Visit {
  /** The observation. */
  s: number
  /** The action taken. */
  a: number
  /** The reward received. */
  r: number
}

/** Options shared by the value-based agents. */
export interface TabularOptions {
  /** The step size $\alpha$. Default 0.5 (Monte Carlo control: sample averaging). */
  learningRate?: number
  /** The exploration rate $\varepsilon$ of the $\varepsilon$-greedy policy. Default 0.1. */
  epsilon?: number
  /** The discount; default the environment's. */
  gamma?: number
  /** The initial action value (optimistic values encourage exploration). Default 0. */
  initialQ?: number
}

/** The state of a value-based agent: the action values and the discount it learns with. */
export interface TabularAgentState {
  /** $Q(o, a)$, observations $\times$ actions. */
  Q: Tensor
  /** The discount $\gamma$ the agent learns with. */
  gamma: number
  /** Transitions learnt from. */
  updates: number
}

/**
 * A value-based agent's first state: every action value at `initialQ`, no updates.
 *
 * @param env The environment, for the table's size and the default discount.
 * @param gamma The discount; the environment's when undefined.
 * @param initialQ The value every entry of $Q$ starts at.
 * @returns The state, with an $S \times A$ table.
 */
const qState = (env: EnvironmentShape, gamma: number | undefined, initialQ: number): TabularAgentState => {
  const { S, A } = sizes(env)
  return { Q: fromData(new Float64Array(S * A).fill(initialQ), [S, A]), gamma: gamma ?? env.gamma, updates: 0 }
}

// ── One-step TD control ──────────────────────────────────────────────────────────────────────────────────────────────

/** The one-step TD control methods. */
export type TdMethod = 'sarsa' | 'q-learning' | 'expected-sarsa'

/** A TD control agent's state; SARSA also holds the transition whose update waits for the next action. */
export interface TdAgentState extends TabularAgentState {
  /** SARSA's transition waiting for the next action to bootstrap from; null otherwise. */
  pending: Transition<number, number> | null
  /** Episodes completed, for the decay of $\varepsilon$. */
  episodes: number
}

/**
 * One-step TD control with an $\varepsilon$-greedy policy:
 * $Q(s, a) \leftarrow Q(s, a) + \alpha (r + \gamma b - Q(s, a))$, with $b = 0$ after a terminal transition. Q-learning
 * bootstraps from $b = \max_{a'} Q(s', a')$ (off-policy; Watkins, 1989), expected SARSA from the expectation under the
 * $\varepsilon$-greedy policy (van Seijen et al., 2009), SARSA from $Q(s', a')$ of the action actually taken next
 * (on-policy), so SARSA holds each transition until the next one arrives. `greedy` is the first action of largest
 * $Q$, and `scalars` reports the mean of the row maxima (and $\varepsilon$ when it decays).
 *
 * @param options The method and its step size, exploration, discount and initial values.
 * @param options.method Which target to bootstrap from: `'q-learning'`, `'sarsa'` or `'expected-sarsa'`.
 * @param options.learningRate The step size $\alpha$.
 * @param options.epsilon The exploration rate $\varepsilon$ (before any decay).
 * @param options.gamma The discount $\gamma$; the environment's when left out.
 * @param options.initialQ The value every action value starts at (optimistic values encourage exploration).
 * @param options.epsilonDecay The episodes $d$ of the decay $\varepsilon / (1 + e/d)$ after $e$ episodes; the default
 *   $\infty$ keeps $\varepsilon$ fixed.
 * @returns The agent, named after the method.
 *
 * @example Choosing the method: expected SARSA on a short corridor
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = tdControlAgent({ method: 'expected-sarsa', learningRate: 0.5 })
 * const s = stream(1)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 30; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('expected SARSA, Q =', g.Q)
 * print('greedy actions in states 0 to 2:', [0, 1, 2].map((o) => agent.greedy(g, o)))
 */
export function tdControlAgent({
  method = 'q-learning',
  learningRate: alpha = 0.5,
  epsilon = 0.1,
  gamma,
  initialQ = 0,
  epsilonDecay = Infinity,
}: TabularOptions & {
  method?: TdMethod
  /**
   * $\varepsilon$ decays as $\varepsilon / (1 + e/d)$ after $e$ episodes, $d$ this value, so exploration fades. Default
   * $\infty$ (no decay).
   */
  epsilonDecay?: number
} = {}): Agent<TdAgentState, number, number> {
  const eps = (g: TdAgentState) => epsilon / (1 + g.episodes / epsilonDecay)
  const update = (Q: Float64Array, A: number, g: number, t: Transition<number, number>, boot: number) => {
    const k = t.observation * A + t.action
    Q[k] += alpha * (t.reward + (t.terminated ? 0 : g * boot) - Q[k])
  }
  return {
    name: method === 'sarsa' ? 'SARSA' : method === 'q-learning' ? 'Q-learning' : 'expected SARSA',
    init: (env) => ({ ...qState(env, gamma, initialQ), pending: null, episodes: 0 }),
    act: (g, o, stream, legal) => epsilonGreedyDecision(g.Q.data, o, g.Q.shape[1], eps(g), stream, legal),
    ...valueBased,
    // ε is a training curve only when it decays.
    scalars: (g) => ({ 'mean max Q': meanMaxQ(g.Q), ...(epsilonDecay < Infinity && { ε: eps(g) }) }),
    learn(g, t) {
      const A = g.Q.shape[1]
      const Q = Float64Array.from(g.Q.data)
      const epsilon = eps(g)
      const episodes = g.episodes + (t.terminated || t.truncated ? 1 : 0)
      let pending: Transition<number, number> | null = null
      if (method === 'sarsa') {
        // The waiting update bootstraps from the action taken now; this one waits for the next action.
        if (g.pending) update(Q, A, g.gamma, g.pending, Q[t.observation * A + t.action])
        if (t.terminated || t.truncated) update(Q, A, g.gamma, t, expectedQ(Q, t.next, A, epsilon, t.nextLegal))
        else pending = t
      } else {
        const boot =
          method === 'q-learning' ? maxQ(Q, t.next, A, t.nextLegal) : expectedQ(Q, t.next, A, epsilon, t.nextLegal)
        update(Q, A, g.gamma, t, t.terminated ? 0 : boot)
      }
      return { Q: fromData(Q, g.Q.shape), gamma: g.gamma, updates: g.updates + 1, pending, episodes }
    },
  }
}

/** Options of the TD control agents: those of every value-based agent, and the decay of $\varepsilon$. */
export type TdControlOptions = TabularOptions & { epsilonDecay?: number }

/**
 * Q-learning (Watkins, 1989): `tdControlAgent` with `method: 'q-learning'`, bootstrapping from
 * $\max_{a'} Q(s', a')$.
 *
 * @param options Step size, exploration, discount, initial values and the decay of $\varepsilon$, as for
 *   `tdControlAgent`.
 * @returns The agent, named `'Q-learning'`.
 *
 * @example One update on a 2-state, 2-action table
 * const agent = qLearningAgent({ learningRate: 0.5 })
 * const env = { observation: { kind: 'discrete', n: 2 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const g0 = agent.init(env, stream(0))
 * // Action 1 in state 0 pays 1 and leads to state 1, whose values are still 0.
 * const g1 = agent.learn(g0, { observation: 0, action: 1, reward: 1, next: 1, terminated: false, truncated: false })
 * print('after one step, Q =', g1.Q)
 * // Action 0 in state 1 pays 2 and ends the episode, so it does not bootstrap.
 * const g2 = agent.learn(g1, { observation: 1, action: 0, reward: 2, next: 0, terminated: true, truncated: false })
 * // The same step from state 0 now bootstraps from the best value of state 1.
 * const g3 = agent.learn(g2, { observation: 0, action: 1, reward: 1, next: 1, terminated: false, truncated: false })
 * print('after three steps, Q =', g3.Q)
 *
 * @example Q-learning on a short corridor
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = qLearningAgent({ learningRate: 0.5 })
 * const s = stream(2)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 30; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('Q =', g.Q)
 * print('greedy actions in states 0 to 2:', [0, 1, 2].map((o) => agent.greedy(g, o)))
 */
export const qLearningAgent = (options: TdControlOptions = {}) => tdControlAgent({ ...options, method: 'q-learning' })
/**
 * SARSA (Rummery and Niranjan, 1994): `tdControlAgent` with `method: 'sarsa'`, bootstrapping from the action actually
 * taken next, so each transition is learnt from one step late.
 *
 * @param options Step size, exploration, discount, initial values and the decay of $\varepsilon$, as for
 *   `tdControlAgent`.
 * @returns The agent, named `'SARSA'`.
 *
 * @example SARSA on a short corridor
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = sarsaAgent({ learningRate: 0.5 })
 * const s = stream(3)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 30; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('Q =', g.Q)
 * print('greedy actions in states 0 to 2:', [0, 1, 2].map((o) => agent.greedy(g, o)))
 */
export const sarsaAgent = (options: TdControlOptions = {}) => tdControlAgent({ ...options, method: 'sarsa' })
/**
 * Expected SARSA (van Seijen et al., 2009): `tdControlAgent` with `method: 'expected-sarsa'`, bootstrapping from the
 * expected action value under the $\varepsilon$-greedy policy.
 *
 * @param options Step size, exploration, discount, initial values and the decay of $\varepsilon$, as for
 *   `tdControlAgent`.
 * @returns The agent, named `'expected SARSA'`.
 *
 * @example Expected SARSA's values sit just below Q-learning's
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = expectedSarsaAgent({ learningRate: 0.5, epsilon: 0.2 })
 * const s = stream(4)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 40; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('Q =', g.Q)
 */
export const expectedSarsaAgent = (options: TdControlOptions = {}) =>
  tdControlAgent({ ...options, method: 'expected-sarsa' })

// ── n-step SARSA ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** An $n$-step SARSA agent's state: the visits whose $n$-step targets are not yet complete. */
export interface NStepAgentState extends TabularAgentState {
  /** The visits of the current episode not yet updated, oldest first: at most $n$ between transitions. */
  window: Visit[]
}

/**
 * $n$-step SARSA (Sutton and Barto, 2018, §7.2): the target of $(s_\tau, a_\tau)$ is the $n$-step return
 * $G = \sum_{i<n} \gamma^i r_{\tau+i+1} + \gamma^n Q(s_{\tau+n}, a_{\tau+n})$, shortened at the end of the episode
 * (with no bootstrap after a terminal state, and the $\varepsilon$-greedy expectation after a truncation). Each visit
 * waits in a window until its target is complete; $Q$ moves towards it by $\alpha (G - Q)$.
 *
 * @param options The number of steps and the step size, exploration, discount and initial values.
 * @param options.n The number of rewards $n$ before bootstrapping; 1 is SARSA.
 * @param options.learningRate The step size $\alpha$.
 * @param options.epsilon The exploration rate $\varepsilon$ (fixed).
 * @param options.gamma The discount $\gamma$; the environment's when left out.
 * @param options.initialQ The value every action value starts at.
 * @returns The agent, named after $n$.
 *
 * @example 3-step SARSA on a short corridor
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = nStepSarsaAgent({ n: 3, learningRate: 0.5 })
 * const s = stream(5)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 30; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('Q =', g.Q)
 * print('greedy actions in states 0 to 2:', [0, 1, 2].map((o) => agent.greedy(g, o)))
 */
export function nStepSarsaAgent({
  n = 4,
  learningRate: alpha = 0.5,
  epsilon = 0.1,
  gamma,
  initialQ = 0,
}: TabularOptions & { n?: number } = {}): Agent<NStepAgentState, number, number> {
  return {
    name: `${n}-step SARSA`,
    init: (env) => ({ ...qState(env, gamma, initialQ), window: [] }),
    act: (g, o, stream, legal) => epsilonGreedyDecision(g.Q.data, o, g.Q.shape[1], epsilon, stream, legal),
    ...valueBased,
    learn(g, t) {
      const A = g.Q.shape[1]
      const Q = Float64Array.from(g.Q.data)
      const gm = g.gamma
      const target = (vs: Visit[], from: number, tail: number) => {
        let G = tail
        for (let i = vs.length - 1; i >= from; i--) G = vs[i].r + gm * G
        return G
      }
      const apply = (v: Visit, G: number) => (Q[v.s * A + v.a] += alpha * (G - Q[v.s * A + v.a]))
      let window = [...g.window, { s: t.observation, a: t.action, r: t.reward }]
      // The oldest visit's target is complete once n more visits follow it: bootstrap from the n-th one's (s, a).
      if (window.length === n + 1) {
        const boot = Q[window[n].s * A + window[n].a]
        apply(window[0], target(window.slice(0, n), 0, boot))
        window = window.slice(1)
      }
      if (t.terminated || t.truncated) {
        const tail = t.terminated ? 0 : expectedQ(Q, t.next, A, epsilon, t.nextLegal)
        for (let i = 0; i < window.length; i++) apply(window[i], target(window, i, tail))
        window = []
      }
      return { Q: fromData(Q, g.Q.shape), gamma: gm, updates: g.updates + 1, window }
    },
  }
}

// ── Monte Carlo control ──────────────────────────────────────────────────────────────────────────────────────────────

/** A Monte Carlo agent's state: action values, first-visit counts and the episode so far. */
export interface MonteCarloAgentState extends TabularAgentState {
  /** First visits of each $(o, a)$, observations $\times$ actions. */
  counts: Tensor
  /** The current episode's visits so far, emptied when it ends. */
  episode: Visit[]
}

/**
 * On-policy first-visit Monte Carlo control with $\varepsilon$-soft policies (Sutton and Barto, 2018, §5.4): when an
 * episode ends, $Q(s, a)$ moves towards the return that followed the first visit of $(s, a)$, by sample averaging
 * (default) or a constant step size $\alpha$. A truncated episode's returns stop at the truncation.
 *
 * @param options The step size, exploration, discount and initial values.
 * @param options.learningRate A constant step size $\alpha$; left out, each update averages, with step $1/N(s, a)$
 *   after $N$ first visits.
 * @param options.epsilon The exploration rate $\varepsilon$ (fixed).
 * @param options.gamma The discount $\gamma$; the environment's when left out.
 * @param options.initialQ The value every action value starts at (overwritten by the first sample average).
 * @returns The agent, named `'Monte Carlo control'`.
 *
 * @example Monte Carlo control on a short corridor
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = monteCarloControlAgent({ epsilon: 0.1 })
 * const s = stream(6)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 40; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('Q =', g.Q)
 * print('first visits:', g.counts)
 */
export function monteCarloControlAgent({
  learningRate: alpha,
  epsilon = 0.1,
  gamma,
  initialQ = 0,
}: TabularOptions = {}): Agent<MonteCarloAgentState, number, number> {
  return {
    name: 'Monte Carlo control',
    init: (env) => {
      const base = qState(env, gamma, initialQ)
      return { ...base, counts: fromData(new Float64Array(base.Q.data.length), base.Q.shape), episode: [] }
    },
    act: (g, o, stream, legal) => epsilonGreedyDecision(g.Q.data, o, g.Q.shape[1], epsilon, stream, legal),
    ...valueBased,
    learn(g, t) {
      const episode = [...g.episode, { s: t.observation, a: t.action, r: t.reward }]
      if (!t.terminated && !t.truncated) return { ...g, updates: g.updates + 1, episode }
      const A = g.Q.shape[1]
      const Q = Float64Array.from(g.Q.data)
      const N = Float64Array.from(g.counts.data)
      const first = new Map<number, number>()
      episode.forEach((v, i) => {
        const k = v.s * A + v.a
        if (!first.has(k)) first.set(k, i)
      })
      let G = 0
      for (let i = episode.length - 1; i >= 0; i--) {
        const v = episode[i]
        G = v.r + g.gamma * G
        const k = v.s * A + v.a
        if (first.get(k) !== i) continue
        N[k] += 1
        Q[k] += (alpha ?? 1 / N[k]) * (G - Q[k])
      }
      return {
        Q: fromData(Q, g.Q.shape),
        gamma: g.gamma,
        updates: g.updates + 1,
        counts: fromData(N, g.Q.shape),
        episode: [],
      }
    },
  }
}

// ── TD(0) prediction ─────────────────────────────────────────────────────────────────────────────────────────────────

/** A TD(0) prediction agent's state: the value estimate and the policy it follows. */
export interface TdPredictionState {
  /** $V(o)$, one per observation. */
  V: Tensor
  /** $\pi(a \mid o)$, observations $\times$ actions. */
  policy: Tensor
  /** The discount $\gamma$. */
  gamma: number
  /** Transitions learnt from. */
  updates: number
}

/**
 * The policy as $\pi(a \mid s)$, $S \times A$: a deterministic policy is an action per state ($-1$ where no action is
 * taken), a stochastic one $S \times A$ probabilities. Throws `ShapeError` for any other length.
 *
 * @param policy The policy: $S$ actions, or $SA$ probabilities row-major (a tensor or an array).
 * @param S The number of states.
 * @param A The number of actions.
 * @returns $SA$ probabilities, row-major; a deterministic policy's rows are one-hot (all zero where it has $-1$).
 */
function policyTable(policy: PolicyInput, S: number, A: number): Float64Array {
  const v = 'shape' in policy ? Array.from(policy.data) : [...policy]
  if (v.length === S * A) return Float64Array.from(v)
  if (v.length !== S) throw new ShapeError('policy', `policy: expected ${S} actions or ${S} × ${A} probabilities`)
  const pi = new Float64Array(S * A)
  for (let s = 0; s < S; s++) if (v[s] >= 0) pi[s * A + v[s]] = 1
  return pi
}

/**
 * A draw from row $s$ of $\pi$ ($S \times A$), by inverting its cumulative sum.
 *
 * @param pi The probabilities, $S \times A$ row-major.
 * @param s The row.
 * @param A The number of actions.
 * @param r The stream the draw comes from.
 * @returns The action drawn; $A - 1$ when rounding leaves the uniform draw past the row's sum.
 */
function drawAction(pi: ArrayLike<number>, s: number, A: number, r: Stream): number {
  let u = uniform(r)
  for (let a = 0; a < A; a++) {
    if (u < pi[s * A + a]) return a
    u -= pi[s * A + a]
  }
  return A - 1
}

/**
 * TD(0) prediction of a fixed policy's value (Sutton, 1988, Machine Learning 3): the agent follows `policy`
 * (deterministic: an action per state; stochastic: states $\times$ actions; default uniformly random) and after every
 * transition $V(s) \leftarrow V(s) + \alpha (r + \gamma V(s') - V(s))$, with $V(s') = 0$ after a terminal transition.
 * `act` ignores any action mask: it draws from the given policy. Throws `ShapeError` at `init` for a policy of the
 * wrong length.
 *
 * @param options The policy to evaluate, the step size and the discount.
 * @param options.policy The policy: an action per state, or $S \times A$ probabilities; uniform over the actions when
 *   left out.
 * @param options.learningRate The step size $\alpha$.
 * @param options.gamma The discount $\gamma$; the environment's when left out.
 * @returns The agent, named `'TD(0) prediction'`.
 *
 * @example The values of always moving right
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = tdPredictionAgent({ policy: [1, 1, 1, -1], learningRate: 0.5 })
 * const s = stream(7)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 30; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('V =', g.V)
 */
export function tdPredictionAgent({
  policy,
  learningRate: alpha = 0.1,
  gamma,
}: Pick<TabularOptions, 'learningRate' | 'gamma'> & { policy?: PolicyInput } = {}): Agent<
  TdPredictionState,
  number,
  number
> {
  return {
    name: 'TD(0) prediction',
    init(env) {
      const { S, A } = sizes(env)
      return {
        V: fromData(new Float64Array(S)),
        policy: fromData(policy ? policyTable(policy, S, A) : new Float64Array(S * A).fill(1 / A), [S, A]),
        gamma: gamma ?? env.gamma,
        updates: 0,
      }
    },
    act: (g, o, stream) => {
      const A = g.policy.shape[1]
      return {
        action: drawAction(g.policy.data, o, A, stream),
        probabilities: Float64Array.from(g.policy.data.subarray(o * A, (o + 1) * A)),
      }
    },
    greedy: (g, o, legal) => argmaxQ(g.policy.data, o, g.policy.shape[1], legal),
    scalars: (g) => ({ 'mean V': Array.from(g.V.data).reduce((a, b) => a + b, 0) / g.V.data.length }),
    learn(g, t) {
      const V = Float64Array.from(g.V.data)
      V[t.observation] += alpha * (t.reward + (t.terminated ? 0 : g.gamma * V[t.next]) - V[t.observation])
      return { ...g, V: fromData(V), updates: g.updates + 1 }
    },
  }
}

// ── REINFORCE ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A REINFORCE agent's state: softmax preferences $\theta$, the baseline $V$ and the episode so far. */
export interface ReinforceState {
  /** $\theta(o, a)$, observations $\times$ actions: $\pi(a \mid o) \propto \exp \theta(o, a)$. */
  theta: Tensor
  /** The baseline $b(o)$ (zeros without a baseline). */
  V: Tensor
  /** The discount $\gamma$. */
  gamma: number
  /** Transitions learnt from. */
  updates: number
  /** The current episode's visits so far, emptied when it ends. */
  episode: Visit[]
}

/**
 * $\operatorname{softmax}(\theta(s, \cdot))$ over the allowed actions (0 elsewhere).
 *
 * @param theta The preferences, $S \times A$ row-major.
 * @param s The row.
 * @param A The number of actions.
 * @param legal The legal actions, when the environment masks some.
 * @returns $A$ probabilities.
 */
function softmaxRow(theta: ArrayLike<number>, s: number, A: number, legal?: readonly number[]): Float64Array {
  const allowed = choices(A, legal)
  let mx = -Infinity
  for (const a of allowed) mx = Math.max(mx, theta[s * A + a])
  const p = new Float64Array(A)
  let z = 0
  for (const a of allowed) z += p[a] = Math.exp(theta[s * A + a] - mx)
  for (const a of allowed) p[a] /= z
  return p
}

/**
 * REINFORCE (Williams, 1992, Machine Learning 8) with a tabular softmax policy
 * $\pi(a \mid s) \propto \exp \theta(s, a)$, all preferences starting at 0: when an episode ends,
 * $\theta(s_t, \cdot) \mathrel{+}= \alpha \gamma^t (G_t - b(s_t)) (\evec_{a_t} - \pi(\cdot \mid s_t))$, the
 * score-function gradient, with every step's $\pi$ taken from the $\theta$ that generated the episode (over every
 * action: the update does not apply an action mask). With `baseline`, $b$ is a state-value estimate learned alongside,
 * $b(s_t) \mathrel{+}= \beta (G_t - b(s_t))$ (Sutton and Barto, 2018, §13.4); without, $b = 0$. `greedy` is the
 * action of largest preference.
 *
 * @param options The step sizes, the baseline switch and the discount.
 * @param options.learningRate The policy's step size $\alpha$.
 * @param options.baseline Whether to subtract a learned state-value baseline.
 * @param options.baselineLearningRate The baseline's step size $\beta$.
 * @param options.gamma The discount $\gamma$; the environment's when left out.
 * @returns The agent, named `'REINFORCE with baseline'` or `'REINFORCE'`.
 *
 * @example REINFORCE learns to walk right
 * // A corridor of states 0 to 3: action 1 moves right, 0 left; reaching state 3 ends the episode with reward 1.
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = reinforceAgent({ learningRate: 0.5 })
 * const s = stream(8)
 * let g = agent.init(env, s)
 * for (let episode = 0; episode < 60; episode++) {
 *   let o = 0
 *   for (let t = 0; t < 20; t++) {
 *     const action = agent.act(g, o, s).action
 *     const next = action === 1 ? o + 1 : Math.max(0, o - 1)
 *     const terminated = next === 3
 *     const truncated = !terminated && t === 19
 *     g = agent.learn(g, { observation: o, action, reward: terminated ? 1 : 0, next, terminated, truncated })
 *     if (terminated) break
 *     o = next
 *   }
 * }
 * print('pi(right) in states 0 to 2:', [0, 1, 2].map((o) => agent.act(g, o, s).probabilities[1]))
 * print('baseline V =', g.V)
 */
export function reinforceAgent({
  learningRate: alpha = 0.1,
  baseline = true,
  baselineLearningRate: baselineAlpha = 0.1,
  gamma,
}: {
  learningRate?: number
  baseline?: boolean
  baselineLearningRate?: number
  gamma?: number
} = {}): Agent<ReinforceState, number, number> {
  return {
    name: baseline ? 'REINFORCE with baseline' : 'REINFORCE',
    init(env) {
      const { S, A } = sizes(env)
      return {
        theta: fromData(new Float64Array(S * A), [S, A]),
        V: fromData(new Float64Array(S)),
        gamma: gamma ?? env.gamma,
        updates: 0,
        episode: [],
      }
    },
    act(g, o, stream, legal) {
      const A = g.theta.shape[1]
      const p = softmaxRow(g.theta.data, o, A, legal)
      return { action: drawAction(p, 0, A, stream), scores: Float64Array.from(p), probabilities: p }
    },
    greedy: (g, o, legal) => argmaxQ(g.theta.data, o, g.theta.shape[1], legal),
    learn(g, t) {
      const episode = [...g.episode, { s: t.observation, a: t.action, r: t.reward }]
      if (!t.terminated && !t.truncated) return { ...g, updates: g.updates + 1, episode }
      const A = g.theta.shape[1]
      const before = g.theta.data
      const theta = Float64Array.from(before)
      const V = Float64Array.from(g.V.data)
      const T = episode.length
      const G = new Float64Array(T + 1)
      for (let i = T - 1; i >= 0; i--) G[i] = episode[i].r + g.gamma * G[i + 1]
      for (let i = 0; i < T; i++) {
        const { s, a } = episode[i]
        const delta = G[i] - (baseline ? V[s] : 0)
        if (baseline) V[s] += baselineAlpha * delta
        const p = softmaxRow(before, s, A)
        const scale = alpha * g.gamma ** i * delta
        for (let b = 0; b < A; b++) theta[s * A + b] += scale * ((b === a ? 1 : 0) - p[b])
      }
      return {
        theta: fromData(theta, g.theta.shape),
        V: fromData(V),
        gamma: g.gamma,
        updates: g.updates + 1,
        episode: [],
      }
    },
  }
}

// ── Reading a learnt table ───────────────────────────────────────────────────────────────────────────────────────────

/**
 * The path a policy follows from `start` when every move takes its most likely outcome (the first of equally likely
 * ones): stops at a terminal state, after `maxLength` states, at a state with action $-1$ or with no outcomes, or on
 * reaching a state already on the path, which it includes so that a loop closes (the last state then appears twice).
 * For drawing the greedy route on a grid.
 *
 * @param mdp The MDP's tables: `outcomes[s * actions + a]` and the terminal flags are read.
 * @param start The state the path starts from.
 * @param policy The action of every state (as `greedyPolicy` returns it), $-1$ where none is taken.
 * @param maxLength The most states the path may hold.
 * @returns The states visited, from `start`, as an int32 tensor.
 *
 * @example The route of a policy, and a loop closing
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * print('always right:', greedyPath(mdp, 0, [1, 1, 1, -1]))
 * print('right, then back:', greedyPath(mdp, 0, [1, 0, 1, -1]))
 */
export function greedyPath(
  mdp: MdpTables,
  start: number,
  policy: Tensor | ArrayLike<number>,
  maxLength = 4 * mdp.states,
): Tensor {
  const pol = 'shape' in policy ? policy.data : policy
  const path = [start]
  const seen = new Set(path)
  let s = start
  while (isActive(mdp, s) && path.length < maxLength) {
    const a = pol[s]
    if (a < 0) break
    const outs = mdp.outcomes[s * mdp.actions + a]
    if (!outs.length) break
    const o = outs.reduce((best, x) => (x.p > best.p ? x : best), outs[0])
    path.push(o.next)
    if (seen.has(o.next)) break
    seen.add(o.next)
    s = o.next
  }
  return fromData(Int32Array.from(path))
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const agent = definer<AgentInfo>('agent', 'gym/agents')
const tabular = { observation: 'discrete', action: 'discrete' } as const
const alphaParam = real(0.01, 1, { default: 0.5, label: 'α' })
const epsilonParam = real(0, 1, { default: 0.1, label: 'ε' })

agent(
  {
    key: 'qLearningAgent',
    name: 'Q-learning',
    summary: 'ε-greedy Q-learning: one off-policy TD update towards r + γ max Q per transition.',
    params: space({ learningRate: alphaParam, epsilon: epsilonParam, initialQ: real(-10, 10, { default: 0 }) }),
    requires: tabular,
    notes: ['q-learning'],
    random: true,
  },
  qLearningAgent,
)
agent(
  {
    key: 'sarsaAgent',
    name: 'SARSA',
    summary: 'On-policy TD control: each update bootstraps from the value of the action actually taken next.',
    params: space({ learningRate: alphaParam, epsilon: epsilonParam }),
    requires: tabular,
    notes: ['sarsa'],
    random: true,
  },
  sarsaAgent,
)
agent(
  {
    key: 'expectedSarsaAgent',
    name: 'Expected SARSA',
    summary: 'TD control bootstrapping from the expected action value under the ε-greedy policy.',
    params: space({ learningRate: alphaParam, epsilon: epsilonParam }),
    requires: tabular,
    notes: ['sarsa'],
    random: true,
  },
  expectedSarsaAgent,
)
agent(
  {
    key: 'tdControlAgent',
    name: 'One-step TD control',
    summary: 'Q-learning, SARSA or expected SARSA by `method`: ε-greedy TD control on discrete observations.',
    params: space({ learningRate: alphaParam, epsilon: epsilonParam }),
    requires: tabular,
    notes: ['q-learning', 'sarsa'],
    random: true,
  },
  tdControlAgent,
)
agent(
  {
    key: 'nStepSarsaAgent',
    name: 'n-step SARSA',
    summary: 'SARSA with n-step returns: n rewards, then the value of the action taken n steps later.',
    params: space({ n: int(1, 32, { default: 4 }), learningRate: alphaParam, epsilon: epsilonParam }),
    requires: tabular,
    notes: ['sarsa'],
    random: true,
  },
  nStepSarsaAgent,
)
agent(
  {
    key: 'monteCarloControlAgent',
    name: 'Monte Carlo control',
    summary: 'First-visit Monte Carlo control with an ε-soft policy: action values average the returns that followed.',
    params: space({ epsilon: epsilonParam }),
    requires: tabular,
    notes: ['reinforcement-learning'],
    random: true,
  },
  monteCarloControlAgent,
)
agent(
  {
    key: 'tdPredictionAgent',
    name: 'TD(0) prediction',
    summary: 'Follows a fixed policy (default uniformly random) and learns its state values by one-step TD updates.',
    params: space({ learningRate: real(0.001, 1, { default: 0.1, label: 'α' }) }),
    requires: tabular,
    notes: ['reinforcement-learning'],
    random: true,
  },
  tdPredictionAgent,
)
agent(
  {
    key: 'reinforceAgent',
    name: 'REINFORCE',
    summary: 'A tabular softmax policy improved by the score-function gradient of each episode’s returns.',
    params: space({ learningRate: real(0.001, 1, { default: 0.1, label: 'α' }), baseline: bool({ default: true }) }),
    requires: tabular,
    notes: ['reinforcement-learning'],
    random: true,
  },
  reinforceAgent,
)
