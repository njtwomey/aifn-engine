/**
 * Tabular learning agents (Sutton and Barto, 2018, chs. 5–7 and 13) on discrete observations and actions: Q-learning,
 * SARSA and expected SARSA (`tdControlAgent`), n-step SARSA, Monte Carlo control, TD(0) prediction of a fixed policy
 * and REINFORCE with a tabular softmax policy. Each is an `Agent`: `act` draws from its stream, `learn` is a pure
 * update from one transition. They need no model of the environment, only its observation and action domains.
 *
 * Conventions. The environment folds a terminal state's value into the reward of arriving there, so a `terminated`
 * transition never bootstraps; a `truncated` one does, from the ε-greedy expectation at `next` for the on-policy
 * methods (the next action is never taken). When the environment masks actions, `act` chooses among the legal ones
 * and maxima and expectations at `next` run over `nextLegal`. Learners that need what comes after a transition buffer
 * it in their state: SARSA one transition (it needs the next action), n-step SARSA n of them, Monte Carlo control and
 * REINFORCE the whole episode, updated when the episode ends.
 */

import type { Agent, AgentInfo, Decision, EnvironmentShape, Transition } from 'aifn-compute/foundation/contracts'
import { integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, domainSize, int, real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { isActive, type MdpTables, type PolicyInput } from '../mdp'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Shared pieces ────────────────────────────────────────────────────────────────────────────────────────────────────

/** The number of observations and actions of an environment with discrete domains. */
function sizes(env: EnvironmentShape): { S: number; A: number } {
  if (env.observation.kind !== 'discrete' || env.action.kind !== 'discrete')
    throw new DomainError('sizes', 'tabular agents need discrete observations and actions')
  return { S: domainSize(env.observation), A: domainSize(env.action) }
}

/** The actions to choose among: the legal ones, or all A. */
const choices = (A: number, legal?: readonly number[]) => legal ?? Array.from({ length: A }, (_, a) => a)

/**
 * ε-greedy on Q(s, ·) over the allowed actions: with probability ε a uniform action, else a greedy one with ties broken
 * uniformly at random (so an untrained agent does not always go one way). Reports the action values and probabilities.
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

/** The first action with the largest Q(s, a) among the allowed ones: the deterministic greedy action. */
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

/** The mean over observations of max_a Q(o, a): the agent's value estimate, a training-curve scalar. */
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

/** max_a Q(s, a) over the allowed actions. */
function maxQ(Q: ArrayLike<number>, s: number, A: number, legal?: readonly number[]): number {
  let best = -Infinity
  for (const a of choices(A, legal)) best = Math.max(best, Q[s * A + a])
  return best
}

/** E_π[Q(s, ·)] under the ε-greedy policy over the allowed actions: ε times the mean plus (1 − ε) times the maximum. */
function expectedQ(Q: ArrayLike<number>, s: number, A: number, eps: number, legal?: readonly number[]): number {
  const allowed = choices(A, legal)
  let mean = 0
  for (const a of allowed) mean += Q[s * A + a] / allowed.length
  return eps * mean + (1 - eps) * maxQ(Q, s, A, legal)
}

/** One step of a buffered episode. */
interface Visit {
  s: number
  a: number
  r: number
}

/** Options shared by the value-based agents. */
export interface TabularOptions {
  /** The step size α. Default 0.5. */
  learningRate?: number
  /** The exploration rate ε of the ε-greedy policy. Default 0.1. */
  epsilon?: number
  /** The discount; default the environment's. */
  gamma?: number
  /** The initial action value (optimistic values encourage exploration). Default 0. */
  initialQ?: number
}

/** The state of a value-based agent: the action values and the discount it learns with. */
export interface TabularAgentState {
  /** Q(o, a), observations × actions. */
  Q: Tensor
  gamma: number
  /** Transitions learnt from. */
  updates: number
}

const qState = (env: EnvironmentShape, gamma: number | undefined, initialQ: number): TabularAgentState => {
  const { S, A } = sizes(env)
  return { Q: fromData(new Float64Array(S * A).fill(initialQ), [S, A]), gamma: gamma ?? env.gamma, updates: 0 }
}

// ── One-step TD control ──────────────────────────────────────────────────────────────────────────────────────────────

/** The one-step TD control methods. */
export type TdMethod = 'sarsa' | 'q-learning' | 'expected-sarsa'

/** A TD control agent's state; SARSA also holds the transition whose update waits for the next action. */
export interface TdAgentState extends TabularAgentState {
  pending: Transition<number, number> | null
  /** Episodes completed, for the decay of ε. */
  episodes: number
}

/**
 * One-step TD control with an ε-greedy policy: Q(s, a) ← Q(s, a) + α (r + γ · bootstrap − Q(s, a)). Q-learning
 * bootstraps from max_a′ Q(s′, a′) (off-policy; Watkins, 1989), expected SARSA from the expectation under the ε-greedy
 * policy (van Seijen et al., 2009), SARSA from Q(s′, a′) of the action actually taken next (on-policy), so SARSA holds
 * each transition until the next one arrives.
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
  /** ε decays as ε / (1 + e / `epsilonDecay`) after e episodes, so exploration fades. Default ∞ (no decay). */
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

/** Options of the TD control agents: those of every value-based agent, and the decay of ε. */
export type TdControlOptions = TabularOptions & { epsilonDecay?: number }

/** Q-learning (see `tdControlAgent`). */
export const qLearningAgent = (options: TdControlOptions = {}) => tdControlAgent({ ...options, method: 'q-learning' })
/** SARSA (see `tdControlAgent`). */
export const sarsaAgent = (options: TdControlOptions = {}) => tdControlAgent({ ...options, method: 'sarsa' })
/** Expected SARSA (see `tdControlAgent`). */
export const expectedSarsaAgent = (options: TdControlOptions = {}) =>
  tdControlAgent({ ...options, method: 'expected-sarsa' })

// ── n-step SARSA ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** An n-step SARSA agent's state: the visits whose n-step targets are not yet complete. */
export interface NStepAgentState extends TabularAgentState {
  window: Visit[]
}

/**
 * n-step SARSA (Sutton and Barto, 2018, §7.2): the target of (s_τ, a_τ) is the n-step return
 * G = Σ_{i<n} γⁱ r_{τ+i+1} + γⁿ Q(s_{τ+n}, a_{τ+n}), shortened at the end of the episode (with no bootstrap after a
 * terminal state, and the ε-greedy expectation after a truncation). Each visit waits in a window until its target is
 * complete.
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
  /** First visits of each (o, a), observations × actions. */
  counts: Tensor
  episode: Visit[]
}

/**
 * On-policy first-visit Monte Carlo control with ε-soft policies (Sutton and Barto, 2018, §5.4): when an episode ends,
 * Q(s, a) moves towards the return that followed the first visit of (s, a), by sample averaging (default) or a
 * constant step size α. A truncated episode's returns stop at the truncation.
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
  /** V(o), one per observation. */
  V: Tensor
  /** π(a | o), observations × actions. */
  policy: Tensor
  gamma: number
  updates: number
}

/**
 * The policy as π(a | s), S × A: a deterministic policy is an action per state (−1 where no action is taken), a
 * stochastic one S × A probabilities.
 */
function policyTable(policy: PolicyInput, S: number, A: number): Float64Array {
  const v = 'shape' in policy ? Array.from(policy.data) : [...policy]
  if (v.length === S * A) return Float64Array.from(v)
  if (v.length !== S) throw new ShapeError('policy', `policy: expected ${S} actions or ${S} × ${A} probabilities`)
  const pi = new Float64Array(S * A)
  for (let s = 0; s < S; s++) if (v[s] >= 0) pi[s * A + v[s]] = 1
  return pi
}

/** A draw from row s of π (S × A). */
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
 * (deterministic: an action per state; stochastic: states × actions; default uniformly random) and after every
 * transition V(s) ← V(s) + α (r + γ V(s′) − V(s)), with V(s′) = 0 after a terminal transition.
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

/** A REINFORCE agent's state: softmax preferences θ, the baseline V and the episode so far. */
export interface ReinforceState {
  /** θ(o, a), observations × actions: π(a | o) ∝ exp θ(o, a). */
  theta: Tensor
  /** The baseline b(o) (zeros without a baseline). */
  V: Tensor
  gamma: number
  updates: number
  episode: Visit[]
}

/** softmax(θ(s, ·)) over the allowed actions (0 elsewhere). */
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
 * REINFORCE (Williams, 1992, Machine Learning 8) with a tabular softmax policy π(a|s) ∝ exp θ(s, a): when an episode
 * ends, θ(s_t, ·) += α γᵗ (G_t − b(s_t)) (e_{a_t} − π(·|s_t)), the score-function gradient, with every step's π taken
 * from the θ that generated the episode. With `baseline`, b is a state-value estimate learned alongside at
 * `baselineLearningRate` (Sutton and Barto, 2018, §13.4); without, b = 0.
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
 * The path a policy follows from `start` when every move takes its most likely outcome: stops at a terminal state,
 * after `maxLength` states, or on reaching a state already on the path, which it includes so that a loop closes (the
 * last state then appears twice). For drawing the greedy route on a grid.
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
