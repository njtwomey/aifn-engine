/**
 * Dynamic programming on a known MDP (Bellman, 1957; Sutton and Barto, 2018, ch. 4): exact policy evaluation, iterative
 * policy evaluation, value iteration and policy iteration, each traceable with the value table and greedy policy per
 * step, on any `MdpTables` (a `TabularMdp`, or an environment's tabular `model`). As agents, `valueIterationAgent` and
 * `policyIterationAgent` plan on the environment's model in `init` and then act greedily; they require a tabular model.
 *
 * Conventions, from `../mdp`: `outcomes[s * actions + a]` lists the outcomes of action $a$ in state $s$ (empty for an
 * illegal action, whose action value is $-\infty$); terminal states take no action and keep their `terminalValue`, so
 * every backup is $Q(s, a) = \sum p \, (r + \gamma V(s'))$. Policies are int32 actions per state, $-1$ at terminals,
 * and value tables start from the terminal values, with 0 at the active states.
 */

import { LinAlgError, solve } from 'aifn-compute/numerics/linalg'
import type { Agent, AgentInfo, EnvironmentShape, Status } from 'aifn-compute/foundation/contracts'
import { integers } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  greedyActions,
  isActive,
  legalActions,
  policyMatrix,
  qFromValues,
  type MdpTables,
  type PolicyInput,
} from '../mdp'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The greedy policy for an action-value table (states $\times$ actions), as int32 actions ($-1$ at terminals). Ties
 * within $10^{-9}$ go to the lowest action index.
 *
 * @param mdp The MDP's tables: its sizes and terminal flags are read.
 * @param Q The action values, $S \times A$ (a tensor, or $SA$ numbers row-major).
 * @returns The action of every state.
 *
 * @example The greedy actions of a table, ties to the lower index
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const Q = tensor([[0.729, 0.81], [0.729, 0.9], [0.5, 0.5], [0, 0]])
 * print('policy:', greedyPolicy(mdp, Q))
 */
export function greedyPolicy(mdp: MdpTables, Q: Tensor | ArrayLike<number>): Tensor {
  return fromData(greedyActions(mdp, 'shape' in Q ? Q.data : Q))
}

/**
 * $V(s) = \max_a Q(s, a)$ at active states and the terminal value elsewhere.
 *
 * @param mdp The MDP's tables: its sizes, terminal flags and terminal values are read.
 * @param Q The action values, $S \times A$ (a tensor, or $SA$ numbers row-major).
 * @returns $V$, one value per state.
 *
 * @example State values from action values
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const Q = tensor([[0.729, 0.81], [0.729, 0.9], [0.81, 1], [0, 0]])
 * print('V:', valuesFromQ(mdp, Q))
 */
export function valuesFromQ(mdp: MdpTables, Q: Tensor | ArrayLike<number>): Tensor {
  const q = 'shape' in Q ? Q.data : Q
  const V = Float64Array.from(mdp.terminalValue)
  for (let s = 0; s < mdp.states; s++) {
    if (!isActive(mdp, s)) continue
    let best = -Infinity
    for (let a = 0; a < mdp.actions; a++) best = Math.max(best, q[s * mdp.actions + a])
    V[s] = best
  }
  return fromData(V)
}

/**
 * The exact value of a policy: solve $(\Imat - \gamma \Pmat_\pi) \vvec = \rvec_\pi$ over the active states by one
 * linear solve, with terminal values fixed. With $\gamma = 1$ a policy that never terminates makes the system singular,
 * and the solve throws `LinAlgError` (as `policyIteration` relies on); a nearly singular one gives very large values.
 *
 * @param mdp The MDP's tables.
 * @param policy The policy: an action per state ($-1$ where none is taken) or $S \times A$ probabilities row-major.
 *   Throws `ShapeError` for any other length.
 * @returns $\vvec_\pi$, one value per state.
 *
 * @example Always moving right, and a uniformly random walk
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * print('always right:', evaluatePolicy(mdp, [1, 1, 1, -1]))
 * print('uniform:', evaluatePolicy(mdp, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]))
 *
 * @example A policy that never terminates has no value without discounting
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const undiscounted = { ...mdp, gamma: 1 }
 * try {
 *   evaluatePolicy(undiscounted, [0, 0, 0, -1])
 * } catch (e) {
 *   print('always left:', e.name)
 * }
 * print('always right:', evaluatePolicy(undiscounted, [1, 1, 1, -1]))
 */
export function evaluatePolicy(mdp: MdpTables, policy: PolicyInput): Tensor {
  const { states: S, actions: A, gamma } = mdp
  const pi = policyMatrix(mdp, policy)
  const active = Array.from({ length: S }, (_, s) => s).filter((s) => isActive(mdp, s))
  const row = new Map(active.map((s, i) => [s, i]))
  const n = active.length
  const M = new Float64Array(n * n)
  const rhs = new Float64Array(n)
  active.forEach((s, i) => {
    M[i * n + i] += 1
    for (let a = 0; a < A; a++) {
      const w = pi[s * A + a]
      if (w === 0) continue
      for (const o of mdp.outcomes[s * A + a]) {
        rhs[i] += w * o.p * o.reward
        const j = row.get(o.next)
        if (j === undefined) rhs[i] += w * o.p * gamma * mdp.terminalValue[o.next]
        else M[i * n + j] -= w * o.p * gamma
      }
    }
  })
  const V = Float64Array.from(mdp.terminalValue)
  if (n > 0) {
    const x = solve(fromData(M, [n, n]), fromData(rhs)).data
    active.forEach((s, i) => (V[s] = x[i]))
  }
  return fromData(V)
}

/** A state of value iteration or policy evaluation: the value table, action values and greedy policy. */
export interface ValueState extends Status {
  /** $V_k$, length $S$. */
  V: Tensor
  /** $Q_k(s, a) = \sum p \, (r + \gamma V_k(s'))$, the one-step backup of $V_k$, $S \times A$. */
  Q: Tensor
  /** The greedy policy for $V_k$ (int32, $-1$ at terminals). */
  policy: Tensor
  /**
   * The Bellman residual $\max_s \lvert (T V_k)(s) - V_k(s) \rvert$: the largest change the next sweep will make.
   * Defined from $k = 0$, so it can be plotted on a log scale; for value iteration the error
   * $\lVert V_k - V^* \rVert_\infty$ is at most $\gamma/(1 - \gamma)$ times it.
   */
  residual: number
  /** Sweeps done. */
  t: number
  /** The residual is below the tolerance. */
  converged: boolean
}

/** A sweep's update of one state: its new value from the action values $Q_k$ ($S \times A$) and the state $s$. */
type Backup = (Q: Float64Array, s: number) => number

/**
 * The state of a sweep algorithm at a value table: its backup $Q$, greedy policy and residual.
 *
 * @param mdp The MDP's tables.
 * @param V The value table $V_k$, one per state; kept in the state (not copied).
 * @param t The number of sweeps done.
 * @param tolerance The residual below which the state is `converged`.
 * @param backup The update the next sweep applies, for the residual.
 * @returns The state, flagged `diverged` when the residual is not finite.
 */
function valueState(mdp: MdpTables, V: Float64Array, t: number, tolerance: number, backup: Backup): ValueState {
  const Q = qFromValues(mdp, V)
  let residual = 0
  for (let s = 0; s < mdp.states; s++)
    if (isActive(mdp, s)) residual = Math.max(residual, Math.abs(backup(Q, s) - V[s]))
  return {
    V: fromData(V),
    Q: fromData(Q, [mdp.states, mdp.actions]),
    policy: fromData(greedyActions(mdp, Q)),
    residual,
    t,
    converged: residual < tolerance,
    diverged: !Number.isFinite(residual),
  }
}

/**
 * Synchronous sweeps $V_{k+1}(s) = \text{backup}(Q_k, s)$ over the active states, from $V_0 = 0$ with terminal values
 * fixed.
 *
 * @param mdp The MDP's tables.
 * @param name The algorithm's readable name.
 * @param tolerance The residual below which a state is `converged`, which stops `run`.
 * @param backup The update of one state from the current action values.
 * @returns The algorithm, which takes no start.
 */
function sweeps(mdp: MdpTables, name: string, tolerance: number, backup: Backup): Algorithm<void, ValueState> {
  return {
    name,
    init: () => valueState(mdp, Float64Array.from(mdp.terminalValue), 0, tolerance, backup),
    step(s) {
      const Q = s.Q.data as Float64Array
      const V = Float64Array.from(s.V.data)
      for (let st = 0; st < mdp.states; st++) if (isActive(mdp, st)) V[st] = backup(Q, st)
      return valueState(mdp, V, s.t + 1, tolerance, backup)
    },
  }
}

/**
 * Value iteration: $V_{k+1}(s) = \max_a \sum p \, (r + \gamma V_k(s'))$ for every active state at once (a synchronous
 * sweep), from $V_0 = 0$ with terminal values fixed. The residual falls at least as fast as $\gamma^k$; done when it is
 * below `tolerance`. Run it with `run(valueIteration(mdp), undefined, steps)`.
 *
 * @param mdp The MDP's tables.
 * @param options When to stop.
 * @param options.tolerance The Bellman residual below which the state is `converged`.
 * @returns The algorithm, whose state carries $V_k$, $Q_k$, the greedy policy and the residual.
 *
 * @example Value iteration on a corridor, against its known optimum
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const state = run(valueIteration(mdp), undefined, 1000)
 * print('V =', state.V)
 * print('optimum:', [0.9 ** 2, 0.9, 1, 0])
 * print('policy:', state.policy)
 * print('sweeps:', state.t, 'converged:', state.converged)
 *
 * @example The residual after each sweep
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * for (const k of [0, 1, 2, 3]) print(`after ${k} sweeps:`, run(valueIteration(mdp), undefined, k).residual)
 */
export function valueIteration(
  mdp: MdpTables,
  { tolerance = 1e-10 }: { tolerance?: number } = {},
): Algorithm<void, ValueState> {
  const A = mdp.actions
  return sweeps(mdp, 'value iteration', tolerance, (Q, s) => {
    let best = -Infinity
    for (let a = 0; a < A; a++) best = Math.max(best, Q[s * A + a])
    return best
  })
}

/**
 * Iterative policy evaluation: $V_{k+1}(s) = \sum_a \pi(a \mid s) \sum p \, (r + \gamma V_k(s'))$, synchronous sweeps
 * from $V_0 = 0$. The `policy` field of the state is the greedy policy with respect to $V_k$ (the improvement step
 * would pick it), not the policy evaluated.
 *
 * @param mdp The MDP's tables.
 * @param policy The policy evaluated: an action per state ($-1$ where none is taken) or $S \times A$ probabilities
 *   row-major. Throws `ShapeError` for any other length.
 * @param options When to stop.
 * @param options.tolerance The residual below which the state is `converged`.
 * @returns The algorithm, whose state carries $V_k$ and the residual.
 *
 * @example Sweeps converge to the exact value
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const walk = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]
 * const state = run(policyEvaluation(mdp, walk), undefined, 1000)
 * print('by sweeps:', state.V, 'after', state.t)
 * print('exact:', evaluatePolicy(mdp, walk))
 * print('greedy with respect to it:', state.policy)
 */
export function policyEvaluation(
  mdp: MdpTables,
  policy: PolicyInput,
  { tolerance = 1e-10 }: { tolerance?: number } = {},
): Algorithm<void, ValueState> {
  const pi = policyMatrix(mdp, policy)
  const A = mdp.actions
  return sweeps(mdp, 'policy evaluation', tolerance, (Q, s) => {
    let v = 0
    for (let a = 0; a < A; a++) v += pi[s * A + a] * Q[s * A + a]
    return v
  })
}

/** A state of policy iteration. */
export interface PolicyIterationState extends Status {
  /** The current policy (int32, $-1$ at terminals). */
  policy: Tensor
  /** Its exact value $v_\pi$, one per state. */
  V: Tensor
  /** Its action values $q_\pi$, $S \times A$. */
  Q: Tensor
  /** States whose action changed in the last improvement. */
  changed: number
  /** Improvements done. */
  t: number
  /** True when greedy improvement no longer changes the policy: it is stable, hence optimal. */
  converged: boolean
}

/**
 * Policy iteration (Howard, 1960): evaluate the policy exactly (`evaluatePolicy`), then improve it greedily (keeping
 * the current action on ties within $10^{-9}$), until the policy is stable. Starts from the first legal action in every
 * state. Converges in finitely many iterations; the step that finds the policy stable counts as one. With
 * $\gamma = 1$, a policy that never reaches a terminal state is evaluated under $\gamma = 1 - 10^{-9}$.
 *
 * @param mdp The MDP's tables.
 * @returns The algorithm, whose state carries the policy, its values and the number of states changed.
 *
 * @example Policy iteration on a corridor
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const state = run(policyIteration(mdp), undefined, 100)
 * print('policy:', state.policy)
 * print('V =', state.V)
 * print('improvements:', state.t, 'converged:', state.converged)
 */
export function policyIteration(mdp: MdpTables): Algorithm<void, PolicyIterationState> {
  const evaluate = (policy: Int32Array, changed: number, t: number): PolicyIterationState => {
    let tables = mdp
    let V: Tensor
    try {
      V = evaluatePolicy(mdp, fromData(policy))
    } catch (e) {
      // An improper policy under γ = 1 (one that never terminates) has no finite value: evaluate it under γ just below
      // 1, which makes its value very low, so the improvement step leaves it.
      if (!(e instanceof LinAlgError) || mdp.gamma < 1) throw e
      tables = { ...mdp, gamma: 1 - 1e-9 }
      V = evaluatePolicy(tables, fromData(policy))
    }
    return {
      policy: fromData(policy),
      V,
      Q: fromData(qFromValues(tables, V.data), [mdp.states, mdp.actions]),
      changed,
      t,
      converged: false,
    }
  }
  return {
    name: 'policy iteration',
    init: () =>
      evaluate(
        Int32Array.from({ length: mdp.states }, (_, s) => (isActive(mdp, s) ? legalActions(mdp, s)[0] : -1)),
        0,
        0,
      ),
    step(s) {
      const current = s.policy.data as Int32Array
      const next = greedyActions(mdp, s.Q.data, current)
      let changed = 0
      for (let st = 0; st < mdp.states; st++) if (next[st] !== current[st]) changed++
      if (changed === 0) return { ...s, changed: 0, t: s.t + 1, converged: true }
      return evaluate(next, changed, s.t + 1)
    },
  }
}

// ── Planners as agents ───────────────────────────────────────────────────────────────────────────────────────────────

/** A planning agent's state: the optimal action values it planned and their greedy policy. */
export interface PlannerState {
  /** $Q^*(s, a)$, states $\times$ actions ($-\infty$ for illegal actions). */
  Q: Tensor
  /** $V^*(s)$, one per state. */
  V: Tensor
  /** The greedy policy (int32, $-1$ at terminals). */
  policy: Tensor
  /** Sweeps (value iteration) or improvements (policy iteration) the plan took. */
  iterations: number
}

/**
 * The environment's tabular model. Throws `DomainError`, naming the agent, when it has none.
 *
 * @param env The environment the agent is initialised for.
 * @param who The agent's name, for the error message.
 * @returns The model's tables.
 */
function tabularModel(env: EnvironmentShape, who: string): MdpTables {
  const m = env.model
  if (m?.kind !== 'tabular') throw new DomainError('tabularModel', `${who} needs an environment with a tabular model`)
  return m
}

/**
 * An agent that plans with `plan` on the environment's tabular model once, in `init`, and then acts greedily on the
 * planned $Q^*$ (observations are the model's state indices), breaking ties within $10^{-9}$ among the best legal
 * actions at random; `greedy` takes the first strictly best. It learns nothing.
 *
 * @param name The agent's readable name, also used in the error when the environment has no tabular model.
 * @param plan The planner: from the model's tables to the `PlannerState`.
 * @returns The agent.
 */
function plannerAgent(name: string, plan: (mdp: MdpTables) => PlannerState): Agent<PlannerState, number, number> {
  return {
    name,
    init: (env) => plan(tabularModel(env, name)),
    act(g, o, stream, legal) {
      const A = g.Q.shape[1]
      const scores = Float64Array.from(g.Q.data.subarray(o * A, (o + 1) * A))
      const allowed = legal ?? Array.from({ length: A }, (_, a) => a)
      const best = Math.max(...allowed.map((a) => scores[a]))
      const ties = allowed.filter((a) => scores[a] >= best - 1e-9)
      const probabilities = new Float64Array(A)
      for (const a of ties) probabilities[a] = 1 / ties.length
      return { action: ties.length === 1 ? ties[0] : ties[integers(stream, ties.length)], scores, probabilities }
    },
    greedy(g, o, legal) {
      const A = g.Q.shape[1]
      const allowed = legal ?? Array.from({ length: A }, (_, a) => a)
      return allowed.reduce((best, a) => (g.Q.data[o * A + a] > g.Q.data[o * A + best] ? a : best), allowed[0])
    },
    learn: (g) => g,
  }
}

/**
 * Value iteration on the environment's tabular model (to `tolerance`, at most 100000 sweeps) in `init`, then the
 * greedy action of $Q^*$, ties broken at random. Throws `DomainError` at `init` when the environment has no tabular
 * model.
 *
 * @param options When planning stops.
 * @param options.tolerance The Bellman residual at which value iteration stops.
 * @returns The agent, named `'value iteration'`.
 *
 * @example A planner acts optimally from its first step
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = valueIterationAgent()
 * const g = agent.init({ ...env, model: { kind: 'tabular', ...mdp } }, stream(0))
 * print('V* =', g.V)
 * print('sweeps:', g.iterations)
 * print('action in state 0:', agent.act(g, 0, stream(1)).action)
 */
export function valueIterationAgent({ tolerance = 1e-10 }: { tolerance?: number } = {}): Agent<
  PlannerState,
  number,
  number
> {
  return plannerAgent('value iteration', (mdp) => {
    const s = run(valueIteration(mdp, { tolerance }), undefined, 100_000)
    return { Q: s.Q, V: s.V, policy: s.policy, iterations: s.t }
  })
}

/**
 * Policy iteration on the environment's tabular model (at most 1000 improvements) in `init`, then the greedy action
 * of the optimal $Q$, ties broken at random. Throws `DomainError` at `init` when the environment has no tabular model.
 *
 * @returns The agent, named `'policy iteration'`.
 *
 * @example The planned policy and its values
 * // A corridor of states 0 to 3 (3 terminal): action 1 moves right, 0 left; reaching state 3 pays 1.
 * const outcomes = []
 * for (let s = 0; s < 4; s++)
 *   for (let a = 0; a < 2; a++) {
 *     const next = a === 1 ? s + 1 : Math.max(0, s - 1)
 *     outcomes.push(s === 3 ? [] : [{ p: 1, next, reward: next === 3 ? 1 : 0 }])
 *   }
 * const terminal = Uint8Array.from([0, 0, 0, 1])
 * const mdp = { states: 4, actions: 2, outcomes, terminal, terminalValue: new Float64Array(4), gamma: 0.9 }
 * const env = { observation: { kind: 'discrete', n: 4 }, action: { kind: 'discrete', n: 2 }, gamma: 0.9 }
 * const agent = policyIterationAgent()
 * const g = agent.init({ ...env, model: { kind: 'tabular', ...mdp } }, stream(0))
 * print('policy:', g.policy)
 * print('V =', g.V)
 * print('improvements:', g.iterations)
 */
export function policyIterationAgent(): Agent<PlannerState, number, number> {
  return plannerAgent('policy iteration', (mdp) => {
    const s = run(policyIteration(mdp), undefined, 1000)
    return { Q: s.Q, V: s.V, policy: s.policy, iterations: s.t }
  })
}

const agent = definer<AgentInfo>('agent', 'gym/agents')
const planner = { observation: 'discrete', action: 'discrete', model: 'tabular' } as const

agent(
  {
    key: 'valueIterationAgent',
    name: 'Value iteration (planner)',
    summary:
      'Solves the environment’s tabular model by value iteration, then acts greedily on Q*: the optimal baseline.',
    params: space({ tolerance: real(1e-12, 1e-2, { default: 1e-10, scale: 'log' }) }),
    requires: planner,
    notes: ['value-iteration'],
  },
  valueIterationAgent,
)
agent(
  {
    key: 'policyIterationAgent',
    name: 'Policy iteration (planner)',
    summary: 'Solves the environment’s tabular model by policy iteration, then acts greedily on the optimal Q.',
    params: space({}),
    requires: planner,
    notes: ['policy-iteration'],
  },
  policyIterationAgent,
)
