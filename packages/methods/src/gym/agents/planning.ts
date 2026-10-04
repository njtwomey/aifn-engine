/**
 * Dynamic programming on a known MDP (Bellman, 1957; Sutton and Barto, 2018, ch. 4): exact policy evaluation, iterative
 * policy evaluation, value iteration and policy iteration, each traceable with the value table and greedy policy per
 * step, on any `MdpTables` (a `TabularMdp`, or an environment's tabular `model`). As agents, `valueIterationAgent` and
 * `policyIterationAgent` plan on the environment's model in `init` and then act greedily; they require a tabular model.
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

/** The greedy policy for an action-value table (states × actions), as int32 actions (−1 at terminals). */
export function greedyPolicy(mdp: MdpTables, Q: Tensor | ArrayLike<number>): Tensor {
  return fromData(greedyActions(mdp, 'shape' in Q ? Q.data : Q))
}

/** V(s) = max_a Q(s, a) at active states and the terminal value elsewhere. */
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
 * The exact value of a policy: solve (I − γ P_π) v = r_π over the active states, with terminal values fixed. With
 * γ = 1 a policy that never terminates makes the system singular; the result is then non-finite.
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
  /** V_k, length S. */
  V: Tensor
  /** Q_k(s, a) = Σ p (r + γ V_k(s′)), the one-step backup of V_k, S × A. */
  Q: Tensor
  /** The greedy policy for V_k (int32, −1 at terminals). */
  policy: Tensor
  /**
   * The Bellman residual max_s |(T V_k)(s) − V_k(s)|: the largest change the next sweep will make. Defined from k = 0,
   * so it can be plotted on a log scale; the error ‖V_k − V*‖∞ is at most γ/(1 − γ) times it.
   */
  residual: number
  /** Sweeps done. */
  t: number
  /** The residual is below the tolerance. */
  converged: boolean
}

type Backup = (Q: Float64Array, s: number) => number

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

/** Synchronous sweeps V_{k+1}(s) = backup(Q_k, s) over the active states, from V₀ = 0 with terminal values fixed. */
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
 * Value iteration: V_{k+1}(s) = max_a Σ p (r + γ V_k(s′)) for every active state at once (a synchronous sweep), from
 * V₀ = 0 with terminal values fixed. The residual falls at least as fast as γᵏ; done when it is below `tolerance`.
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
 * Iterative policy evaluation: V_{k+1}(s) = Σ_a π(a|s) Σ p (r + γ V_k(s′)), synchronous sweeps from V₀ = 0. The
 * `policy` field of the state is the greedy policy with respect to V_k (the improvement step would pick it).
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
  /** The current policy (int32, −1 at terminals). */
  policy: Tensor
  /** Its exact value v_π and action values q_π. */
  V: Tensor
  Q: Tensor
  /** States whose action changed in the last improvement. */
  changed: number
  /** Improvements done. */
  t: number
  /** True when greedy improvement no longer changes the policy: it is stable, hence optimal. */
  converged: boolean
}

/**
 * Policy iteration (Howard, 1960): evaluate the policy exactly, then improve it greedily (keeping the current action
 * on ties), until the policy is stable. Starts from the first legal action in every state. Converges in finitely
 * many iterations. With γ = 1, a policy that never reaches a terminal state is evaluated under γ = 1 − 10⁻⁹.
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
  /** Q*(s, a), states × actions (−∞ for illegal actions). */
  Q: Tensor
  /** V*(s). */
  V: Tensor
  /** The greedy policy (int32, −1 at terminals). */
  policy: Tensor
  /** Sweeps (value iteration) or improvements (policy iteration) the plan took. */
  iterations: number
}

/** The environment's tabular model, or an error naming the agent. */
function tabularModel(env: EnvironmentShape, who: string): MdpTables {
  const m = env.model
  if (m?.kind !== 'tabular') throw new DomainError('tabularModel', `${who} needs an environment with a tabular model`)
  return m
}

/**
 * An agent that plans with `plan` on the environment's tabular model once, in `init`, and then acts greedily on the
 * planned Q* (observations are the model's state indices), breaking ties among the best legal actions at random. It
 * learns nothing.
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

/** Value iteration on the environment's tabular model (to `tolerance`), then greedy action (see `plannerAgent`). */
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

/** Policy iteration on the environment's tabular model, then greedy action (see `plannerAgent`). */
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
