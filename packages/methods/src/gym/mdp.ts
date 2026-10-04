/**
 * Finite Markov decision processes as plain data, shared by the gridworld environments (`./environments`) and the
 * planning and tabular agents (`./agents`): the `TabularMdp` type and constructor, outcome sampling, legal actions,
 * one-step backups, policy helpers and the optimal values by value iteration.
 *
 * Conventions. States are integers; a grid cell (x, y) is state y · width + x with y = 0 at the bottom row. Actions on
 * grids are up, right, down, left (0–3). Terminal states take no action and have a fixed value `terminalValue` (0
 * unless an environment pays an exit reward there), so every backup is r + γ V(s′) with V(s′) = terminalValue(s′) at a
 * terminal s′, and an episode's return adds γ^T · terminalValue on arrival.
 */

import type { Outcome, TabularModel } from 'aifn-compute/foundation/contracts'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

export type { Outcome } from 'aifn-compute/foundation/contracts'

/** What a grid cell is, for drawing and for the rules. */
export type CellKind = 'open' | 'wall' | 'start' | 'goal' | 'trap' | 'hole' | 'cliff' | 'terminal'

/** A finite MDP. */
export interface TabularMdp {
  name: string
  states: number
  actions: number
  /**
   * `outcomes[s * actions + a]`: the distribution over (next state, reward). Empty for terminal states and walls, and
   * for an action that is not legal in an active state.
   */
  outcomes: Outcome[][]
  /** The start state of every episode. */
  start: number
  /** 1 for states where no action is taken (terminals and walls). */
  terminal: Uint8Array
  /** The fixed value of each terminal state (0 elsewhere). */
  terminalValue: Float64Array
  gamma: number
  actionNames: string[]
  /** Grid layout, when the MDP is a grid. */
  grid?: { width: number; height: number; kinds: CellKind[]; actionVectors: [number, number][] }
}

/** Up, right, down, left as (dx, dy) with y pointing up. */
export const GRID_ACTIONS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]
export const GRID_ACTION_NAMES = ['up', 'right', 'down', 'left']

/** The state of grid cell (x, y). */
export const cellState = (width: number, x: number, y: number): number => y * width + x
/** The (x, y) cell of a grid state. */
export const stateCell = (width: number, s: number): [number, number] => [s % width, Math.floor(s / width)]

/** The tables planners read: what a `TabularMdp` and an environment's tabular model have in common. */
export type MdpTables = Pick<
  TabularModel<unknown>,
  'states' | 'actions' | 'outcomes' | 'terminal' | 'terminalValue' | 'gamma'
>

/** States where an action is taken. */
export const isActive = (mdp: MdpTables, s: number): boolean => !mdp.terminal[s]

/** The actions legal in active state s: those with outcomes (all of them in a grid). */
export function legalActions(mdp: MdpTables, s: number): number[] {
  const out: number[] = []
  for (let a = 0; a < mdp.actions; a++) if (mdp.outcomes[s * mdp.actions + a].length > 0) out.push(a)
  return out
}

/** True when some active state has an illegal action (an empty outcome list), so environments must mask actions. */
export function hasIllegalActions(mdp: MdpTables): boolean {
  for (let s = 0; s < mdp.states; s++)
    if (isActive(mdp, s))
      for (let a = 0; a < mdp.actions; a++) if (mdp.outcomes[s * mdp.actions + a].length === 0) return true
  return false
}

/** Sample an outcome of action a in state s from the stream r (one uniform draw). */
export function sampleOutcome(r: Stream, mdp: MdpTables, s: number, a: number): Outcome {
  const outs = mdp.outcomes[s * mdp.actions + a]
  let u = uniform(r)
  for (const o of outs) {
    if (u < o.p) return o
    u -= o.p
  }
  return outs[outs.length - 1]
}

/**
 * A general finite MDP from dense arrays: P[s][a][s′] transition probabilities and R[s][a][s′] rewards (or R[s][a]
 * expected rewards), with optional terminal states. An all-zero row P[s][a] at an active state makes action a illegal
 * there (its outcome list is empty).
 */
export function tabularMdp(options: {
  transitions: readonly (readonly (readonly number[])[])[]
  rewards: readonly (readonly (readonly number[] | number)[])[]
  gamma: number
  start?: number
  terminal?: readonly number[]
  terminalValue?: readonly number[]
  name?: string
  actionNames?: string[]
}): TabularMdp {
  const P = options.transitions
  const S = P.length
  const A = P[0].length
  const terminal = new Uint8Array(S)
  for (const s of options.terminal ?? []) terminal[s] = 1
  const outcomes: Outcome[][] = []
  for (let s = 0; s < S; s++)
    for (let a = 0; a < A; a++) {
      if (terminal[s]) {
        outcomes.push([])
        continue
      }
      const r = options.rewards[s][a]
      outcomes.push(
        P[s][a].flatMap((p, t) => (p > 0 ? [{ p, next: t, reward: typeof r === 'number' ? r : r[t] }] : [])),
      )
    }
  return {
    name: options.name ?? 'MDP',
    states: S,
    actions: A,
    outcomes,
    start: options.start ?? 0,
    terminal,
    terminalValue: Float64Array.from({ length: S }, (_, s) => (terminal[s] ? (options.terminalValue?.[s] ?? 0) : 0)),
    gamma: options.gamma,
    actionNames: options.actionNames ?? Array.from({ length: A }, (_, a) => `a${a}`),
  }
}

// ── Policies (shared by planning and learning) ─────────────────────────────────────────────────────────────────────

/** A policy: a deterministic action per state (int32, −1 at terminals) or a stochastic one, states × actions. */
export type PolicyInput = Tensor | readonly number[]

/**
 * The greedy action per state (lowest index on ties within 1e-9, or `prefer[s]` if it is tied), −1 at terminals.
 * Illegal actions have Q = −∞ (see `qFromValues`), so they are never greedy.
 */
export function greedyActions(mdp: MdpTables, Q: ArrayLike<number>, prefer?: ArrayLike<number>): Int32Array {
  const { states: S, actions: A } = mdp
  const out = new Int32Array(S).fill(-1)
  for (let s = 0; s < S; s++) {
    if (!isActive(mdp, s)) continue
    let best = -Infinity
    for (let a = 0; a < A; a++) best = Math.max(best, Q[s * A + a])
    const keep = prefer?.[s]
    if (keep !== undefined && keep >= 0 && Q[s * A + keep] >= best - 1e-9) out[s] = keep
    else
      for (let a = 0; a < A; a++)
        if (Q[s * A + a] >= best - 1e-9) {
          out[s] = a
          break
        }
  }
  return out
}

/** π(a | s) as a dense states × actions array from a deterministic or stochastic policy. */
export function policyMatrix(mdp: MdpTables, policy: PolicyInput): Float64Array {
  const { states: S, actions: A } = mdp
  const v = 'shape' in policy ? Array.from(policy.data) : [...policy]
  if (v.length === S * A) return Float64Array.from(v)
  if (v.length !== S) throw new ShapeError('policy', `policy: expected ${S} actions or ${S} × ${A} probabilities`)
  const pi = new Float64Array(S * A)
  for (let s = 0; s < S; s++) if (isActive(mdp, s) && v[s] >= 0) pi[s * A + v[s]] = 1
  return pi
}

// ── Backups and optimal values ───────────────────────────────────────────────────────────────────────────────────

/** Q(s, a) = Σ_{s′} p (r + γ V(s′)) for every active state and legal action; −∞ for illegal actions, 0 at terminals. */
export function qFromValues(mdp: MdpTables, V: ArrayLike<number>): Float64Array {
  const { states: S, actions: A, gamma } = mdp
  const Q = new Float64Array(S * A)
  for (let s = 0; s < S; s++) {
    if (!isActive(mdp, s)) continue
    for (let a = 0; a < A; a++) {
      const outs = mdp.outcomes[s * A + a]
      let q = outs.length ? 0 : -Infinity
      for (const o of outs) q += o.p * (o.reward + gamma * V[o.next])
      Q[s * A + a] = q
    }
  }
  return Q
}

/**
 * The optimal state values V* by value iteration from V₀ = 0 (terminal values fixed), until the Bellman residual is
 * below `tolerance` or after `maxSweeps` sweeps. The environments' oracle; the traceable form is the planning agents'
 * `valueIteration`.
 */
export function optimalValues(mdp: MdpTables, { tolerance = 1e-10, maxSweeps = 100_000 } = {}): Float64Array {
  const A = mdp.actions
  let V = Float64Array.from(mdp.terminalValue)
  for (let k = 0; k < maxSweeps; k++) {
    const Q = qFromValues(mdp, V)
    const next = Float64Array.from(mdp.terminalValue)
    let residual = 0
    for (let s = 0; s < mdp.states; s++) {
      if (!isActive(mdp, s)) continue
      let best = -Infinity
      for (let a = 0; a < A; a++) best = Math.max(best, Q[s * A + a])
      next[s] = best
      residual = Math.max(residual, Math.abs(best - V[s]))
    }
    V = next
    if (residual < tolerance) break
  }
  return V
}
