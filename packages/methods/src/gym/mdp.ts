/**
 * Finite Markov decision processes as plain data, shared by the gridworld environments (`./environments`) and the
 * planning and tabular agents (`./agents`): the `TabularMdp` type and constructor, outcome sampling, legal actions,
 * one-step backups, policy helpers and the optimal values by value iteration.
 *
 * Conventions. States are integers; a grid cell $(x, y)$ is state $y w + x$ ($w$ the width) with $y = 0$ at the bottom
 * row. Actions on grids are up, right, down, left (0 to 3). Tables of states by actions are flat and row-major: entry
 * `s * actions + a` belongs to state $s$ and action $a$. Terminal states take no action and have a fixed value
 * `terminalValue` (0 unless an environment pays an exit reward there), so every backup is $r + \gamma V(s')$ with
 * $V(s')$ the terminal value at a terminal $s'$, and an episode's discounted return adds $\gamma^T$ times the terminal
 * value on arrival at step $T$.
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
  /** A readable name. */
  name: string
  /** The number of states $S$. */
  states: number
  /** The number of actions $A$. */
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
  /** The discount factor $\gamma$. */
  gamma: number
  /** A display name per action, in order. */
  actionNames: string[]
  /**
   * Grid layout, when the MDP is a grid: its size, the kind of each cell by state index, and the
   * $(\Delta x, \Delta y)$ of each action.
   */
  grid?: { width: number; height: number; kinds: CellKind[]; actionVectors: [number, number][] }
}

/** Up, right, down, left as $(\Delta x, \Delta y)$ with $y$ pointing up. */
export const GRID_ACTIONS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]
/** The names of the grid actions, in the order of `GRID_ACTIONS`. */
export const GRID_ACTION_NAMES = ['up', 'right', 'down', 'left']

/**
 * The state of grid cell $(x, y)$, $y w + x$.
 *
 * @param width The grid's width $w$, in cells.
 * @param x The column, from 0 at the left.
 * @param y The row, from 0 at the bottom.
 * @returns The state index.
 *
 * @example Cells and states of a 4-wide grid
 * print('cell (3, 2) is state', cellState(4, 3, 2))
 * print('state 11 is cell', stateCell(4, 11))
 */
export const cellState = (width: number, x: number, y: number): number => y * width + x
/**
 * The $(x, y)$ cell of a grid state: the inverse of `cellState`.
 *
 * @param width The grid's width $w$, in cells.
 * @param s The state index.
 * @returns The column $x$ (from the left) and the row $y$ (from the bottom).
 *
 * @example Every cell of a 2-by-2 grid
 * print([0, 1, 2, 3].map((s) => stateCell(2, s)))
 */
export const stateCell = (width: number, s: number): [number, number] => [s % width, Math.floor(s / width)]

/** The tables planners read: what a `TabularMdp` and an environment's tabular model have in common. */
export type MdpTables = Pick<
  TabularModel<unknown>,
  'states' | 'actions' | 'outcomes' | 'terminal' | 'terminalValue' | 'gamma'
>

/**
 * Whether an action is taken in state $s$: true unless it is terminal (or a wall).
 *
 * @param mdp The MDP's tables; only `terminal` is read.
 * @param s The state index.
 * @returns True at an active state.
 *
 * @example The cells of the gridworld that end an episode
 * const mdp = gymEnvironment('gridworldEnvironment', {}).model
 * print('active', Array.from({ length: mdp.states }, (_, s) => isActive(mdp, s)))
 */
export const isActive = (mdp: MdpTables, s: number): boolean => !mdp.terminal[s]

/**
 * The actions legal in active state $s$: those with outcomes (all of them in a grid). Empty at a terminal state.
 *
 * @param mdp The MDP's tables; `actions` and `outcomes` are read.
 * @param s The state index.
 * @returns The legal actions' indices, in increasing order.
 *
 * @example An action with no outcomes is illegal
 * const mdp = tabularMdp({
 *   transitions: [[[1, 0], [0, 0]], [[0, 1], [0, 1]]],
 *   rewards: [[0, 0], [0, 0]],
 *   gamma: 0.9,
 * })
 * print('legal in state 0', legalActions(mdp, 0), '; in state 1', legalActions(mdp, 1))
 */
export function legalActions(mdp: MdpTables, s: number): number[] {
  const out: number[] = []
  for (let a = 0; a < mdp.actions; a++) if (mdp.outcomes[s * mdp.actions + a].length > 0) out.push(a)
  return out
}

/**
 * True when some active state has an illegal action (an empty outcome list), so environments must mask actions.
 *
 * @param mdp The MDP's tables.
 * @returns Whether any active state has an action with no outcomes.
 *
 * @example A grid has none; a table may have some
 * const masked = tabularMdp({
 *   transitions: [[[1, 0], [0, 0]], [[0, 1], [0, 1]]],
 *   rewards: [[0, 0], [0, 0]],
 *   gamma: 0.9,
 * })
 * print('gridworld', hasIllegalActions(gymEnvironment('gridworldEnvironment', {}).model))
 * print('table', hasIllegalActions(masked))
 */
export function hasIllegalActions(mdp: MdpTables): boolean {
  for (let s = 0; s < mdp.states; s++)
    if (isActive(mdp, s))
      for (let a = 0; a < mdp.actions; a++) if (mdp.outcomes[s * mdp.actions + a].length === 0) return true
  return false
}

/**
 * Sample an outcome of action $a$ in state $s$ by inversion, with one uniform draw from the stream.
 *
 * @param r The stream to draw from; advanced by one draw.
 * @param mdp The MDP's tables.
 * @param s The state index, an active state.
 * @param a The action index, legal in `s`.
 * @returns One of `mdp.outcomes[s * actions + a]`, chosen with its probability `p`.
 *
 * @example Twenty draws of a slippery move
 * const mdp = gymEnvironment('gridworldEnvironment', {}).model
 * const s = stream(1)
 * print('outcomes of up from 0', mdp.outcomes[0].map((o) => [o.next, o.p]))
 * print('twenty draws', Array.from({ length: 20 }, () => sampleOutcome(s, mdp, 0, 0).next))
 */
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
 * A general finite MDP from dense arrays: $P[s][a][s']$ transition probabilities and $R[s][a][s']$ rewards (or
 * $R[s][a]$ expected rewards), with optional terminal states. An all-zero row $P[s][a]$ at an active state makes action
 * $a$ illegal there (its outcome list is empty). Only the transitions with positive probability become outcomes; the
 * rows are not checked to sum to 1.
 *
 * @param options The tables and the MDP's settings: `transitions`, $S$ by $A$ by $S$ probabilities $P[s][a][s']$
 *   (their sizes give $S$ and $A$); `rewards`, $S$ by $A$ entries, each a reward per next state or one number for
 *   every next state; `gamma`, the discount factor; `start`, the start state (default 0); `terminal`, the indices of
 *   the terminal states (default none), whose rows are ignored; `terminalValue`, a value per state, read at the
 *   terminal ones (default 0); `name` (default `MDP`); and `actionNames` (default `a0`, `a1`, ...).
 * @returns The MDP, with no grid layout.
 *
 * @example Stay for 0.5 a step, or leave for 1
 * const mdp = tabularMdp({
 *   transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]],
 *   rewards: [[0.5, 1], [0, 0]],
 *   gamma: 0.9,
 *   terminal: [1],
 *   actionNames: ['stay', 'leave'],
 * })
 * print('outcomes of state 0', mdp.outcomes.slice(0, 2))
 * print('V*', optimalValues(mdp))
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

/**
 * A policy: a deterministic action per state ($S$ integers, $-1$ at terminals) or a stochastic one, $S \times A$
 * probabilities in row-major order. A tensor or a plain array.
 */
export type PolicyInput = Tensor | readonly number[]

/**
 * The greedy action per state (lowest index on ties within $10^{-9}$, or `prefer[s]` if it is tied), $-1$ at
 * terminals. Illegal actions have $Q = -\infty$ (see `qFromValues`), so they are never greedy.
 *
 * @param mdp The MDP's tables; `states`, `actions` and `terminal` are read.
 * @param Q The action values, $S \times A$ in row-major order.
 * @param prefer An action per state to keep when it is among the best (the current policy, so that policy iteration
 *   does not switch between tied actions); a negative entry, or none, prefers nothing.
 * @returns The greedy action of each state.
 *
 * @example The greedy policy of the optimal values
 * const mdp = gymEnvironment('gridworldEnvironment', {}).model
 * const pi = greedyActions(mdp, qFromValues(mdp, optimalValues(mdp)))
 * print('actions', Array.from(pi, (a) => (a < 0 ? '-' : GRID_ACTION_NAMES[a])))
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

/**
 * $\pi(a \mid s)$ as a dense $S \times A$ array from a deterministic or stochastic policy. Throws `ShapeError` when the
 * policy has neither $S$ nor $S A$ entries.
 *
 * @param mdp The MDP's tables; `states`, `actions` and `terminal` are read.
 * @param policy The policy: $S$ actions (a one-hot row each; a negative action, or a terminal state, gives a row of
 *   zeros), or $S A$ probabilities, copied as they are.
 * @returns The probabilities, $S \times A$ in row-major order.
 *
 * @example A deterministic policy as probabilities
 * const mdp = tabularMdp({
 *   transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]],
 *   rewards: [[0.5, 1], [0, 0]],
 *   gamma: 0.9,
 *   terminal: [1],
 * })
 * print('pi', policyMatrix(mdp, [1, -1]))
 */
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

/**
 * One backup of state values to action values, $Q(s, a) = \sum p \, (r + \gamma V(s'))$ over the outcomes
 * $(p, s', r)$ of $a$ in $s$, for every active state and legal action; $-\infty$ for illegal actions, 0 at terminals.
 *
 * @param mdp The MDP's tables.
 * @param V The state values, one per state; at a terminal state, its terminal value.
 * @returns $Q$, $S \times A$ in row-major order.
 *
 * @example Action values of the optimal values
 * const mdp = tabularMdp({
 *   transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]],
 *   rewards: [[0.5, 1], [0, 0]],
 *   gamma: 0.9,
 *   terminal: [1],
 * })
 * print('Q', qFromValues(mdp, optimalValues(mdp)))
 */
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
 * The optimal state values $V^*$ by value iteration from $V_0 = 0$ (terminal values fixed), until the Bellman residual
 * $\max_s \lvert V_{k+1}(s) - V_k(s) \rvert$ is below `tolerance` or after `maxSweeps` sweeps. The environments'
 * oracle; the traceable form is the planning agents' `valueIteration`.
 *
 * @param mdp The MDP's tables.
 * @param options When to stop.
 * @param options.tolerance The residual below which the iteration stops.
 * @param options.maxSweeps The most sweeps; the values of the last are returned even if the residual is still above
 *   `tolerance` (with $\gamma = 1$ and no terminal reachable it never falls).
 * @returns $V^*$, one value per state.
 *
 * @example The gridworld's optimal values, top row first
 * const V = optimalValues(gymEnvironment('gridworldEnvironment', {}).model)
 * print('y = 2', V.slice(8, 12))
 * print('y = 1', V.slice(4, 8))
 * print('y = 0', V.slice(0, 4))
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
