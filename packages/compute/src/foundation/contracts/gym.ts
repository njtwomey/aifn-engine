/**
 * Sequential decisions: the `Environment` an agent acts in, the `Agent` that acts and learns, and the `Domain` of their
 * observations and actions (docs/aifn-gym.md §3). Gymnasium's semantics in aifn's style: environment state is
 * plain data, randomness is an explicit `Stream`, and `terminated` (a true end: values do not bootstrap past it) is
 * kept apart from `truncated` (cut short by a time limit: values still bootstrap). Implementations live in
 * applications (`aifn-methods/gym`).
 */

import type { Index, Scalar, Shape, Size, Tensor, Value } from './numbers'
import type { Stream } from './random'

// ── Domains ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The integers $\{0, \dots, n - 1\}$ (Gymnasium's `Discrete(n)`), with optional display names, one per value. */
export interface DiscreteDomain {
  /** The brand of a discrete domain. */
  readonly kind: 'discrete'
  /** The number of values, at least 1. */
  readonly n: Size
  /** A display name per value, in order. */
  readonly names?: readonly string[]
}

/**
 * A box of real arrays of shape `shape` (Gymnasium's `Box(low, high, shape)`): element $i$ lies in
 * $[\text{low}_i, \text{high}_i]$, the bounds flat in row-major order (bounds may be infinite). Optional display names,
 * one per element.
 */
export interface BoxDomain {
  /** The brand of a box domain. */
  readonly kind: 'box'
  /** The lower bound of each element, flat and row-major. */
  readonly low: readonly number[]
  /** The upper bound of each element, flat and row-major; at least `low`. */
  readonly high: readonly number[]
  /** The shape of one value; `[]` for a scalar. */
  readonly shape: Shape
  /** A display name per element, in row-major order. */
  readonly names?: readonly string[]
}

/** A set of observations or actions: discrete values or a continuous box. */
export type Domain = DiscreteDomain | BoxDomain

/** The kinds of domain, as the registry pairs agents with environments. */
export type DomainKind = Domain['kind']

// ── Environment ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** What `step` returns: the next state, what the agent sees of it, the reward, and whether the episode ended. */
export interface Step<S, O> {
  /** The environment's next state. */
  state: S
  /** What the agent sees of the next state. */
  observation: O
  /** The reward of the step just taken. */
  reward: Scalar
  /** A terminal state: the return ends here and values do not bootstrap past it. */
  terminated: boolean
  /** Cut short by the environment itself (the rollout also truncates at `horizon`): values still bootstrap. */
  truncated: boolean
}

/** One possible result of an action in a finite MDP: probability `p` of reaching state `next` with `reward`. */
export interface Outcome {
  /** The probability of this outcome. */
  p: number
  /** The table index of the next state. */
  next: Index
  /** The reward received. */
  reward: Scalar
}

/**
 * An explicit finite MDP, for planners: `outcomes[s * actions + a]` is the distribution over (next state, reward) of
 * action $a$ in state $s$, empty at terminal states. A terminal state has the fixed value `terminalValue[s]`. `encode`
 * and `decode` map environment states to table indices.
 */
export interface TabularModel<S> {
  /** The brand of a tabular model. */
  readonly kind: 'tabular'
  /** The number of states. */
  readonly states: Size
  /** The number of actions. */
  readonly actions: Size
  /** The outcomes of each state and action, at index `s * actions + a`. */
  readonly outcomes: readonly (readonly Outcome[])[]
  /** 1 for states where no action is taken. */
  readonly terminal: Uint8Array
  /** The fixed value of each terminal state (read only where `terminal` is 1). */
  readonly terminalValue: Float64Array
  /** The discount factor. */
  readonly gamma: Scalar
  /** The table index of an environment state. */
  encode(state: S): Index
  /** The environment state of a table index. */
  decode(index: Index): S
}

/**
 * Deterministic, differentiable dynamics, for model-based control (LQR about an equilibrium, iLQR, MPC): the state as
 * a vector $\xvec$ of length `stateSize` and the action as a vector $\uvec$ of length `actionSize`. `transition` gives
 * the next state and `reward` the reward of taking $\uvec$ in $\xvec$; both are written with tensor primitives, so
 * $\xvec$ and $\uvec$ may be traced (`jacobian`, `grad`). `encode` and `decode` map environment states to and from
 * $\xvec$.
 */
export interface DynamicsModel<S> {
  /** The brand of a dynamics model. */
  readonly kind: 'dynamics'
  /** The length of the state vector $\xvec$. */
  readonly stateSize: Size
  /** The length of the action vector $\uvec$. */
  readonly actionSize: Size
  /** The next state vector after action $\uvec$ in state $\xvec$. */
  transition(x: Value, u: Value): Value
  /** The reward of action $\uvec$ in state $\xvec$, a scalar. */
  reward(x: Value, u: Value): Value
  /** The state vector of an environment state. */
  encode(state: S): Tensor
  /** The environment state of a state vector. */
  decode(x: Value): S
}

/** The explicit dynamics of an environment: a finite MDP's tables, or differentiable deterministic dynamics. */
export type EnvironmentModel<S> = TabularModel<S> | DynamicsModel<S>

/** Ground truth for evaluation, never shown to agents. */
export interface EnvironmentOracle<S, A> {
  /** $\expect[\text{reward} \mid \text{state}, \text{action}]$ (a bandit's arm means). */
  expectedReward?(state: S, action: A): Scalar
  /** The best achievable expected reward in a state (for pseudo-regret). */
  bestExpectedReward?(state: S): Scalar
  /** Optimal state values, when known exactly. */
  optimalValues?(): Tensor
}

/**
 * A numeric series of an environment's state, plotted against step under an episode's playback (an angle, a
 * position).
 */
export interface StateSeries<S> {
  /** The series' label. */
  readonly name: string
  /** The series' value at a state. */
  value(state: S): number
}

/**
 * How the lab draws a state. Every kind may name `series` of the state to plot against step. `grid`: a grid of
 * `width` by `height` cells, cell $(x, y)$ at index $y w + x$ ($w$ the width) with $y = 0$ at the bottom row; `cells`
 * names each cell's kind (`wall`, `goal`, ...), `cell` gives a state's cell index, and `actionVectors` the
 * $(\Delta x, \Delta y)$ of each action.
 */
export interface GridRender<S> {
  /** The brand of a grid drawing. */
  readonly kind: 'grid'
  /** The number of columns. */
  readonly width: Size
  /** The number of rows. */
  readonly height: Size
  /** The kind of each cell, by cell index. */
  readonly cells: readonly string[]
  /** The index of the cell a state occupies. */
  cell(state: S): Index
  /** The move of each action, by action index. */
  readonly actionVectors: readonly (readonly [number, number])[]
  /** Series of the state to plot against step. */
  readonly series?: readonly StateSeries<S>[]
}

/** `pendulum`: a rod of `length` pivoted at the origin, at `angle(state)` radians from upright (anticlockwise). */
export interface PendulumRender<S> {
  /** The brand of a pendulum drawing. */
  readonly kind: 'pendulum'
  /** The rod's length. */
  readonly length: number
  /** The rod's angle from upright, in radians, anticlockwise. */
  angle(state: S): number
  /** Series of the state to plot against step. */
  readonly series?: readonly StateSeries<S>[]
}

/**
 * `cartpole`: a cart at `cart(state)` on a track from `-trackLimit` to `trackLimit`, carrying a pole of `poleLength`
 * at `angle(state)` radians from upright (positive clockwise, towards $+x$).
 */
export interface CartPoleRender<S> {
  /** The brand of a cart-pole drawing. */
  readonly kind: 'cartpole'
  /** The pole's length. */
  readonly poleLength: number
  /** Half the track's length: the track runs from `-trackLimit` to `trackLimit`. */
  readonly trackLimit: number
  /** The cart's position on the track. */
  cart(state: S): number
  /** The pole's angle from upright, in radians, positive clockwise. */
  angle(state: S): number
  /** Series of the state to plot against step. */
  readonly series?: readonly StateSeries<S>[]
}

/** How to draw an environment's state. */
export type RenderSpec<S> = GridRender<S> | PendulumRender<S> | CartPoleRender<S>

/**
 * What an agent may know of an environment before acting: its domains, discount and episode cap, and its explicit
 * model when it has one (planning agents require it).
 */
export interface EnvironmentShape {
  /** A readable name. */
  readonly name: string
  /** The domain of observations. */
  readonly observation: Domain
  /** The domain of actions. */
  readonly action: Domain
  /** The discount the problem is posed with (agents may use their own). */
  readonly gamma: Scalar
  /** The longest episode: the rollout truncates there (`Infinity` for none). */
  readonly horizon: number
  /** The explicit dynamics, when the environment has them. */
  readonly model?: EnvironmentModel<unknown>
}

/** How an episode ended, for display: a success (a goal reached, survived the time limit) or a failure, and why. */
export interface EpisodeEnd {
  /** True for a success, false for a failure. */
  success: boolean
  /** A short reason, e.g. "pole fell: θ = 13.1°", "reached the goal", "survived 500 steps". */
  reason: string
}

/**
 * An environment. `S` is its state (plain data), `O` what the agent observes, `A` an action. `reset` and `step` are
 * pure and draw only from their stream. The optional capabilities serve planners (`model`), evaluation (`oracle`) and
 * the lab (`render`).
 */
export interface Environment<S, O, A> extends EnvironmentShape {
  /** A first state and its observation, drawn from the stream. */
  reset(stream: Stream): { state: S; observation: O }
  /** The result of taking `action` in `state`, drawing from the stream; `state` is not modified. */
  step(state: S, action: A, stream: Stream): Step<S, O>
  /** The actions allowed in a state, when not every action of the domain is. */
  legal?(state: S): readonly A[]
  /**
   * How an episode that ended at `state` went: `terminated` there, or `truncated` after `steps` steps. Null when the
   * environment has no notion of success (a bandit's round).
   */
  ending?(state: S, how: 'terminated' | 'truncated', steps: number): EpisodeEnd | null
  /** The explicit dynamics, for planners. */
  readonly model?: EnvironmentModel<S>
  /** Ground truth for evaluation, never shown to agents. */
  readonly oracle?: EnvironmentOracle<S, A>
  /** How the lab draws a state. */
  readonly render?: RenderSpec<S>
}

// ── Agent ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One environment step as the agent sees it. */
export interface Transition<O, A> {
  /** What the agent saw before acting. */
  observation: O
  /** The action it took. */
  action: A
  /** The reward it received. */
  reward: Scalar
  /** What it saw after the step. */
  next: O
  /** The step reached a terminal state: no bootstrapping from `next`. */
  terminated: boolean
  /** The episode was cut short at `next`: values still bootstrap from it. */
  truncated: boolean
  /** The actions legal at `next`, when the environment masks actions (`legal`). */
  nextLegal?: readonly A[]
}

/** An agent's choice, with what the lab may show: the scores it ranked actions by and its action probabilities. */
export interface Decision<A> {
  /** The action chosen. */
  action: A
  /** The score of each action, by action index (action values, upper confidence bounds, ...). */
  scores?: Float64Array
  /** The probability of choosing each action, by action index. */
  probabilities?: Float64Array
}

/**
 * A learning agent. Its state `G` is plain data; `init`, `act` and `learn` are pure, and only `init` and `act` draw
 * randomness. Episode-level learners buffer transitions in `learn` and update at the episode's end.
 */
export interface Agent<G, O, A> {
  /** A readable name. */
  readonly name: string
  /** The agent's first state for an environment, drawing from the stream. */
  init(env: EnvironmentShape, stream: Stream): G
  /** Choose an action; `legal` lists the allowed actions when the environment masks some. */
  act(agent: G, observation: O, stream: Stream, legal?: readonly A[]): Decision<A>
  /** The agent's state after learning from one transition; `agent` is not modified. */
  learn(agent: G, transition: Transition<O, A>): G
  /** The deterministic action of the learnt policy (no exploration), for evaluating it; `act` is used when absent. */
  greedy?(agent: G, observation: O, legal?: readonly A[]): A
  /** Scalars that track learning (a value estimate, an exploration rate, a loss), for training curves. */
  scalars?(agent: G): Readonly<Record<string, number>>
}

// ── Traces ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * One episode in full, as the rollouts of `aifn-methods/gym` produce it and the views play it: environment states and
 * observations (both from the reset to the arrival), actions and rewards.
 */
export interface Trajectory<S, O, A> {
  /** The environment states, from the reset to the last: one more than the actions. */
  states: S[]
  /** The observations, aligned with `states`. */
  observations: O[]
  /** The actions taken, one per step. */
  actions: A[]
  /** The reward of each step, aligned with `actions`. */
  rewards: number[]
  /** The undiscounted sum of the rewards. */
  episodeReturn: number
  /** It ended at a terminal state, not by truncation. */
  reachedTerminal: boolean
  /** The total pseudo-regret of its actions, when the environment's oracle knows expected rewards; else 0. */
  regret: number
  /** How it ended (success or failure, and why), when the environment says (`ending`); else null. */
  ending: EpisodeEnd | null
}

/** A checkpoint of a training run: the agent's state after `episode` episodes (0 is the initial state). */
export interface Checkpoint<G> {
  /** The number of episodes completed. */
  episode: number
  /** The agent's state then. */
  agent: G
}

/** A training run so far: per-episode columns (one entry per completed episode) and checkpoints. Plain data. */
export interface Training<G> {
  /** The seed of the run's root stream. */
  seed: number | string
  /** Episodes requested (NaN under a step budget). */
  total: number
  /** Episodes completed. */
  episodes: number
  /** The step budget (NaN under an episode budget). */
  budget: number
  /** The environment steps done. */
  steps: number
  /** The checkpoint spacing in episodes (doubled whenever a step-budget run would exceed `maxCheckpoints`). */
  every: number
  /** The undiscounted return of each episode. */
  returns: Float64Array
  /** The length of each episode, in steps. */
  lengths: Float64Array
  /** 1 for an episode that ended at a terminal state, 0 for one truncated. */
  terminated: Uint8Array
  /** The total pseudo-regret of each episode, when the environment's oracle knows expected rewards; else null. */
  regret: Float64Array | null
  /** 1 for a success, $-1$ for a failure, 0 when the environment does not say (`ending`). */
  outcome: Int8Array
  /** The episode's first action, for discrete actions (a bandit's pull); NaN otherwise. */
  firstAction: Float64Array
  /** The agent's `scalars` after each episode (NaN before a scalar first appears). */
  scalars: Record<string, Float64Array>
  /** The agent's state after every `every` episodes, from episode 0. */
  checkpoints: Checkpoint<G>[]
  /** The agent's state after the last completed episode. */
  final: G
  /** True once the budget is spent. */
  done: boolean
}

/** A call of a registered factory by its worker address (`<module>/<key>`) with plain parameters. */
export interface FactoryCall {
  /** The factory's worker address, `<module>/<key>`. */
  readonly address: string
  /** Its parameters, as plain data. */
  readonly params: object
}

/**
 * What a training view needs to train an agent in an environment and play its episodes, with no knowledge of where
 * either is defined: both as values on the page (to replay and evaluate episodes), the same two as worker calls with
 * the training generator's address (to train off the main thread), and the functions that re-run a run's episodes.
 * `aifn-methods/gym` `gymSetup` builds one from registry keys.
 */
export interface GymSetup {
  /** The environment, as a value on the page. */
  readonly env: Environment<unknown, unknown, unknown>
  /** The agent, as a value on the page. */
  readonly agent: Agent<unknown, unknown, unknown>
  /** The call that builds the same environment in a worker. */
  readonly envCall: FactoryCall
  /** The call that builds the same agent in a worker. */
  readonly agentCall: FactoryCall
  /** The worker address of the training generator, called `(env, agent, { episodes | steps, seed })`. */
  readonly trainingAddress: string
  /** The environment `evaluate` plays in, when it differs from the trained one (another start state). */
  readonly evaluationEnv?: Environment<unknown, unknown, unknown>
  /** Training episode `episode` (1-based) exactly as it happened. */
  replay(training: Training<unknown>, episode: number): Trajectory<unknown, unknown, unknown>
  /** A fresh greedy episode in `env` of the agent as it was after `episode` episodes, on `seed`. */
  evaluate(
    env: Environment<unknown, unknown, unknown>,
    training: Training<unknown>,
    episode: number,
    seed: number | string,
  ): Trajectory<unknown, unknown, unknown>
  /** The agent's state after `episode` episodes. */
  agentAfter(training: Training<unknown>, episode: number): unknown
}
