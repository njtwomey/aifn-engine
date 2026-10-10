/**
 * `aifn-methods/gym`: sequential decisions on one protocol, environments and agents run against each other
 * (docs/aifn-gym.md), in the manner of Gymnasium with explicit random streams.
 *
 * Child modules:
 *
 * - `aifn-methods/gym/environments`: what agents act in. Bandits (Bernoulli, Gaussian, linear contextual), finite MDPs
 *   on grids (gridworld, cliff walking, mazes, FrozenLake) and `mdpEnvironment` for any table, and, in its child
 *   `control`, the pendulum and the cart-pole with differentiable dynamics models.
 * - `aifn-methods/gym/agents`: what acts and learns. The random baseline, bandit policies, tabular learners (TD and
 *   Monte Carlo control, REINFORCE), planners (value and policy iteration), the deep Q-network, and, in its children,
 *   classic-control and policy-gradient agents.
 *
 * Shared by both, at this level:
 *
 * - Finite-MDP tables (`mdp.ts`): `TabularMdp` and `tabularMdp`, outcome sampling (`sampleOutcome`), legal actions
 *   (`legalActions`, `hasIllegalActions`, `isActive`), the backup `qFromValues`, `greedyActions` and `policyMatrix`,
 *   the oracle's `optimalValues`, and the grid conventions (`cellState`, `stateCell`, `GRID_ACTIONS`).
 * - Running an agent (`rollout.ts`): `rollout` (one environment step per algorithm step), `episodes` (one episode per
 *   step), `runEpisode` (a single episode, learning or greedy) and `compare` (several agents over replicates, with
 *   common random numbers).
 * - Training without a display (`train.ts`): `train` and its generator form `training`, with checkpoints
 *   (`checkpointSpacing`) from which `replay`, `agentAfter` and `evaluateEpisode` re-run any episode exactly.
 * - Registries: `environmentRegistry` and `agentRegistry`, keyed by `info.key`; `compatible` and `validPairs` pair
 *   environments with the agents whose declared domains, model and family agree; `gymEnvironment` and `gymSetup` build
 *   them, and the compute `GymSetup` a training view consumes, from registry keys; `gymFunctions` registers the
 *   functions of this level.
 *
 * Environments and agents are plain data with pure functions that draw only from the stream they are given, so a run
 * is reproduced exactly from its seed.
 */

import type { Agent, AgentInfo, Environment, EnvironmentInfo, GymSetup } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { entries, type Entry } from 'aifn-compute/foundation/registry'
import * as agents from './agents'
import * as environments from './environments'
import { agentAfter as afterEpisodes, evaluateEpisode as evaluateAfter, replay as replayEpisode } from './train'

export {
  compare,
  episodes,
  rollout,
  type CompareOptions,
  type Comparison,
  type EpisodeState,
  type RolloutOptions,
  type RolloutState,
  type RolloutTransition,
  runEpisode,
  type EpisodeMode,
  type Spread,
} from './rollout'
export { agentAfter, checkpointSpacing, evaluateEpisode, replay, train, training, type TrainOptions } from './train'
export {
  cellState,
  GRID_ACTION_NAMES,
  GRID_ACTIONS,
  greedyActions,
  hasIllegalActions,
  isActive,
  legalActions,
  optimalValues,
  policyMatrix,
  qFromValues,
  sampleOutcome,
  stateCell,
  tabularMdp,
  type CellKind,
  type MdpTables,
  type Outcome,
  type PolicyInput,
  type TabularMdp,
} from './mdp'

/** A registered environment factory, called `(params)`. */
export type EnvironmentEntry = Entry<(params: never) => unknown, EnvironmentInfo>
/** A registered agent factory, called `(params)`. */
export type AgentEntry = Entry<(params: never) => unknown, AgentInfo>

/** Every registered environment (kind `environment`). */
export const environmentRegistry = entries('environment', environments) as Readonly<Record<string, EnvironmentEntry>>

/** Every registered agent (kind `agent`). */
export const agentRegistry = entries('agent', agents) as Readonly<Record<string, AgentEntry>>

/**
 * Whether an agent can run in an environment by their declared metadata: the agent's required observation and action
 * domain kinds, tabular or dynamics model (a capability `model`; the kind is checked when the model is built) and
 * environment families all match.
 *
 * @param env The environment's registry metadata (`environmentRegistry[key].info`).
 * @param agent The agent's registry metadata (`agentRegistry[key].info`); each of its `requires` fields left out
 *   accepts any environment.
 * @returns True when every requirement the agent declares is met by the environment.
 *
 * @example Q-learning runs on a gridworld but not on the continuous pendulum
 * const grid = environmentRegistry.gridworldEnvironment.info
 * const pendulum = environmentRegistry.pendulumEnvironment.info
 * print('gridworld:', compatible(grid, agentRegistry.qLearningAgent.info))
 * print('pendulum:', compatible(pendulum, agentRegistry.qLearningAgent.info))
 */
export function compatible(env: EnvironmentInfo, agent: AgentInfo): boolean {
  const r = agent.requires
  if (r.observation && env.observation !== r.observation) return false
  if (r.action && env.action !== r.action) return false
  if (r.model && !env.capabilities?.includes('model')) return false
  if (r.families && !r.families.includes(env.family)) return false
  return true
}

/**
 * Every pair of a registered environment and a registered agent that `compatible` allows, as registry keys, in
 * registry order (environments outer, agents inner).
 *
 * @returns The allowed pairs, each `{ environment, agent }` naming two registry keys.
 *
 * @example How many pairs the registries allow, and the first few
 * const pairs = validPairs()
 * print('pairs:', pairs.length)
 * print('first:', pairs.slice(0, 3).map((p) => `${p.environment} + ${p.agent}`))
 */
export function validPairs(): { environment: string; agent: string }[] {
  const out: { environment: string; agent: string }[] = []
  for (const e of Object.values(environmentRegistry))
    for (const a of Object.values(agentRegistry))
      if (compatible(e.info, a.info)) out.push({ environment: e.info.key, agent: a.info.key })
  return out
}

// ── Setups for training views ─────────────────────────────────────────────────────────────────────────────────────

/** An environment of any state, observation and action types, as the registries build them. */
type AnyEnv = Environment<unknown, unknown, unknown>
/** An agent of any state, observation and action types, as the registries build them. */
type AnyAgent = Agent<unknown, unknown, unknown>

/**
 * An environment of the gym registry built from its parameters, for instance an evaluation variant of the environment
 * a setup trains on. Throws `DomainError` for a key the registry doesn't hold.
 *
 * @param envKey The environment's registry key, e.g. `'gridworldEnvironment'`.
 * @param envParams The parameters its factory takes; `{}` gives its defaults.
 * @returns The environment.
 *
 * @example The default gridworld
 * const env = gymEnvironment('gridworldEnvironment', {})
 * print('name:', env.name, ' discount:', env.gamma)
 */
export function gymEnvironment(envKey: string, envParams: object): AnyEnv {
  const e = environmentRegistry[envKey]
  if (!e) throw new DomainError('gymEnvironment', `gymEnvironment: no environment '${envKey}' in the gym registry`)
  return (e as unknown as (p: object) => AnyEnv)(envParams)
}

/**
 * The setup for environment `envKey` and agent `agentKey` of the gym registries, built from their parameters: the two
 * built objects, the calls that rebuild them in a worker, the training address, and the `replay`, `evaluate` and
 * `agentAfter` functions a training view uses to re-run any episode. Throws `DomainError` for a key either registry
 * doesn't hold; it does not check that the pair is `compatible`.
 *
 * @param envKey The environment's registry key.
 * @param envParams The environment factory's parameters.
 * @param agentKey The agent's registry key.
 * @param agentParams The agent factory's parameters.
 * @param extra Fields copied into the setup as they are.
 * @param extra.evaluationEnv An environment to evaluate the trained agent on, when it differs from the training one.
 * @returns The `GymSetup` a training view consumes.
 *
 * @example Q-learning on the default gridworld
 * const setup = gymSetup('gridworldEnvironment', {}, 'qLearningAgent', {})
 * print('environment call:', setup.envCall.address)
 * print('agent call:', setup.agentCall.address)
 */
export function gymSetup(
  envKey: string,
  envParams: object,
  agentKey: string,
  agentParams: object,
  extra: { evaluationEnv?: AnyEnv } = {},
): GymSetup {
  const e = environmentRegistry[envKey]
  const a = agentRegistry[agentKey]
  if (!e) throw new DomainError('gymSetup', `gymSetup: no environment '${envKey}' in the gym registry`)
  if (!a) throw new DomainError('gymSetup', `gymSetup: no agent '${agentKey}' in the gym registry`)
  const env = (e as unknown as (p: object) => AnyEnv)(envParams)
  const agent = (a as unknown as (p: object) => AnyAgent)(agentParams)
  return {
    env,
    agent,
    envCall: { address: `${e.info.module}/${envKey}`, params: envParams },
    agentCall: { address: `${a.info.module}/${agentKey}`, params: agentParams },
    trainingAddress: 'gym/training',
    ...extra,
    replay: (training, episode) => replayEpisode(env, agent, training, episode).trajectory,
    evaluate: (on, training, episode, seed) => evaluateAfter(on, agent, training, episode, seed).trajectory,
    agentAfter: (training, episode) => afterEpisodes(env, agent, training, episode),
  }
}

export { gymFunctions } from './registry'
