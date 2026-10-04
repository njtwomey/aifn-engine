/**
 * `aifn-methods/gym`: sequential decisions on one protocol (docs/aifn-gym.md): environments (`./environments`: bandits,
 * finite MDPs, classic control) and agents (`./agents`: bandit policies, tabular learners, planners) meet in the loops
 * of `rollout.ts` (`rollout`, `episodes`, `compare`); `train.ts` trains headlessly (`train`, `training`) and replays
 * or evaluates any training episode (`replay`, `evaluateEpisode`). `mdp.ts` holds the finite-MDP tables both sides
 * share.
 * Registries: `environmentRegistry` and `agentRegistry`, keyed by `info.key`; `validPairs` lists the environment ×
 * agent pairs whose declared domains, model and family agree; `gymSetup` makes the compute `GymSetup` a training view
 * consumes from two registry keys.
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
 */
export function compatible(env: EnvironmentInfo, agent: AgentInfo): boolean {
  const r = agent.requires
  if (r.observation && env.observation !== r.observation) return false
  if (r.action && env.action !== r.action) return false
  if (r.model && !env.capabilities?.includes('model')) return false
  if (r.families && !r.families.includes(env.family)) return false
  return true
}

/** Every registered environment × agent pair that `compatible` allows, as registry keys. */
export function validPairs(): { environment: string; agent: string }[] {
  const out: { environment: string; agent: string }[] = []
  for (const e of Object.values(environmentRegistry))
    for (const a of Object.values(agentRegistry))
      if (compatible(e.info, a.info)) out.push({ environment: e.info.key, agent: a.info.key })
  return out
}

// ── Setups for training views ─────────────────────────────────────────────────────────────────────────────────────

type AnyEnv = Environment<unknown, unknown, unknown>
type AnyAgent = Agent<unknown, unknown, unknown>

/** An environment of the gym registry built from its parameters (an evaluation variant, say). */
export function gymEnvironment(envKey: string, envParams: object): AnyEnv {
  const e = environmentRegistry[envKey]
  if (!e) throw new DomainError('gymEnvironment', `gymEnvironment: no environment '${envKey}' in the gym registry`)
  return (e as unknown as (p: object) => AnyEnv)(envParams)
}

/** The setup for environment `envKey` and agent `agentKey` of the gym registries, built from their parameters. */
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
