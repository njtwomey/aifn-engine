/**
 * `aifn-methods/gym/agents/policy`: policy-gradient and actor–critic agents with neural networks. On-policy, for
 * discrete actions: REINFORCE with a learned baseline, A2C and PPO (`on-policy.ts`). Off-policy, for one continuous
 * action: DDPG (`ddpg.ts`). Offline, from logged transitions: conservative Q-learning against offline DQN
 * (`offline.ts`).
 */

export {
  a2cAgent,
  ppoAgent,
  reinforceBaselineAgent,
  type OnPolicyOptions,
  type OnPolicyState,
  type PpoOptions,
} from './on-policy'
export { ddpgAgent, type DdpgOptions, type DdpgState } from './ddpg'
export {
  logTransitions,
  offlineComparison,
  offlineQLearning,
  type LogOptions,
  type OfflineCheckpoint,
  type OfflineLog,
  type OfflineQOptions,
} from './offline'
