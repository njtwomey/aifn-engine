/**
 * `aifn-methods/gym/agents/policy`: policy-gradient and actor–critic agents with neural networks.
 *
 * - On-policy, for a box observation and discrete actions, with a categorical policy network and a value network:
 *   `reinforceBaselineAgent` (an update per episode), `a2cAgent` (every $n$ steps, on $n$-step returns) and `ppoAgent`
 *   (epochs of minibatches per rollout, on the clipped surrogate with GAE($\lambda$) advantages).
 * - Off-policy, for one continuous action: `ddpgAgent`, a tanh actor and a Q critic from a replay buffer.
 * - Offline, from logged transitions: `logTransitions` logs a behaviour policy, `offlineQLearning` trains a Q-network
 *   on the log with or without the conservative (CQL) term, and `offlineComparison` runs both on one log; the two
 *   trainers are generators that a worker can stream.
 *
 * The agents' states are plain data, and every draw in `learn` comes from a seed fixed in `init`, so learning is a
 * pure function of the state and the transition. Their `init` throws `TypeError` for domains they do not support.
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
