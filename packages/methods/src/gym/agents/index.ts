/**
 * `aifn-methods/gym/agents`: agents on the gym protocol, from bandit policies to deep reinforcement learning.
 *
 * - The baseline: `randomAgent`, uniform over the legal actions.
 * - Bandit policies on arm statistics: `uniformPolicy`, `exploreThenCommit`, `epsilonGreedy`, `ucb1`, `klUcb`,
 *   `thompsonBernoulli`, `thompsonGaussian` and `exp3` (adversarial); on a $K \times d$ context, `linUcb` and
 *   `linearThompson`. With them, the KL-UCB index (`klBernoulli`, `klUcbIndex`) and the Lai–Robbins regret bound
 *   (`laiRobbinsBound`).
 * - Tabular learning on discrete observations and actions: `qLearningAgent`, `sarsaAgent`, `expectedSarsaAgent` (all
 *   `tdControlAgent`), `nStepSarsaAgent`, `monteCarloControlAgent`, `tdPredictionAgent` and the softmax
 *   `reinforceAgent`; `greedyPath` reads a route off a policy.
 * - Planning on a known MDP's tables, as traceable algorithms run by `run`: `valueIteration`, `policyIteration` and
 *   `policyEvaluation`, with the exact `evaluatePolicy`, `greedyPolicy` and `valuesFromQ`; as agents that plan in
 *   `init`, `valueIterationAgent` and `policyIterationAgent`, which need a tabular model.
 * - The deep Q-network, `dqnAgent`, with its pieces: the persistent replay buffer (`replayBuffer`, `pushTransition`,
 *   `transitionAt`, `bufferSize`, `sampleIndices`, `gatherMinibatch`), `qNetwork`, `qValues`, `tdTargets`, and the
 *   RL Baselines3 Zoo's CartPole recipe (`SB3_CARTPOLE`, `epsilonStepsFor`).
 * - From the child modules: `control` (LQR from autodiff Jacobians, the pendulum's swing-up, the cross-entropy
 *   method) and `policy` (REINFORCE with a baseline, A2C, PPO, DDPG, and offline CQL).
 * - The registry tables `planningAlgorithms` and `agentFunctions`.
 *
 * Agents' states are plain data: `init` and `act` draw from the stream they are given, and `learn` is a pure update
 * that returns a new state. Agents that need particular domains or a model check them in `init` and throw otherwise.
 */

export { randomAgent, type RandomAgentState } from './random'
export {
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  laiRobbinsBound,
  linearThompson,
  linUcb,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
  type ArmStatistics,
  type Exp3State,
  type RidgeState,
} from './bandits'
export {
  expectedSarsaAgent,
  greedyPath,
  monteCarloControlAgent,
  nStepSarsaAgent,
  qLearningAgent,
  reinforceAgent,
  sarsaAgent,
  tdControlAgent,
  tdPredictionAgent,
  type MonteCarloAgentState,
  type NStepAgentState,
  type ReinforceState,
  type TabularAgentState,
  type TabularOptions,
  type TdControlOptions,
  type TdAgentState,
  type TdMethod,
  type TdPredictionState,
} from './tabular'
export {
  evaluatePolicy,
  greedyPolicy,
  policyEvaluation,
  policyIteration,
  policyIterationAgent,
  valueIteration,
  valueIterationAgent,
  valuesFromQ,
  type PlannerState,
  type PolicyIterationState,
  type ValueState,
} from './planning'
export {
  lineariseDynamics,
  pendulumLqr,
  swingUpAgent,
  type Linearisation,
  type PendulumPlant,
  type SwingUpOptions,
  type SwingUpState,
  type TorqueAction,
} from './control'
export {
  crossEntropyAgent,
  linearPolicyAgent,
  lqrBangBangAgent,
  type CrossEntropyOptions,
  type CrossEntropyState,
  type GenerationSummary,
  type LqrBangBangOptions,
  type LqrBangBangState,
} from './control'
export {
  bufferSize,
  CHUNK,
  dqnAgent,
  SB3_CARTPOLE,
  SB3_CARTPOLE_STEPS,
  SB3_CARTPOLE_EPSILON_FRACTION,
  epsilonStepsFor,
  gatherMinibatch,
  pushTransition,
  qNetwork,
  qValues,
  replayBuffer,
  sampleIndices,
  tdTargets,
  transitionAt,
  type DqnOptions,
  type DqnState,
  type Minibatch,
  type ReplayBuffer,
  type StoredTransition,
} from './dqn'
export {
  a2cAgent,
  ddpgAgent,
  logTransitions,
  offlineComparison,
  offlineQLearning,
  ppoAgent,
  reinforceBaselineAgent,
  type DdpgOptions,
  type DdpgState,
  type LogOptions,
  type OfflineCheckpoint,
  type OfflineLog,
  type OfflineQOptions,
  type OnPolicyOptions,
  type OnPolicyState,
  type PpoOptions,
} from './policy'
export { planningAlgorithms, agentFunctions } from './registry'
