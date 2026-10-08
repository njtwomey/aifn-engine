/**
 * `aifn-methods/neural/post-training`: post-training a policy after pre-training, on toy policies small enough to see
 * whole: supervised fine-tuning, preference optimisation and group-relative policy optimisation, with the arithmetic
 * of their advantages and evaluation.
 *
 * - Advantages: `groupAdvantages` (GRPO's group mean and standard deviation, RLOO's leave-one-out baseline, Dr. GRPO
 *   without the scale) and `batchAdvantages` (REINFORCE++'s batch scale); `tokenWeights` spreads an advantage over a
 *   response's tokens (sequence mean, token mean, or a constant).
 * - KL and evaluation: `klEstimators` (Schulman's $k_1$, $k_2$, $k_3$ from one sample), `passAtK` (the unbiased
 *   estimator from $n$ samples) and `passAtKExact` ($1 - (1 - p)^k$).
 * - One prompt: `postTrainingTrace` trains a softmax over $K$ responses by SFT, DPO (or IPO, or SimPO) or GRPO and
 *   records every step with TRL's statistics, the KL to the start and the entropy.
 * - Many prompts: `grpoTrace` runs GRPO and its relatives (Dr. GRPO, DAPO's clip-higher and dynamic sampling) on any
 *   `GroupPolicyProblem`; `coverageProblem` builds the problem on which reinforcement learning with verifiable rewards
 *   sharpens a policy, and `solveProbabilities` gives each prompt's chance of being solved by one sample.
 *
 * The losses are `aifn-compute/learning/losses`' `dpo`, `ipo` and `simpo`, the gradients come from `valueAndGrad`, and
 * the optimisers are `adamRule` and `sgdRule` from `aifn-compute/optim/first-order`. Runs are deterministic from their
 * seed.
 */

export {
  batchAdvantages,
  groupAdvantages,
  klEstimators,
  passAtK,
  passAtKExact,
  tokenWeights,
  type AdvantageOptions,
  type Aggregation,
  type Baseline,
  type Scale,
} from './advantages'
export {
  coverageProblem,
  grpoTrace,
  solveProbabilities,
  type ClipRange,
  type CoverageOptions,
  type GroupPolicyProblem,
  type GrpoOptions,
  type GrpoSnapshot,
} from './grpo'
export {
  postTrainingTrace,
  type PostTrainingMethod,
  type PostTrainingOptions,
  type PostTrainingSnapshot,
  type ToyPolicy,
} from './toy'
export { postTrainingFunctions } from './registry'
