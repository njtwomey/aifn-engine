/** The registry of `aifn-methods/neural/post-training`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as advantages from './advantages'
import * as grpo from './grpo'
import * as toy from './toy'

const fn = definer<FunctionInfo>('function', 'neural/post-training')
const GRPO = ['group-relative-policy-optimisation']
const RLVR = ['reinforcement-learning-with-verifiable-rewards']

fn(
  {
    key: 'groupAdvantages',
    name: 'Group-relative advantages',
    tex: 'A_i = \\frac{r_i - \\bar r}{\\operatorname{std}(r) + \\epsilon}',
    summary: 'Each reward of a group scored against the group: GRPO, RLOO and Dr. GRPO by the baseline and scale.',
    role: 'transform',
    notes: [...GRPO, ...RLVR],
    cite: ['shao2024grpo', 'ahmadian2024', 'liu2025drgrpo'],
  },
  advantages.groupAdvantages,
)
fn(
  {
    key: 'batchAdvantages',
    name: 'Batch-normalised advantages',
    summary: 'Rewards centred within each group and scaled by the whole batch’s spread, as REINFORCE++ does.',
    role: 'transform',
    notes: GRPO,
    cite: ['hu2025reinforcepp'],
  },
  advantages.batchAdvantages,
)
fn(
  {
    key: 'tokenWeights',
    name: 'Token weights of a response’s advantage',
    summary: 'A response’s advantage spread over its tokens: per-response mean, token mean, or a constant.',
    role: 'transform',
    notes: GRPO,
    cite: ['shao2024grpo', 'yu2025dapo', 'liu2025drgrpo'],
  },
  advantages.tokenWeights,
)
fn(
  {
    key: 'klEstimators',
    name: 'KL estimators from one sample',
    tex: 'k_3 = \\rho - 1 - \\log\\rho',
    summary: 'Schulman’s k1, k2 and k3 estimates of KL(π ‖ π_ref) from one sampled response.',
    role: 'property',
    notes: GRPO,
    cite: ['schulman2020kl'],
  },
  advantages.klEstimators,
)
fn(
  {
    key: 'passAtK',
    name: 'pass@k (unbiased estimator)',
    tex: '1 - \\binom{n - c}{k} / \\binom{n}{k}',
    summary: 'The chance that at least one of k samples is correct, estimated from n samples of which c are correct.',
    role: 'property',
    notes: RLVR,
    cite: ['yue2025'],
  },
  advantages.passAtK,
)
fn(
  {
    key: 'passAtKExact',
    name: 'pass@k from a success probability',
    tex: '1 - (1 - p)^k',
    role: 'property',
    notes: RLVR,
    cite: ['yue2025'],
  },
  advantages.passAtKExact,
)
fn(
  {
    key: 'postTrainingTrace',
    name: 'Post-training a toy policy',
    summary: 'A softmax over a few responses trained by SFT, DPO (or IPO, SimPO) or GRPO, with every step recorded.',
    role: 'simulation',
    notes: [
      'adapting-pretrained-language-models',
      'direct-preference-optimisation',
      'group-relative-policy-optimisation',
      'reinforcement-learning-with-verifiable-rewards',
    ],
    cite: ['rafailov2023', 'shao2024grpo', 'razin2025'],
  },
  toy.postTrainingTrace,
)
fn(
  {
    key: 'grpoTrace',
    name: 'Group-relative policy optimisation',
    summary: 'GRPO and its relatives (Dr. GRPO, DAPO, RLOO baselines) on a many-prompt softmax policy, step by step.',
    role: 'simulation',
    notes: [...GRPO, ...RLVR],
    cite: ['shao2024grpo', 'liu2025drgrpo', 'yu2025dapo'],
  },
  grpo.grpoTrace,
)
fn(
  {
    key: 'coverageProblem',
    name: 'Strategy coverage problem',
    summary: 'Prompts sharing a few strategies, each solving a random subset: where RLVR sharpens rather than teaches.',
    role: 'construction',
    notes: RLVR,
    cite: ['yue2025'],
  },
  grpo.coverageProblem,
)
fn(
  {
    key: 'solveProbabilities',
    name: 'Per-prompt solve probability',
    summary: 'The chance that one sample from the policy solves each prompt.',
    role: 'property',
    notes: RLVR,
    cite: ['yue2025'],
  },
  grpo.solveProbabilities,
)

/** The functions of the module, keyed by name. */
export const postTrainingFunctions = entries<FunctionInfo>('function', advantages, grpo, toy) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
