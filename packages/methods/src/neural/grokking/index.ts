/**
 * `aifn-methods/neural/grokking`: delayed generalisation on modular arithmetic. An embedding MLP (`ModularMlp`) trained
 * by full-batch AdamW on part of the table of a ∘ b mod p (`grokkingRun`, a generator of snapshots with the training
 * and test curves, the weight norm and checkpointed parameters).
 */

export {
  grokkingRun,
  ModularMlp,
  pairsOf,
  type GrokkingCurves,
  type GrokkingRunOptions,
  type GrokkingSnapshot,
  type ModularMlpConfig,
  type ModularMlpParams,
  type Pairs,
} from './grokking'
