/**
 * `aifn-methods/neural/grokking`: delayed generalisation on modular arithmetic, a network that fits its training
 * pairs early and generalises to the held-out ones much later.
 *
 * - The model: `ModularMlp`, an embedding MLP for $a \circ b \bmod p$ (shared residue embedding, one ReLU layer,
 *   logits over the $p$ answers), and `pairsOf`, which reads a part of the table as the operand columns it takes.
 * - The run: `grokkingRun` trains it by full-batch AdamW (or L-BFGS on the weight-decay penalty) and yields snapshots
 *   with the training and test accuracy and loss, the weight norm, and checkpointed parameters.
 *
 * The data are the parts of `aifn-methods/data`'s `modularArithmetic`: features $n \times 2$ (the operands) and labels
 * (the answers). Runs are deterministic from their seed.
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
