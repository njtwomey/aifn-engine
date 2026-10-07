/**
 * `aifn-compute/signal/sparse`: sparse representations over a dictionary, as scikit-learn's `orthogonal_mp`,
 * `sparse_encode` and `DictionaryLearning`, and the pursuits of Mallat's "A Wavelet Tour of Signal Processing".
 *
 * - Greedy pursuits, one atom per step: `matchingPursuit` (an atom may recur) and `orthogonalMatchingPursuit` (least
 *   squares on the support, exactly $s$ non-zeros after $s$ steps), with their step-through forms
 *   `matchingPursuitSteps` and `orthogonalMatchingPursuitSteps`.
 * - Convex and thresholding methods: `basisPursuit` (least $\ell_1$ norm with $\Dmat\xvec = \yvec$, a linear program),
 *   `basisPursuitDenoising` (the lasso, by FISTA or ISTA) and `iterativeHardThresholding` (at most $s$ non-zeros), with
 *   `basisPursuitDenoisingSteps` and `iterativeHardThresholdingSteps`.
 * - Many signals at once: `sparseCode` codes every column of $\Ymat$ by any of these methods.
 * - Learning the dictionary: `dictionaryLearning` and `dictionaryLearningSteps`, alternating orthogonal matching
 *   pursuit with a K-SVD or method-of-optimal-directions update.
 * - Dictionaries and sparse vectors: `mutualCoherence` (which bounds when the pursuits recover the sparsest
 *   representation), `normaliseAtoms` and `hardThreshold`.
 *
 * Dictionaries hold their atoms as columns, $\Dmat$ is $m \times k$, and a signal is $\yvec \approx \Dmat\xvec$; a
 * matrix of signals $\Ymat$ ($m \times n$) holds one per column, and its codes $\Xmat$ ($k \times n$) one per column.
 * The solvers reuse `aifn-compute/numerics/linalg` (`lstsq`, `svd`) and `aifn-compute/optim` (`linprog`,
 * `proximalGradient` with `proxL1`).
 */

export { hardThreshold, mutualCoherence, normaliseAtoms } from './atoms'
export {
  matchingPursuit,
  matchingPursuitSteps,
  orthogonalMatchingPursuit,
  orthogonalMatchingPursuitSteps,
  type PursuitOptions,
  type PursuitState,
  type SparseApproximation,
} from './pursuit'
export {
  basisPursuit,
  basisPursuitDenoising,
  basisPursuitDenoisingSteps,
  iterativeHardThresholding,
  iterativeHardThresholdingSteps,
  type BasisPursuitDenoisingOptions,
  type BasisPursuitResult,
  type HardThresholdingOptions,
} from './convex'
export { sparseCode, type SparseCoder, type SparseCodes } from './code'
export {
  dictionaryLearning,
  dictionaryLearningSteps,
  type DictionaryLearningOptions,
  type DictionaryLearningState,
  type DictionaryUpdate,
} from './dictionary'
export { sparseAlgorithms, sparseFunctions } from './registry'
