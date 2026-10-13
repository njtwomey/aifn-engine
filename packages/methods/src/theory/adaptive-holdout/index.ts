/**
 * `aifn-methods/theory/adaptive-holdout`: holdouts that stay honest under adaptive reuse (the Ladder and
 * Thresholdout), the boosting attack that breaks a plain one, and the winner's curse of picking the best of $J$.
 *
 * - Leaderboards: `ladder` (Blum and Hardt's Ladder with step $\eta$) and `ladderParameterFree` (its step set by a
 *   paired $t$-test), on a $k \times n$ matrix of per-example losses; `LadderResult` holds what they report.
 * - A reusable holdout: `thresholdout` (Dwork et al., 2015), on the queries' values on a training set and a holdout,
 *   with Laplace noise on the threshold and on the answers and a budget of holdout answers.
 * - The attack: `boostingAttack` submits random label vectors to an `exact`, `rounded`, `ladder`, `parameter-free` or
 *   `thresholdout` leaderboard and votes over those that score well; it reports the vote's holdout loss, its true loss
 *   and what the leaderboard says, after every submission.
 * - Selection without adaptivity: `expectedGaussianMaximum`, $\expect\max_{j \le J} Z_j$ for independent standard
 *   normals, against which the best of $J$ fixed candidates' score is inflated.
 *
 * Every random function takes a stream and draws from named children of it, so a seed fixes the whole transcript, and
 * returns typed arrays and plain objects that cross to a worker unchanged. `adaptiveHoldoutFunctions` registers them.
 */

export { boostingAttack, type AttackMechanism, type BoostingAttackOptions, type BoostingAttackResult } from './attack'
export { ladder, ladderParameterFree, type LadderResult } from './ladder'
export { expectedGaussianMaximum } from './maximum'
export { adaptiveHoldoutFunctions } from './registry'
export { thresholdout, type ThresholdoutOptions, type ThresholdoutResult } from './thresholdout'
