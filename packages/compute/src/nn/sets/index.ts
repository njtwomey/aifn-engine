/**
 * `aifn-compute/nn/sets`: permutation-invariant pooling over sets, for Deep Sets and PointNet.
 *
 * - `padSets`: sets of different sizes as one padded `[B, K, d]` tensor with a `[B, K]` mask.
 * - `setPool`: the masked sum, mean or maximum over each set's elements, `[B, K, m]` to `[B, m]`.
 * - `criticalSet`: the elements that win a max-pool in at least one channel (PointNet's critical points).
 *
 * Padding never contributes, so a pool equals the pool of the set alone and does not depend on the order of its
 * elements. An empty set is an error (`DomainError`), not a silent 0 or NaN. `setPool` is differentiable in the
 * features.
 */

export { criticalSet, padSets, setPool, type PaddedSets, type SetPooling } from './sets'
export { setsFunctions } from './registry'
