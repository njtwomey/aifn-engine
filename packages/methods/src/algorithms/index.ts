/**
 * `aifn-methods/algorithms`: classic algorithms as worked problems on the engines of `aifn-compute`.
 *
 * - `dynamic-programming`: the 0/1 and unbounded knapsacks (`knapsack`, re-exported here, and `unboundedKnapsack`),
 *   each also given as a `DynamicProgram` for the `dp` engine of `aifn-compute/optim/programming`, with the table and
 *   the traceback.
 *
 * The sequence programmes (longest common subsequence, edit distance, alignments) live in
 * `aifn-compute/optim/programming` itself.
 */

export { knapsack } from './dynamic-programming'
