/**
 * `aifn-methods/algorithms/dynamic-programming`: the 0/1 and unbounded knapsacks on the `dp` engine of
 * `aifn-compute/optim/programming` (which also holds the sequence programmes: LCS, edit distance and alignments).
 *
 * - Solve: `knapsack` (each item at most once) and `unboundedKnapsack` (any number of each) return a
 *   `KnapsackResult`: the best value, its weight, the items taken, the table and the traceback.
 * - The problems as `DynamicProgram`s, to fill with `dp` or step through with `dynamicProgram`: `knapsackProgram`
 *   (an $(n + 1) \times (C + 1)$ table) and `unboundedKnapsackProgram` (a table of $C + 1$ cells).
 * - `dynamicProgrammingFunctions`: the registry entries of the four.
 *
 * Weights and the capacity $C$ are non-negative integers and the solve is $O(nC)$ for $n$ items; invalid input throws
 * `DomainError` or `ShapeError`.
 */
export { knapsackProgram, type KnapsackResult, knapsack, unboundedKnapsackProgram, unboundedKnapsack } from './problems'
export { dynamicProgrammingFunctions } from './registry'
