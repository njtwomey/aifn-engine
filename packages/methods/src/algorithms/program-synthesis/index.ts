/**
 * `aifn-methods/algorithms/program-synthesis`: programming by example for string transformations, by a version-space
 * algebra in the manner of FlashFill.
 *
 * - The language: a program is a concatenation of `StringAtom`s (constants, tokens of the input with a case
 *   transformation, first characters of tokens); `stringTokens` splits an input, `evalAtom` and `runStringProgram`
 *   run atoms and programs, `atomText` prints an atom.
 * - The version space: `exampleVersionSpace` is the graph of every program consistent with one example,
 *   `intersectVersionSpace` adds an example (the product graph), and `versionSpace` does both for a list.
 * - Reading it: `countByLength` counts the consistent programs by length, and `topPrograms` returns the $k$ cheapest
 *   under `atomCost`.
 * - `programSynthesisFunctions`: the module's registry entries.
 *
 * The graph holds exponentially many programs in polynomial size, and counting and ranking are dynamic programmes over
 * it. Everything is plain data, and a program that does not apply to an input returns null rather than throwing.
 */

export {
  atomCost,
  atomText,
  countByLength,
  evalAtom,
  exampleVersionSpace,
  intersectVersionSpace,
  runStringProgram,
  stringTokens,
  topPrograms,
  versionSpace,
  type CaseTransform,
  type RankedProgram,
  type StringAtom,
  type StringExample,
  type StringVersionSpace,
} from './strings'
export { programSynthesisFunctions } from './registry'
