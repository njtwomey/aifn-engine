/**
 * `aifn-compute/interpreter`: run small JavaScript programs against aifn, for data made from code.
 *
 * - Running: `runProgram(source, { seed, prelude, entry, args })` compiles the program with `new Function`, the
 *   prelude's names passed in, and returns `{ ok, value | error, output, ms }`: the top-level `return` value, or the
 *   value of the entry function (`make` by default) when there is none. `checkProgram(source)` reports a syntax error
 *   only, without running anything. Errors come back as values, with a line and column where the engine gives one.
 * - The prelude: `corePrelude` holds the namespaces `CORE_NAMESPACES` that mirror aifn's tree: `math` (elementwise,
 *   generated from the primitive registry), `array`, `random` (seeded), `stats`, `linalg` and `signal`; `seed`,
 *   `print`, `Math` (mapped to `math`, with a seeded `Math.random`) and `console.log` are top-level. Applications add
 *   namespaces (`learn`) with `makePrelude` and `withPrelude`. Each entry records its source module, parameters and
 *   doc, which drive evaluation, completion and reference lists alike: `lookup`, `members`, `qualified` and
 *   `signature` read them, and `params` writes a parameter list from a compact spec.
 * - Values: prelude functions take numbers, plain arrays or tensors and return numbers or plain arrays (`toValue`,
 *   `toTensor`, `toData`, `toShape`, with `describe` for error messages); `math.add`, `math.mul` and the other
 *   elementwise operators broadcast.
 * - Controls: `entrySignature(source)` reads the entry function's parameters without running it (literal defaults,
 *   types inferred or from a JSDoc `@param {int} n [10, 1000] doc` block, the type names in `PARAM_TYPES`), and
 *   `entrySpace` gives a `Space` of those that can have a control.
 *
 * Seeds: each random call draws from its own child of the run's root stream (`rootStream`), keyed by call order, so
 * the same program and seed give the same data; the program sees no streams, and can call `seed(s)`. A program runs
 * on the caller's thread with no step budget: run untrusted or long ones in a worker that can be terminated.
 */
export { checkProgram, runProgram, type RunError, type RunOptions, type RunResult } from './run'
export { entrySignature, entrySpace, PARAM_TYPES, type EntryParam, type EntrySignature, type ParamType } from './entry'
export { CORE_NAMESPACES, corePrelude, rootStream } from './core-prelude'
export {
  lookup,
  members,
  params,
  makePrelude,
  qualified,
  signature,
  withPrelude,
  type Context,
  type Namespace,
  type Param,
  type Prelude,
  type PreludeEntry,
} from './prelude'
export { describe, toData, toShape, toTensor, toValue, type Data, type Input } from './values'
