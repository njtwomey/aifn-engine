/**
 * `aifn-compute/interpreter`: run small JavaScript programs against aifn, for data made from code.
 *
 * ```ts
 * const result = runProgram(`
 *   seed(7)
 *   function make(n = 200, noise = 0.3) {
 *     const x = array.linspace(0, 6, n)
 *     const e = random.normal(n)
 *     const y = x.map((v, i) => 1.5 * v + 2 * math.sin(v) + noise * e[i])
 *     return [x, y]
 *   }`)
 * ```
 *
 * - `runProgram(source, { seed, prelude, entry, args })` compiles the program with `new Function`, the prelude's
 *   namespaces passed in, and returns `{ ok, value | error, output, ms }`; `checkProgram(source)` reports a syntax
 *   error only.
 * - The prelude's namespaces mirror aifn's tree: `math` (elementwise, generated from the primitive registry), `array`,
 *   `random` (seeded), `stats`, `linalg`, `signal`; applications add more (`learn`) with `withPrelude`. Each entry
 *   records its source module, parameters and doc, which drive evaluation, completion and reference lists alike.
 *   `seed`, `print`, `Math` (mapped to `math`, with a seeded `Math.random`) and `console.log` are top-level.
 * - Prelude functions take numbers, plain arrays or tensors and return numbers or plain arrays; `math.add`,
 *   `math.mul` and the other elementwise operators broadcast.
 * - `entrySignature(source)` reads the entry function's parameters (literal defaults, types inferred or from a JSDoc
 *   `@param {int} n [10, 1000] doc` block) and gives a `Space` of them, for controls.
 * - Seeds: each random call draws from its own child of the run's root stream, keyed by call order; no streams are
 *   exposed. A program can call `seed(s)`.
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
