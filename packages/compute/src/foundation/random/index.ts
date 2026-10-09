/**
 * `aifn-compute/foundation/random`: keyed, counter-based random streams (Philox4x32-10) as plain data, and samplers
 * that take the stream first.
 *
 * - Streams: `stream(seed)` is a root; `child(s, ...path)` an independent stream named under it, whose draws do not
 *   depend on how much the parent has drawn; `fork` a copy that replays the same draws. A `Stream` is
 *   `{ key, position }`, and a draw depends only on the two.
 * - The primitive: `randomBits` draws 32-bit words and advances the position; `philox4x32` is the block cipher under
 *   it.
 * - Continuous draws: `uniform`, `normal` (Box–Muller) and `normals`, `exponential`.
 * - Discrete draws: `bernoulli`, `integers` (without modulo bias), `categorical` (a linear scan per draw), or
 *   `aliasTable` then `aliasSample` for many draws from the same weights; `shuffle` (in place), `permutation` and
 *   `choice` (with or without replacement, optionally weighted).
 * - Replicates: `replicate(n, s, fn)` runs `fn` on `child(s, k)` for each $k$, caching by key, for prefix reuse and
 *   common random numbers.
 * - Building blocks for samplers elsewhere (`aifn-compute/probability`): `units` and `standardNormals` (blocks of base
 *   variates), `boundedIntegers`, `drawBlock` (a fixed number of words per draw) and `drawEach` (rejection loops, one
 *   child stream per element), and `checkBroadcast`, `eventRows`, `batchIndex` for parameter shapes. `randomFunctions`
 *   is the registry of the draws.
 *
 * Samplers take number or tensor parameters with broadcasting and an optional `{ shape }`; numbers in give a number
 * out, otherwise a tensor filled in row-major order. Every draw advances the stream it is given, in place, by a number
 * of words fixed in advance (samplers.ts lists them). None is differentiable.
 */

export { child, fork, randomBits, stream, type Key, type Stream } from './stream'
export { philox4x32 } from './philox'
export {
  aliasSample,
  aliasTable,
  bernoulli,
  categorical,
  choice,
  exponential,
  integers,
  normal,
  normals,
  permutation,
  shuffle,
  uniform,
  type AliasTable,
  type ChoiceOptions,
  type Drawn,
  type Param,
  type SampleOptions,
  type Spread,
} from './samplers'
export { replicate, type ReplicateCache } from './replicate'

// Building blocks for samplers written on blocks and child keys (`aifn-compute/probability/samplers`, distributions).
export {
  batchIndex,
  boundedIntegers,
  checkBroadcast,
  drawBlock,
  drawEach,
  eventRows,
  standardNormals,
  units,
} from './samplers'
export { randomFunctions } from './registry'
