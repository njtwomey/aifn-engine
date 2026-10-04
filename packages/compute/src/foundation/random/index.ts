/**
 * `aifn-compute/foundation/random`: keyed, counter-based random streams (Philox4x32-10) as plain data, and samplers that take
 * the stream first.
 *
 * ```ts
 * const s = stream(7)
 * const chain = child(s, 'chain', 3) // key "7/chain:3"; the same draws however much `s` has drawn
 * const x = normal(chain, 0, 1)
 * const runs = replicate(20, child(s, 'runs'), (r) => simulate(r))
 * ```
 *
 * A `Stream` is `{ key, position }`: a draw depends only on the key and the position, and advances the position.
 * `randomBits` is the primitive every draw is built on. Samplers take number or tensor parameters with broadcasting
 * and an optional `{ shape }`; numbers in give a number out, otherwise a tensor filled in row-major order. They are not
 * differentiable (see samplers.ts for the words each draw uses).
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
