/**
 * Randomness: keys and streams as plain data (design K §6). A `Key` names a counter space, a `Stream` is a cursor over
 * it, and a draw depends only on (key, position). Both are structured-cloneable, so a stream sent to a worker or stored
 * in a checkpoint reproduces its draws exactly. `aifn-compute/foundation/random` implements the functions over them.
 */

import type { Shape, Size } from './numbers'

/**
 * A key: a path of names (e.g. `7/chain:3/env`) and its 128-bit hash, which supplies the Philox key and the upper half
 * of the counter. Plain and structured-cloneable.
 */
export interface Key {
  /** The human-readable path. Separators inside names are percent-encoded. */
  readonly path: string
  /** The path's hash as four 32-bit words. */
  readonly hash: readonly [number, number, number, number]
}

/**
 * A stream: a cursor over a key's counter space, in 32-bit words. Two numbers and a string, so a stream sent to a
 * worker reproduces its draws exactly. Randomness in any registered entry comes only from a stream. Drawing advances
 * `position`; nothing else about a stream changes, and no draw depends on anything but `key` and `position`.
 */
export interface Stream {
  /** The key whose counter space the stream reads. */
  readonly key: Key
  /** Words drawn so far; `randomBits(s, n)` advances it by $n$. */
  position: Size
}

/** Options of a sampler or a distribution's `sample`. */
export type SampleOptions = {
  /**
   * The sample shape. Draws have shape `[...shape, ...batchShape, ...eventShape]` (PyTorch's convention). Omitted: one
   * draw per batch member (a number for an unbatched univariate law or all-number sampler parameters).
   */
  shape?: Shape
}
