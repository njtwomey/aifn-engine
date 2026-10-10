/**
 * Keyed, counter-based random streams as plain data (design K §6).
 *
 * A key is identified by a path of names, e.g. `7/chain:3/env` for `child(child(stream(7), 'chain', 3), 'env')`. The
 * path is hashed to 128 bits (see `absorb` in philox.ts); the hash supplies the Philox key and the upper half of the
 * counter, and the lower 64 bits of the counter count the key's output blocks of four words. A stream is a key and a
 * position, so its $n$-th word depends only on the path and $n$: drawing from a parent or a sibling never changes a
 * child's values, every path gets its own sequence, and a stream copied (or sent to a worker) draws the same values.
 * Drawing advances the stream's `position` in place; `fork` gives an independent copy.
 */

import type { Key, Size, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { absorb, philox4x32, ROOT_HASH } from './philox'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Key, Stream } from 'aifn-compute/foundation/contracts'

const LEVEL = 0x2f2f2f2f // marks the start of a path level
const ROOT = 0x5eed5eed // marks the root

/**
 * Encode one level of a path injectively as 32-bit words, for `absorb`.
 *
 * @param marker The word that starts the level: one for the root, another for a child level.
 * @param parts The names of the level's parts, as strings.
 * @returns The marker, the number of parts, then for each part its length and its UTF-16 code units.
 */
function encodeLevel(marker: number, parts: string[]): number[] {
  const words = [marker, parts.length]
  for (const p of parts) {
    words.push(p.length)
    for (let i = 0; i < p.length; i++) words.push(p.charCodeAt(i))
  }
  return words
}

/**
 * The display form of one path part: `%`, `/` and `:` are percent-encoded so that keys map one-to-one to paths.
 *
 * @param part The part's name.
 * @returns The name with `%` as `%25`, `/` as `%2F` and `:` as `%3A`.
 */
function escapePart(part: string): string {
  return part.replace(/[%/:]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`)
}

/**
 * The name of one path part (or seed): a number by its decimal form, so `3` and `'3'` name the same part. Throws
 * `DomainError` for a number that is not finite.
 *
 * @param part The part as the caller gave it.
 * @returns Its name, as a string.
 */
function partName(part: string | number): string {
  if (typeof part === 'number' && !Number.isFinite(part))
    throw new DomainError('stream', `stream: path part ${part} is not finite`)
  return String(part)
}

const TWO_POW_32 = 4294967296

/**
 * The root stream for a seed, at position 0: every draw from it, or from its children, is fixed by the seed. Numbers
 * and strings name the same root when their decimal forms agree: `stream(7)` equals `stream('7')`. Throws
 * `DomainError` for a number that is not finite.
 *
 * @param seed The seed: any finite number or any string.
 * @returns A new stream `{ key, position: 0 }`, whose `key.path` is the seed's name.
 *
 * @example A seed fixes every draw
 * const s = stream(7)
 * print('path =', s.key.path)
 * print('draws =', uniform(s, 0, 1, { shape: [3] }))
 * print('again =', uniform(stream(7), 0, 1, { shape: [3] }))
 *
 * @example A number and its decimal string are the same seed
 * print('stream(7):', randomBits(stream(7), 2))
 * print("stream('7'):", randomBits(stream('7'), 2))
 * print('stream(8):', randomBits(stream(8), 2))
 */
export function stream(seed: number | string): Stream {
  const name = partName(seed)
  const hash = absorb(ROOT_HASH, encodeLevel(ROOT, [name]))
  return { key: { path: escapePart(name), hash }, position: 0 }
}

/**
 * An independent stream named by `path` under a stream's key (one level; several parts are joined by `:` in the
 * path), at position 0. It never collides with its siblings, its parent or any other path, and its values do not
 * depend on how much the parent has drawn. Numbers name the same child as their decimal strings: `child(s, 3)` equals
 * `child(s, '3')`. Throws `DomainError` for a number part that is not finite.
 *
 * @param s The parent: a stream, or just its key (its position is ignored either way).
 * @param path The parts naming the child: strings or finite numbers.
 * @returns A new stream at position 0 whose key path is the parent's, then `/`, then the parts joined by `:` (a lone
 *   `%` when there are no parts, so that it differs from the path of one empty part).
 *
 * @example A child's draws do not depend on its parent's
 * const s = stream(7)
 * const chain = child(s, 'chain', 3)
 * print('path =', chain.key.path)
 * print('draw =', uniform(chain))
 * const parentDraws = uniform(s, 0, 1, { shape: [100] })
 * print('after the parent drew 100:', uniform(child(s, 'chain', 3)))
 *
 * @example Siblings differ, and nest to any depth
 * const s = stream(7)
 * print('chain 0:', uniform(child(s, 'chain', 0)))
 * print('chain 1:', uniform(child(s, 'chain', 1)))
 * print('nested path =', child(child(s, 'chain', 0), 'env').key.path)
 */
export function child(s: Stream | Key, ...path: (string | number)[]): Stream {
  const key = 'key' in s ? s.key : s
  const parts = path.map(partName)
  const hash = absorb(key.hash, encodeLevel(LEVEL, parts))
  // No parts shows as a lone `%`, which no escaped name contains, so `child(s)` and `child(s, '')` differ in path.
  const level = parts.length === 0 ? '%' : parts.map(escapePart).join(':')
  return { key: { path: `${key.path}/${level}`, hash }, position: 0 }
}

/**
 * An independent copy of a stream (same key and position): drawing from one does not advance the other, so the two
 * draw the same values. Use it to replay draws; use `child` for new, independent ones.
 *
 * @param s The stream to copy; not modified.
 * @returns A new stream with the key and position of `s`.
 *
 * @example A fork replays the draws
 * const s = stream(3)
 * const replay = fork(s)
 * print('s:', uniform(s, 0, 1, { shape: [2] }))
 * print('fork:', uniform(replay, 0, 1, { shape: [2] }))
 * print('positions:', s.position, replay.position)
 */
export function fork(s: Stream): Stream {
  return { key: s.key, position: s.position }
}

/**
 * The random primitive: the next $n$ 32-bit words of a stream, words $[p, p + n)$ of its key's counter space for
 * position $p$, as a `Uint32Array`. Advances `s.position` by $n$. Every other draw in aifn is a function of these
 * words; none is differentiable. Integer outputs are bit-identical on every platform. Throws `DomainError` unless
 * $n$ is a non-negative integer.
 *
 * @param s The stream to draw from; its `position` is advanced in place.
 * @param n The number of words to draw.
 * @returns A new `Uint32Array` of the $n$ words.
 *
 * @example Words depend only on the position, not on how they were split
 * const s = stream(0)
 * print('4 words =', randomBits(s, 4))
 * print('2 more =', randomBits(s, 2))
 * print('position =', s.position)
 * print('6 at once =', randomBits(stream(0), 6))
 */
export function randomBits(s: Stream, n: Size): Uint32Array {
  if (!(Number.isInteger(n) && n >= 0))
    throw new DomainError('randomBits', `randomBits: n must be a non-negative integer, got ${n}`)
  const out = new Uint32Array(n)
  const [h0, h1, h2, h3] = s.key.hash
  const start = s.position
  const block = new Uint32Array(4)
  let i = 0
  while (i < n) {
    const p = start + i
    const b = Math.floor(p / 4)
    // Counter = (block low 32 bits, block high bits, h2, h3); key = (h0, h1).
    philox4x32(b >>> 0, Math.floor(b / TWO_POW_32) >>> 0, h2, h3, h0, h1, block)
    for (let w = p & 3; w < 4 && i < n; w++, i++) out[i] = block[w]
  }
  s.position = start + n
  return out
}
