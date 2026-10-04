/**
 * Keyed, counter-based random streams as plain data (design K §6).
 *
 * A key is identified by a path of names, e.g. `7/chain:3/env` for `child(child(stream(7), 'chain', 3), 'env')`. The
 * path is hashed to 128 bits (see `absorb` in philox.ts); the hash supplies the Philox key and the upper half of the
 * counter, and the lower 64 bits of the counter count the key's output blocks of four words. A stream is a key and a
 * position, so its n-th word depends only on the path and n: drawing from a parent or a sibling never changes a
 * child's values, every path gets its own sequence, and a stream copied (or sent to a worker) draws the same values.
 */

import type { Key, Size, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { absorb, philox4x32, ROOT_HASH } from './philox'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Key, Stream } from 'aifn-compute/foundation/contracts'

const LEVEL = 0x2f2f2f2f // marks the start of a path level
const ROOT = 0x5eed5eed // marks the root

/** Encode one level of a path injectively as 32-bit words: marker, part count, then each part's length and code units. */
function encodeLevel(marker: number, parts: string[]): number[] {
  const words = [marker, parts.length]
  for (const p of parts) {
    words.push(p.length)
    for (let i = 0; i < p.length; i++) words.push(p.charCodeAt(i))
  }
  return words
}

/** The display form of one path part: `%`, `/` and `:` are percent-encoded so that keys map one-to-one to paths. */
function escapePart(part: string): string {
  return part.replace(/[%/:]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`)
}

function partName(part: string | number): string {
  if (typeof part === 'number' && !Number.isFinite(part))
    throw new DomainError('stream', `stream: path part ${part} is not finite`)
  return String(part)
}

const TWO_POW_32 = 4294967296

/**
 * The root stream for a seed, at position 0. Numbers and strings name the same root when their decimal forms agree:
 * `stream(7)` equals `stream('7')`.
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
 * `child(s, '3')`.
 */
export function child(s: Stream | Key, ...path: (string | number)[]): Stream {
  const key = 'key' in s ? s.key : s
  const parts = path.map(partName)
  const hash = absorb(key.hash, encodeLevel(LEVEL, parts))
  return { key: { path: `${key.path}/${parts.map(escapePart).join(':')}`, hash }, position: 0 }
}

/** An independent copy of a stream (same key and position): drawing from one does not advance the other. */
export function fork(s: Stream): Stream {
  return { key: s.key, position: s.position }
}

/**
 * The random primitive: the next `n` 32-bit words of a stream, words [position, position + n) of its key's counter
 * space, as a Uint32Array. Advances `s.position` by n. Every other draw in aifn is a function of these words; none is
 * differentiable. Integer outputs are bit-identical on every platform.
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
