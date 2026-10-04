/**
 * Philox4x32-10, the counter-based generator of Salmon, Moraes, Dror and Shaw (2011), "Parallel random numbers: as easy
 * as 1, 2, 3", SC'11. It is a 10-round keyed bijection of a 128-bit counter: output block i of a stream is
 * philox(counter = i, key). Nothing is carried between blocks, so any block can be computed directly, and streams with
 * different keys are statistically independent (Philox4x32-10 passes TestU01's BigCrush for every key).
 *
 * Everything is done in 32-bit integer arithmetic: `Math.imul` for the low half of each 32×32-bit product, and 16-bit
 * limbs for the high half.
 */

const M0 = 0xd2511f53
const M1 = 0xcd9e8d57
const W0 = 0x9e3779b9 // the golden ratio, the first Weyl key increment
const W1 = 0xbb67ae85 // √3 − 1, the second

/** The high 32 bits of the 64-bit product of two uint32 values. */
function mulhi(a: number, b: number): number {
  const al = a & 0xffff
  const ah = a >>> 16
  const bl = b & 0xffff
  const bh = b >>> 16
  const lh = al * bh
  const hl = ah * bl
  const mid = ((al * bl) >>> 16) + (lh & 0xffff) + (hl & 0xffff)
  return (ah * bh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0
}

/**
 * Philox4x32-10: encrypt the counter (c0, c1, c2, c3) under the key (k0, k1), all uint32, writing four uint32 words to
 * `out` (length ≥ 4). Matches the Random123 known-answer vectors.
 */
export function philox4x32(
  c0: number,
  c1: number,
  c2: number,
  c3: number,
  k0: number,
  k1: number,
  out: Uint32Array | number[],
): void {
  for (let round = 0; round < 10; round++) {
    const hi0 = mulhi(M0, c0)
    const lo0 = Math.imul(M0, c0) >>> 0
    const hi1 = mulhi(M1, c2)
    const lo1 = Math.imul(M1, c2) >>> 0
    c0 = (hi1 ^ c1 ^ k0) >>> 0
    c1 = lo1
    c2 = (hi0 ^ c3 ^ k1) >>> 0
    c3 = lo0
    k0 = (k0 + W0) >>> 0
    k1 = (k1 + W1) >>> 0
  }
  out[0] = c0
  out[1] = c1
  out[2] = c2
  out[3] = c3
}

/** A 128-bit stream identity: the Philox key (h0, h1) and the two upper counter words (h2, h3). */
export type KeyHash = readonly [number, number, number, number]

/** The initial hash state: the first 128 fractional bits of π (as in the Random123 test vectors). */
export const ROOT_HASH: KeyHash = [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]
/** The fixed key of the absorbing permutation (the next 64 bits of π). */
const HASH_KEY0 = 0xa4093822
const HASH_KEY1 = 0x299f31d0

/**
 * Absorb 32-bit words into a 128-bit hash state, sponge fashion: XOR each word into the state and apply Philox4x32-10
 * under a fixed key (a fixed bijection of 128 bits). Callers encode their input injectively (lengths first), so distinct
 * key paths give distinct inputs; distinct inputs collide with probability about 2⁻⁹⁶ per pair.
 */
export function absorb(state: KeyHash, words: readonly number[]): KeyHash {
  const s = [state[0], state[1], state[2], state[3]]
  for (const w of words) {
    s[0] = (s[0] ^ w) >>> 0
    philox4x32(s[0], s[1], s[2], s[3], HASH_KEY0, HASH_KEY1, s)
  }
  return [s[0], s[1], s[2], s[3]]
}
