/**
 * Philox4x32-10, the counter-based generator of Salmon, Moraes, Dror and Shaw (2011), "Parallel random numbers: as easy
 * as 1, 2, 3", SC'11, and the hash that turns stream paths into its keys.
 *
 * Philox is a 10-round keyed bijection of a 128-bit counter: output block $i$ of a stream is the encryption of counter
 * $i$ under the stream's key. Nothing is carried between blocks, so any block can be computed directly, and streams
 * with different keys are statistically independent (Philox4x32-10 passes TestU01's BigCrush for every key tested).
 *
 * Everything is done in 32-bit integer arithmetic: `Math.imul` for the low half of each $32 \times 32$-bit product, and
 * 16-bit limbs for the high half.
 */

const M0 = 0xd2511f53
const M1 = 0xcd9e8d57
const W0 = 0x9e3779b9 // the golden ratio, the first Weyl key increment
const W1 = 0xbb67ae85 // √3 − 1, the second

/**
 * The high 32 bits of the 64-bit product of two uint32 values, $\lfloor ab / 2^{32} \rfloor$, from 16-bit limbs (a
 * float64 holds only 53 bits of the product).
 *
 * @param a The first factor, an integer in $[0, 2^{32})$.
 * @param b The second factor, an integer in $[0, 2^{32})$.
 * @returns The high word of $ab$, as a uint32.
 */
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
 * Philox4x32-10: encrypt a 128-bit counter under a 64-bit key, ten rounds of two multiply-and-XOR S-boxes with the key
 * bumped by Weyl increments between rounds. Matches the Random123 known-answer vectors. A pure function of its
 * arguments: `randomBits` calls it with the block index as the counter, and most code should draw through streams
 * instead.
 *
 * @param c0 Word 0 (lowest) of the counter, a uint32.
 * @param c1 Word 1 of the counter, a uint32.
 * @param c2 Word 2 of the counter, a uint32.
 * @param c3 Word 3 (highest) of the counter, a uint32.
 * @param k0 Word 0 of the key, a uint32.
 * @param k1 Word 1 of the key, a uint32.
 * @param out Where the four output words are written, in entries 0 to 3 (length at least 4; other entries are left
 *   alone). It may be an array also used for the inputs, since they are read first.
 *
 * @example Random123's known-answer test: counter and key all zero
 * const out = new Uint32Array(4)
 * philox4x32(0, 0, 0, 0, 0, 0, out)
 * print('block =', Array.from(out, (w) => w.toString(16)))
 *
 * @example Neighbouring counters give unrelated blocks
 * const out = new Uint32Array(4)
 * philox4x32(1, 0, 0, 0, 0, 0, out)
 * print('counter 1 =', Array.from(out, (w) => w.toString(16)))
 * philox4x32(2, 0, 0, 0, 0, 0, out)
 * print('counter 2 =', Array.from(out, (w) => w.toString(16)))
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

/**
 * A 128-bit stream identity, four uint32 words: the Philox key (words 0 and 1) and the two upper counter words (words 2
 * and 3).
 */
export type KeyHash = readonly [number, number, number, number]

/** The initial hash state: the first 128 fractional bits of $\pi$ (as in the Random123 test vectors). */
export const ROOT_HASH: KeyHash = [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]
/** The fixed key of the absorbing permutation (the next 64 bits of $\pi$). */
const HASH_KEY0 = 0xa4093822
const HASH_KEY1 = 0x299f31d0

/**
 * Absorb 32-bit words into a 128-bit hash state, sponge fashion: XOR each word into the first state word and apply
 * Philox4x32-10 under a fixed key (a fixed bijection of 128 bits). Callers encode their input injectively (lengths
 * first), so distinct key paths give distinct inputs; distinct inputs collide with probability about $2^{-96}$ per
 * pair.
 *
 * @param state The hash state to start from (`ROOT_HASH`, or a parent key's hash); not modified.
 * @param words The words to absorb, in order, each a uint32.
 * @returns The new hash state.
 */
export function absorb(state: KeyHash, words: readonly number[]): KeyHash {
  const s = [state[0], state[1], state[2], state[3]]
  for (const w of words) {
    s[0] = (s[0] ^ w) >>> 0
    philox4x32(s[0], s[1], s[2], s[3], HASH_KEY0, HASH_KEY1, s)
  }
  return [s[0], s[1], s[2], s[3]]
}
