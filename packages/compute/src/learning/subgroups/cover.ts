/**
 * Covers as bitsets: the rows a description selects, one bit per row in 32-bit words. Conjunction is a word-wise AND
 * and support a population count, so evaluating a refinement costs n/32 word operations.
 */

/** A set of row indices in [0, n): bit i of word i >> 5. Bits at or past n are always 0. */
export type Bitset = Uint32Array

const words = (n: number) => (n + 31) >>> 5

/** The rows i in [0, n) for which `member(i)` holds. */
export function bitset(n: number, member: (i: number) => boolean): Bitset {
  const out = new Uint32Array(words(n))
  for (let i = 0; i < n; i++) if (member(i)) out[i >>> 5] |= 1 << (i & 31)
  return out
}

/** Every row of [0, n). */
export function bitsetFull(n: number): Bitset {
  return bitset(n, () => true)
}

/** True when row i is in the set. */
export function bitsetHas(a: Bitset, i: number): boolean {
  return ((a[i >>> 5] >>> (i & 31)) & 1) === 1
}

/** a ∩ b. */
export function bitsetAnd(a: Bitset, b: Bitset): Bitset {
  const out = new Uint32Array(a.length)
  for (let w = 0; w < a.length; w++) out[w] = a[w] & b[w]
  return out
}

/** a \ b. */
export function bitsetAndNot(a: Bitset, b: Bitset): Bitset {
  const out = new Uint32Array(a.length)
  for (let w = 0; w < a.length; w++) out[w] = a[w] & ~b[w]
  return out
}

/** The complement of a within [0, n). */
export function bitsetNot(a: Bitset, n: number): Bitset {
  return bitsetAndNot(bitsetFull(n), a)
}

const popcount32 = (v: number) => {
  v = v - ((v >>> 1) & 0x55555555)
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333)
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24
}

/** |a|. */
export function bitsetCount(a: Bitset): number {
  let c = 0
  for (let w = 0; w < a.length; w++) c += popcount32(a[w])
  return c
}

/** |a ∩ b|, without building the intersection. */
export function bitsetAndCount(a: Bitset, b: Bitset): number {
  let c = 0
  for (let w = 0; w < a.length; w++) c += popcount32(a[w] & b[w])
  return c
}

/** The rows of a, ascending. */
export function bitsetIndices(a: Bitset): Int32Array {
  const out = new Int32Array(bitsetCount(a))
  let k = 0
  for (let w = 0; w < a.length; w++) {
    let v = a[w]
    while (v !== 0) {
      const low = v & -v
      out[k++] = (w << 5) + 31 - Math.clz32(low)
      v ^= low
    }
  }
  return out
}

/** |a ∩ b| / |a ∪ b| (1 for two empty sets). */
export function bitsetJaccard(a: Bitset, b: Bitset): number {
  const both = bitsetAndCount(a, b)
  const union = bitsetCount(a) + bitsetCount(b) - both
  return union === 0 ? 1 : both / union
}
