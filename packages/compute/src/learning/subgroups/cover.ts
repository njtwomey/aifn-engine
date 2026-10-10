/**
 * Covers as bitsets: the rows a description selects, one bit per row in 32-bit words. Conjunction is a word-wise AND
 * and support a population count, so evaluating a refinement costs $n/32$ word operations.
 *
 * The binary operations take two bitsets over the same $n$ rows (equal word counts) and return a new one; none checks
 * that, or modifies its inputs.
 */

/**
 * A set of row indices in $[0, n)$: row $i$ is bit $i \bmod 32$ of word $\lfloor i/32 \rfloor$ (`i >> 5`). Bits at
 * or past $n$ are always 0.
 */
export type Bitset = Uint32Array

/**
 * The number of 32-bit words a bitset of $n$ rows takes, $\lceil n/32 \rceil$.
 *
 * @param n The number of rows.
 * @returns The word count.
 */
const words = (n: number) => (n + 31) >>> 5

/**
 * The rows $i$ in $[0, n)$ for which `member(i)` holds.
 *
 * @param n The number of rows of the table.
 * @param member The test of a row index, called once for each of $0, \dots, n - 1$.
 * @returns The set of rows that pass.
 *
 * @example The even rows of seven
 * const even = bitset(7, (i) => i % 2 === 0)
 * print('rows =', bitsetIndices(even), 'count =', bitsetCount(even))
 */
export function bitset(n: number, member: (i: number) => boolean): Bitset {
  const out = new Uint32Array(words(n))
  for (let i = 0; i < n; i++) if (member(i)) out[i >>> 5] |= 1 << (i & 31)
  return out
}

/**
 * Every row of $[0, n)$: the cover of the empty description.
 *
 * @param n The number of rows.
 * @returns The full set.
 *
 * @example All five rows
 * print('rows =', bitsetIndices(bitsetFull(5)))
 */
export function bitsetFull(n: number): Bitset {
  return bitset(n, () => true)
}

/**
 * True when row $i$ is in the set.
 *
 * @param a The set.
 * @param i The row index, in $[0, n)$.
 * @returns Whether bit $i$ is set.
 *
 * @example Membership of rows 2 and 3
 * const a = bitset(6, (i) => i < 3)
 * print('has 2:', bitsetHas(a, 2), ' has 3:', bitsetHas(a, 3))
 */
export function bitsetHas(a: Bitset, i: number): boolean {
  return ((a[i >>> 5] >>> (i & 31)) & 1) === 1
}

/**
 * The intersection $a \cap b$: the cover of a conjunction.
 *
 * @param a A set over $n$ rows.
 * @param b A set over the same rows.
 * @returns A new set.
 *
 * @example Rows below 4 that are even
 * const a = bitset(8, (i) => i < 4)
 * const b = bitset(8, (i) => i % 2 === 0)
 * print('a and b =', bitsetIndices(bitsetAnd(a, b)))
 */
export function bitsetAnd(a: Bitset, b: Bitset): Bitset {
  const out = new Uint32Array(a.length)
  for (let w = 0; w < a.length; w++) out[w] = a[w] & b[w]
  return out
}

/**
 * The difference $a \setminus b$: the rows of $a$ not in $b$.
 *
 * @param a A set over $n$ rows.
 * @param b A set over the same rows.
 * @returns A new set.
 *
 * @example Rows below 4 that are odd
 * const a = bitset(8, (i) => i < 4)
 * const b = bitset(8, (i) => i % 2 === 0)
 * print('a and not b =', bitsetIndices(bitsetAndNot(a, b)))
 */
export function bitsetAndNot(a: Bitset, b: Bitset): Bitset {
  const out = new Uint32Array(a.length)
  for (let w = 0; w < a.length; w++) out[w] = a[w] & ~b[w]
  return out
}

/**
 * The complement of $a$ within $[0, n)$: the rows outside a subgroup.
 *
 * @param a A set over $n$ rows.
 * @param n The number of rows, so the bits at or past $n$ stay 0.
 * @returns A new set.
 *
 * @example The complement of rows 1 and 2 among five
 * print('not a =', bitsetIndices(bitsetNot(bitset(5, (i) => i === 1 || i === 2), 5)))
 */
export function bitsetNot(a: Bitset, n: number): Bitset {
  return bitsetAndNot(bitsetFull(n), a)
}

/**
 * The number of set bits of a 32-bit word, by the parallel bit count (Warren, "Hacker's Delight", chapter 5).
 *
 * @param v The word, as an unsigned 32-bit integer.
 * @returns Its population count, 0 to 32.
 */
const popcount32 = (v: number) => {
  v = v - ((v >>> 1) & 0x55555555)
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333)
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24
}

/**
 * The size $\lvert a \rvert$: the support of a cover.
 *
 * @param a The set.
 * @returns The number of rows in it.
 *
 * @example Rows divisible by 3 among ten
 * print('count =', bitsetCount(bitset(10, (i) => i % 3 === 0)))
 */
export function bitsetCount(a: Bitset): number {
  let c = 0
  for (let w = 0; w < a.length; w++) c += popcount32(a[w])
  return c
}

/**
 * The size $\lvert a \cap b \rvert$, without building the intersection: the true positives of a cover against a
 * target.
 *
 * @param a A set over $n$ rows.
 * @param b A set over the same rows.
 * @returns The number of rows in both.
 *
 * @example Positives inside a subgroup
 * const subgroup = bitset(8, (i) => i < 4)
 * const positives = bitset(8, (i) => [1, 2, 6].includes(i))
 * print('true positives =', bitsetAndCount(subgroup, positives))
 */
export function bitsetAndCount(a: Bitset, b: Bitset): number {
  let c = 0
  for (let w = 0; w < a.length; w++) c += popcount32(a[w] & b[w])
  return c
}

/**
 * The rows of $a$, ascending.
 *
 * @param a The set.
 * @returns The row indices, $\lvert a \rvert$ of them.
 *
 * @example The rows of a set
 * print('rows =', bitsetIndices(bitset(40, (i) => i % 13 === 0)))
 */
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

/**
 * The Jaccard index $\lvert a \cap b \rvert / \lvert a \cup b \rvert$ (1 for two empty sets): how much two covers
 * overlap, as the redundancy filter of `subgroupDiscovery` uses it.
 *
 * @param a A set over $n$ rows.
 * @param b A set over the same rows.
 * @returns The index, from 0 (disjoint) to 1 (equal).
 *
 * @example Two covers sharing two of their four rows
 * const a = bitset(6, (i) => i < 3)
 * const b = bitset(6, (i) => i >= 1 && i < 4)
 * print('jaccard =', bitsetJaccard(a, b))
 */
export function bitsetJaccard(a: Bitset, b: Bitset): number {
  const both = bitsetAndCount(a, b)
  const union = bitsetCount(a) + bitsetCount(b) - both
  return union === 0 ? 1 : both / union
}
