/**
 * Shingles and resemblance (Broder 1997): a document as the set of its contiguous $k$-character or $w$-word
 * substrings, and the resemblance of two documents as the Jaccard similarity
 * $\lvert \Acal \cap \Bcal \rvert / \lvert \Acal \cup \Bcal \rvert$ of their shingle sets. Shingle sets are returned as
 * sorted arrays of distinct strings.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Compare two strings by UTF-16 code unit, as `<` does (code-point order unless a surrogate pair is involved).
 *
 * @param a The first string.
 * @param b The second string.
 * @returns $-1$, 0 or 1 as `a` sorts before, with or after `b`.
 */
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/**
 * Check a shingle length: throws `DomainError` unless it is an integer of at least 1.
 *
 * @param k The shingle length.
 * @param op The caller's name, for the error message.
 */
function positive(k: number, op: string): void {
  if (!(Number.isInteger(k) && k >= 1)) throw new DomainError(op, `${op}: the shingle length must be an integer ≥ 1`)
}

/**
 * The set of character $k$-shingles of a text, in code-point order. Runs of white space are first collapsed to one
 * space, so layout does not change the set; a text shorter than $k$ gives itself as its only shingle (none when empty).
 *
 * @param text The text; shingles are formed over its code points.
 * @param k The shingle length, an integer of at least 1.
 * @returns The distinct shingles, sorted.
 *
 * @example Repeats count once
 * print(characterShingles('banana', 3))
 * print(characterShingles('a  rose\nis', 4))
 */
export function characterShingles(text: string, k: number): string[] {
  positive(k, 'characterShingles')
  const chars = [...text.replace(/\s+/gu, ' ')]
  if (chars.length === 0) return []
  if (chars.length < k) return [chars.join('')]
  const out = new Set<string>()
  for (let i = 0; i + k <= chars.length; i++) out.add(chars.slice(i, i + k).join(''))
  return [...out].sort(byCodePoint)
}

/**
 * The set of word $w$-shingles of a token list (runs of $w$ tokens joined by `joiner`), in code-point order. A list
 * shorter than $w$ gives itself, joined, as its only shingle (none when empty).
 *
 * @param tokens The tokens, in order.
 * @param w The shingle length in tokens, an integer of at least 1.
 * @param joiner The string placed between the tokens of a shingle.
 * @returns The distinct shingles, sorted.
 *
 * @example Two-word shingles
 * print(wordShingles(['a', 'rose', 'is', 'a', 'rose', 'is', 'a', 'rose'], 2))
 */
export function wordShingles(tokens: readonly string[], w: number, joiner = ' '): string[] {
  positive(w, 'wordShingles')
  if (tokens.length === 0) return []
  if (tokens.length < w) return [tokens.join(joiner)]
  const out = new Set<string>()
  for (let i = 0; i + w <= tokens.length; i++) out.add(tokens.slice(i, i + w).join(joiner))
  return [...out].sort(byCodePoint)
}

/**
 * The Jaccard similarity $\lvert \Acal \cap \Bcal \rvert / \lvert \Acal \cup \Bcal \rvert$ of two sets (given as
 * iterables; repeats are ignored), in $[0, 1]$. Two empty sets have similarity 1 by convention (as datasketch).
 *
 * @param a The first set $\Acal$.
 * @param b The second set $\Bcal$.
 * @returns The share of the union that is in both.
 *
 * @example The resemblance of two sentences by their 3-shingles
 * const a = characterShingles('the cat sat on the mat', 3)
 * const b = characterShingles('the cat sat on a mat', 3)
 * print('shared', a.filter((x) => b.includes(x)).length, 'of', new Set([...a, ...b]).size)
 * print('Jaccard', jaccardSimilarity(a, b))
 */
export function jaccardSimilarity(a: Iterable<string>, b: Iterable<string>): number {
  const A = new Set(a)
  const B = new Set(b)
  if (A.size === 0 && B.size === 0) return 1
  let both = 0
  for (const x of A) if (B.has(x)) both++
  return both / (A.size + B.size - both)
}
