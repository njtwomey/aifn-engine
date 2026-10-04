/**
 * Shingles and resemblance (Broder 1997): a document as the set of its contiguous k-character or w-word substrings, and
 * the resemblance of two documents as the Jaccard similarity |A ∩ B| / |A ∪ B| of their shingle sets.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

function positive(k: number, op: string): void {
  if (!(Number.isInteger(k) && k >= 1)) throw new DomainError(op, `${op}: the shingle length must be an integer ≥ 1`)
}

/**
 * The set of character k-shingles of a text, in code-point order. Runs of white space are first collapsed to one
 * space, so layout does not change the set; a text shorter than k gives itself as its only shingle (none when empty).
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

/** The set of word w-shingles of a token list (runs of w tokens joined by `joiner`), in code-point order. */
export function wordShingles(tokens: readonly string[], w: number, joiner = ' '): string[] {
  positive(w, 'wordShingles')
  if (tokens.length === 0) return []
  if (tokens.length < w) return [tokens.join(joiner)]
  const out = new Set<string>()
  for (let i = 0; i + w <= tokens.length; i++) out.add(tokens.slice(i, i + w).join(joiner))
  return [...out].sort(byCodePoint)
}

/**
 * The Jaccard similarity |A ∩ B| / |A ∪ B| of two sets (given as iterables; repeats are ignored), in [0, 1]. Two empty
 * sets have similarity 1 by convention (as datasketch).
 */
export function jaccardSimilarity(a: Iterable<string>, b: Iterable<string>): number {
  const A = new Set(a)
  const B = new Set(b)
  if (A.size === 0 && B.size === 0) return 1
  let both = 0
  for (const x of A) if (B.has(x)) both++
  return both / (A.size + B.size - both)
}
