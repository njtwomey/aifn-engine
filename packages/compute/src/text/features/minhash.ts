/**
 * MinHash (Broder 1997; Broder et al. 2000), the locality-sensitive family for sets; its signatures are banded into
 * candidate pairs by `aifn-compute/numerics/neighbours` (`lshBands`, `lshCandidates`; Indyk & Motwani 1998; Leskovec,
 * Rajaraman & Ullman, ch. 3). For a random hash h, P[min h(A) = min h(B)] = J(A, B), the Jaccard similarity, so the
 * share of k independent hashes whose minima agree is an unbiased estimate of J with variance J(1 − J)/k. Banding cuts
 * a k = b·r signature into b bands of r rows; two sets become a candidate pair when any band agrees, which happens with
 * probability 1 − (1 − Jʳ)ᵇ: an S-curve in J that rises around (1/b)^{1/r}.
 *
 * Hash function i is 32-bit MurmurHash3 of the shingle's UTF-8 bytes, seeded by a hash of i and the signature's seed,
 * so signatures are reproducible from (seed, k) alone and a longer signature extends a shorter one.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { murmurHash3 } from './hashing'

/** The signature value of an empty set: above every 32-bit hash. */
const EMPTY = 2 ** 32

/** Options of {@link minHashSignature}. */
export interface MinHashOptions {
  /** The number k of hash functions (default 128). */
  hashes?: number
  /** The seed of the family of hash functions (default 1, as datasketch). */
  seed?: number
}

const seedsOf = (k: number, seed: number) =>
  Array.from({ length: k }, (_, i) => murmurHash3(`minhash:${i}`, seed) >>> 0)

/**
 * The MinHash signature of a set (float64 [k]): entry i is the least value of hash function i over the set's
 * elements, an unsigned 32-bit integer; 2³² for an empty set.
 */
export function minHashSignature(set: Iterable<string>, options: MinHashOptions = {}): Tensor {
  const { hashes = 128, seed = 1 } = options
  if (!(Number.isInteger(hashes) && hashes >= 1))
    throw new DomainError('minHashSignature', 'minHashSignature: hashes must be a positive integer')
  const seeds = seedsOf(hashes, seed)
  const out = new Float64Array(hashes).fill(EMPTY)
  for (const x of new Set(set))
    for (let i = 0; i < hashes; i++) {
      const h = murmurHash3(x, seeds[i]) >>> 0
      if (h < out[i]) out[i] = h
    }
  return fromData(out)
}

/** The MinHash signatures of several sets, one row each (float64 [N, k]). */
export function minHashSignatures(sets: readonly Iterable<string>[], options: MinHashOptions = {}): Tensor {
  const k = options.hashes ?? 128
  const out = new Float64Array(sets.length * k)
  sets.forEach((s, n) => out.set(minHashSignature(s, options).data as Float64Array, n * k))
  return fromData(out, [sets.length, k])
}

/**
 * The MinHash estimate of the Jaccard similarity of two sets from their signatures: the share of the first `hashes`
 * positions (default all) where they agree.
 */
export function minHashSimilarity(a: VectorLike, b: VectorLike, options: { hashes?: number } = {}): number {
  const x = dense.toF64(a, 'minHashSimilarity')
  const y = dense.toF64(b, 'minHashSimilarity')
  if (x.length !== y.length)
    throw new DomainError('minHashSimilarity', 'minHashSimilarity: the signatures differ in length')
  const k = options.hashes ?? x.length
  if (!(Number.isInteger(k) && k >= 1 && k <= x.length))
    throw new DomainError('minHashSimilarity', `minHashSimilarity: hashes must be an integer in [1, ${x.length}]`)
  let same = 0
  for (let i = 0; i < k; i++) if (x[i] === y[i]) same++
  return same / k
}

/** The standard error √(J(1 − J)/k) of the MinHash estimate of a Jaccard similarity J from k hashes. */
export function minHashStandardError(similarity: number, hashes: number): number {
  if (!(similarity >= 0 && similarity <= 1) || !(hashes >= 1))
    throw new DomainError('minHashStandardError', 'minHashStandardError: need J in [0, 1] and k ≥ 1')
  return Math.sqrt((similarity * (1 - similarity)) / hashes)
}
