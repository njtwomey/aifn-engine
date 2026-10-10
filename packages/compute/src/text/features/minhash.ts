/**
 * MinHash (Broder 1997; Broder et al. 2000), the locality-sensitive family for sets; its signatures are banded into
 * candidate pairs by `aifn-compute/numerics/neighbours` (`lshBands`, `lshCandidates`; Indyk & Motwani 1998; Leskovec,
 * Rajaraman & Ullman, ch. 3). For a random hash $h$, $\Pr[\min h(\Acal) = \min h(\Bcal)] = J(\Acal, \Bcal)$, the
 * Jaccard similarity, so the share of $k$ independent hashes whose minima agree is an unbiased estimate of $J$ with
 * variance $J(1 - J)/k$. Banding cuts a $k = br$ signature into $b$ bands of $r$ rows; two sets become a candidate pair
 * when any band agrees, which happens with probability $1 - (1 - J^r)^b$: an S-curve in $J$ that rises around
 * $(1/b)^{1/r}$.
 *
 * Hash function $i$ is 32-bit MurmurHash3 of the shingle's UTF-8 bytes, seeded by a hash of $i$ and the signature's
 * seed, so signatures are reproducible from the seed and $k$ alone and a longer signature extends a shorter one.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { murmurHash3 } from './hashing'

/** The signature value of an empty set: above every 32-bit hash. */
const EMPTY = 2 ** 32

/** Options of {@link minHashSignature}. */
export interface MinHashOptions {
  /** The number $k$ of hash functions (default 128). */
  hashes?: number
  /** The seed of the family of hash functions (default 1, as datasketch). */
  seed?: number
}

/**
 * The seeds of the first $k$ hash functions of a family: seed $i$ is the MurmurHash3 of `minhash:i` under the family's
 * seed.
 *
 * @param k The number of hash functions.
 * @param seed The seed of the family.
 * @returns The $k$ seeds, as unsigned 32-bit integers.
 */
const seedsOf = (k: number, seed: number) =>
  Array.from({ length: k }, (_, i) => murmurHash3(`minhash:${i}`, seed) >>> 0)

/**
 * The MinHash signature of a set (float64 [k]): entry $i$ is the least value of hash function $i$ over the set's
 * elements, an unsigned 32-bit integer; $2^{32}$ for an empty set. Throws `DomainError` unless `hashes` is a positive
 * integer.
 *
 * @param set The set's elements, such as shingles; repeats are ignored.
 * @param options The number of hash functions and the family's seed; see {@link MinHashOptions}.
 * @returns The signature.
 *
 * @example A short signature, and an empty set's
 * print(minHashSignature(['a', 'b', 'c'], { hashes: 4 }))
 * print(minHashSignature([], { hashes: 4 }))
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

/**
 * The MinHash signatures of several sets, one row each (float64 [N, k]), all from the same hash functions.
 *
 * @param sets The sets, such as the shingle sets of $N$ documents.
 * @param options The number of hash functions and the family's seed; see {@link MinHashOptions}.
 * @returns One signature per row.
 *
 * @example Signatures of three shingle sets
 * const docs = ['the cat sat on the mat', 'the cat sat on a mat', 'a dog barked']
 * print(minHashSignatures(docs.map((d) => characterShingles(d, 3)), { hashes: 5 }))
 */
export function minHashSignatures(sets: readonly Iterable<string>[], options: MinHashOptions = {}): Tensor {
  const k = options.hashes ?? 128
  const out = new Float64Array(sets.length * k)
  sets.forEach((s, n) => out.set(minHashSignature(s, options).data as Float64Array, n * k))
  return fromData(out, [sets.length, k])
}

/**
 * The MinHash estimate of the Jaccard similarity of two sets from their signatures: the share of the first `hashes`
 * positions (default all) where they agree. Throws `DomainError` when the signatures differ in length or `hashes` is
 * not an integer from 1 to their length.
 *
 * @param a The first signature.
 * @param b The second signature, made with the same seed.
 * @param options How many leading positions to compare.
 * @param options.hashes The number of positions compared (default the signatures' length).
 * @returns The estimate, in $[0, 1]$.
 *
 * @example The estimate approaches the exact Jaccard similarity as hashes are added
 * const a = characterShingles('the cat sat on the mat', 3)
 * const b = characterShingles('the cat sat on a mat', 3)
 * const [sa, sb] = [minHashSignature(a, { hashes: 256 }), minHashSignature(b, { hashes: 256 })]
 * print('exact', jaccardSimilarity(a, b))
 * for (const k of [16, 64, 256]) print(k, 'hashes', minHashSimilarity(sa, sb, { hashes: k }))
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

/**
 * The standard error $\sqrt{J(1 - J)/k}$ of the MinHash estimate of a Jaccard similarity $J$ from $k$ hashes. Throws
 * `DomainError` unless $0 \le J \le 1$ and $k \ge 1$.
 *
 * @param similarity The Jaccard similarity $J$.
 * @param hashes The number of hash functions $k$.
 * @returns The standard deviation of the estimate.
 *
 * @example The error shrinks as $1/\sqrt{k}$
 * for (const k of [16, 64, 256]) print(k, 'hashes', minHashStandardError(0.5, k))
 */
export function minHashStandardError(similarity: number, hashes: number): number {
  if (!(similarity >= 0 && similarity <= 1) || !(hashes >= 1))
    throw new DomainError('minHashStandardError', 'minHashStandardError: need J in [0, 1] and k ≥ 1')
  return Math.sqrt((similarity * (1 - similarity)) / hashes)
}
