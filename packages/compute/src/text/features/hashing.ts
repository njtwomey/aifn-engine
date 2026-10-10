/**
 * Feature hashing (Weinberger et al. 2009): a term's column is a hash of its characters, so no vocabulary is stored;
 * a second hash bit gives each term a sign, which makes collisions add zero-mean noise instead of a bias. The hash and
 * conventions are scikit-learn's `HashingVectorizer`: 32-bit MurmurHash3 (x86, seed 0) of the term's UTF-8 bytes,
 * column $\lvert h \rvert \bmod m$, sign $+1$ for $h \ge 0$ and $-1$ otherwise.
 *
 * aifn has no sparse tensor type: `hashedFeatures` gives each document's non-zero columns and values (a sparse row),
 * and `featureHash` the dense [D, m] matrix, for the small $m$ of figures and tests.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

const encoder = new TextEncoder()

/**
 * MurmurHash3 x86 32-bit (Appleby) of the UTF-8 bytes of `text`, as a signed 32-bit integer: the value of
 * scikit-learn's `murmurhash3_32(text, seed)`.
 *
 * @param text The string to hash, encoded as UTF-8.
 * @param seed The seed, read as an unsigned 32-bit integer.
 * @returns The hash, from $-2^{31}$ to $2^{31} - 1$.
 *
 * @example The same values as scikit-learn's `murmurhash3_32`
 * print('hello', murmurHash3('hello'))
 * print('café ', murmurHash3('café'))
 * print('hello, seed 1', murmurHash3('hello', 1))
 */
export function murmurHash3(text: string, seed = 0): number {
  const bytes = encoder.encode(text)
  const c1 = 0xcc9e2d51
  const c2 = 0x1b873593
  let h = seed >>> 0
  const blocks = bytes.length >>> 2
  for (let i = 0; i < blocks; i++) {
    const j = 4 * i
    let k = bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24)
    k = Math.imul(k, c1)
    k = (k << 15) | (k >>> 17)
    k = Math.imul(k, c2)
    h ^= k
    h = (h << 13) | (h >>> 19)
    h = (Math.imul(h, 5) + 0xe6546b64) | 0
  }
  // The 1–3 trailing bytes, little-endian, mixed like a block.
  const tail = 4 * blocks
  const rest = bytes.length & 3
  if (rest > 0) {
    let k = 0
    for (let r = rest - 1; r >= 0; r--) k = (k << 8) | bytes[tail + r]
    k = Math.imul(k, c1)
    k = (k << 15) | (k >>> 17)
    k = Math.imul(k, c2)
    h ^= k
  }
  h ^= bytes.length
  h ^= h >>> 16
  h = Math.imul(h, 0x85ebca6b)
  h ^= h >>> 13
  h = Math.imul(h, 0xc2b2ae35)
  h ^= h >>> 16
  return h | 0
}

/** Options of {@link hashedFeatures} and {@link featureHash}. */
export interface HashingOptions {
  /** The number of columns $m$ (default 1024 for `featureHash`, $2^{20}$ for `hashedFeatures`). */
  features?: number
  /** Give each term the sign of its hash (default true; false gives unsigned hashing, biased by collisions). */
  signed?: boolean
  /** Count each distinct term once per document, with its sign when `signed` (default false). */
  binary?: boolean
  /** The hash seed (default 0). */
  seed?: number
}

/**
 * The column and sign of a term under $m$ columns: $\lvert h \rvert \bmod m$ and the sign of $h$, for $h$ the term's
 * MurmurHash3.
 *
 * @param term The term.
 * @param features The number of columns $m$.
 * @param seed The hash seed.
 * @returns The column, from 0 to $m - 1$, and the sign, $+1$ or $-1$.
 *
 * @example Where three words land among 8 columns
 * for (const w of ['cat', 'dog', 'mat']) print(w, hashColumn(w, 8))
 */
export function hashColumn(term: string, features: number, seed = 0): { column: number; sign: number } {
  const h = murmurHash3(term, seed)
  return { column: Math.abs(h) % features, sign: h >= 0 ? 1 : -1 }
}

/**
 * The hashed vector of one tokenised document as a sparse row: its non-zero `columns` in increasing order (int32 [k])
 * and their `values` (float64 [k]). Signed terms that cancel in a column leave no entry. Throws `DomainError` unless
 * `features` is a positive integer.
 *
 * @param tokens The document's tokens; each occurrence adds its sign to its column (once per distinct term with
 *   `binary`).
 * @param options The number of columns (default $2^{20}$), signing, binary counting and seed; see
 *   {@link HashingOptions}.
 * @returns The non-zero columns and their values.
 *
 * @example A sparse row of $2^{20}$ columns
 * const r = hashedFeatures(['the', 'cat', 'sat', 'on', 'the', 'mat'])
 * print('columns', r.columns)
 * print('values ', r.values)
 */
export function hashedFeatures(
  tokens: readonly string[],
  options: HashingOptions = {},
): { columns: Tensor; values: Tensor } {
  const { features = 2 ** 20, signed = true, binary = false, seed = 0 } = options
  if (!(Number.isInteger(features) && features >= 1))
    throw new DomainError('hashedFeatures', 'hashedFeatures: features must be a positive integer')
  const acc = new Map<number, number>()
  for (const t of binary ? new Set(tokens) : tokens) {
    const { column, sign } = hashColumn(t, features, seed)
    acc.set(column, (acc.get(column) ?? 0) + (signed ? sign : 1))
  }
  const entries = [...acc.entries()].filter(([, v]) => v !== 0).sort((a, b) => a[0] - b[0])
  return {
    columns: fromData(Int32Array.from(entries, (e) => e[0])),
    values: fromData(Float64Array.from(entries, (e) => e[1])),
  }
}

/**
 * The hashed document–feature matrix (float64 [D, m]), each row normalised by `norm` (default `l2`, as
 * `HashingVectorizer`; `none` keeps signed counts). A row of zeros is left as it is. Throws `DomainError` unless
 * `features` is a positive integer.
 *
 * @param documents The tokenised documents, one row each.
 * @param options The hashing options (see {@link HashingOptions}; `features` defaults to 1024 here) and `norm`, the
 *   row normalisation: `l2`, `l1` or `none`.
 * @returns The dense matrix.
 *
 * @example Two documents in 8 columns
 * const docs = [['the', 'cat', 'sat'], ['the', 'dog', 'sat', 'down']]
 * print('signed counts', featureHash(docs, { features: 8, norm: 'none' }))
 * print('l2 rows      ', featureHash(docs, { features: 8 }))
 */
export function featureHash(
  documents: readonly (readonly string[])[],
  options: HashingOptions & { norm?: 'none' | 'l1' | 'l2' } = {},
): Tensor {
  const { features = 1024, norm = 'l2' } = options
  const out = new Float64Array(documents.length * features)
  documents.forEach((doc, d) => {
    const { columns, values } = hashedFeatures(doc, { ...options, features })
    let z = 0
    for (const v of values.data) z += norm === 'l1' ? Math.abs(v) : v * v
    if (norm === 'l2') z = Math.sqrt(z)
    const scale = norm === 'none' || z === 0 ? 1 : 1 / z
    for (let k = 0; k < columns.data.length; k++) out[d * features + columns.data[k]] = values.data[k] * scale
  })
  return fromData(out, [documents.length, features])
}
