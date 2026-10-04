/**
 * Random indexing (Kanerva, Kristoferson & Holst 2000; Sahlgren 2005): every context word gets a fixed sparse ternary
 * index vector of d dimensions, a few entries +1 or −1 and the rest 0, and a word's vector is the sum of the index
 * vectors of the words around it. Index vectors in high dimension are nearly orthogonal, so the sum approximates a
 * random projection of the word's co-occurrence row (Achlioptas 2003) without ever building the V × V matrix, and a
 * new word or document only adds to the sums. It is the incremental alternative to the SVD of a co-occurrence matrix.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, stream, units } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { cooccurrence, type CooccurrenceOptions } from 'aifn-compute/text/cooccurrence'
import { buildVocabulary, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** Options of {@link indexVector}. */
export interface IndexVectorOptions {
  /** The dimension d (default 64). */
  dimensions?: number
  /** The number of non-zero entries, half +1 and half −1 (default 4; an odd count gives the extra one +1). */
  nonZeros?: number
  /** The seed (default 0): the same token and seed always give the same vector. */
  seed?: number | string
}

function checkIndex(d: number, nz: number, op: string): void {
  if (!(Number.isInteger(d) && d >= 1 && Number.isInteger(nz) && nz >= 1 && nz <= d))
    throw new DomainError(op, `${op}: need integers 1 ≤ nonZeros ≤ dimensions`)
}

/**
 * The sparse ternary index vector of a token (float64 [d]): `nonZeros` distinct positions drawn uniformly, the first
 * half set to +1 and the rest to −1. It depends only on the token and the seed, not on the vocabulary.
 */
export function indexVector(token: string, options: IndexVectorOptions = {}): Tensor {
  const { dimensions = 64, nonZeros = 4, seed = 0 } = options
  checkIndex(dimensions, nonZeros, 'indexVector')
  const out = new Float64Array(dimensions)
  const s = child(stream(seed), 'random-indexing', token)
  // A partial Fisher–Yates shuffle picks distinct positions.
  const order = Array.from({ length: dimensions }, (_, i) => i)
  const u = units(s, nonZeros)
  for (let k = 0; k < nonZeros; k++) {
    const j = k + Math.floor(u[k] * (dimensions - k))
    ;[order[k], order[j]] = [order[j], order[k]]
    out[order[k]] = k < Math.ceil(nonZeros / 2) ? 1 : -1
  }
  return fromData(out)
}

/** Options of {@link randomIndexing}: index vectors, and the window of {@link cooccurrence}. */
export interface RandomIndexingOptions extends IndexVectorOptions, Omit<CooccurrenceOptions, 'words' | 'contexts'> {
  /** The rows (default every word, most frequent first, without specials). */
  words?: Vocabulary
}

/** Word vectors by random indexing, with the index vectors of the contexts they sum. */
export interface RandomIndexing {
  readonly kind: 'random-indexing'
  /** Context vectors (float64 [V, d]): row w is Σ_c #(w, c) · index(c). */
  readonly vectors: Tensor
  /** The index vectors of the vocabulary's words (float64 [V, d]). */
  readonly index: Tensor
  readonly words: Vocabulary
}

/**
 * Random indexing of tokenised documents: each word's context vector is the (distance-weighted) sum of the index vectors
 * of the words in its window. Equal to the co-occurrence matrix times the matrix of index vectors, computed here
 * through it for clarity; a streaming implementation adds one index vector per context token instead.
 */
export function randomIndexing(
  documents: readonly (readonly string[])[],
  options: RandomIndexingOptions = {},
): RandomIndexing {
  const { dimensions = 64, nonZeros = 4, seed = 0 } = options
  checkIndex(dimensions, nonZeros, 'randomIndexing')
  const words = options.words ?? buildVocabulary(documents, { specials: [] })
  const V = words.tokens.length
  const index = new Float64Array(V * dimensions)
  words.tokens.forEach((t, i) =>
    index.set(indexVector(t, { dimensions, nonZeros, seed }).data as Float64Array, i * dimensions),
  )
  const counts = cooccurrence(documents, { ...options, words, contexts: words }).matrix.data as Float64Array
  const out = new Float64Array(V * dimensions)
  for (let w = 0; w < V; w++)
    for (let c = 0; c < V; c++) {
      const x = counts[w * V + c]
      if (x === 0) continue
      for (let j = 0; j < dimensions; j++) out[w * dimensions + j] += x * index[c * dimensions + j]
    }
  return {
    kind: 'random-indexing',
    vectors: fromData(out, [V, dimensions]),
    index: fromData(index, [V, dimensions]),
    words,
  }
}
