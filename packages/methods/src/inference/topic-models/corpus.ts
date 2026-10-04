/**
 * Shared views of a bag-of-words corpus for the topic models: the document × term count matrix, the most probable
 * words of each topic, and documents as word-id arrays.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { tokenise } from 'aifn-compute/text/tokenise'

/** Documents as arrays of word ids in 0 … V − 1. */
export type Documents = readonly (readonly number[])[]

/** The document × term count matrix [D, V] of a corpus. */
export function documentTermCounts(documents: Documents, vocabulary: Size): Tensor {
  const D = documents.length
  const out = new Float64Array(D * vocabulary)
  documents.forEach((doc, d) => {
    for (const w of doc) out[d * vocabulary + w]++
  })
  return fromData(out, [D, vocabulary])
}

/** The ids of the n largest entries of each row of a topic × word matrix [K, V], largest first. */
export function topWords(topicWord: Tensor, n = 10): number[][] {
  const [K, V] = topicWord.shape
  const d = toFlat(topicWord)
  return Array.from({ length: K }, (_, k) =>
    Array.from({ length: V }, (_, w) => w)
      .sort((a, b) => d[k * V + b] - d[k * V + a] || a - b)
      .slice(0, n),
  )
}

/**
 * Join short labelled texts into documents of `perDocument` texts sharing a label (in order, dropping the remainder of
 * each label), so every document has one dominant topic: sentence-level corpora are too short for topic models.
 */
export function groupByLabel(
  documents: Documents,
  labels: ArrayLike<number>,
  perDocument: Size,
): { documents: number[][]; labels: number[] } {
  const L = Math.max(0, ...Array.from(labels)) + 1
  const out: number[][] = []
  const outLabels: number[] = []
  for (let l = 0; l < L; l++) {
    const ids: number[] = []
    for (let i = 0; i < labels.length; i++) if (labels[i] === l) ids.push(i)
    for (let g = 0; g + perDocument <= ids.length; g += perDocument) {
      out.push(ids.slice(g, g + perDocument).flatMap((i) => [...documents[i]]))
      outLabels.push(l)
    }
  }
  return { documents: out, labels: outLabels }
}

/** A corpus as word ids and the vocabulary they index. */
export type BagOfWordsCorpus = { documents: number[][]; vocabulary: string[] }

/**
 * Texts as word-id documents: lower-cased word tokens (`aifn-compute/text/tokenise`), with `stopWords` removed, numbered in
 * order of first appearance.
 */
export function bagOfWordsCorpus(
  texts: readonly string[],
  options: { stopWords?: readonly string[] } = {},
): BagOfWordsCorpus {
  const stop = new Set(options.stopWords ?? [])
  const ids = new Map<string, number>()
  const documents = texts.map((text) =>
    tokenise(text)
      .tokens.map((w) => w.toLowerCase())
      .filter((w) => !stop.has(w))
      .map((w) => {
        let id = ids.get(w)
        if (id === undefined) ids.set(w, (id = ids.size))
        return id
      }),
  )
  return { documents, vocabulary: [...ids.keys()] }
}
