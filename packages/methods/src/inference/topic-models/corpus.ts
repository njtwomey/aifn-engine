/**
 * Shared views of a bag-of-words corpus for the topic models: texts as word-id documents, the $D \times V$
 * document-term count matrix, the most probable words of each topic, and short labelled texts grouped into documents.
 *
 * A document is an array of word ids in $\{0, \dots, V - 1\}$, with $V$ the vocabulary size; word order is kept but
 * the models here only count words.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { tokenise } from 'aifn-compute/text/tokenise'

/** Documents as arrays of word ids in $\{0, \dots, V - 1\}$, $V$ the vocabulary size. */
export type Documents = readonly (readonly number[])[]

/**
 * The $D \times V$ document-term count matrix of a corpus: entry $(d, w)$ is the number of times word $w$ occurs in
 * document $d$.
 *
 * @param documents The $D$ documents, as word ids. A word id outside $\{0, \dots, V - 1\}$ is not checked, and lands
 *   in another row's entries or nowhere.
 * @param vocabulary The vocabulary size $V$.
 * @returns The counts, a float64 $D \times V$ matrix.
 *
 * @example Two documents over three words
 * print(documentTermCounts([[0, 0, 2], [1, 2, 2, 2]], 3))
 */
export function documentTermCounts(documents: Documents, vocabulary: Size): Tensor {
  const D = documents.length
  const out = new Float64Array(D * vocabulary)
  documents.forEach((doc, d) => {
    for (const w of doc) out[d * vocabulary + w]++
  })
  return fromData(out, [D, vocabulary])
}

/**
 * The ids of the `n` largest entries of each row of a $K \times V$ topic-word matrix, largest first, ties broken by the
 * smaller id.
 *
 * @param topicWord The $K \times V$ topic-word weights or probabilities, one topic per row.
 * @param n The number of word ids per topic (all $V$ when `n` exceeds it).
 * @returns $K$ arrays of word ids, one per topic.
 *
 * @example The two heaviest words of each topic
 * print(topWords(tensor([[0.1, 0.6, 0.3, 0], [0.5, 0, 0.1, 0.4]]), 2))
 */
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
 * each label), so every document has one dominant topic: sentence-level corpora are too short for topic models. The
 * output is ordered by label, then by position.
 *
 * @param documents The short texts, as word ids.
 * @param labels The label of each text, an integer from 0 up; labels $0, \dots, L - 1$ are visited, $L - 1$ the
 *   largest label.
 * @param perDocument The number of texts joined into each document.
 * @returns The joined `documents`, and the label of each in `labels`.
 *
 * @example Five texts in two labels, joined in pairs
 * const { documents, labels } = groupByLabel([[0], [1], [2], [3], [4]], [0, 1, 0, 1, 0], 2)
 * print('documents', documents)
 * print('labels', labels)
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

/**
 * A corpus as word ids and the vocabulary they index: `documents` holds each text's word ids, and `vocabulary[w]` is
 * the word with id `w`.
 */
export type BagOfWordsCorpus = { documents: number[][]; vocabulary: string[] }

/**
 * Texts as word-id documents: the word tokens of `tokenise` (`aifn-compute/text/tokenise`), lower-cased, with
 * `stopWords` removed, numbered in order of first appearance.
 *
 * @param texts The texts, one document each.
 * @param options The words to drop.
 * @param options.stopWords Lower-case words removed after lower-casing (none when left out).
 * @returns The `documents` as word ids and the `vocabulary` they index.
 *
 * @example Two sentences, without "the"
 * const corpus = bagOfWordsCorpus(['The cat sat.', 'The dog sat on the cat.'], { stopWords: ['the'] })
 * print('documents', corpus.documents)
 * print('vocabulary', corpus.vocabulary)
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
