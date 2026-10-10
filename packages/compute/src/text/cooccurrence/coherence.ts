/**
 * Topic coherence: how often a topic's top words occur together in documents. NPMI coherence (Bouma, 2009; Aletras
 * and Stevenson, 2013; Lau, Newman and Baldwin, 2014) averages the normalised pointwise mutual information of every
 * pair of top words over document co-occurrence; UMass coherence (Mimno et al., 2011) averages
 * $\log((D(w_i, w_j) + 1)/D(w_j))$ over ordered pairs. The estimates follow gensim's `CoherenceModel` with boolean
 * document windows.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The coherence measure: `npmi` (from $-1$ to 1, higher is more coherent) or `umass` ($\le 0$ in practice). */
export type CoherenceMeasure = 'npmi' | 'umass'

/** Options of `topicCoherence`. */
export type CoherenceOptions = {
  /** The measure (default `npmi`). */
  measure?: CoherenceMeasure
  /** Added inside the logs of NPMI so pairs that never co-occur stay finite (default 1e-12, gensim's). */
  epsilon?: number
}

/** The coherence of each topic and their mean. */
export type Coherence = {
  /** One value per topic (NaN for a topic of fewer than two words). */
  topics: Float64Array
  /** The mean over topics. */
  mean: number
}

/**
 * The coherence of topics given by their top words (`topics[k]` = word ids, most probable first) on a reference corpus
 * of documents (arrays of word ids), with document co-occurrence: $P(w)$ is the share of documents containing $w$ and
 * $P(w_i, w_j)$ the share containing both.
 * $\operatorname{NPMI}(w_i, w_j) = \log((P(w_i, w_j) + \epsilon)/(P(w_i)P(w_j))) / -\log(P(w_i, w_j) + \epsilon)$,
 * averaged over the pairs $i < j$; UMass averages $\log((D(w_i, w_j) + 1)/D(w_j))$ over $i > j$ ($w_j$ ranked above
 * $w_i$), with $D$ a count of documents. As in gensim, a pair present in every document has NPMI $-1$ ($\epsilon$ in
 * the denominator); a word that occurs in no document gives $+\infty$ under NPMI, and under UMass when it is the
 * higher-ranked word of a pair, rather than a finite value. Throws `DomainError` when there are no documents.
 *
 * @param topics The top words of each topic as word ids, most probable first.
 * @param documents The reference corpus, each document a list of word ids (repeats count once).
 * @param options The measure and the $\epsilon$ of NPMI; see {@link CoherenceOptions}.
 * @returns The coherence of each topic and their mean.
 *
 * @example A topic whose words co-occur, and one whose words do not
 * // Word ids: 0 cat, 1 dog, 2 pet, 3 stock, 4 market, 5 price.
 * const documents = [[0, 1, 2], [0, 2], [1, 2], [3, 4, 5], [3, 5], [4, 5]]
 * const topics = [[2, 0, 1], [2, 3, 0]]
 * print('NPMI ', topicCoherence(topics, documents).topics)
 * print('UMass', topicCoherence(topics, documents, { measure: 'umass' }).topics)
 */
export function topicCoherence(
  topics: readonly (readonly number[])[],
  documents: readonly (readonly number[])[],
  options: CoherenceOptions = {},
): Coherence {
  const { measure = 'npmi', epsilon = 1e-12 } = options
  const D = documents.length
  if (D === 0) throw new DomainError('topicCoherence', 'topicCoherence: no documents')
  // Document sets of the words that appear in some topic.
  const words = new Set<number>()
  for (const t of topics) for (const w of t) words.add(w)
  const containing = new Map<number, Set<Size>>()
  for (const w of words) containing.set(w, new Set())
  documents.forEach((doc, d) => {
    for (const w of doc) containing.get(w)?.add(d)
  })
  const count = (w: number) => containing.get(w)!.size
  const both = (a: number, b: number) => {
    const [small, large] = count(a) <= count(b) ? [a, b] : [b, a]
    let k = 0
    for (const d of containing.get(small)!) if (containing.get(large)!.has(d)) k++
    return k
  }
  const values = topics.map((top) => {
    let total = 0
    let pairs = 0
    for (let i = 0; i < top.length; i++)
      for (let j = 0; j < i; j++) {
        const wi = top[i]
        const wj = top[j]
        if (measure === 'umass') {
          total += Math.log((both(wi, wj) + 1) / count(wj))
        } else {
          const pij = both(wi, wj) / D + epsilon
          total += Math.log(pij / ((count(wi) / D) * (count(wj) / D))) / -Math.log(pij)
        }
        pairs++
      }
    return pairs > 0 ? total / pairs : NaN
  })
  const scores = Float64Array.from(values)
  return { topics: scores, mean: scores.reduce((a, b) => a + b, 0) / Math.max(1, scores.length) }
}
