/**
 * Topics from matrix factorisations of the document × term matrix: latent semantic analysis (Deerwester et al., 1990),
 * a truncated SVD through `aifn-compute/text/representations`, whose components mix signs; and non-negative matrix
 * factorisation (Lee and Seung, 1999) through `aifn-compute/numerics/factorisation`, whose non-negative parts read as topics.
 * NMF with the generalised Kullback–Leibler divergence fits the same likelihood as pLSA (Gaussier and Goutte, 2005).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { nmfSteps, type NmfLoss } from 'aifn-compute/numerics/factorisation'
import { run } from 'aifn-compute/foundation/trace'
import { lsa } from 'aifn-compute/text/representations'
import { documentTermCounts, type Documents } from './corpus'

/** Topics as weights over words and documents as weights over topics. */
export type FactorTopics = {
  /** Topic × word weights [K, V]: for NMF a distribution per topic; for LSA the signed term loadings. */
  topicWord: Tensor
  /** Document × topic weights [D, K]: for NMF a distribution per document; for LSA the signed coordinates. */
  docTopic: Tensor
}

/**
 * LSA topics: the rank-K truncated SVD of the term × document count matrix; topic k's word weights are the terms'
 * coordinates on component k (signed, so a topic's top words are its largest positive loadings) and the documents'
 * weights their coordinates.
 */
export function lsaTopics(documents: Documents, vocabulary: Size, topics: Size): FactorTopics {
  const counts = documentTermCounts(documents, vocabulary)
  const r = lsa(transpose(counts), topics)
  return { topicWord: transpose(r.rows) as Tensor, docTopic: r.columns }
}

/** Options of `nmfTopicSteps`. */
export type NmfTopicOptions = { documents: Documents; vocabulary: Size; topics: Size; loss?: NmfLoss }

/** NMF of the document × term counts, X ≈ WH, stepped: W [D, K] holds the documents' topic weights, H [K, V] the topics. */
export function nmfTopicSteps(options: NmfTopicOptions) {
  const { documents, vocabulary, topics, loss = 'kullback-leibler' } = options
  return nmfSteps(documentTermCounts(documents, vocabulary), { rank: topics, loss, tolerance: 0 })
}

/** Normalise an NMF state's factors into topic and document distributions (rows sum to 1). */
export function nmfTopics(state: { W: Tensor; H: Tensor }): FactorTopics {
  const rows = (t: Tensor) => {
    const [r, c] = t.shape
    const d = Float64Array.from(toFlat(t))
    for (let i = 0; i < r; i++) {
      let s = 0
      for (let j = 0; j < c; j++) s += d[i * c + j]
      for (let j = 0; j < c; j++) d[i * c + j] = s > 0 ? d[i * c + j] / s : 1 / c
    }
    return fromData(d, [r, c])
  }
  // A document's weight on topic k counts the topic's total mass: W_dk Σ_w H_kw.
  const [D, K] = state.W.shape
  const V = state.H.shape[1]
  const H = toFlat(state.H)
  const W = Float64Array.from(toFlat(state.W))
  for (let k = 0; k < K; k++) {
    let s = 0
    for (let w = 0; w < V; w++) s += H[k * V + w]
    for (let d = 0; d < D; d++) W[d * K + k] *= s
  }
  return { topicWord: rows(state.H), docTopic: rows(fromData(W, [D, K])) }
}

/** Run NMF topics to the end (default 200 sweeps). */
export function nmfTopicModel(options: NmfTopicOptions & { steps?: Size; stream?: Stream }): FactorTopics {
  const s = run(nmfTopicSteps(options), undefined, options.steps ?? 200, {
    stream: options.stream ?? stream('nmf-topics'),
  })
  return nmfTopics(s)
}
