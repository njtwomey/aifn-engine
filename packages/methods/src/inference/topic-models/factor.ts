/**
 * Topics from matrix factorisations of the $D \times V$ document-term count matrix: latent semantic analysis
 * (Deerwester et al., 1990), a truncated SVD through `aifn-compute/text/representations`, whose components mix signs;
 * and non-negative matrix factorisation $\Xmat \approx \Wmat\Hmat$ (Lee and Seung, 1999) through
 * `aifn-compute/numerics/factorisation`, whose non-negative parts read as topics. NMF with the generalised
 * Kullback-Leibler divergence fits the same likelihood as pLSA (Gaussier and Goutte, 2005).
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
  /** Topic-word weights, $K \times V$: for NMF a distribution per topic; for LSA the signed term loadings. */
  topicWord: Tensor
  /** Document-topic weights, $D \times K$: for NMF a distribution per document; for LSA the signed coordinates. */
  docTopic: Tensor
}

/**
 * LSA topics: the rank-$K$ truncated SVD of the $V \times D$ term-document count matrix; topic $k$'s word weights are
 * the terms' coordinates $\Umat_K\Sigmamat_K$ on component $k$ (signed, so a topic's top words are its largest positive
 * loadings) and the documents' weights their coordinates $\Vmat_K\Sigmamat_K$. The sign of each component is
 * arbitrary.
 *
 * @param documents The corpus, as word ids.
 * @param vocabulary The vocabulary size $V$.
 * @param topics The rank $K$, the number of components kept.
 * @returns The term loadings as `topicWord` ($K \times V$) and the document coordinates as `docTopic` ($D \times K$).
 *
 * @example Two vocabularies give two components
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const { topicWord, docTopic } = lsaTopics(documents, 6, 2)
 * print('term loadings', topicWord)
 * print('document coordinates', docTopic)
 */
export function lsaTopics(documents: Documents, vocabulary: Size, topics: Size): FactorTopics {
  const counts = documentTermCounts(documents, vocabulary)
  const r = lsa(transpose(counts), topics)
  return { topicWord: transpose(r.rows) as Tensor, docTopic: r.columns }
}

/**
 * Options of `nmfTopicSteps`: the corpus `documents`, the `vocabulary` size $V$, the number of `topics` $K$, and the
 * `loss` NMF minimises (default `'kullback-leibler'`; `'frobenius'` for squared error).
 */
export type NmfTopicOptions = { documents: Documents; vocabulary: Size; topics: Size; loss?: NmfLoss }

/**
 * NMF of the document-term counts, $\Xmat \approx \Wmat\Hmat$, stepped by `nmfSteps` (multiplicative updates, one
 * sweep per step): $\Wmat$ ($D \times K$) holds the documents' topic weights and $\Hmat$ ($K \times V$) the topics. The
 * tolerance is 0, so a run takes every step it is given. Normalise a state into distributions with `nmfTopics`.
 *
 * @param options The corpus, its vocabulary size, the number of topics and the loss.
 * @returns The `nmfSteps` algorithm, run with no start; its `init` draws random factors from its stream.
 *
 * @example The KL objective falls as the factors separate the vocabularies
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const alg = nmfTopicSteps({ documents, vocabulary: 6, topics: 2 })
 * print('objective at 0, 5 and 50 steps:', run(alg, undefined, 0).objective, run(alg, undefined, 5).objective,
 *   run(alg, undefined, 50).objective)
 * print('topics', nmfTopics(run(alg, undefined, 50)).topicWord)
 */
export function nmfTopicSteps(options: NmfTopicOptions) {
  const { documents, vocabulary, topics, loss = 'kullback-leibler' } = options
  return nmfSteps(documentTermCounts(documents, vocabulary), { rank: topics, loss, tolerance: 0 })
}

/**
 * Normalise an NMF state's factors into topic and document distributions (rows sum to 1). A document's weight on
 * topic $k$ is first scaled by the topic's total mass, $W_{dk} \sum_w H_{kw}$, so the proportions count tokens. A row
 * that is all zero becomes uniform.
 *
 * @param state The factors: `W` ($D \times K$) and `H` ($K \times V$), as an `nmfTopicSteps` state holds them. Not
 *   modified.
 * @returns `topicWord`, the rows of $\Hmat$ normalised ($K \times V$), and `docTopic`, the scaled rows of $\Wmat$
 *   normalised ($D \times K$).
 *
 * @example The factors of a small state, as distributions
 * print(nmfTopics({ W: tensor([[2, 0], [1, 1]]), H: tensor([[1, 1, 0], [0, 0, 4]]) }))
 */
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

/**
 * Run NMF topics to the end and return them as distributions: `nmfTopicSteps` for `steps` sweeps, then `nmfTopics`.
 *
 * @param options The options of `nmfTopicSteps`, with `steps`, the number of sweeps (default 200), and `stream`, the
 *   root stream of the run (default `stream('nmf-topics')`).
 * @returns The topics and the documents' proportions.
 *
 * @example Two vocabularies, two topics
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const { topicWord, docTopic } = nmfTopicModel({ documents, vocabulary: 6, topics: 2, steps: 100 })
 * print('top words of each topic', topWords(topicWord, 3))
 * print('topic proportions', docTopic)
 */
export function nmfTopicModel(options: NmfTopicOptions & { steps?: Size; stream?: Stream }): FactorTopics {
  const s = run(nmfTopicSteps(options), undefined, options.steps ?? 200, {
    stream: options.stream ?? stream('nmf-topics'),
  })
  return nmfTopics(s)
}
