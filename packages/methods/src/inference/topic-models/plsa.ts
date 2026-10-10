/**
 * Probabilistic latent semantic analysis (Hofmann, 1999) fitted by expectation-maximisation: each document $d$ is a
 * mixture $P(w \mid d) = \sum_z P(z \mid d) P(w \mid z)$ of $K$ topics. The E-step gives each (document, word) pair its
 * topic responsibilities $P(z \mid d, w) \propto P(z \mid d) P(w \mid z)$; the M-step re-estimates both distributions
 * from the expected counts $n(d, w) P(z \mid d, w)$. The log-likelihood $\sum_{d, w} n(d, w) \log P(w \mid d)$ never
 * decreases. Unlike LDA, pLSA has no prior on the distributions, so it fits each document's proportions freely.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, units } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { documentTermCounts, type Documents } from './corpus'

/** The state of `plsaSteps`. */
export interface PlsaState extends Status {
  /** EM steps done. */
  t: Size
  /** The topics $P(w \mid z)$, $K \times V$; each row sums to 1. */
  topicWord: Tensor
  /** The documents' proportions $P(z \mid d)$, $D \times K$; each row sums to 1. */
  docTopic: Tensor
  /** The log-likelihood $\sum_{d, w} n(d, w) \log P(w \mid d)$ of the corpus. */
  logLikelihood: number
}

/** Options of `plsaSteps`. */
export type PlsaOptions = {
  /** The corpus, as word ids. */
  documents: Documents
  /** The number of topics $K$. */
  topics: Size
  /** The vocabulary size $V$. */
  vocabulary: Size
  /**
   * Initial $P(w \mid z)$ ($K \times V$) and $P(z \mid d)$ ($D \times K$), row-major, used as given (default random
   * from the init stream).
   */
  init?: { topicWord: ArrayLike<number>; docTopic: ArrayLike<number> }
}

/**
 * The pLSA log-likelihood $\sum_{d, w} n(d, w) \log \sum_k \theta_{dk}\phi_{kw}$, skipping zero counts.
 *
 * @param n The counts $n(d, w)$, $D \times V$ row-major.
 * @param theta The proportions $\theta_{dk} = P(k \mid d)$, $D \times K$ row-major.
 * @param phi The topics $\phi_{kw} = P(w \mid k)$, $K \times V$ row-major.
 * @param D The number of documents.
 * @param K The number of topics.
 * @param V The vocabulary size.
 * @returns The log-likelihood.
 */
function logLikelihoodOf(n: ArrayLike<number>, theta: Float64Array, phi: Float64Array, D: Size, K: Size, V: Size) {
  let ll = 0
  for (let d = 0; d < D; d++)
    for (let w = 0; w < V; w++) {
      const c = n[d * V + w]
      if (c === 0) continue
      let p = 0
      for (let k = 0; k < K; k++) p += theta[d * K + k] * phi[k * V + w]
      ll += c * Math.log(p)
    }
  return ll
}

/**
 * pLSA by EM as a step-through algorithm, one E-step and M-step per step. `init` draws random distributions from its
 * stream (each entry $0.5 + u$ with $u$ uniform, then normalised) unless `options.init` gives them. A state is
 * `converged` when the log-likelihood changes by less than $10^{-10}$ of its size.
 *
 * @param options The corpus, the number of topics and, optionally, the starting distributions.
 * @returns The algorithm, run with no start.
 *
 * @example Two vocabularies, two topics
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const final = run(plsaSteps({ documents, topics: 2, vocabulary: 6 }), undefined, 50, { stream: stream(0) })
 * print('top words of each topic', topWords(final.topicWord, 3))
 * print('P(z | d)', final.docTopic)
 * print('steps', final.t, 'converged', final.converged, 'log-likelihood', final.logLikelihood)
 */
export function plsaSteps(options: PlsaOptions): Algorithm<void, PlsaState> {
  const { topics: K, vocabulary: V, documents } = options
  const D = documents.length
  const n = toFlat(documentTermCounts(documents, V))
  const normalise = (a: Float64Array, rows: Size, cols: Size) => {
    for (let r = 0; r < rows; r++) {
      let s = 0
      for (let c = 0; c < cols; c++) s += a[r * cols + c]
      for (let c = 0; c < cols; c++) a[r * cols + c] /= s
    }
    return a
  }
  return {
    name: 'plsa-em',
    init: (_start, s) => {
      const phi = options.init
        ? Float64Array.from(options.init.topicWord)
        : normalise(
            Float64Array.from(units(child(s, 'topics'), K * V), (u) => 0.5 + u),
            K,
            V,
          )
      const theta = options.init
        ? Float64Array.from(options.init.docTopic)
        : normalise(
            Float64Array.from(units(child(s, 'documents'), D * K), (u) => 0.5 + u),
            D,
            K,
          )
      return {
        t: 0,
        topicWord: fromData(phi, [K, V]),
        docTopic: fromData(theta, [D, K]),
        logLikelihood: logLikelihoodOf(n, theta, phi, D, K, V),
      }
    },
    step: (state) => {
      const theta = Float64Array.from(state.docTopic.data)
      const phi = Float64Array.from(state.topicWord.data)
      const newPhi = new Float64Array(K * V)
      const newTheta = new Float64Array(D * K)
      const r = new Float64Array(K)
      for (let d = 0; d < D; d++)
        for (let w = 0; w < V; w++) {
          const c = n[d * V + w]
          if (c === 0) continue
          let z = 0
          for (let k = 0; k < K; k++) z += r[k] = theta[d * K + k] * phi[k * V + w]
          for (let k = 0; k < K; k++) {
            const e = (c * r[k]) / z
            newPhi[k * V + w] += e
            newTheta[d * K + k] += e
          }
        }
      normalise(newPhi, K, V)
      normalise(newTheta, D, K)
      const ll = logLikelihoodOf(n, newTheta, newPhi, D, K, V)
      return {
        t: state.t + 1,
        topicWord: fromData(newPhi, [K, V]),
        docTopic: fromData(newTheta, [D, K]),
        logLikelihood: ll,
        converged: Math.abs(ll - state.logLikelihood) < 1e-10 * Math.abs(ll),
      }
    },
  }
}
