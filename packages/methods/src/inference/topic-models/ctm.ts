/**
 * A correlated topic model (Blei and Lafferty, 2007) fitted by a MAP variant of its variational EM ("CTM-lite"): each
 * document's topic proportions are $\thetavec_d = \operatorname{softmax}(\etavec_d)$ with
 * $\etavec_d \sim \Gauss(\muvec, \Sigmamat)$, so topics can co-occur more or less than independent Dirichlet draws
 * allow.
 *
 * The E-step finds each $\etavec_d$ at the mode of its log posterior given the topics, which is, up to a constant,
 * $-\tfrac{1}{2}(\etavec - \muvec)^\top \Sigmamat^{-1} (\etavec - \muvec)$ plus
 * $\sum_w n_{dw} \log \sum_k \theta_k\beta_{kw}$,
 * by L-BFGS with automatic gradients (Blei and Lafferty fit a Gaussian variational posterior instead; the mode is its
 * point-estimate limit). The M-step re-estimates the topics $\betavec_k$ from the expected counts and
 * $(\muvec, \Sigmamat)$ from the modes, with $\Sigmamat$ shrunk towards the identity.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, units } from 'aifn-compute/foundation/random'
import {
  add,
  expandDims,
  fromData,
  logsumexp,
  matmul,
  mul,
  neg,
  sub,
  sum,
  take,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import { logSoftmax } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import type { Documents } from './corpus'

/** Options of `correlatedTopicSteps`. */
export type CorrelatedTopicOptions = {
  /** The corpus, as word ids. */
  documents: Documents
  /** The number of topics $K$. */
  topics: Size
  /** The vocabulary size $V$. */
  vocabulary: Size
  /** Pseudo-count added to every topic-word count in the M-step (default 0.01). */
  smoothing?: number
  /**
   * $s$, the weight of the identity in the covariance update
   * $\Sigmamat \leftarrow (1 - s)\operatorname{cov}(\etavec) + s\Imat$ (default 0.1).
   */
  shrinkage?: number
  /** L-BFGS steps per document in the E-step (default 15). */
  innerSteps?: Size
}

/** The state of `correlatedTopicSteps`. */
export interface CorrelatedTopicState extends Status {
  /** EM steps done. */
  t: Size
  /** The topics $\betavec_k$, $K \times V$, each a distribution over words. */
  topicWord: Tensor
  /**
   * Each document's logistic-normal coordinates $\etavec_d$, $D \times K$; its proportions are
   * $\operatorname{softmax}(\etavec_d)$.
   */
  eta: Tensor
  /** The topic proportions $\thetavec_d = \operatorname{softmax}(\etavec_d)$, $D \times K$. */
  docTopic: Tensor
  /** The mean $\muvec$ of the logistic normal ($K$ values). */
  mean: Tensor
  /** The covariance $\Sigmamat$ of the logistic normal, $K \times K$. */
  covariance: Tensor
  /** $\sum_d \sum_w n_{dw} \log \sum_k \theta_{dk} \beta_{kw}$ at the current estimates. */
  logLikelihood: number
}

/**
 * The softmax of each row of a $D \times K$ row-major array, computed stably.
 *
 * @param eta The rows $\etavec_d$, $D \times K$ row-major; not modified.
 * @param D The number of rows.
 * @param K The number of columns.
 * @returns The rows $\operatorname{softmax}(\etavec_d)$, a new $D \times K$ row-major array.
 */
const softmaxRows = (eta: Float64Array, D: Size, K: Size) => {
  const out = new Float64Array(D * K)
  for (let d = 0; d < D; d++) {
    let m = -Infinity
    for (let k = 0; k < K; k++) m = Math.max(m, eta[d * K + k])
    let z = 0
    for (let k = 0; k < K; k++) z += out[d * K + k] = Math.exp(eta[d * K + k] - m)
    for (let k = 0; k < K; k++) out[d * K + k] /= z
  }
  return out
}

/**
 * CTM-lite as a step-through algorithm: each step is one E-step over every document and one M-step. `init` draws
 * random topics from its stream (each entry $0.5 + u$ with $u$ uniform, normalised) and starts with
 * $\etavec_d = \zeros$, $\muvec = \zeros$ and $\Sigmamat = \Imat$. A state is flagged `diverged` when its
 * log-likelihood is not finite; nothing sets `converged`.
 *
 * @param options The corpus, the number of topics and the fitting settings.
 * @returns The algorithm, run with no start.
 *
 * @example Two vocabularies, two topics that do not co-occur
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const alg = correlatedTopicSteps({ documents, topics: 2, vocabulary: 6 })
 * const final = run(alg, undefined, 10, { stream: stream(0) })
 * print('top words of each topic', topWords(final.topicWord, 3))
 * print('topic proportions', final.docTopic)
 * print('topic correlations', topicCorrelations(final.covariance))
 */
export function correlatedTopicSteps(options: CorrelatedTopicOptions): Algorithm<void, CorrelatedTopicState> {
  const { documents, topics: K, vocabulary: V, smoothing = 0.01, shrinkage = 0.1, innerSteps = 15 } = options
  const D = documents.length
  // Each document's distinct words and their counts.
  const docs = documents.map((doc) => {
    const counts = new Map<number, number>()
    for (const w of doc) counts.set(w, (counts.get(w) ?? 0) + 1)
    return { words: [...counts.keys()], counts: fromData(Float64Array.from(counts.values()), [counts.size]) }
  })
  const loglik = (beta: Float64Array, theta: Float64Array) => {
    let ll = 0
    documents.forEach((doc, d) => {
      for (const w of doc) {
        let p = 0
        for (let k = 0; k < K; k++) p += theta[d * K + k] * beta[k * V + w]
        ll += Math.log(p)
      }
    })
    return ll
  }
  const state = (
    t: Size,
    beta: Float64Array,
    eta: Float64Array,
    mu: Float64Array,
    cov: Float64Array,
  ): CorrelatedTopicState => {
    const theta = softmaxRows(eta, D, K)
    return {
      t,
      topicWord: fromData(beta, [K, V]),
      eta: fromData(eta, [D, K]),
      docTopic: fromData(theta, [D, K]),
      mean: fromData(mu, [K]),
      covariance: fromData(cov, [K, K]),
      logLikelihood: loglik(beta, theta),
    }
  }
  return {
    name: 'correlated-topic-model-map-em',
    init: (_start, s) => {
      const beta = Float64Array.from(units(child(s, 'topics'), K * V), (u) => 0.5 + u)
      for (let k = 0; k < K; k++) {
        let z = 0
        for (let w = 0; w < V; w++) z += beta[k * V + w]
        for (let w = 0; w < V; w++) beta[k * V + w] /= z
      }
      const cov = new Float64Array(K * K)
      for (let k = 0; k < K; k++) cov[k * K + k] = 1
      return state(0, beta, new Float64Array(D * K), new Float64Array(K), cov)
    },
    step: (s) => {
      const beta = Float64Array.from(s.topicWord.data)
      const logBeta = fromData(Float64Array.from(beta, Math.log), [K, V])
      const mu = fromData(Float64Array.from(s.mean.data), [K])
      const precision = inverse(s.covariance) as Tensor
      // E-step: the posterior mode of each η_d with the topics fixed.
      const eta = Float64Array.from(s.eta.data)
      docs.forEach(({ words, counts }, d) => {
        const lb = take(transpose(logBeta), words) // [W, K]: log β_kw for the document's words
        const objective = {
          kind: 'objective' as const,
          name: 'ctm-document',
          dim: K,
          value: (x: Value) => {
            const centred = sub(x, mu)
            const prior = mul(0.5, sum(mul(centred, matmul(precision, centred))))
            const words_ = logsumexp(add(lb, expandDims(logSoftmax(x), 0)), -1) // [W]
            return add(prior, neg(sum(mul(counts, words_))))
          },
        }
        const r = minimize(objective, eta.subarray(d * K, (d + 1) * K), { method: 'lbfgs', maxSteps: innerSteps })
        eta.set(toFlat(r.x), d * K)
      })
      // M-step: topics from the expected counts, (μ, Σ) from the modes.
      const theta = softmaxRows(eta, D, K)
      const next = new Float64Array(K * V).fill(smoothing)
      const resp = new Float64Array(K)
      documents.forEach((doc, d) => {
        for (const w of doc) {
          let z = 0
          for (let k = 0; k < K; k++) z += resp[k] = theta[d * K + k] * beta[k * V + w]
          for (let k = 0; k < K; k++) next[k * V + w] += resp[k] / z
        }
      })
      for (let k = 0; k < K; k++) {
        let z = 0
        for (let w = 0; w < V; w++) z += next[k * V + w]
        for (let w = 0; w < V; w++) next[k * V + w] /= z
      }
      const mean = new Float64Array(K)
      for (let d = 0; d < D; d++) for (let k = 0; k < K; k++) mean[k] += eta[d * K + k] / D
      const cov = new Float64Array(K * K)
      for (let d = 0; d < D; d++)
        for (let a = 0; a < K; a++)
          for (let b = 0; b < K; b++) cov[a * K + b] += ((eta[d * K + a] - mean[a]) * (eta[d * K + b] - mean[b])) / D
      for (let a = 0; a < K; a++)
        for (let b = 0; b < K; b++) cov[a * K + b] = (1 - shrinkage) * cov[a * K + b] + (a === b ? shrinkage : 0)
      const out = state(s.t + 1, next, eta, mean, cov)
      return { ...out, diverged: !Number.isFinite(out.logLikelihood) }
    },
  }
}

/**
 * The correlation matrix $C_{ab} = \Sigma_{ab} / \sqrt{\Sigma_{aa}\Sigma_{bb}}$ of the logistic normal's covariance:
 * how strongly topics co-occur across documents.
 *
 * @param covariance The covariance $\Sigmamat$, $K \times K$, with a positive diagonal.
 * @returns The $K \times K$ correlation matrix, with a unit diagonal.
 *
 * @example A covariance as correlations
 * print(topicCorrelations(tensor([[4, -1], [-1, 1]])))
 */
export function topicCorrelations(covariance: Tensor): Tensor {
  const [K] = covariance.shape
  const c = toFlat(covariance)
  return fromData(
    Float64Array.from(
      { length: K * K },
      (_, i) => c[i] / Math.sqrt(c[Math.floor(i / K) * (K + 1)] * c[(i % K) * (K + 1)]),
    ),
    [K, K],
  )
}
