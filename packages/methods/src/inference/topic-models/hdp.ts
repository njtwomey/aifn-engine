/**
 * The hierarchical Dirichlet process topic model (Teh, Jordan, Beal and Blei, 2006, "Hierarchical Dirichlet
 * processes", JASA 101(476)): LDA with an unbounded number of topics. Global topic weights
 * $\betavec \sim \operatorname{GEM}(\gamma)$ are shared by every document, each document's proportions are
 * $\thetavec_d \sim \operatorname{DP}(\alpha, \betavec)$, and each topic is $\phivec_k \sim \Dir(\eta)$. The number of
 * topics in use is inferred: a corpus with five themes ends up with about five topics.
 *
 * Fitted by the direct-assignment Gibbs sampler (§5.3 of the paper): $\thetavec$ and $\phivec$ are integrated out, and
 * only the finitely many topics in use are represented, with $\betavec = (\beta_1, \dots, \beta_K, \beta_u)$ where
 * $\beta_u$ is the mass of all unused topics.
 * - Each token's topic is drawn from
 *   $p(z = k \mid \text{rest}) \propto (n_{dk} + \alpha\beta_k)(n_{kw} + \eta)/(n_k + V\eta)$ for a used topic, and
 *   $\propto \alpha\beta_u/V$ for a new one; a new topic takes a share $b \sim \Beta(1, \gamma)$ of $\beta_u$ (the
 *   stick-breaking construction).
 * - After the tokens of a sweep, topics left with no tokens are removed, their weight returned to $\beta_u$.
 * - The table counts $m_{dk}$ (how many "tables" of the Chinese restaurant franchise serve topic $k$ in document $d$)
 *   are drawn as $m_{dk} = \sum_{j=0}^{n_{dk}-1} \Bern(\alpha\beta_k/(\alpha\beta_k + j))$ (Antoniak's distribution),
 *   and then $\betavec \sim \Dir(m_{\cdot 1}, \dots, m_{\cdot K}, \gamma)$.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { categorical, child, integers, units } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { beta as betaDraw, dirichlet } from 'aifn-compute/probability/samplers'
import type { Documents } from './corpus'
import { ldaLogLikelihood } from './lda'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `hdpGibbs`. */
export type HdpOptions = {
  /** The corpus, as word ids. */
  documents: Documents
  /** The vocabulary size $V$. */
  vocabulary: Size
  /** The document-level concentration $\alpha$ (default 1). */
  alpha?: number
  /** The top-level concentration $\gamma$ (default 1). */
  gamma?: number
  /** The symmetric topic-word Dirichlet concentration $\eta$ (default 0.05). */
  eta?: number
  /** Topics the chain starts with, tokens assigned to them uniformly at random (default 10). */
  initialTopics?: Size
}

/** The state of `hdpGibbs`. */
export interface HdpState extends Status {
  /** Sweeps done. */
  t: Size
  /** The number of topics in use, $K$. */
  topics: Size
  /** The topic $z_{dn}$ of every token, one int32 vector per document, in $\{0, \dots, K - 1\}$. */
  assignments: Tensor[]
  /** The counts $n_{dk}$ of each document's tokens in each topic, $D \times K$. */
  docTopic: Matrix
  /** The counts $n_{kw}$ of each word's tokens in each topic, $K \times V$. */
  topicWord: Matrix
  /** The tokens in each topic, $n_k$ ($K$ values). */
  topicTotals: Tensor
  /**
   * The global weights $(\beta_1, \dots, \beta_K, \beta_u)$, $K + 1$ values summing to 1; the last is the mass of
   * every unused topic.
   */
  weights: Tensor
  /** $\log p(\wvec \mid \zvec)$ with the topics integrated out (`ldaLogLikelihood` with $\eta$, at the current $K$). */
  logLikelihood: number
}

/**
 * The HDP topic model fitted by direct-assignment Gibbs sampling, one sweep (tokens, then tables and $\betavec$) per
 * step. `init` assigns each token one of `initialTopics` topics uniformly from its stream, with equal weights
 * $1/(K + 1)$, and drops the topics left empty. A state is flagged `diverged` when its log-likelihood is not finite.
 *
 * @param options The corpus and the concentrations. Throws `DomainError` when $\alpha$, $\gamma$ or $\eta$ is not
 *   positive, or `initialTopics` is not a positive integer.
 * @returns The algorithm, run with no start; read point estimates of a state with `hdpEstimates`.
 *
 * @example Two vocabularies: the chain settles on two topics from five
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const alg = hdpGibbs({ documents, vocabulary: 6, initialTopics: 5 })
 * print('topics at the start', run(alg, undefined, 0, { stream: stream(1) }).topics)
 * const final = run(alg, undefined, 20, { stream: stream(1) })
 * print('topics after 20 sweeps', final.topics, 'with tokens', final.topicTotals)
 * print('global weights (the last unused)', final.weights)
 */
export function hdpGibbs(options: HdpOptions): Algorithm<void, HdpState> {
  const { documents, vocabulary: V, alpha = 1, gamma = 1, eta = 0.05, initialTopics = 10 } = options
  if (!(alpha > 0 && gamma > 0 && eta > 0)) throw new DomainError('hdpGibbs', 'hdpGibbs: α, γ and η must be positive')
  if (!(Number.isInteger(initialTopics) && initialTopics >= 1))
    throw new DomainError('hdpGibbs', 'hdpGibbs: initialTopics must be a positive integer')
  const D = documents.length

  // Working tables with one row (or entry) per topic, so topics can be added and removed.
  type Tables = { z: Int32Array[]; ndk: number[][]; nkw: Float64Array[]; nk: number[]; beta: number[]; betaU: number }
  const toState = (t: Size, tb: Tables): HdpState => {
    const K = tb.nk.length
    const nkw = new Float64Array(K * V)
    tb.nkw.forEach((row, k) => nkw.set(row, k * V))
    const ndk = new Float64Array(D * K)
    tb.ndk.forEach((row, d) => row.forEach((v, k) => (ndk[d * K + k] = v)))
    const logLikelihood = ldaLogLikelihood(nkw, tb.nk, K, V, eta)
    return {
      t,
      topics: K,
      assignments: tb.z.map((z) => fromData(Int32Array.from(z), [z.length])),
      docTopic: fromData(ndk, [D, K]),
      topicWord: fromData(nkw, [K, V]),
      topicTotals: fromData(Float64Array.from(tb.nk), [K]),
      weights: fromData(Float64Array.from([...tb.beta, tb.betaU]), [K + 1]),
      logLikelihood,
      diverged: !Number.isFinite(logLikelihood),
    }
  }
  const fromState = (s: HdpState): Tables => {
    const K = s.topics
    const nkw = toFlat(s.topicWord)
    const ndk = toFlat(s.docTopic)
    const w = toFlat(s.weights)
    return {
      z: s.assignments.map((a) => Int32Array.from(toFlat(a))),
      ndk: Array.from({ length: D }, (_, d) => Array.from(ndk.slice(d * K, (d + 1) * K))),
      nkw: Array.from({ length: K }, (_, k) => Float64Array.from(nkw.slice(k * V, (k + 1) * V))),
      nk: Array.from(toFlat(s.topicTotals)),
      beta: Array.from(w.slice(0, K)),
      betaU: w[K],
    }
  }

  return {
    name: 'hdp-direct-assignment-gibbs',
    init: (_start, s) => {
      const K = initialTopics
      const tb: Tables = {
        z: [],
        ndk: documents.map(() => new Array<number>(K).fill(0)),
        nkw: Array.from({ length: K }, () => new Float64Array(V)),
        nk: new Array<number>(K).fill(0),
        beta: new Array<number>(K).fill(1 / (K + 1)),
        betaU: 1 / (K + 1),
      }
      tb.z = documents.map((doc, d) =>
        Int32Array.from(doc, (w, n) => {
          const k = integers(child(s, d, n), K)
          tb.ndk[d][k]++
          tb.nkw[k][w]++
          tb.nk[k]++
          return k
        }),
      )
      return toState(0, removeEmpty(tb))
    },
    step: (s, ctx) => {
      const tb = fromState(s)
      const p: number[] = []
      documents.forEach((doc, d) => {
        const z = tb.z[d]
        doc.forEach((w, n) => {
          const old = z[n]
          tb.ndk[d][old]--
          tb.nkw[old][w]--
          tb.nk[old]--
          const K = tb.nk.length
          p.length = K + 1
          for (let k = 0; k < K; k++)
            p[k] = ((tb.ndk[d][k] + alpha * tb.beta[k]) * (tb.nkw[k][w] + eta)) / (tb.nk[k] + V * eta)
          p[K] = (alpha * tb.betaU) / V
          let k = categorical(child(ctx.stream, 'z', d, n), p)
          if (k === K) {
            // A new topic: break a piece off the unused mass.
            const b = betaDraw(child(ctx.stream, 'stick', d, n), 1, gamma) as number
            tb.beta.push(b * tb.betaU)
            tb.betaU *= 1 - b
            tb.nkw.push(new Float64Array(V))
            tb.nk.push(0)
            for (const row of tb.ndk) row.push(0)
            k = K
          }
          z[n] = k
          tb.ndk[d][k]++
          tb.nkw[k][w]++
          tb.nk[k]++
        })
      })
      const live = removeEmpty(tb)
      // Table counts by Antoniak's distribution, then the global weights.
      const K = live.nk.length
      const m = new Array<number>(K).fill(0)
      live.ndk.forEach((row, d) =>
        row.forEach((count, k) => {
          if (count === 0) return
          const a = alpha * live.beta[k]
          const u = units(child(ctx.stream, 'tables', d, k), count)
          for (let j = 0; j < count; j++) if (u[j] < a / (a + j)) m[k]++
        }),
      )
      const drawn = toFlat(dirichlet(child(ctx.stream, 'weights'), [...m, gamma]))
      live.beta = Array.from(drawn.slice(0, K))
      live.betaU = drawn[K]
      return toState(s.t + 1, live)
    },
  }
}

/**
 * The tables without topics that hold no tokens, renumbered in order; their weight goes back to the unused mass.
 *
 * @param tb The working tables: assignments `z`, counts `ndk`, `nkw` and `nk`, weights `beta` and the unused mass
 *   `betaU`. Returned as they are when every topic holds a token; otherwise not modified.
 * @returns The tables with the empty topics removed and the assignments renumbered.
 */
function removeEmpty<
  T extends { z: Int32Array[]; ndk: number[][]; nkw: Float64Array[]; nk: number[]; beta: number[]; betaU: number },
>(tb: T): T {
  const keep = tb.nk.map((n) => n > 0)
  if (keep.every(Boolean)) return tb
  const newIndex: number[] = []
  let next = 0
  keep.forEach((k, i) => (newIndex[i] = k ? next++ : -1))
  return {
    ...tb,
    z: tb.z.map((z) => z.map((k) => newIndex[k])),
    ndk: tb.ndk.map((row) => row.filter((_, k) => keep[k])),
    nkw: tb.nkw.filter((_, k) => keep[k]),
    nk: tb.nk.filter((_, k) => keep[k]),
    beta: tb.beta.filter((_, k) => keep[k]),
    betaU: tb.betaU + tb.beta.reduce((a, b, k) => (keep[k] ? a : a + b), 0),
  }
}

/**
 * Point estimates from an HDP state: $\phi_{kw} = (n_{kw} + \eta)/(n_k + V\eta)$ ($K \times V$), and each document's
 * proportions over the topics in use, $\theta_{dk} \propto n_{dk} + \alpha\beta_k$ ($D \times K$, the unused mass left
 * out, so each row sums to 1).
 *
 * @param s A state of `hdpGibbs`.
 * @param options The `vocabulary` size and the concentrations the sampler used: `alpha` (default 1) and `eta` (default
 *   0.05).
 * @returns `topicWord`, $\phivec$ ($K \times V$), and `docTopic`, $\thetavec$ ($D \times K$).
 *
 * @example The topics and proportions the chain found
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const final = run(hdpGibbs({ documents, vocabulary: 6, initialTopics: 5 }), undefined, 20, { stream: stream(1) })
 * const { topicWord, docTopic } = hdpEstimates(final, { vocabulary: 6 })
 * print('top words of each topic', topWords(topicWord, 3))
 * print('topic proportions', docTopic)
 */
export function hdpEstimates(
  s: HdpState,
  options: Pick<HdpOptions, 'vocabulary' | 'alpha' | 'eta'>,
): {
  topicWord: Matrix
  docTopic: Matrix
} {
  const { vocabulary: V, alpha = 1, eta = 0.05 } = options
  const K = s.topics
  const nkw = toFlat(s.topicWord)
  const nk = toFlat(s.topicTotals)
  const ndk = toFlat(s.docTopic)
  const w = toFlat(s.weights)
  const D = ndk.length / Math.max(1, K)
  const phi = Float64Array.from(nkw, (v, i) => (v + eta) / (nk[Math.floor(i / V)] + V * eta))
  const theta = new Float64Array(D * K)
  for (let d = 0; d < D; d++) {
    let z = 0
    for (let k = 0; k < K; k++) z += theta[d * K + k] = ndk[d * K + k] + alpha * w[k]
    for (let k = 0; k < K; k++) theta[d * K + k] /= z
  }
  return { topicWord: fromData(phi, [K, V]), docTopic: fromData(theta, [D, K]) }
}
