/**
 * The hierarchical Dirichlet process topic model (Teh, Jordan, Beal & Blei, 2006, "Hierarchical Dirichlet processes",
 * JASA 101(476)): LDA with an unbounded number of topics. Global topic weights β ~ GEM(γ) are shared by every
 * document, each document's proportions are θ_d ~ DP(α, β), and each topic is φ_k ~ Dir(η). The number of topics in
 * use is inferred: a corpus with five themes ends up with about five topics.
 *
 * Fitted by the direct-assignment Gibbs sampler (§5.3 of the paper): θ and φ are integrated out, and only the finitely
 * many topics in use are represented, with β = (β_1, …, β_K, β_u) where β_u is the mass of all unused topics.
 * - Each token's topic is drawn from p(z = k | rest) ∝ (n_dk + αβ_k)(n_kw + η)/(n_k + Vη) for a used topic, and
 *   ∝ αβ_u/V for a new one; a new topic takes a share b ~ Beta(1, γ) of β_u (the stick-breaking construction).
 * - Topics left with no tokens are removed, their weight returned to β_u.
 * - The table counts m_dk (how many "tables" of the Chinese restaurant franchise serve topic k in document d) are drawn
 *   as m_dk = Σ_{j<n_dk} Bernoulli(αβ_k/(αβ_k + j)) (Antoniak's distribution), and then β ~ Dir(m_·1, …, m_·K, γ).
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
  documents: Documents
  vocabulary: Size
  /** Document-level concentration α (default 1), top-level concentration γ (default 1), topic–word Dirichlet η (0.05). */
  alpha?: number
  gamma?: number
  eta?: number
  /** Topics the chain starts with, tokens assigned at random (default 10). */
  initialTopics?: Size
}

/** The state of `hdpGibbs`. */
export interface HdpState extends Status {
  /** Sweeps done. */
  t: Size
  /** Topics in use, K. */
  topics: Size
  /** z_dn per document (int32), in 0 … K − 1. */
  assignments: Tensor[]
  /** n_dk [D, K], n_kw [K, V] and n_k [K]. */
  docTopic: Matrix
  topicWord: Matrix
  topicTotals: Tensor
  /** The global weights (β_1, …, β_K, β_u) [K + 1]; the last is the mass of every unused topic. */
  weights: Tensor
  /** log p(w | z) with the topics integrated out (as LDA's, at the current K). */
  logLikelihood: number
}

/** The HDP topic model fitted by direct-assignment Gibbs sampling, one sweep (tokens, then tables and β) per step. */
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

/** The tables without topics that hold no tokens; their weight goes back to the unused mass. */
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
 * Point estimates from an HDP state: φ_kw = (n_kw + η)/(n_k + Vη) [K, V], and each document's proportions over the
 * topics in use, θ_dk ∝ n_dk + αβ_k [D, K] (the unused mass left out).
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
