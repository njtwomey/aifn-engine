/**
 * A dynamic topic model (Blei & Lafferty, 2006, "Dynamic topic models", ICML): the corpus is split into time slices,
 * and each topic's natural parameters drift between slices as a Gaussian random walk, β_{t,k} ~ N(β_{t−1,k}, σ²I), with
 * word distribution φ_{t,k} = softmax(β_{t,k}). Documents of slice t are LDA documents over the slice's topics, with
 * proportions θ_d ~ Dir(α) (fixed α, as gensim's `LdaSeqModel`).
 *
 * Fitted by a MAP variant of the paper's variational EM ("DTM-lite"), so that one objective rises at every step:
 * - E-step: mean-field variational LDA for each document given its slice's topics (Blei, Ng & Jordan, 2003): word
 *   responsibilities r_wk ∝ exp(E[log θ_k]) φ_{t,k,w} and γ_dk = α + Σ_w n_dw r_wk, warm-started from the last step.
 * - M-step: for each topic, the whole chain β_{1:T,k} at the mode of Σ_t Σ_w n_tkw log φ_{t,k,w} − Σ_t ‖β_t − β_{t−1}‖²/2σ²
 *   − ‖β_1‖²/2σ₀², n_tkw the expected counts, by L-BFGS with automatic gradients. Blei and Lafferty instead keep a
 *   Gaussian posterior over the chain, computed by a variational Kalman filter and smoother; the mode is the
 *   point-estimate limit of that posterior, and the random-walk prior is what ties neighbouring slices together.
 * The topics start from a static LDA fitted to all slices pooled. The objective (the evidence lower bound plus the log
 * prior of the topics) never decreases.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, stream, units } from 'aifn-compute/foundation/random'
import {
  expandDims,
  fromData,
  logsumexp,
  matmul,
  mul,
  reshape,
  square,
  sub,
  sum,
  add,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { digamma, logGamma } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import type { Documents } from './corpus'
import { ldaCollapsedGibbs, ldaEstimates, type LdaOptions } from './lda'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `dynamicTopicSteps`. */
export type DynamicTopicOptions = {
  documents: Documents
  /** Each document's time slice, an integer in 0 … T − 1. */
  times: ArrayLike<number>
  topics: Size
  vocabulary: Size
  /** Dirichlet concentration of the documents' proportions (default 0.1). */
  alpha?: number
  /** Variance σ² of a topic's drift between neighbouring slices (default 0.5) and σ₀² of the first slice (10). */
  variance?: number
  initialVariance?: number
  /** Variational iterations per document in each E-step (default 10) and L-BFGS steps per topic in the M-step (25). */
  innerSteps?: Size
  topicSteps?: Size
  /**
   * Collapsed Gibbs sweeps of a static LDA over all slices pooled, whose topics start every slice (default 50, as
   * gensim's `LdaSeqModel` starts from an LDA fit); 0 starts from random topics.
   */
  initialSweeps?: Size
}

/** The state of `dynamicTopicSteps`. */
export interface DynamicTopicState extends Status {
  t: Size
  /** The topics' natural parameters β [T, K, V] and word distributions φ = softmax(β) [T, K, V]. */
  logTopics: Tensor
  topicWord: Tensor
  /** Variational Dirichlet parameters γ [D, K], and the proportions E[θ] = γ/Σγ [D, K]. */
  gamma: Tensor
  docTopic: Tensor
  /** The evidence lower bound plus the topics' log prior (up to a constant); non-decreasing. */
  objective: number
  /** Σ_d Σ_w n_dw log Σ_k E[θ_dk] φ_{t_d,k,w} / N: the per-token log-likelihood at the point estimates. */
  logLikelihood: number
}

/** DTM-lite as a step-through algorithm: each step is one E-step over every document and one M-step per topic. */
export function dynamicTopicSteps(options: DynamicTopicOptions): Algorithm<void, DynamicTopicState> {
  const { documents, topics: K, vocabulary: V, alpha = 0.1, variance = 0.5, initialVariance = 10 } = options
  const { innerSteps = 10, topicSteps = 25, initialSweeps = 50 } = options
  const D = documents.length
  const times = Array.from(options.times)
  if (times.length !== D) throw new ShapeError('dynamicTopicSteps', 'dynamicTopicSteps: one time slice per document')
  if (times.some((t) => !(Number.isInteger(t) && t >= 0)))
    throw new DomainError('dynamicTopicSteps', 'dynamicTopicSteps: time slices must be non-negative integers')
  if (!(alpha > 0 && variance > 0 && initialVariance > 0))
    throw new DomainError(
      'dynamicTopicSteps',
      'dynamicTopicSteps: α, the variance and the initial variance must be positive',
    )
  const T = Math.max(...times) + 1
  const docs = documents.map((doc) => {
    const counts = new Map<number, number>()
    for (const w of doc) counts.set(w, (counts.get(w) ?? 0) + 1)
    return { words: [...counts.keys()], counts: [...counts.values()], length: doc.length }
  })
  const tokens = documents.reduce((a, d) => a + d.length, 0)

  const logSoftmaxRows = (beta: Float64Array) => {
    const out = new Float64Array(beta.length)
    for (let r = 0; r < T * K; r++) {
      let m = -Infinity
      for (let w = 0; w < V; w++) m = Math.max(m, beta[r * V + w])
      let z = 0
      for (let w = 0; w < V; w++) z += Math.exp(beta[r * V + w] - m)
      const lz = m + Math.log(z)
      for (let w = 0; w < V; w++) out[r * V + w] = beta[r * V + w] - lz
    }
    return out
  }
  const priorPenalty = (beta: Float64Array) => {
    let p = 0
    for (let k = 0; k < K; k++)
      for (let w = 0; w < V; w++) {
        p += beta[k * V + w] ** 2 / (2 * initialVariance)
        for (let t = 1; t < T; t++)
          p += (beta[(t * K + k) * V + w] - beta[((t - 1) * K + k) * V + w]) ** 2 / (2 * variance)
      }
    return p
  }

  /**
   * The responsibilities of one document given γ and the topics, and its ELBO; with `update`, then γ's optimum given
   * them. Expected counts are added to `expected` [T, K, V] when given.
   */
  const document = (
    d: number,
    gamma: Float64Array,
    logPhi: Float64Array,
    iterations: number,
    expected?: Float64Array,
  ) => {
    const { words, counts } = docs[d]
    const g = gamma.subarray(d * K, (d + 1) * K)
    const base = times[d] * K * V
    const r = new Float64Array(words.length * K)
    const elog = new Float64Array(K)
    const responsibilities = () => {
      let total = 0
      for (let k = 0; k < K; k++) total += g[k]
      const dt = digamma(total)
      for (let k = 0; k < K; k++) elog[k] = digamma(g[k]) - dt
      words.forEach((w, i) => {
        let m = -Infinity
        for (let k = 0; k < K; k++) m = Math.max(m, (r[i * K + k] = elog[k] + logPhi[base + k * V + w]))
        let z = 0
        for (let k = 0; k < K; k++) z += r[i * K + k] = Math.exp(r[i * K + k] - m)
        for (let k = 0; k < K; k++) r[i * K + k] /= z
      })
    }
    for (let it = 0; it < iterations; it++) {
      responsibilities()
      for (let k = 0; k < K; k++) {
        let s = alpha
        words.forEach((_, i) => (s += counts[i] * r[i * K + k]))
        g[k] = s
      }
    }
    if (iterations === 0) responsibilities()
    // The ELBO at (r, γ): E[log p(θ)] + E[log p(z, w | θ, φ)] − E[log q(θ)] − E[log q(z)].
    let total = 0
    for (let k = 0; k < K; k++) total += g[k]
    const dt = digamma(total)
    for (let k = 0; k < K; k++) elog[k] = digamma(g[k]) - dt
    let elbo = logGamma(K * alpha) - K * logGamma(alpha) - logGamma(total)
    for (let k = 0; k < K; k++) elbo += (alpha - g[k]) * elog[k] + logGamma(g[k])
    words.forEach((w, i) => {
      for (let k = 0; k < K; k++) {
        const q = r[i * K + k]
        if (q <= 0) continue
        elbo += counts[i] * q * (elog[k] + logPhi[base + k * V + w] - Math.log(q))
        if (expected) expected[base + k * V + w] += counts[i] * q
      }
    })
    return elbo
  }

  const state = (t: Size, beta: Float64Array, gamma: Float64Array, elbo: number): DynamicTopicState => {
    const logPhi = logSoftmaxRows(beta)
    const theta = Float64Array.from(gamma)
    for (let d = 0; d < D; d++) {
      let z = 0
      for (let k = 0; k < K; k++) z += theta[d * K + k]
      for (let k = 0; k < K; k++) theta[d * K + k] /= z
    }
    let ll = 0
    docs.forEach(({ words, counts }, d) =>
      words.forEach((w, i) => {
        let p = 0
        for (let k = 0; k < K; k++) p += theta[d * K + k] * Math.exp(logPhi[times[d] * K * V + k * V + w])
        ll += counts[i] * Math.log(p)
      }),
    )
    const objective = elbo - priorPenalty(beta)
    return {
      t,
      logTopics: fromData(beta, [T, K, V]),
      topicWord: fromData(logPhi.map(Math.exp), [T, K, V]),
      gamma: fromData(gamma, [D, K]),
      docTopic: fromData(theta, [D, K]),
      objective,
      logLikelihood: ll / Math.max(1, tokens),
      diverged: !Number.isFinite(objective),
    }
  }

  // The difference operator (T − 1) × T of the random walk, for the M-step's prior.
  const diff = new Float64Array(Math.max(0, T - 1) * T)
  for (let t = 1; t < T; t++) {
    diff[(t - 1) * T + t] = 1
    diff[(t - 1) * T + t - 1] = -1
  }
  const diffT = fromData(diff, [Math.max(0, T - 1), T])

  return {
    name: 'dynamic-topic-model-map-em',
    init: (_start, s) => {
      // Topics shared by every slice at the start: a static LDA's, or random; proportions spread evenly.
      const start = new Float64Array(K * V)
      if (initialSweeps > 0) {
        const lda: LdaOptions = { documents, topics: K, vocabulary: V, alpha, beta: 0.05 }
        const fitted = run(ldaCollapsedGibbs(lda), undefined, initialSweeps, { stream: child(s, 'lda') })
        const phi = toFlat(ldaEstimates(fitted, lda).topicWord)
        for (let i = 0; i < K * V; i++) start[i] = Math.log(phi[i])
      } else {
        const u = units(child(s, 'topics'), K * V)
        for (let i = 0; i < K * V; i++) start[i] = Math.log(0.5 + u[i])
      }
      const beta = new Float64Array(T * K * V)
      for (let t = 0; t < T; t++) beta.set(start, t * K * V)
      const gamma = new Float64Array(D * K)
      docs.forEach(({ length }, d) => gamma.fill(alpha + length / K, d * K, (d + 1) * K))
      const logPhi = logSoftmaxRows(beta)
      let elbo = 0
      for (let d = 0; d < D; d++) elbo += document(d, gamma, logPhi, 0)
      return state(0, beta, gamma, elbo)
    },
    step: (s) => {
      const beta = Float64Array.from(toFlat(s.logTopics))
      const gamma = Float64Array.from(toFlat(s.gamma))
      // E-step: r then γ for each document, warm-started from the last γ; the expected counts.
      const logPhi = logSoftmaxRows(beta)
      const expected = new Float64Array(T * K * V)
      for (let d = 0; d < D; d++) document(d, gamma, logPhi, innerSteps, expected)
      // The responsibilities at the final γ define the bound the M-step raises.
      const rExpected = new Float64Array(T * K * V)
      let rest = 0
      for (let d = 0; d < D; d++) rest += document(d, gamma, logPhi, 0, rExpected)
      // M-step: each topic's chain at the mode of its expected log-likelihood plus the random-walk prior.
      for (let k = 0; k < K; k++) {
        const counts = new Float64Array(T * V)
        const start = new Float64Array(T * V)
        for (let t = 0; t < T; t++)
          for (let w = 0; w < V; w++) {
            counts[t * V + w] = rExpected[(t * K + k) * V + w]
            start[t * V + w] = beta[(t * K + k) * V + w]
          }
        const n = fromData(counts, [T, V])
        const negative = (x: Value): Value => {
          const B = reshape(x, [T, V])
          const logLik = sum(mul(n, sub(B, expandDims(logsumexp(B, -1), -1))))
          const b0 = matmul(
            fromData(
              Float64Array.from({ length: T }, (_, t) => (t === 0 ? 1 : 0)),
              [1, T],
            ),
            B,
          )
          let prior: Value = mul(1 / (2 * initialVariance), sum(square(b0)))
          if (T > 1) prior = add(prior, mul(1 / (2 * variance), sum(square(matmul(diffT, B)))))
          return sub(prior, logLik)
        }
        const value = (x: Float64Array) => {
          const v = negative(fromData(x, [T * V]))
          return typeof v === 'number' ? v : toFlat(v as Tensor)[0]
        }
        const r = minimize(
          { kind: 'objective' as const, name: 'dtm-topic-chain', dim: T * V, value: negative },
          start,
          { method: 'lbfgs', maxSteps: topicSteps },
        )
        const x = Float64Array.from(toFlat(r.x))
        // L-BFGS's line search only accepts descent; keep the old chain if the result is no better.
        const chosen = value(x) <= value(start) ? x : start
        for (let t = 0; t < T; t++) for (let w = 0; w < V; w++) beta[(t * K + k) * V + w] = chosen[t * V + w]
      }
      // The bound at the new topics with the E-step's responsibilities (same r, new log φ).
      const newLogPhi = logSoftmaxRows(beta)
      let elbo = rest
      for (let i = 0; i < T * K * V; i++) elbo += rExpected[i] * (newLogPhi[i] - logPhi[i])
      return state(s.t + 1, beta, gamma, elbo)
    },
  }
}

/** Options of `dynamicTopicRun`. */
export type DynamicTopicRunOptions = DynamicTopicOptions & {
  /** EM steps (default 30) and a checkpoint every this many (default steps/15). */
  steps?: Size
  every?: Size
  seed?: string | number
}

/** One checkpoint of `dynamicTopicRun`: the topics of every slice [T, K, V] and the proportions [D, K], row-major. */
export type DynamicTopicCheckpoint = {
  readonly step: Size
  readonly topicWord: Float64Array
  readonly docTopic: Float64Array
}

/** A snapshot of `dynamicTopicRun`. */
export type DynamicTopicSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  /** Slices T, topics K and vocabulary V. */
  readonly shape: { T: Size; K: Size; V: Size }
  readonly history: { step: number[]; objective: number[]; logLikelihood: number[] }
  readonly checkpoints: readonly DynamicTopicCheckpoint[]
}

/** Fit DTM-lite step by step and yield a snapshot at every checkpoint: a generator, so a worker can stream it. */
export function* dynamicTopicRun(options: DynamicTopicRunOptions): Generator<DynamicTopicSnapshot> {
  const { steps = 30, seed = 'dtm' } = options
  const every = Math.max(1, options.every ?? Math.round(steps / 15))
  const alg = dynamicTopicSteps(options)
  const root = stream(seed)
  const history = { step: [] as number[], objective: [] as number[], logLikelihood: [] as number[] }
  const checkpoints: DynamicTopicCheckpoint[] = []
  let s = alg.init(undefined, child(root, 'init'))
  const [T, K, V] = s.topicWord.shape
  const record = (keep: boolean) => {
    history.step.push(s.t)
    history.objective.push(s.objective)
    history.logLikelihood.push(s.logLikelihood)
    if (keep)
      checkpoints.push({
        step: s.t,
        topicWord: Float64Array.from(toFlat(s.topicWord)),
        docTopic: Float64Array.from(toFlat(s.docTopic)),
      })
  }
  const snapshot = (done: boolean): DynamicTopicSnapshot => ({
    step: s.t,
    steps,
    done,
    shape: { T, K, V },
    history: { step: [...history.step], objective: [...history.objective], logLikelihood: [...history.logLikelihood] },
    checkpoints: [...checkpoints],
  })
  record(true)
  yield snapshot(steps === 0)
  for (let t = 1; t <= steps; t++) {
    s = alg.step(s, { t: t - 1, stream: child(root, 'step', t) })
    const keep = t % every === 0 || t === steps
    record(keep)
    if (keep) yield snapshot(t === steps)
  }
}
