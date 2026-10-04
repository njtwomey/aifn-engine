/**
 * `topicModelRun`: fit one topic model of the module to a corpus (the HDP with its number of topics inferred) and stream its progress for a page to plot and play:
 * the log-likelihood per token and the NPMI coherence of the topics' top words (`aifn-compute/text/cooccurrence`) after each
 * step, and checkpoints holding the topics, the documents' topic proportions and the top words.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { topicCoherence } from 'aifn-compute/text/cooccurrence'
import { topWords, type Documents } from './corpus'
import { correlatedTopicSteps } from './ctm'
import { lsaTopics, nmfTopics, nmfTopicSteps } from './factor'
import { hdpEstimates, hdpGibbs, type HdpOptions, type HdpState } from './hdp'
import { ldaCollapsedGibbs, ldaEstimates, type LdaOptions, type LdaState } from './lda'
import { plsaSteps } from './plsa'

/** The topic models `topicModelRun` can fit. */
export type TopicMethod = 'lda' | 'labelled-lda' | 'hdp' | 'plsa' | 'lsa' | 'nmf' | 'ctm'

/** The methods in the order a page lists them, with a display name. */
export const TOPIC_METHODS: readonly { method: TopicMethod; name: string }[] = [
  { method: 'lda', name: 'LDA (collapsed Gibbs)' },
  { method: 'hdp', name: 'HDP (direct-assignment Gibbs)' },
  { method: 'plsa', name: 'pLSA (EM)' },
  { method: 'nmf', name: 'NMF (KL)' },
  { method: 'lsa', name: 'LSA (SVD)' },
  { method: 'ctm', name: 'Correlated topic model' },
  { method: 'labelled-lda', name: 'Labelled LDA' },
]

/** Options of `topicModelRun`. */
export type TopicModelRunOptions = {
  documents: Documents
  vocabulary: Size
  method: TopicMethod
  /**
   * Topics K (labelled LDA: the number of labels, plus one shared background topic; HDP: the topics the chain starts
   * with, after which their number is inferred). Default 5.
   */
  topics?: Size
  /** Sweeps, EM or NMF steps (default 50; LSA has none). */
  steps?: Size
  /** Keep a checkpoint every this many steps (default steps/25). */
  every?: Size
  /** LDA's Dirichlet concentrations (default 0.5 and 0.05); HDP's document concentration α and topic–word η. */
  alpha?: number
  beta?: number
  /** HDP's top-level concentration γ (default 1). */
  gamma?: number
  /** Labelled LDA: each document's label in 0 … L − 1. */
  labels?: ArrayLike<number>
  /** Top words per topic for coherence and display (default 8). */
  top?: Size
  seed?: string | number
}

/** One checkpoint of a topic-model run. */
export type TopicCheckpoint = {
  readonly step: Size
  /** Topics at this checkpoint (fixed except for the HDP). */
  readonly topics: Size
  /** Topic × word weights [K, V] and document × topic weights [D, K], row-major. */
  readonly topicWord: Float64Array
  readonly docTopic: Float64Array
  /** The top words of each topic. */
  readonly top: number[][]
  /** The NPMI coherence of each topic. */
  readonly coherence: number[]
}

/** A snapshot of `topicModelRun`. */
export type TopicSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  readonly method: TopicMethod
  /** Topics at the latest checkpoint. */
  readonly topics: Size
  /** Per step: the fit, the mean coherence and the topics in use. */
  readonly history: { step: number[]; logLikelihood: number[]; coherence: number[]; topics: number[] }
  readonly checkpoints: readonly TopicCheckpoint[]
}

/** The training log-likelihood per token Σ_d Σ_w log Σ_k θ_dk φ_kw / N (NaN when the weights are not distributions). */
export function logLikelihoodPerToken(documents: Documents, topicWord: Tensor, docTopic: Tensor): number {
  const [K, V] = topicWord.shape
  const phi = toFlat(topicWord)
  const theta = toFlat(docTopic)
  let ll = 0
  let n = 0
  documents.forEach((doc, d) => {
    for (const w of doc) {
      let p = 0
      for (let k = 0; k < K; k++) p += theta[d * K + k] * phi[k * V + w]
      ll += Math.log(p)
      n++
    }
  })
  return ll / Math.max(1, n)
}

/** Fit a topic model step by step and yield a snapshot at every checkpoint: a generator, so a worker can stream it. */
export function* topicModelRun(options: TopicModelRunOptions): Generator<TopicSnapshot> {
  const {
    documents,
    vocabulary: V,
    method,
    steps: stepOption = 50,
    alpha = 0.5,
    beta = 0.05,
    top = 8,
    seed = 'topics',
  } = options
  const labelCount = options.labels ? Math.max(...Array.from(options.labels)) + 1 : 0
  const K = method === 'labelled-lda' ? labelCount + 1 : (options.topics ?? 5)
  const steps = method === 'lsa' ? 0 : stepOption
  const every = Math.max(1, options.every ?? Math.round(steps / 25))
  const root = stream(seed)
  const history = {
    step: [] as number[],
    logLikelihood: [] as number[],
    coherence: [] as number[],
    topics: [] as number[],
  }
  const checkpoints: TopicCheckpoint[] = []
  const record = (step: Size, topicWord: Tensor, docTopic: Tensor, keep: boolean) => {
    const words = topWords(topicWord, top)
    const coh = topicCoherence(words, documents)
    history.step.push(step)
    history.logLikelihood.push(method === 'lsa' ? NaN : logLikelihoodPerToken(documents, topicWord, docTopic))
    history.coherence.push(coh.mean)
    history.topics.push(topicWord.shape[0])
    if (keep)
      checkpoints.push({
        step,
        topics: topicWord.shape[0],
        topicWord: Float64Array.from(toFlat(topicWord)),
        docTopic: Float64Array.from(toFlat(docTopic)),
        top: words,
        coherence: Array.from(coh.topics),
      })
  }
  const snapshot = (step: Size, done: boolean): TopicSnapshot => ({
    step,
    steps,
    done,
    method,
    topics: history.topics.at(-1) ?? K,
    history: {
      step: [...history.step],
      logLikelihood: [...history.logLikelihood],
      coherence: [...history.coherence],
      topics: [...history.topics],
    },
    checkpoints: [...checkpoints],
  })

  if (method === 'lsa') {
    const r = lsaTopics(documents, V, K)
    record(0, r.topicWord, r.docTopic, true)
    yield snapshot(0, true)
    return
  }

  // Each method as an algorithm and the topic/document weights of its states.
  let alg: Algorithm<void, never>
  let weights: (s: never) => { topicWord: Tensor; docTopic: Tensor }
  if (method === 'lda' || method === 'labelled-lda') {
    const ldaOptions: LdaOptions = {
      documents,
      topics: K,
      vocabulary: V,
      alpha,
      beta,
      allowed: method === 'labelled-lda' ? Array.from(options.labels!, (l) => [l, K - 1]) : undefined,
    }
    alg = ldaCollapsedGibbs(ldaOptions) as never
    weights = (s) => ldaEstimates(s as LdaState, ldaOptions)
  } else if (method === 'hdp') {
    const hdpOptions: HdpOptions = {
      documents,
      vocabulary: V,
      alpha,
      eta: beta,
      gamma: options.gamma ?? 1,
      initialTopics: K,
    }
    alg = hdpGibbs(hdpOptions) as never
    weights = (s) => hdpEstimates(s as HdpState, hdpOptions)
  } else if (method === 'plsa') {
    alg = plsaSteps({ documents, topics: K, vocabulary: V }) as never
    weights = (s) => s
  } else if (method === 'nmf') {
    alg = nmfTopicSteps({ documents, vocabulary: V, topics: K }) as never
    weights = (s) => nmfTopics(s)
  } else {
    alg = correlatedTopicSteps({ documents, topics: K, vocabulary: V }) as never
    weights = (s) => s
  }
  let s = alg.init(undefined, child(root, 'init'))
  const show = (t: Size, keep: boolean) => {
    const w = weights(s)
    record(t, w.topicWord, w.docTopic, keep)
  }
  show(0, true)
  yield snapshot(0, false)
  for (let t = 1; t <= steps; t++) {
    s = alg.step(s, { t: t - 1, stream: child(root, 'step', t) })
    const keep = t % every === 0 || t === steps
    show(t, keep)
    if (keep) yield snapshot(t, t === steps)
  }
}
