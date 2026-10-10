/**
 * Fitting any one topic model of the module to a corpus while streaming its progress, for a page to plot and play
 * (`topicModelRun`).
 *
 * After each step the run records the log-likelihood per token and the NPMI coherence of the topics' top words
 * (`aifn-compute/text/cooccurrence`), and at each checkpoint the topics, the documents' topic proportions and the top
 * words. Every method is reported in the same terms: a $K \times V$ topic-word matrix and a $D \times K$
 * document-topic matrix, with $K$ changing over the run only for the HDP, whose number of topics is inferred.
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
  /** The corpus, as word ids. */
  documents: Documents
  /** The vocabulary size $V$. */
  vocabulary: Size
  /** The model to fit. */
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
  /** LDA's $\alpha$, and the HDP's document concentration $\alpha$ (default 0.5). */
  alpha?: number
  /** LDA's $\beta$, and the HDP's topic-word concentration $\eta$ (default 0.05). */
  beta?: number
  /** The HDP's top-level concentration $\gamma$ (default 1). */
  gamma?: number
  /**
   * Labelled LDA: each document's label in $\{0, \dots, L - 1\}$; document $d$ may use its label's topic and the shared
   * background topic $L$. Required for `'labelled-lda'`.
   */
  labels?: ArrayLike<number>
  /** Top words per topic for coherence and display (default 8). */
  top?: Size
  /** The seed of the root stream (default `'topics'`). */
  seed?: string | number
}

/** One checkpoint of a topic-model run. */
export type TopicCheckpoint = {
  /** The step it was taken at. */
  readonly step: Size
  /** Topics at this checkpoint (fixed except for the HDP). */
  readonly topics: Size
  /** Topic-word weights, $K \times V$ row-major. */
  readonly topicWord: Float64Array
  /** Document-topic weights, $D \times K$ row-major. */
  readonly docTopic: Float64Array
  /** The top words of each topic. */
  readonly top: number[][]
  /** The NPMI coherence of each topic. */
  readonly coherence: number[]
}

/** A snapshot of `topicModelRun`. */
export type TopicSnapshot = {
  /** The step it was taken at. */
  readonly step: Size
  /** The total number of steps of the run (0 for LSA). */
  readonly steps: Size
  /** True for the last snapshot. */
  readonly done: boolean
  /** The model being fitted. */
  readonly method: TopicMethod
  /** Topics at the latest checkpoint. */
  readonly topics: Size
  /** Per step: the fit (log-likelihood per token; NaN for LSA), the mean coherence and the topics in use. */
  readonly history: { step: number[]; logLikelihood: number[]; coherence: number[]; topics: number[] }
  /** Every checkpoint so far, oldest first. */
  readonly checkpoints: readonly TopicCheckpoint[]
}

/**
 * The training log-likelihood per token, $\frac{1}{N} \sum_d \sum_{n} \log \sum_k \theta_{dk} \phi_{k w_{dn}}$ over the
 * $N$ tokens (NaN when a mixture probability is negative, as with LSA's signed weights).
 *
 * @param documents The corpus, as word ids.
 * @param topicWord The topics $\phivec$, $K \times V$, each row a distribution over words.
 * @param docTopic The proportions $\thetavec$, $D \times K$, one row per document.
 * @returns The mean log probability of a token; 0 for an empty corpus.
 *
 * @example Two topics that fit the corpus, against uniform ones
 * const documents = [[0, 1, 0, 1], [2, 3, 3, 2]]
 * const topics = tensor([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
 * print('fitted', logLikelihoodPerToken(documents, topics, tensor([[1, 0], [0, 1]])))
 * print('uniform', logLikelihoodPerToken(documents, tensor([[0.25, 0.25, 0.25, 0.25]]), tensor([[1], [1]])))
 * print('log(1/2) =', Math.log(0.5), 'log(1/4) =', Math.log(0.25))
 */
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

/**
 * Fit a topic model step by step and yield a snapshot at every checkpoint: a generator, so a worker can stream it. The
 * first snapshot is the initial state (step 0); one follows every `every` steps and at the last step. LSA has no steps
 * and yields one snapshot. Labelled LDA runs `ldaCollapsedGibbs` with $K = L + 1$ topics, each document allowed its
 * label's topic and the background topic $L$. The run's draws derive from `stream(seed)`: `init` from its child
 * `'init'`, step $t$ from its child `('step', t)`.
 *
 * @param options The corpus, the method and its settings.
 * @returns A generator of snapshots, each holding the whole history and every checkpoint so far.
 *
 * @example LDA on two vocabularies: the fit and the coherence rise
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const options = { documents, vocabulary: 6, method: 'lda', topics: 2, steps: 10, every: 5, top: 3 }
 * const snapshots = [...topicModelRun(options)]
 * const last = snapshots.at(-1)
 * print('snapshots at steps', snapshots.map((s) => s.step), 'done', last.done)
 * print('log-likelihood per token', last.history.logLikelihood[0], '->', last.history.logLikelihood.at(-1))
 * print('top words', last.checkpoints.at(-1).top, 'coherence', last.checkpoints.at(-1).coherence)
 */
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
