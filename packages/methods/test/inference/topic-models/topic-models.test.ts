/**
 * The topic models: pLSA against a direct NumPy EM (`fixtures/inference/topic-models.json`); laws (pLSA's likelihood
 * never falls; labelled LDA uses only each document's topics; NMF topics are distributions; the CTM's likelihood
 * rises); and every method of `topicModelRun` on the seeded topic corpus, where the topics found must separate the
 * corpus's own topics (documents' dominant topic against their labels) and be coherent.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { tokenise } from 'aifn-compute/text/tokenise'
import { topicCorpus } from 'aifn-methods/text/corpora'
import {
  correlatedTopicSteps,
  groupByLabel,
  ldaCollapsedGibbs,
  lsaTopics,
  nmfTopicModel,
  plsaSteps,
  topicModelAlgorithms,
  topicModelFunctions,
  topicModelRun,
  type TopicMethod,
  type TopicSnapshot,
} from 'aifn-methods/inference/topic-models'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type F = {
  documents: number[][]
  topics: number
  vocabulary: number
  topicWord0: number[][]
  docTopic0: number[][]
  plsa: { steps: number; topicWord: number[][]; docTopic: number[][]; logLikelihood: number }[]
}
const T = fixture<F>('inference/topic-models')
const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `${i}`).toBeLessThanOrEqual(tol))

describe('pLSA', () => {
  const options = {
    documents: T.documents,
    topics: T.topics,
    vocabulary: T.vocabulary,
    init: { topicWord: T.topicWord0.flat(), docTopic: T.docTopic0.flat() },
  }
  for (const r of T.plsa)
    it(`matches a direct EM after ${r.steps} step(s)`, () => {
      const s = run(plsaSteps(options), undefined, r.steps)
      close(toFlat(s.topicWord), r.topicWord.flat(), 1e-12)
      close(toFlat(s.docTopic), r.docTopic.flat(), 1e-12)
      expect(s.logLikelihood).toBeCloseTo(r.logLikelihood, 9)
    })
  it('never lowers the log-likelihood', () => {
    const t = trace(plsaSteps({ ...options, init: undefined }), undefined, 40, { stream: stream(2) })
    for (let i = 1; i < t.steps.length; i++)
      expect(t.steps[i].logLikelihood).toBeGreaterThanOrEqual(t.steps[i - 1].logLikelihood - 1e-9)
  })
})

/** The seeded topic corpus as word ids, with each sentence's topic. */
function corpus(sentences = 600, perDocument = 4) {
  const c = topicCorpus(stream(3), { sentences })
  const vocab = new Map<string, number>()
  const stop = new Set(['the', 'a', 'some', 'in', 'to', 'near'])
  const documents = c.documents.map((doc) =>
    tokenise(doc)
      .tokens.map((w) => w.toLowerCase())
      .filter((w) => !stop.has(w))
      .map((w) => (vocab.has(w) ? vocab.get(w)! : (vocab.set(w, vocab.size), vocab.size - 1))),
  )
  const g = groupByLabel(documents, Array.from(c.labels ?? []), perDocument)
  return { documents: g.documents, vocabulary: vocab.size, labels: g.labels }
}

/** Purity: the share of documents whose dominant topic's majority label is their own. */
function purity(docTopic: Float64Array, K: number, labels: number[]) {
  const D = labels.length
  const dominant = Array.from({ length: D }, (_, d) => {
    let b = 0
    for (let k = 1; k < K; k++) if (docTopic[d * K + k] > docTopic[d * K + b]) b = k
    return b
  })
  let correct = 0
  for (let k = 0; k < K; k++) {
    const counts = new Map<number, number>()
    dominant.forEach((t, d) => t === k && counts.set(labels[d], (counts.get(labels[d]) ?? 0) + 1))
    correct += Math.max(0, ...counts.values())
  }
  return correct / D
}

describe('on the topic corpus', () => {
  const c = corpus()
  const last = (g: Generator<TopicSnapshot>) => {
    let s: TopicSnapshot | undefined
    for (const x of g) s = x
    return s!
  }
  const methods: [TopicMethod, number, number][] = [
    ['lda', 100, 0.95],
    ['hdp', 100, 0.9],
    ['plsa', 60, 0.7],
    ['nmf', 150, 0.9],
    ['ctm', 15, 0.9],
    ['labelled-lda', 40, 0.95],
    ['lsa', 0, 0.8],
  ]
  for (const [method, steps, minPurity] of methods)
    it(
      `${method} separates the corpus topics (purity ≥ ${minPurity}) with finite coherence`,
      { timeout: 60_000 },
      () => {
        const s = last(topicModelRun({ ...c, method, topics: 5, steps, seed: 1 }))
        expect(s.done).toBe(true)
        const cp = s.checkpoints[s.checkpoints.length - 1]
        expect(cp.top.length).toBe(s.topics)
        expect(Number.isFinite(s.history.coherence[s.history.coherence.length - 1])).toBe(true)
        if (method !== 'lsa') {
          const ll = s.history.logLikelihood
          expect(ll[ll.length - 1]).toBeGreaterThan(ll[0])
        }
        // Labelled LDA's last topic is the shared background; purity is over the label topics.
        const K = s.topics
        expect(purity(cp.docTopic, K, c.labels)).toBeGreaterThanOrEqual(minPurity)
      },
    )

  it('labelled LDA assigns each document only its label topic and the background', () => {
    const allowed = c.labels.map((l) => [l, 5])
    const s = run(
      ldaCollapsedGibbs({
        documents: c.documents,
        topics: 6,
        vocabulary: c.vocabulary,
        alpha: 0.5,
        beta: 0.05,
        allowed,
      }),
      undefined,
      5,
    )
    s.assignments.forEach((z, d) => toFlat(z).forEach((k) => expect(allowed[d]).toContain(k)))
  })

  it('NMF topics are distributions; LSA topics have one row per topic', () => {
    const n = nmfTopicModel({ documents: c.documents, vocabulary: c.vocabulary, topics: 4, steps: 50 })
    const tw = toFlat(n.topicWord)
    for (let k = 0; k < 4; k++)
      expect(tw.slice(k * c.vocabulary, (k + 1) * c.vocabulary).reduce((a, b) => a + b)).toBeCloseTo(1, 12)
    expect(lsaTopics(c.documents, c.vocabulary, 3).topicWord.shape).toEqual([3, c.vocabulary])
  })

  it('the CTM’s covariance stays symmetric positive definite', () => {
    const s = run(
      correlatedTopicSteps({ documents: c.documents.slice(0, 80), topics: 4, vocabulary: c.vocabulary }),
      undefined,
      3,
    )
    const S = toFlat(s.covariance)
    for (let a = 0; a < 4; a++) {
      expect(S[a * 5]).toBeGreaterThan(0)
      for (let b = 0; b < 4; b++) expect(S[a * 4 + b]).toBeCloseTo(S[b * 4 + a], 12)
    }
  })
})

describe('registry', () => {
  it('entries are well formed', () => {
    expectInfo(topicModelAlgorithms, 'algorithm')
    expectInfo(topicModelFunctions, 'function')
  })
})
