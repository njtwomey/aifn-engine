/**
 * The HDP and the dynamic topic model, by laws and planted structure: the HDP sampler keeps its count tables
 * consistent and, started from ten topics on a corpus planted with four, ends with about four, each concentrated on one
 * planted block; DTM-lite's objective (bound plus topic prior) never falls, and on the drifting corpus each fitted
 * topic's word distribution follows the planted theme through every slice.
 */
import { describe, expect, it } from 'vitest'
import { categorical, child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { dirichlet } from 'aifn-compute/probability/samplers'
import { driftingTopicCorpus } from 'aifn-methods/text/corpora'
import {
  dynamicTopicRun,
  dynamicTopicSteps,
  hdpEstimates,
  hdpGibbs,
  type DynamicTopicSnapshot,
  type HdpState,
} from 'aifn-methods/inference/topic-models'
import { expectProtocol } from '../../protocol'

/** D documents over B disjoint blocks of `width` words; each document mostly one block (θ ~ Dir(0.1)). */
function blockCorpus(B = 4, width = 10, D = 60, length = 50) {
  const s = stream('hdp-planted')
  return Array.from({ length: D }, (_, d) => {
    const theta = toFlat(dirichlet(child(s, 'theta', d), new Array<number>(B).fill(0.1)))
    return Array.from({ length }, (_, n) => {
      const b = categorical(child(s, 'z', d, n), theta)
      return b * width + categorical(child(s, 'w', d, n), new Array<number>(width).fill(1))
    })
  })
}

describe('HDP (direct-assignment Gibbs)', () => {
  const documents = blockCorpus()
  const V = 40
  it('keeps its tables consistent and finds about the planted number of topics', { timeout: 60_000 }, () => {
    const alg = hdpGibbs({ documents, vocabulary: V, alpha: 1, gamma: 1, eta: 0.05, initialTopics: 10 })
    const s: HdpState = run(alg, undefined, 80, { stream: stream(4) })
    const K = s.topics
    // Counts: Σ_k n_k = tokens, each document's row sums to its length, no empty topic, weights a distribution.
    const nk = toFlat(s.topicTotals)
    expect(nk.reduce((a, b) => a + b, 0)).toBe(documents.reduce((a, d) => a + d.length, 0))
    expect(Math.min(...nk)).toBeGreaterThan(0)
    const ndk = toFlat(s.docTopic)
    documents.forEach((doc, d) => expect(ndk.slice(d * K, (d + 1) * K).reduce((a, b) => a + b, 0)).toBe(doc.length))
    const w = toFlat(s.weights)
    expect(w.length).toBe(K + 1)
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    // The planted four, perhaps with a small extra topic or two.
    expect(K).toBeGreaterThanOrEqual(4)
    expect(K).toBeLessThanOrEqual(7)
    // The four largest topics each put at least 90% of their tokens in one block, and the blocks differ.
    const nkw = toFlat(s.topicWord)
    const largest = Array.from({ length: K }, (_, k) => k)
      .sort((a, b) => nk[b] - nk[a])
      .slice(0, 4)
    const blocks = largest.map((k) => {
      const mass = [0, 1, 2, 3].map((b) => nkw.slice(k * V + 10 * b, k * V + 10 * b + 10).reduce((a, c) => a + c, 0))
      const top = Math.max(...mass)
      expect(top / nk[k]).toBeGreaterThan(0.9)
      return mass.indexOf(top)
    })
    expect(new Set(blocks).size).toBe(4)
    const est = hdpEstimates(s, { vocabulary: V, alpha: 1, eta: 0.05 })
    expect(est.topicWord.shape).toEqual([K, V])
    toFlat(est.docTopic)
      .reduce<number[]>((rows, v, i) => ((rows[Math.floor(i / K)] = (rows[Math.floor(i / K)] ?? 0) + v), rows), [])
      .forEach((r) => expect(r).toBeCloseTo(1, 10))
  })
  it('follows the trace protocol', () => {
    expectProtocol(hdpGibbs({ documents: documents.slice(0, 6), vocabulary: V, initialTopics: 3 }), undefined, { n: 4 })
  })
})

describe('dynamic topic model (MAP variational EM)', () => {
  const c = driftingTopicCorpus(stream('dtm-test'), { slices: 5, documentsPerSlice: 30, length: 40 })
  const vocabulary = c.topics!.vocabulary
  const index = new Map(vocabulary.map((w, i) => [w, i]))
  const documents = c.documents.map((d) => d.split(' ').map((w) => index.get(w)!))
  const V = vocabulary.length
  const options = { documents, times: c.times!, topics: 4, vocabulary: V }

  it('never lowers its objective', { timeout: 60_000 }, () => {
    const tr = trace(dynamicTopicSteps(options), undefined, 12, {
      stream: stream(2),
      record: { objective: (s) => s.objective },
    })
    const o = toFlat(tr.series.objective)
    for (let i = 1; i < o.length; i++) expect(o[i]).toBeGreaterThanOrEqual(o[i - 1] - 1e-6 * Math.abs(o[i - 1]))
    expect(o[o.length - 1]).toBeGreaterThan(o[0])
  })

  it('recovers the drifting themes slice by slice', { timeout: 60_000 }, () => {
    let s: DynamicTopicSnapshot | undefined
    for (const x of dynamicTopicRun({ ...options, steps: 40, seed: 3 })) s = x
    const { T, K } = s!.shape
    const phi = s!.checkpoints.at(-1)!.topicWord
    const truth = c.topics!.topicWord
    // Match each fitted topic to the planted theme it puts most mass on (pooled over slices).
    const match = Array.from({ length: K }, (_, k) => {
      const mass = [0, 1, 2, 3].map((b) => {
        let m = 0
        for (let t = 0; t < T; t++) for (let i = 0; i < 12; i++) m += phi[(t * K + k) * V + 12 * b + i]
        return m
      })
      return mass.indexOf(Math.max(...mass))
    })
    expect(new Set(match).size).toBe(4)
    // The fitted topics follow their themes through time: mean total variation from the planted distribution over
    // slices is small, and well below that of the static LDA topics the run starts from (the same in every slice).
    const tv = (p: Float64Array) => {
      let total = 0
      for (let k = 0; k < K; k++)
        for (let t = 0; t < T; t++)
          for (let w = 0; w < V; w++) total += Math.abs(p[(t * K + k) * V + w] - truth[t][match[k]][w]) / 2
      return total / (K * T)
    }
    const fitted = tv(phi)
    const staticTopics = tv(s!.checkpoints[0].topicWord)
    expect(fitted).toBeLessThan(0.25)
    expect(fitted).toBeLessThan(staticTopics - 0.15)
    let shift = 0
    for (let w = 0; w < V; w++) shift += Math.abs(truth[0][0][w] - truth[T - 1][0][w]) / 2
    expect(shift).toBeGreaterThan(0.8)
  })
})
