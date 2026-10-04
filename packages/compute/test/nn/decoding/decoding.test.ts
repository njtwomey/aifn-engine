/**
 * aifn-compute/nn/decoding: the logit processors against transcriptions of Hugging Face's (`fixtures/nn/decoding.json`), and
 * the decoders by their laws on a toy language model (a fixed table of next-token logits by the last token): greedy
 * decoding picks argmaxes; sampling is reproducible and respects top-k; beam search with B = V^n finds the exhaustive
 * optimum and with B = 1 equals greedy; speculative decoding keeps the target's distribution (Monte Carlo) and its
 * acceptance rate matches Σ min(p, q).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  applyRepetitionPenalty,
  applyTopK,
  applyTopP,
  beamSearch,
  expectedTokensPerCall,
  greedyDecoding,
  nextTokenDistribution,
  samplingDecoding,
  softmaxOf,
  speculativeDecoding,
  type LogitsFn,
} from 'aifn-compute/nn/decoding'
import { fixture } from '../../fixtures'

type Case = {
  logits: number[]
  prefix: number[]
  k: number
  p: number
  temperature: number
  penalty: number
  topK: number[]
  topP: number[]
  repetition: number[]
  combined: number[]
  filtered: number[]
}
const F = fixture<{ cases: Case[] }>('nn/decoding')

const closeArr = (a: readonly number[], b: readonly number[], tol = 1e-12) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => (Number.isFinite(b[i]) ? expect(Math.abs(v - b[i])).toBeLessThan(tol) : expect(v).toBe(b[i])))
}

describe('logit processors', () => {
  it('softmaxOf handles a vocabulary larger than the spread-argument limit (review)', () => {
    const logits = Array.from({ length: 300_000 }, () => 0)
    logits[7] = Math.log(299_999)
    expect(softmaxOf(logits)[7]).toBeCloseTo(0.5, 10)
  })
  for (const [i, c] of F.cases.entries())
    it(`match Hugging Face’s warpers (case ${i})`, () => {
      closeArr(applyTopK(c.logits, c.k), c.topK)
      closeArr(applyTopP(c.logits, c.p), c.topP)
      closeArr(applyRepetitionPenalty(c.logits, c.prefix, c.penalty), c.repetition)
      const d = nextTokenDistribution(c.logits, c.prefix, {
        repetitionPenalty: c.penalty,
        temperature: c.temperature,
        topK: c.k,
        topP: c.p,
      })
      closeArr(d.filtered, c.filtered)
      expect(d.kept).toEqual(c.combined.map((v) => v !== -Infinity))
    })
})

// A toy language model over V = 4 tokens: the logits depend on the last token only (a bigram table).
const TABLE = [
  [2.0, 1.5, 0.2, -1.0],
  [0.1, 0.3, 2.2, 1.9],
  [1.2, -0.5, 0.4, 2.5],
  [0.0, 0.0, 0.0, 3.0],
]
const bigram: LogitsFn = (prefix) => tensor(TABLE[prefix.length ? prefix[prefix.length - 1] : 0])
const flatter: LogitsFn = (prefix) => tensor(TABLE[prefix.length ? prefix[prefix.length - 1] : 0].map((l) => 0.5 * l))

describe('decoders', () => {
  it('greedy decoding appends argmaxes and stops at a stop token', () => {
    const s = run(greedyDecoding(bigram, { prompt: [1], maxTokens: 10, stop: [3] }), undefined, 20)
    expect(s.tokens).toEqual([1, 2, 3])
    expect(s.terminated).toBe(true)
    const lp = Math.log(softmaxOf(TABLE[1])[2]) + Math.log(softmaxOf(TABLE[2])[3])
    expect(s.logProb).toBeCloseTo(lp, 12)
  })

  it('sampling is reproducible from the root stream and stays in the top-k support', () => {
    const alg = samplingDecoding(bigram, { prompt: [0], maxTokens: 30, topK: 2, temperature: 1.5 })
    const a = run(alg, undefined, 30, { stream: stream('lm') })
    const b = run(alg, undefined, 30, { stream: stream('lm') })
    expect(a.tokens).toEqual(b.tokens)
    const tr = trace(alg, undefined, 30, { stream: stream('lm'), keep: 'all' })
    for (const st of tr.steps.slice(1)) expect(st.kept[st.token]).toBe(true)
    // Top-2 keeps two tokens, or more when the second-largest logit is tied (row 3 of the table).
    expect(tr.steps.slice(1).every((st) => st.kept.filter(Boolean).length === (st.tokens.at(-2) === 3 ? 4 : 2))).toBe(
      true,
    )
  })

  it('beam search with one beam is greedy, and with V^n beams finds the most probable sequence', () => {
    const greedy = run(greedyDecoding(bigram, { prompt: [2], maxTokens: 3 }), undefined, 3)
    const one = run(beamSearch(bigram, { prompt: [2], maxTokens: 3, beams: 1 }), undefined, 3)
    expect(one.best.tokens).toEqual(greedy.tokens)
    // Exhaustive: every sequence of 3 tokens after the prompt.
    let best = { lp: -Infinity, seq: [] as number[] }
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 4; b++)
        for (let c = 0; c < 4; c++) {
          const seq = [2, a, b, c]
          let lp = 0
          for (let i = 1; i < 4; i++) lp += Math.log(softmaxOf(TABLE[seq[i - 1]])[seq[i]])
          if (lp > best.lp) best = { lp, seq }
        }
    const full = run(beamSearch(bigram, { prompt: [2], maxTokens: 3, beams: 64, expand: 4 }), undefined, 3)
    expect(full.best.tokens).toEqual(best.seq)
    expect(full.best.logProb).toBeCloseTo(best.lp, 12)
    expect(full.tree.filter((n) => n.step === 1)).toHaveLength(4)
  })

  it('a length penalty favours longer finished hypotheses', () => {
    const short = run(beamSearch(bigram, { prompt: [1], maxTokens: 6, beams: 3, stop: [3] }), undefined, 6)
    const long = run(
      beamSearch(bigram, { prompt: [1], maxTokens: 6, beams: 3, stop: [3], lengthPenalty: 2 }),
      undefined,
      6,
    )
    expect(long.best.tokens.length).toBeGreaterThanOrEqual(short.best.tokens.length)
  })

  it('speculative decoding keeps the target distribution, at the predicted acceptance rate', () => {
    // First generated token after the prompt [1], many seeds: its frequency must match the target's p.
    const p = softmaxOf(TABLE[1])
    const q = softmaxOf(TABLE[1].map((l) => 0.5 * l))
    const counts = [0, 0, 0, 0]
    const n = 4000
    let proposed = 0
    let accepted = 0
    for (let i = 0; i < n; i++) {
      const s = run(speculativeDecoding(bigram, flatter, { prompt: [1], maxTokens: 3, lookahead: 2 }), undefined, 1, {
        stream: stream(`spec-${i}`),
      })
      counts[s.tokens[1]]++
      proposed += s.round!.drafted.length > 0 ? 1 : 0
      accepted += s.round!.accepted > 0 ? 1 : 0
    }
    counts.forEach((c, j) => expect(Math.abs(c / n - p[j])).toBeLessThan(4 * Math.sqrt((p[j] * (1 - p[j])) / n)))
    // The first draft token is accepted with probability Σ_j min(p_j, q_j).
    const alpha = p.reduce((a, pj, j) => a + Math.min(pj, q[j]), 0)
    expect(Math.abs(accepted / proposed - alpha)).toBeLessThan(4 * Math.sqrt((alpha * (1 - alpha)) / n))
    expect(expectedTokensPerCall(0.5, 2)).toBeCloseTo(1.75, 14)
  })

  it('speculative decoding with the target as its own draft accepts everything', () => {
    const s = run(speculativeDecoding(bigram, bigram, { prompt: [0], maxTokens: 9, lookahead: 4 }), undefined, 10)
    expect(s.acceptedTotal).toBe(s.proposed)
    expect(s.generated).toBe(9)
    expect(s.targetCalls).toBe(2)
  })
})
