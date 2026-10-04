import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { mobyHyphenation } from 'aifn-methods/data/real/hyphenation'
import {
  BiRnnTagger,
  gapLabels,
  hyphenScores,
  letterWindow,
  liangLearningRun,
  rnnProbabilities,
  rnnSaliency,
  rowExamples,
  taggerTrainingRun,
  thresholdScores,
  windowExamples,
  windowSaliency,
  WindowTagger,
  windowProbabilities,
} from 'aifn-methods/text/hyphenation'
import { parseHyphenated } from 'aifn-compute/text/hyphenation'

const words = ['hy-phen-a-tion', 'ta-ble', 'peo-ple', 'win-ter'].map((w) => parseHyphenated(w))

describe('hyphen scores', () => {
  it('count hits, false alarms and misses over every gap', () => {
    // hy-phen-a-tion: predicted after 1 (hit) and 2 (wrong); 5 and 6 missed.
    const s = hyphenScores([words[0]], [[1, 2]])
    expect([s.tp, s.fp, s.fn, s.tn]).toEqual([1, 1, 2, 6])
    expect(s.precision).toBeCloseTo(0.5)
    expect(s.recall).toBeCloseTo(1 / 3)
    expect(s.fHalf).toBeCloseTo((1.25 * 0.5 * (1 / 3)) / (0.25 * 0.5 + 1 / 3))
    expect(gapLabels([words[1]])).toEqual([0, 1, 0, 0])
    const t = thresholdScores([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.6], 0.5)
    expect([t.tp, t.fp, t.fn]).toEqual([1, 1, 1])
  })
})

describe('tagger inputs', () => {
  it('windows and rows put a letter and its labels in place', () => {
    expect(letterWindow('table', 0, 3).slice(0, 4)).toEqual([0, 0, 1, 2 + 19])
    const w = windowExamples([words[1]], 3)
    expect(w.x.shape).toEqual([4, 7])
    expect(Array.from(w.y.data)).toEqual([0, 1, 0, 0])
    const r = rowExamples([words[1]], 8)
    expect(Array.from(r.w.data)).toEqual([0, 1, 1, 1, 1, 0, 0, 0])
    expect(Array.from(r.y.data)).toEqual([0, 0, 1, 0, 0, 0, 0, 0])
  })

  it('models give a probability per gap and an occlusion map', () => {
    const win = WindowTagger()
    const wp = win.init(stream('w'))
    const rnn = BiRnnTagger({ width: 12 })
    const rp = rnn.init(stream('r'))
    const [a, b] = windowProbabilities(win, wp, ['table', 'hyphen'])
    expect([a.length, b.length]).toEqual([4, 5])
    expect(rnnProbabilities(rnn, rp, ['table', 'hyphen']).map((p) => p.length)).toEqual([4, 5])
    expect(windowSaliency(win, wp, 'table').map((r) => r.length)).toEqual([7, 7, 7, 7])
    expect(rnnSaliency(rnn, rp, 'table').map((r) => r.length)).toEqual([5, 5, 5, 5])
  })
})

describe('runs on the Moby words', () => {
  const data = mobyHyphenation(stream(0), { words: 1500 })

  it('PATGEN raises held-out F₀.₅ and training precision over the levels', () => {
    const shots = [...liangLearningRun(data)]
    const last = shots.at(-1)!
    expect(last.done).toBe(true)
    expect(last.history.length).toBe(last.step + 1)
    const f = (c: { tp: number; fp: number; fn: number }) => (1.25 * c.tp) / (1.25 * c.tp + 0.25 * c.fn + c.fp)
    expect(f(last.test)).toBeGreaterThan(0.6)
    expect(last.train.tp / (last.train.tp + last.train.fp)).toBeGreaterThan(0.97)
  }, 60_000)

  it('the taggers learn: test F₁ rises from step 0', () => {
    const shots = [...taggerTrainingRun(data, { steps: 60, every: 30 })]
    const final = shots.at(-1)!
    expect(final.checkpoints.map((c) => c.step)).toEqual([0, 30, 60])
    const labels = gapLabels(data.test.words)
    const [first, end] = [final.checkpoints[0], final.checkpoints.at(-1)!]
    expect(end.windowTest.length).toBe(labels.length)
    expect(thresholdScores(labels, end.windowTest, 0.5).f1).toBeGreaterThan(
      thresholdScores(labels, first.windowTest, 0.5).f1,
    )
    expect(final.rnnLosses.at(-1)!).toBeLessThan(final.rnnLosses[0])
  }, 120_000)
})
