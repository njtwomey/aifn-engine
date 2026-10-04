import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { parseTemplates } from 'aifn-compute/text/features'
import { mobyHyphenation } from 'aifn-methods/data/real/hyphenation'
import {
  crfHyphenate,
  crfHyphenationRun,
  HYPHENATION_TEMPLATES,
  hyphenationRows,
  hyphenationSequence,
} from 'aifn-methods/text/hyphenation'

describe('hyphenation with a template CRF', () => {
  it('makes letter rows and HYPH/O labels, and every preset parses', () => {
    expect(hyphenationRows('hyp')).toEqual([
      ['h', 'c'],
      ['y', 'v'],
      ['p', 'c'],
    ])
    expect(hyphenationSequence({ word: 'table', hyphens: [1] }).labels).toEqual(['O', 'HYPH', 'O', 'O', 'O'])
    for (const src of Object.values(HYPHENATION_TEMPLATES))
      expect(parseTemplates(src).templates.length).toBeGreaterThan(1)
  })

  it('learns to hyphenate held-out words from a few hundred', () => {
    const data = mobyHyphenation(stream('crf-test'), { words: 600 })
    const snaps = [...crfHyphenationRun(data, { maxSteps: 25, every: 25 })]
    const last = snaps.at(-1)!
    expect(snaps[0].step).toBe(0)
    expect(last.testScores!.precision).toBeGreaterThan(0.6)
    expect(last.testScores!.recall).toBeGreaterThan(0.5)
    const r = crfHyphenate(last.crf, 'information')
    expect(r.probabilities).toHaveLength(10)
    r.probabilities.forEach((p) => expect(p).toBeGreaterThanOrEqual(0))
  })
})
