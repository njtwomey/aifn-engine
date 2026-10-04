import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { hyphenationDatasetRegistry, MOBY_WORDS, mobyHyphenation } from 'aifn-methods/data/real/hyphenation'
import { generate } from 'aifn-methods/data'
import { defaults } from 'aifn-compute/foundation/space'
import { expectInfo } from '../../registry'
import * as vendored from '../../../src/data/real/hyphenation/words'

describe('mobyHyphenation', () => {
  const data = mobyHyphenation(stream(1))

  it('records its provenance and public-domain licence', () => {
    expect(data.meta.source).toMatch(/Moby Hyphenator II \(public domain, 2001\)/)
    expect(data.meta.url).toBe('https://www.gutenberg.org/ebooks/3204')
    expect(data.meta.sha256).toBe('eeb30474c86b8af3469035ec1a0913e35905325ca885290db9dfed9881e230ac')
    expect(vendored.WORDS.length).toBeLessThan(150_000)
    // The vendored list is byte for byte what scripts/hyphenation.py wrote (guards against hand edits).
    expect(createHash('sha256').update(vendored.WORDS).digest('hex')).toBe(
      '22fcf971d019c2afe7be0e18408930bb451b97607ce16bc7ced576abaadd7304',
    )
  })

  it('holds the 8000 most frequent words, in rank order, with dictionary points', () => {
    expect(data.all.words.length).toBe(MOBY_WORDS)
    expect(data.all.words.slice(0, 3).map((w) => w.word)).toEqual(['that', 'with', 'this'])
    expect(data.all.words.every((w, i) => w.rank === i && /^[a-z]{4,15}$/.test(w.word))).toBe(true)
    expect(data.all.points).toBe(11401)
    expect(data.truth.hyphens('people')).toEqual([2])
    expect(data.truth.labels('table')).toEqual([0, 1, 0, 0, 0])
    expect(data.truth.hyphens('hyphenation')).toBeNull()
    for (const w of data.all.words) for (const i of w.hyphens) expect(i >= 0 && i < w.word.length - 1).toBe(true)
  })

  it('splits by word, deterministically from its stream', () => {
    const { train, test } = data
    expect(train.words.length + test.words.length).toBe(MOBY_WORDS)
    expect(test.words.length).toBe(1200)
    const held = new Set(test.words.map((w) => w.word))
    expect(train.words.some((w) => held.has(w.word))).toBe(false)
    const again = mobyHyphenation(stream(1))
    expect(again.test.words.map((w) => w.word)).toEqual(test.words.map((w) => w.word))
    const other = mobyHyphenation(stream(2))
    expect(other.test.words.map((w) => w.word)).not.toEqual(test.words.map((w) => w.word))
    const small = mobyHyphenation(stream(1), { words: 100, testFraction: 0.2 })
    expect([small.train.words.length, small.test.words.length]).toEqual([80, 20])
  })

  it('is registered as a split with truth', () => {
    expectInfo(hyphenationDatasetRegistry, 'dataset')
    const entry = hyphenationDatasetRegistry.mobyHyphenation
    expect(entry.info).toMatchObject({ output: 'split', truth: true, task: 'sequence' })
    const made = generate(entry, child(stream(3), 'x'), defaults(entry.info.knobs)) as ReturnType<
      typeof mobyHyphenation
    >
    expect(made.all.words.length).toBe(MOBY_WORDS)
  })
})
