import { describe, expect, it } from 'vitest'
import { run } from 'aifn-compute/foundation/trace'
import {
  formatPattern,
  hyphenationPatterns,
  liangHyphenate,
  liangResult,
  liangSteps,
  markHyphens,
  parseHyphenated,
  parsePattern,
  patgenSteps,
  type HyphenatedWord,
} from 'aifn-compute/text/hyphenation'

// The patterns that match "hyphenation" in TeX's plain.tex patterns, as listed in Liang (1983, §2.3) and the TeXbook
// (appendix H): .hy-phen-ation.
const HYPHENATION = ['hy3ph', 'he2n', 'hena4', 'hen5at', '1na', 'n2at', '1tio', '2io', 'o2n']

describe('Liang patterns', () => {
  it('parses and writes TeX patterns', () => {
    expect(parsePattern('hen5at')).toEqual({ letters: 'henat', values: [0, 0, 0, 5, 0, 0] })
    expect(parsePattern('.hy3ph')).toEqual({ letters: '.hyph', values: [0, 0, 0, 3, 0, 0] })
    expect(parsePattern('1tio')).toEqual({ letters: 'tio', values: [1, 0, 0, 0] })
    expect(parsePattern('hena4').values).toEqual([0, 0, 0, 0, 4])
    for (const p of HYPHENATION) {
      const { letters, values } = parsePattern(p)
      expect(formatPattern(letters, values)).toBe(p)
    }
  })

  it('hyphenates "hyphenation" as hy-phen-ation (Liang 1983; the TeXbook)', () => {
    const h = liangHyphenate(hyphenationPatterns(HYPHENATION), 'hyphenation')
    // The classic layout: . h y3p h e2n5a4t2i o2n .
    expect(h.gaps).toEqual([0, 3, 0, 0, 2, 5, 4, 2, 0, 2])
    expect(h.hyphens).toEqual([1, 5])
    expect(markHyphens('hyphenation', h.hyphens)).toBe('hy-phen-ation')
    expect(h.matches.map((m) => m.pattern)).toEqual([
      'hy3ph',
      'he2n',
      'hena4',
      'hen5at',
      '1na',
      'n2at',
      '1tio',
      '2io',
      'o2n',
    ])
  })

  it('applies the largest digit per gap and respects the margins', () => {
    const set = hyphenationPatterns(['a1b', 'a4b', 'b3c', '1d'])
    // a4b merges with a1b; the larger digit wins.
    expect(set.patterns).toContain('a4b')
    expect(liangHyphenate(set, 'abcd', { leftMin: 1, rightMin: 1 }).hyphens).toEqual([1, 2])
    expect(liangHyphenate(set, 'abcd').hyphens).toEqual([])
  })

  it('step by step equals the one-pass result, one start position per step', () => {
    const set = hyphenationPatterns(HYPHENATION)
    const alg = liangSteps(set, 'hyphenation')
    let s = alg.init(undefined, undefined as never)
    const fired: string[][] = []
    while (!s.done) {
      s = alg.step(s, undefined as never)
      fired.push(s.fired.map((m) => m.pattern))
    }
    expect(s.t).toBe('.hyphenation.'.length)
    expect(fired[1]).toEqual(['hy3ph'])
    expect(fired[4]).toEqual(['he2n', 'hena4', 'hen5at'])
    expect(liangResult(s)).toEqual(liangHyphenate(set, 'hyphenation'))
  })

  it('reads hyphenated words', () => {
    expect(parseHyphenated('hy-phen-a-tion')).toEqual({ word: 'hyphenation', hyphens: [1, 5, 6] })
    expect(markHyphens('hyphenation', [1, 5, 6])).toBe('hy-phen-a-tion')
  })
})

describe('PATGEN', () => {
  const words: HyphenatedWord[] = [
    'hy-phen-a-tion',
    'com-put-er',
    'pat-tern',
    'let-ter',
    'bet-ter',
    'mat-ter',
    'sum-mer',
    'din-ner',
    'win-ter',
    'cen-ter',
    'num-ber',
    'tim-ber',
    'ac-tion',
    'mo-tion',
    'na-tion',
    'sta-tion',
    'ques-tion',
    'por-tion',
    'pen-cil',
    'gar-den',
  ].map((w) => parseHyphenated(w))

  it('improves training accuracy across levels, and the learned patterns reproduce its counts', () => {
    const alg = patgenSteps(words, {
      leftMin: 1,
      rightMin: 1,
      levels: [
        { goodWeight: 1, badWeight: 1, threshold: 2, lengths: [2, 3] },
        { goodWeight: 1, badWeight: 1, threshold: 1, lengths: [2, 4] },
        { goodWeight: 1, badWeight: 2, threshold: 1, lengths: [3, 5] },
      ],
    })
    const states = [alg.init(undefined, undefined as never)]
    while (!states.at(-1)!.done) states.push(alg.step(states.at(-1)!, undefined as never))
    const f = (c: { tp: number; fp: number; fn: number }) => (2 * c.tp) / (2 * c.tp + c.fp + c.fn)
    expect(states[0].counts.tp).toBe(0)
    // One step per pass: lengths 2–3, 2–4, 3–5.
    expect(states.length - 1).toBe(2 + 3 + 3)
    const byLevel = [0, 2, 5, 8].map((t) => f(states[t].counts))
    for (let k = 1; k < byLevel.length; k++) expect(byLevel[k]).toBeGreaterThanOrEqual(byLevel[k - 1])
    expect(byLevel.at(-1)!).toBeGreaterThan(0.9)
    const last = states.at(-1)!
    const found = words.reduce(
      (acc, w) => {
        const h = new Set(liangHyphenate(last.patterns, w.word, { leftMin: 1, rightMin: 1 }).hyphens)
        const t = new Set(w.hyphens)
        for (const i of h)
          if (t.has(i)) acc.tp++
          else acc.fp++
        for (const i of t) if (!h.has(i)) acc.fn++
        return acc
      },
      { tp: 0, fp: 0, fn: 0 },
    )
    expect(found).toEqual(last.counts)
    // Patterns of level 1 carry odd digits, of level 2 even ones.
    for (const p of last.passes[0].added) expect(Math.max(...parsePattern(p).values)).toBe(1)
    for (const p of last.passes[2].added) expect(Math.max(...parsePattern(p).values)).toBe(2)
  })

  it('stops at the pattern budget', () => {
    const final = run(patgenSteps(words, { leftMin: 1, rightMin: 1, maxPatterns: 5 }), undefined, 100)
    expect(final.patterns.patterns.length).toBeLessThanOrEqual(5)
    expect(final.done).toBe(true)
  })
})
