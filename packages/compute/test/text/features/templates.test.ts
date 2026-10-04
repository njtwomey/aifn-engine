import { describe, expect, it } from 'vitest'
import {
  encodeTemplateRows,
  expandTemplate,
  expandTemplates,
  featureIndex,
  parseTemplates,
  TemplateSyntaxError,
} from 'aifn-compute/text/features'

// The example of the CRF++ documentation ("Preparing feature templates"): the current token is "the".
const rows = [
  ['He', 'PRP', 'B-NP'],
  ['reckons', 'VBZ', 'B-VP'],
  ['the', 'DT', 'B-NP'],
  ['current', 'JJ', 'I-NP'],
  ['account', 'NN', 'I-NP'],
]

describe('CRF++ feature templates', () => {
  it('expands the documentation examples at the current token', () => {
    const t = parseTemplates(
      [
        '# Unigram',
        'U00:%x[0,0]',
        'U01:%x[0,1]',
        'U02:%x[-1,0]',
        'U03:%x[-2,1]',
        'U04:%x[0,0]/%x[0,1]',
        'UABC%x[0,1]123',
        '',
        'B',
      ].join('\n'),
    )
    expect(t.templates.map((x) => expandTemplate(x, rows, 2))).toEqual([
      'U00:the',
      'U01:DT',
      'U02:reckons',
      'U03:PRP',
      'U04:the/DT',
      'UABCDT123',
      'B',
    ])
    expect(t.templates.map((x) => x.kind)).toEqual([...Array(6).fill('unigram'), 'bigram'])
    expect(t.templates[4].id).toBe('U04')
    expect(t.templates[6].id).toBe('B')
    expect(t.columns).toBe(2)
    expect(t.reach).toBe(2)
    expect(t.templates[0].line).toBe(2)
  })

  it('reads _B-k before the start and _B+k after the end', () => {
    const t = parseTemplates('U:%x[-2,0]|%x[-1,0]|%x[1,0]|%x[2,0]')
    expect(expandTemplate(t.templates[0], rows, 0)).toBe('U:_B-2|_B-1|reckons|the')
    expect(expandTemplate(t.templates[0], rows, 4)).toBe('U:the|current|_B+1|_B+2')
  })

  it('expands bigram templates only where there is an edge (n ≥ 1)', () => {
    const e = expandTemplates(parseTemplates('U:%x[0,0]\nB\nB01:%x[0,1]'), rows)
    expect(e.unigram[0]).toEqual(['U:He'])
    expect(e.bigram[0]).toEqual([])
    expect(e.bigram[1]).toEqual(['B', 'B01:VBZ'])
  })

  it.each([
    ['X01:%x[0,0]', 1, 1, /unknown template type/],
    ['U01:%y[0,0]', 1, 5, /followed by x/],
    ['U01:%x(0,0)', 1, 7, /expected '\['/],
    ['U01:%x[a,0]', 1, 5, /malformed macro/],
    ['U01:%x[0,-1]', 1, 5, /malformed macro/],
    ['U01:%x[0,0', 1, 5, /unterminated/],
    ['U:%x[0,0]\n\nU:%x[9,0]', 3, 6, /beyond ±8/],
    ['# only a comment', 1, 1, /no templates/],
  ])('reports %j at its line and column', (src, line, column, message) => {
    let error: unknown
    try {
      parseTemplates(src)
    } catch (e) {
      error = e
    }
    expect(error).toBeInstanceOf(TemplateSyntaxError)
    const e = error as TemplateSyntaxError
    expect([e.line, e.column]).toEqual([line, column])
    expect(e.message).toMatch(message)
  })

  it('indexes strings with counts and drops rare ones (-f)', () => {
    const t = parseTemplates('U00:%x[0,1]\nB')
    const data = [rows, rows.slice(0, 2)]
    const all = featureIndex(t, data)
    expect(all.unigram).toEqual(['U00:PRP', 'U00:VBZ', 'U00:DT', 'U00:JJ', 'U00:NN'])
    expect(Array.from(all.unigramCounts)).toEqual([2, 2, 1, 1, 1])
    expect(all.bigram).toEqual(['B'])
    expect(Array.from(all.bigramCounts)).toEqual([5])
    const two = featureIndex(t, data, { minFrequency: 2 })
    expect(two.unigram).toEqual(['U00:PRP', 'U00:VBZ'])
    expect(two.dropped).toEqual({ unigram: 3, bigram: 0 })
    const enc = encodeTemplateRows(two, rows)
    expect(Array.from(enc.unigramStart)).toEqual([0, 1, 2, 2, 2, 2])
    expect(Array.from(enc.unigramIds)).toEqual([0, 1])
    expect(Array.from(enc.bigramStart)).toEqual([0, 0, 1, 2, 3, 4])
  })
})
