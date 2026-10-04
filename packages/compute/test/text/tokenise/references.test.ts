/**
 * The Treebank, casual and sentence tokenisers against NLTK 3.9.1 (TreebankWordTokenizer, TweetTokenizer, the
 * pre-trained English Punkt model), and the GPT-2, cl100k and o200k patterns against the `regex` module on the
 * published patterns (fixture `text/tokenise`).
 */
import { describe, expect, it } from 'vitest'
import { toRows } from 'aifn-compute/foundation/tensor'
import { casualTokenise, sentenceSplit, tokenise, treebankTokenise, treebankTokens } from 'aifn-compute/text/tokenise'
import { fixture } from '../../fixtures'

type Fx = {
  treebank: { text: string; tokens: string[]; parens: string[]; spans: [number, number][] }[]
  casual: {
    text: string
    options: { preserve_case?: boolean; reduce_len?: boolean; strip_handles?: boolean; match_phone_numbers?: boolean }
    tokens: string[]
  }[]
  sentences: { text: string; sentences: string[] }[]
  patterns: ({ text: string } & Record<'gpt2' | 'cl100k' | 'o200k', string[]>)[]
}
const fx = fixture<Fx>('text/tokenise')

/** Code-point offsets (Python) to UTF-16 offsets. */
const utf16 = (text: string, cp: number) => [...text].slice(0, cp).join('').length

describe('treebankTokens', () => {
  it.each(fx.treebank.map((c) => [c.text, c] as const))('matches NLTK on %j', (_, c) => {
    expect(treebankTokens(c.text)).toEqual(c.tokens)
    expect(treebankTokens(c.text, { convertParentheses: true })).toEqual(c.parens)
  })
  it.each(fx.treebank.map((c) => [c.text, c] as const))('locates the tokens as span_tokenize on %j', (_, c) => {
    const t = treebankTokenise(c.text)
    expect(toRows(t.offsets)).toEqual(c.spans.map(([s, e]) => [utf16(c.text, s), utf16(c.text, e)]))
  })
})

describe('casualTokenise', () => {
  it.each(fx.casual.map((c) => [c.text, JSON.stringify(c.options), c] as const))(
    'matches NLTK on %j %s',
    (_, __, c) => {
      const t = casualTokenise(c.text, {
        preserveCase: c.options.preserve_case,
        reduceLength: c.options.reduce_len,
        stripHandles: c.options.strip_handles,
        phoneNumbers: c.options.match_phone_numbers,
      })
      expect(t.tokens).toEqual(c.tokens)
    },
  )
  it('points decoded entities and shortened runs at the text they came from', () => {
    const t = casualTokenise('a &amp; b!!!!!!')
    expect(t.tokens).toEqual(['a', '&', 'b', '!', '!', '!'])
    expect(toRows(t.offsets).slice(0, 2)).toEqual([
      [0, 1],
      [2, 7],
    ])
  })
})

// Where the rules knowingly differ from the pre-trained Punkt model: "Mt." is an abbreviation here, not to Punkt.
const DIVERGES: Record<string, string[]> = {
  'I saw Mt. Everest. It was tall. (It really was.) We went home.': [
    'I saw Mt. Everest.',
    'It was tall.',
    '(It really was.)',
    'We went home.',
  ],
}

describe('sentenceSplit', () => {
  it.each(fx.sentences.map((c) => [c.text, c] as const))('agrees with Punkt on %j', (_, c) => {
    expect(sentenceSplit(c.text).tokens).toEqual(DIVERGES[c.text] ?? c.sentences)
  })
  it('gives each sentence its offsets, white space between sentences excluded', () => {
    const text = '  One here.  Two there!\nThree.  '
    const t = sentenceSplit(text)
    expect(t.tokens).toEqual(['One here.', 'Two there!', 'Three.'])
    expect(toRows(t.offsets).map(([s, e]) => text.slice(s, e))).toEqual(t.tokens)
  })
})

describe('pre-tokeniser patterns', () => {
  for (const name of ['gpt2', 'cl100k', 'o200k'] as const)
    it.each(fx.patterns.map((c) => [c.text, c] as const))(`${name} matches the regex module on %j`, (_, c) => {
      const t = tokenise(c.text, { pattern: name })
      expect(t.tokens).toEqual(c[name])
      expect(t.tokens.join('')).toBe(c.text)
    })
})
