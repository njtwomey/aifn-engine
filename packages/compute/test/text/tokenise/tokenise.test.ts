/** `aifn-compute/text/tokenise`: patterns, offsets, characters and detokenisation. */
import { describe, expect, it } from 'vitest'
import { characterTokenise, detokenise, tokenise, whitespaceTokenise } from 'aifn-compute/text/tokenise'
import { toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type U = { text: string; sklearn_tokens: string[] }
const F = fixture<{ unicode: { strings: U[] } }>('text/features').unicode.strings

describe('tokenise', () => {
  it('splits words with inner apostrophes and hyphens, with offsets that slice the source', () => {
    const t = tokenise("Don't stop: state-of-the-art, 42 times.")
    expect(t.tokens).toEqual(["Don't", 'stop', 'state-of-the-art', '42', 'times'])
    for (const [k, [s, e]] of toRows(t.offsets).entries()) expect(t.source.slice(s, e)).toBe(t.tokens[k])
    expect(t.offsets.dtype).toBe('int32')
    expect(t.offsets.shape).toEqual([5, 2])
  })
  it('keeps punctuation as tokens with wordsAndPunctuation', () => {
    expect(tokenise('Hello, world!', { pattern: 'wordsAndPunctuation' }).tokens).toEqual(['Hello', ',', 'world', '!'])
  })
  it.each(F.map((u) => [u.text, u] as const))("matches scikit-learn's token pattern on %s", (_, u) => {
    expect(tokenise(u.text, { pattern: 'alphanumeric' }).tokens).toEqual(u.sklearn_tokens)
  })
  it('splits like the GPT-2 pre-tokeniser and concatenates back exactly', () => {
    const text = "Hello world, it's 2024!  Bye"
    const t = tokenise(text, { pattern: 'gpt2' })
    expect(t.tokens).toEqual(['Hello', ' world', ',', ' it', "'s", ' 2024', '!', ' ', ' Bye'])
    expect(t.tokens.join('')).toBe(text)
  })
  it('accepts a regular expression without the g flag', () => {
    expect(tokenise('a1b22c333', { pattern: /\d+/ }).tokens).toEqual(['1', '22', '333'])
  })
  it('splits on white space', () => {
    expect(whitespaceTokenise(' a  b\tc\n').tokens).toEqual(['a', 'b', 'c'])
  })
})

describe('characterTokenise', () => {
  it('gives code points, with surrogate pairs whole', () => {
    const t = characterTokenise('a🙂b')
    expect(t.tokens).toEqual(['a', '🙂', 'b'])
    expect(toRows(t.offsets)).toEqual([
      [0, 1],
      [1, 3],
      [3, 4],
    ])
  })
  it('gives grapheme clusters', () => {
    const decomposed = 'été'
    expect(characterTokenise(decomposed).tokens).toHaveLength(5)
    expect(characterTokenise(decomposed, { unit: 'grapheme' }).tokens).toEqual(['é', 't', 'é'])
    expect(characterTokenise('a b', { skipWhitespace: true }).tokens).toEqual(['a', 'b'])
  })
})

describe('detokenise', () => {
  it('restores the source from offsets', () => {
    const t = tokenise('  Hello,   world!  ', { pattern: 'wordsAndPunctuation' })
    expect(detokenise(t)).toBe('Hello,   world!')
  })
  it('joins a token list by spacing rules', () => {
    expect(detokenise(['Hello', ',', 'world', '!'])).toBe('Hello, world!')
    expect(detokenise(['I', 'do', "n't", 'know', '(', 'yet', ')', '.'])).toBe("I don't know (yet).")
    expect(detokenise([])).toBe('')
  })
})
