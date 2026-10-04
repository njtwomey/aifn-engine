/** `aifn-compute/text/normalise` against Python's `unicodedata` and `str.casefold` (fixture `text/features`). */
import { describe, expect, it } from 'vitest'
import { caseFold, collapseWhitespace, normalise, stripAccents } from 'aifn-compute/text/normalise'
import { fixture } from '../../fixtures'

type U = { text: string; nfkc: string; casefold: string; nfkc_casefold: string; strip_accents: string }
const F = fixture<{ unicode: { strings: U[] } }>('text/features').unicode.strings

describe('normalise', () => {
  it.each(F.map((u) => [u.text, u] as const))('matches Python on %s', (_, u) => {
    expect(caseFold(u.text)).toBe(u.casefold)
    expect(stripAccents(u.text)).toBe(u.strip_accents)
    expect(normalise(u.text, { caseFold: false, whitespace: false })).toBe(u.nfkc)
    expect(normalise(u.text, { whitespace: false })).toBe(u.nfkc_casefold)
  })
  it('folds the characters whose folding is not their lower case', () => {
    expect(caseFold('Straße')).toBe('strasse')
    expect(caseFold('ΣΊΣΥΦΟΣ')).toBe('σίσυφοσ') // no final sigma: folding is context-free
    expect(caseFold('ﬁ')).toBe('fi')
    expect(caseFold('ꭰ')).toBe('Ꭰ') // Cherokee folds to upper case
    expect(caseFold('MASSE') === caseFold('Maße')).toBe(true)
  })
  it('collapses and trims white space', () => {
    expect(collapseWhitespace('  a \t\n b  ')).toBe('a b')
    expect(normalise('  Ｆｕｌｌ  WIDTH ')).toBe('full width')
  })
  it('strips accents only on request', () => {
    expect(normalise('Café', { stripAccents: true })).toBe('cafe')
    expect(normalise('Café')).toBe('café')
    expect(normalise('ǰ', { form: 'NFC' }).normalize('NFC')).toBe(normalise('ǰ', { form: 'NFC' }))
  })
})
