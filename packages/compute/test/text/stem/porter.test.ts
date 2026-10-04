/**
 * The Porter stemmer against the published algorithm: every example of Porter (1980), checked in the step it
 * illustrates, the measure table, and whole-word stems including the over- and under-stemming pairs of the stemming note.
 */
import { describe, expect, it } from 'vitest'
import {
  consonantVowelForm,
  porterMeasure,
  porterStem,
  porterStemTrace,
  removeStopWords,
  STOP_WORDS,
} from 'aifn-compute/text/stem'

/** Porter (1980), the examples beside each rule: [step, word, the word after that step]. */
const PAPER: [string, string, string][] = [
  ['1a', 'caresses', 'caress'],
  ['1a', 'ponies', 'poni'],
  ['1a', 'ties', 'ti'],
  ['1a', 'caress', 'caress'],
  ['1a', 'cats', 'cat'],
  ['1b', 'feed', 'feed'],
  ['1b', 'agreed', 'agree'],
  ['1b', 'plastered', 'plaster'],
  ['1b', 'bled', 'bled'],
  ['1b', 'motoring', 'motor'],
  ['1b', 'sing', 'sing'],
  ['1b', 'conflated', 'conflate'],
  ['1b', 'troubled', 'trouble'],
  ['1b', 'sized', 'size'],
  ['1b', 'hopping', 'hop'],
  ['1b', 'tanned', 'tan'],
  ['1b', 'falling', 'fall'],
  ['1b', 'hissing', 'hiss'],
  ['1b', 'fizzed', 'fizz'],
  ['1b', 'failing', 'fail'],
  ['1b', 'filing', 'file'],
  ['1c', 'happy', 'happi'],
  ['1c', 'sky', 'sky'],
  ['2', 'relational', 'relate'],
  ['2', 'conditional', 'condition'],
  ['2', 'rational', 'rational'],
  ['2', 'valenci', 'valence'],
  ['2', 'hesitanci', 'hesitance'],
  ['2', 'digitizer', 'digitize'],
  ['2', 'conformabli', 'conformable'],
  ['2', 'radicalli', 'radical'],
  ['2', 'differentli', 'different'],
  ['2', 'vileli', 'vile'],
  ['2', 'analogousli', 'analogous'],
  ['2', 'vietnamization', 'vietnamize'],
  ['2', 'predication', 'predicate'],
  ['2', 'operator', 'operate'],
  ['2', 'feudalism', 'feudal'],
  ['2', 'decisiveness', 'decisive'],
  ['2', 'hopefulness', 'hopeful'],
  ['2', 'callousness', 'callous'],
  ['2', 'formaliti', 'formal'],
  ['2', 'sensitiviti', 'sensitive'],
  ['2', 'sensibiliti', 'sensible'],
  ['3', 'triplicate', 'triplic'],
  ['3', 'formative', 'form'],
  ['3', 'formalize', 'formal'],
  ['3', 'electriciti', 'electric'],
  ['3', 'electrical', 'electric'],
  ['3', 'hopeful', 'hope'],
  ['3', 'goodness', 'good'],
  ['4', 'revival', 'reviv'],
  ['4', 'allowance', 'allow'],
  ['4', 'inference', 'infer'],
  ['4', 'airliner', 'airlin'],
  ['4', 'gyroscopic', 'gyroscop'],
  ['4', 'adjustable', 'adjust'],
  ['4', 'defensible', 'defens'],
  ['4', 'irritant', 'irrit'],
  ['4', 'replacement', 'replac'],
  ['4', 'adjustment', 'adjust'],
  ['4', 'dependent', 'depend'],
  ['4', 'adoption', 'adopt'],
  ['4', 'homologou', 'homolog'],
  ['4', 'communism', 'commun'],
  ['4', 'activate', 'activ'],
  ['4', 'angulariti', 'angular'],
  ['4', 'effective', 'effect'],
  ['4', 'bowdlerize', 'bowdler'],
  ['5a', 'probate', 'probat'],
  ['5a', 'rate', 'rate'],
  ['5a', 'cease', 'ceas'],
  ['5b', 'controll', 'control'],
  ['5b', 'roll', 'roll'],
]

describe('porterStemTrace', () => {
  it.each(PAPER)('step %s: %s → %s', (step, word, after) => {
    const { steps } = porterStemTrace(word)
    const fired = steps.filter((s) => s.step === step)
    const before = steps.filter((s) => s.step < step)
    expect(before).toEqual([]) // the example reaches its step unchanged
    if (after === word) expect(fired).toEqual([])
    else {
      expect(fired[0].before).toBe(word)
      expect(fired[fired.length - 1].after).toBe(after)
    }
  })
  it('names the rules as written in the paper', () => {
    expect(porterStemTrace('generalizations').steps.map((s) => `${s.step} ${s.rule}`)).toEqual([
      '1a S → ∅',
      '2 (m>0) IZATION → IZE',
      '3 (m>0) ALIZE → AL',
      '4 (m>1) AL → ∅',
    ])
  })
})

describe('porterMeasure', () => {
  it.each([
    [0, ['tr', 'ee', 'tree', 'y', 'by']],
    [1, ['trouble', 'oats', 'trees', 'ivy']],
    [2, ['troubles', 'private', 'oaten', 'orrery']],
  ] as const)('m = %i', (m, words) => {
    for (const w of words) expect(porterMeasure(w)).toBe(m)
  })
  it('classifies y by its neighbour', () => {
    expect(consonantVowelForm('trouble')).toBe('CCVVCCV')
    expect(consonantVowelForm('syzygy')).toBe('CVCVCV')
    expect(consonantVowelForm('toy')).toBe('CVC')
  })
})

describe('porterStem', () => {
  it.each([
    ['generalizations', 'gener'],
    ['oscillators', 'oscil'],
    ['filing', 'file'],
    ['argument', 'argument'],
    // The paper lists it under step 4's OUS rule; in the full algorithm step 1a takes the s first.
    ['homologous', 'homolog'],
    ['relational', 'relat'],
    ['universe', 'univers'],
    ['university', 'univers'],
    ['universal', 'univers'],
    ['general', 'gener'],
    ['generous', 'gener'],
    ['generate', 'gener'],
    ['news', 'new'],
    ['organization', 'organ'],
    ['absorption', 'absorpt'],
    ['alumnus', 'alumnu'],
    ['alumni', 'alumni'],
    ['Europe', 'europ'],
    ['European', 'european'],
    ['was', 'wa'],
    ['is', 'is'],
    ['naïve', 'naïve'],
  ])('%s → %s', (word, stem) => expect(porterStem(word)).toBe(stem))
})

describe('stop words', () => {
  it('keeps the published lists', () => {
    expect(STOP_WORDS.nltk).toHaveLength(198)
    expect(STOP_WORDS.scikitLearn).toHaveLength(318)
    expect(new Set(STOP_WORDS.scikitLearn).size).toBe(318)
  })
  it('removes stop words case-insensitively', () => {
    expect(removeStopWords(['The', 'cat', 'is', 'on', 'THE', 'mat'])).toEqual(['cat', 'mat'])
    expect(removeStopWords(['call', 'me', 'bill'], 'scikitLearn')).toEqual([])
    expect(removeStopWords(['a', 'b'], ['b'])).toEqual(['a'])
  })
})
