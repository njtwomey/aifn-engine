/**
 * The Porter stemmer as published (Porter 1980, "An algorithm for suffix stripping", Program 14(3)): five steps of
 * suffix rules `(condition) S1 → S2`, each step firing at most the rule with the longest matching suffix, with
 * conditions on the measure m of the remaining stem. Words of one or two letters are left alone, as in Porter's own
 * implementation. The later revisions (Porter2/Snowball, the "logi" and "bli" rules of the C release) are not applied.
 */

/** One rule that fired: the step (`1a` … `5b`), the rule as written in the paper, and the word before and after. */
export interface PorterStep {
  readonly step: string
  readonly rule: string
  readonly before: string
  readonly after: string
}

/** A stem with the rules that produced it. */
export interface PorterTrace {
  readonly word: string
  readonly stem: string
  readonly steps: readonly PorterStep[]
}

function isConsonant(w: string, i: number): boolean {
  const c = w[i]
  if ('aeiou'.includes(c)) return false
  // y is a consonant at the start of a word or after a vowel, and a vowel after a consonant ("toy" C V C, "syzygy").
  if (c === 'y') return i === 0 ? true : !isConsonant(w, i - 1)
  return true
}

/** The consonant/vowel pattern of a word in Porter's sense, e.g. "trouble" → "CCVVCCV". */
export function consonantVowelForm(word: string): string {
  let out = ''
  for (let i = 0; i < word.length; i++) out += isConsonant(word, i) ? 'C' : 'V'
  return out
}

/** Porter's measure m of a stem: the number of VC pairs in its form [C](VC)^m[V] ("tree" 0, "trouble" 1, "oaten" 2). */
export function porterMeasure(stem: string): number {
  const form = consonantVowelForm(stem).replace(/C+/g, 'C').replace(/V+/g, 'V')
  return (form.match(/VC/g) ?? []).length
}

const hasVowel = (s: string) => consonantVowelForm(s).includes('V')
const endsDoubleConsonant = (s: string) =>
  s.length >= 2 && s[s.length - 1] === s[s.length - 2] && isConsonant(s, s.length - 1)
/** *o: the stem ends consonant–vowel–consonant, and the last consonant is not w, x or y ("hop", "fil"). */
const endsCvc = (s: string) => {
  const n = s.length
  return n >= 3 && isConsonant(s, n - 3) && !isConsonant(s, n - 2) && isConsonant(s, n - 1) && !'wxy'.includes(s[n - 1])
}

type Rule = readonly [suffix: string, replacement: string]

const STEP2: readonly Rule[] = [
  ['ational', 'ate'],
  ['tional', 'tion'],
  ['enci', 'ence'],
  ['anci', 'ance'],
  ['izer', 'ize'],
  ['abli', 'able'],
  ['alli', 'al'],
  ['entli', 'ent'],
  ['eli', 'e'],
  ['ousli', 'ous'],
  ['ization', 'ize'],
  ['ation', 'ate'],
  ['ator', 'ate'],
  ['alism', 'al'],
  ['iveness', 'ive'],
  ['fulness', 'ful'],
  ['ousness', 'ous'],
  ['aliti', 'al'],
  ['iviti', 'ive'],
  ['biliti', 'ble'],
]
const STEP3: readonly Rule[] = [
  ['icate', 'ic'],
  ['ative', ''],
  ['alize', 'al'],
  ['iciti', 'ic'],
  ['ical', 'ic'],
  ['ful', ''],
  ['ness', ''],
]
const STEP4: readonly Rule[] = 'al ance ence er ic able ible ant ement ment ent ion ou ism ate iti ous ive ize'
  .split(' ')
  .map((s): Rule => [s, ''])

const upper = (s: string) => s.toUpperCase() || '∅'

/** Porter's stemmer with the rules that fired, step by step. */
export function porterStemTrace(word: string): PorterTrace {
  const original = word
  let w = word.toLowerCase()
  const steps: PorterStep[] = []
  if (w.length <= 2 || !/^[a-z]+$/.test(w)) return { word: original, stem: w, steps }
  const fire = (step: string, rule: string, after: string) => {
    steps.push({ step, rule, before: w, after })
    w = after
  }
  /** Steps 2–4: only the rule with the longest matching suffix is considered; if its condition fails, nothing. */
  const longest = (rules: readonly Rule[], step: string, minMeasure: number) => {
    let best: Rule | null = null
    for (const r of rules) if (w.endsWith(r[0]) && (!best || r[0].length > best[0].length)) best = r
    if (!best) return
    const [suffix, rep] = best
    const stem = w.slice(0, w.length - suffix.length)
    if (porterMeasure(stem) <= minMeasure) return
    if (suffix === 'ion' && !/[st]$/.test(stem)) return
    const condition = suffix === 'ion' ? `m>${minMeasure} and (*S or *T)` : `m>${minMeasure}`
    fire(step, `(${condition}) ${upper(suffix)} → ${upper(rep)}`, stem + rep)
  }

  // Step 1a: plurals. SS → SS changes nothing but stops the S rule.
  if (w.endsWith('sses')) fire('1a', 'SSES → SS', w.slice(0, -2))
  else if (w.endsWith('ies')) fire('1a', 'IES → I', w.slice(0, -2))
  else if (!w.endsWith('ss') && w.endsWith('s')) fire('1a', 'S → ∅', w.slice(0, -1))

  // Step 1b: -eed, -ed, -ing; after -ed or -ing the stem is repaired.
  let repair = false
  if (w.endsWith('eed')) {
    if (porterMeasure(w.slice(0, -3)) > 0) fire('1b', '(m>0) EED → EE', w.slice(0, -1))
  } else if (w.endsWith('ed') && hasVowel(w.slice(0, -2))) {
    fire('1b', '(*v*) ED → ∅', w.slice(0, -2))
    repair = true
  } else if (w.endsWith('ing') && hasVowel(w.slice(0, -3))) {
    fire('1b', '(*v*) ING → ∅', w.slice(0, -3))
    repair = true
  }
  if (repair) {
    if (w.endsWith('at')) fire('1b', 'AT → ATE', w + 'e')
    else if (w.endsWith('bl')) fire('1b', 'BL → BLE', w + 'e')
    else if (w.endsWith('iz')) fire('1b', 'IZ → IZE', w + 'e')
    else if (endsDoubleConsonant(w) && !'lsz'.includes(w[w.length - 1]))
      fire('1b', '(*d and not (*L or *S or *Z)) → single letter', w.slice(0, -1))
    else if (porterMeasure(w) === 1 && endsCvc(w)) fire('1b', '(m=1 and *o) → E', w + 'e')
  }

  // Step 1c: a final y after a vowel-bearing stem.
  if (w.endsWith('y') && hasVowel(w.slice(0, -1))) fire('1c', '(*v*) Y → I', w.slice(0, -1) + 'i')

  longest(STEP2, '2', 0)
  longest(STEP3, '3', 0)
  longest(STEP4, '4', 1)

  // Step 5a: a final e; step 5b: a final ll.
  if (w.endsWith('e')) {
    const stem = w.slice(0, -1)
    const m = porterMeasure(stem)
    if (m > 1) fire('5a', '(m>1) E → ∅', stem)
    else if (m === 1 && !endsCvc(stem)) fire('5a', '(m=1 and not *o) E → ∅', stem)
  }
  if (porterMeasure(w) > 1 && w.endsWith('ll')) fire('5b', '(m>1 and *d and *L) → single letter', w.slice(0, -1))
  return { word: original, stem: w, steps }
}

/** The Porter stem of one word (lower-cased; words that are not all letters a–z are returned lower-cased). */
export function porterStem(word: string): string {
  return porterStemTrace(word).stem
}
