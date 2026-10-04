/**
 * Sentence splitting by the decision rules of Punkt (Kiss & Strunk 2006) with fixed English parameters: a "?" or "!"
 * ends a sentence; a word ending in a period ends one unless it is a known abbreviation, an initial or a number, and
 * those end one only when the next word is capitalised and usually starts sentences. The abbreviations and sentence
 * starters are those of NLTK's pre-trained English Punkt model; the orthographic evidence Punkt learns from a corpus
 * (which words appear in lower case) is replaced by a list of common function words.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import type { Tokenisation } from './tokenise'

/** Abbreviations of NLTK's English Punkt model (lower case, without the final period). */
export const ENGLISH_ABBREVIATIONS: readonly string[] = (
  'a.a a.c a.d a.g a.h a.m a.m.e a.s a.t adm ala ariz aug ave b.f b.v bros c c.i.t c.o.m.b c.v calif chg cie co col ' +
  'colo conn corp cos ct d d.c d.h d.w dec dr e e.f e.h e.l e.m f f.g f.j feb fla fri ft g g.d g.f g.k ga gen h h.c ' +
  'h.f h.m i.m.s ill inc j.b j.c j.j j.k j.p j.r jan jr k kan ky l l.a l.f l.p lt ltd m m.b.a m.d.c m.j maj messrs mg ' +
  'mich minn mr mrs ms n n.c n.d n.h n.j n.m n.v n.y nev nov oct ok okla ore p p.a.m p.m pa ph.d prof r r.a r.h r.i ' +
  'r.j r.k r.t rep reps s s.a s.a.y s.c s.g s.p.a s.s sen sep sept sr st sw t t.j tenn tues u.k u.n u.s u.s.a ' +
  'u.s.s.r v va vs vt w w.c w.r w.va w.w wash wed wis yr ' +
  // Common abbreviations Punkt's newswire training missed.
  'e.g i.e etc vol no fig eq approx dept est mt jun jul apr mar'
).split(' ')

/** Words that, capitalised after an abbreviation, start a new sentence (NLTK's English Punkt sentence starters). */
export const ENGLISH_SENTENCE_STARTERS: readonly string[] = (
  'according although among both but despite even he however i if in indeed instead it many meanwhile moreover most ' +
  'nevertheless nonetheless nor sales separately similarly since so some the there these they this though thus ' +
  'under when while yet ' +
  // Pronouns and determiners that occur in lower case mid-sentence, standing in for Punkt's learned orthographic
  // context (conjunctions and wh-words are left out: Punkt also reads them as continuing after an abbreviation).
  'a an as at by for from her his its my no not of on our she that their then to was we with you your all after ' +
  'before each every one two'
).split(' ')

/** Options of {@link sentenceSplit}. */
export interface SentenceOptions {
  /** Abbreviations, lower case without the final period (default {@link ENGLISH_ABBREVIATIONS}). */
  abbreviations?: readonly string[]
  /** Sentence starters, lower case (default {@link ENGLISH_SENTENCE_STARTERS}). */
  starters?: readonly string[]
}

const CLOSING = /["'”’)\]}»]+$/u
const OPENING = /^["'“‘([{«]+/u
const NUMBER = /^-?[.,]?\p{Nd}[\p{Nd},.-]*$/u
const INITIAL = /^\p{L}$/u
const ELLIPSIS = /\.\.+$/u

/**
 * Split text into sentences, each a token with its offsets (leading and trailing white space excluded). Candidates are
 * words (runs of non-space) ending in ".", "?" or "!", possibly followed by closing quotes or brackets, and followed by
 * white space.
 */
export function sentenceSplit(text: string, options: SentenceOptions = {}): Tokenisation {
  const abbreviations = new Set(options.abbreviations ?? ENGLISH_ABBREVIATIONS)
  const starters = new Set(options.starters ?? ENGLISH_SENTENCE_STARTERS)
  const words = [...text.matchAll(/\S+/gu)].map((m) => ({ w: m[0], s: m.index, e: m.index + m[0].length }))
  const ends: number[] = []
  for (let k = 0; k + 1 < words.length; k++) {
    const core = words[k].w.replace(CLOSING, '')
    const last = core.slice(-1)
    if (last !== '.' && last !== '?' && last !== '!') continue
    const next = words[k + 1].w.replace(OPENING, '')
    const nextType = next.toLowerCase().replace(/[^\p{L}\p{N}'-]+$/u, '')
    const capitalised = /^\p{Lu}/u.test(next)
    const startsSentence = capitalised && starters.has(nextType)
    let isBreak: boolean
    if (last !== '.') isBreak = true
    else if (ELLIPSIS.test(core)) isBreak = startsSentence
    else {
      const type = core.replace(/\.$/u, '').replace(OPENING, '').toLowerCase()
      if (NUMBER.test(type)) isBreak = !/^\p{Ll}/u.test(next)
      else if (abbreviations.has(type)) isBreak = startsSentence
      else if (INITIAL.test(type)) isBreak = startsSentence
      else isBreak = true
    }
    if (isBreak) ends.push(k)
  }
  const tokens: string[] = []
  const offsets: number[] = []
  let first = 0
  for (const k of [...ends, words.length - 1]) {
    if (words.length === 0 || first > k) break
    const s = words[first].s
    const e = words[k].e
    tokens.push(text.slice(s, e))
    offsets.push(s, e)
    first = k + 1
  }
  return { kind: 'tokens', source: text, tokens, offsets: fromData(Int32Array.from(offsets), [tokens.length, 2]) }
}
