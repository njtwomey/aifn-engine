/**
 * A toy part-of-speech tagging task for template CRFs: short English sentences tagged with a small universal tag set
 * (DET, NOUN, VERB, ADJ, ADP, PRON, ADV, CONJ, and `.` for punctuation), written for teaching. Several words take two
 * tags ("runs", "fish", "can", "like", "watch", "light"), so the tag of a word depends on its neighbours and the label
 * transitions matter. Each token is a row of three columns, as in CRF++'s input format: the word (lower case), its
 * last two letters, and its shape (`Xx` capitalised, `x` lower case, `.` punctuation).
 */

import type { LabelledSequence } from './template-crf'

/** The tag set of `toyPosCorpus`, in the order its CRF labels take. */
export const TOY_POS_TAGS = ['DET', 'NOUN', 'VERB', 'ADJ', 'ADP', 'PRON', 'ADV', 'CONJ', '.'] as const

// word/TAG tokens; the first word of each sentence is capitalised in the source.
const SENTENCES = [
  'The/DET dog/NOUN runs/VERB in/ADP the/DET park/NOUN ./.',
  'A/DET small/ADJ cat/NOUN sees/VERB the/DET bird/NOUN ./.',
  'She/PRON likes/VERB the/DET red/ADJ ball/NOUN ./.',
  'The/DET old/ADJ man/NOUN can/VERB fish/VERB in/ADP the/DET river/NOUN ./.',
  'He/PRON keeps/VERB fish/NOUN in/ADP a/DET glass/NOUN can/NOUN ./.',
  'The/DET children/NOUN watch/VERB the/DET light/NOUN slowly/ADV fade/VERB ./.',
  'My/PRON watch/NOUN runs/VERB fast/ADV ./.',
  'They/PRON like/VERB light/ADJ meals/NOUN and/CONJ long/ADJ walks/NOUN ./.',
  'A/DET cat/NOUN like/ADP that/DET runs/VERB quickly/ADV ./.',
  'The/DET morning/NOUN runs/NOUN were/VERB long/ADJ ./.',
  'We/PRON walk/VERB to/ADP the/DET old/ADJ bridge/NOUN ./.',
  'The/DET bird/NOUN sings/VERB and/CONJ the/DET dog/NOUN barks/VERB ./.',
  'He/PRON quietly/ADV reads/VERB a/DET long/ADJ book/NOUN ./.',
  'The/DET park/NOUN is/VERB green/ADJ and/CONJ quiet/ADJ ./.',
  'She/PRON can/VERB see/VERB the/DET boats/NOUN on/ADP the/DET lake/NOUN ./.',
  'They/PRON fish/VERB on/ADP the/DET lake/NOUN at/ADP night/NOUN ./.',
  'A/DET big/ADJ dog/NOUN chases/VERB the/DET small/ADJ cat/NOUN ./.',
  'The/DET light/NOUN in/ADP the/DET room/NOUN is/VERB warm/ADJ ./.',
  'We/PRON watch/VERB the/DET birds/NOUN from/ADP the/DET bridge/NOUN ./.',
  'He/PRON runs/VERB the/DET shop/NOUN with/ADP his/PRON sister/NOUN ./.',
  'The/DET quick/ADJ fox/NOUN jumps/VERB over/ADP the/DET lazy/ADJ dog/NOUN ./.',
  'Her/PRON sister/NOUN likes/VERB green/ADJ tea/NOUN ./.',
  'I/PRON can/VERB walk/VERB to/ADP the/DET park/NOUN ./.',
  'The/DET man/NOUN with/ADP the/DET watch/NOUN reads/VERB slowly/ADV ./.',
  'Birds/NOUN sing/VERB in/ADP the/DET morning/NOUN ./.',
  'The/DET cat/NOUN sleeps/VERB on/ADP a/DET warm/ADJ chair/NOUN ./.',
  'They/PRON paint/VERB the/DET old/ADJ boats/NOUN red/ADJ ./.',
  'A/DET fish/NOUN swims/VERB in/ADP the/DET cold/ADJ river/NOUN ./.',
  'She/PRON walks/VERB her/PRON dog/NOUN and/CONJ reads/VERB ./.',
  'The/DET light/ADJ rain/NOUN falls/VERB on/ADP the/DET roof/NOUN ./.',
  'We/PRON like/VERB the/DET quiet/ADJ lake/NOUN ./.',
  'He/PRON sees/VERB a/DET light/NOUN on/ADP the/DET hill/NOUN ./.',
  'The/DET boats/NOUN can/VERB sail/VERB at/ADP night/NOUN ./.',
  'My/PRON sister/NOUN runs/VERB every/DET morning/NOUN ./.',
  'The/DET dog/NOUN and/CONJ the/DET cat/NOUN sleep/VERB ./.',
  'A/DET long/ADJ walk/NOUN calms/VERB the/DET mind/NOUN ./.',
  'They/PRON watch/VERB fish/NOUN swim/VERB slowly/ADV ./.',
  'The/DET shop/NOUN sells/VERB tea/NOUN and/CONJ cans/NOUN ./.',
  'She/PRON quickly/ADV opens/VERB the/DET can/NOUN ./.',
  'His/PRON old/ADJ watch/NOUN keeps/VERB time/NOUN ./.',
  'The/DET children/NOUN walk/VERB to/ADP the/DET river/NOUN ./.',
  'I/PRON like/VERB a/DET walk/NOUN in/ADP the/DET rain/NOUN ./.',
  'The/DET fox/NOUN runs/VERB from/ADP the/DET dogs/NOUN ./.',
  'We/PRON can/VERB fish/VERB with/ADP a/DET long/ADJ rod/NOUN ./.',
  'A/DET bird/NOUN like/ADP a/DET crow/NOUN sings/VERB badly/ADV ./.',
  'The/DET room/NOUN has/VERB a/DET small/ADJ light/NOUN ./.',
]

/** The rows of a sentence's tokens: [word, last two letters, shape]. */
export function posRows(words: readonly string[]): string[][] {
  return words.map((w) => {
    const shape = /^[A-Z]/.test(w) ? 'Xx' : /^[a-z]/.test(w) ? 'x' : '.'
    const lower = w.toLowerCase()
    return [lower, lower.length > 2 ? lower.slice(-2) : lower, shape]
  })
}

/** The toy corpus as labelled sequences (see the module comment), in a fixed order. */
export function toyPosCorpus(): LabelledSequence[] {
  return SENTENCES.map((s) => {
    const tokens = s.split(' ').map((t) => {
      const cut = t.lastIndexOf('/')
      return { word: t.slice(0, cut), tag: t.slice(cut + 1) }
    })
    return { rows: posRows(tokens.map((t) => t.word)), labels: tokens.map((t) => t.tag) }
  })
}

/** CRF++ template sets for the toy tagger: from the word alone to a window with combined columns. */
export const TOY_POS_TEMPLATES: Readonly<Record<string, string>> = {
  minimal: ['# The current word, and label transitions.', 'U00:%x[0,0]', 'B'].join('\n'),
  'unigram window': [
    '# Words at offsets −2 … +2, each on its own.',
    'U00:%x[-2,0]',
    'U01:%x[-1,0]',
    'U02:%x[0,0]',
    'U03:%x[1,0]',
    'U04:%x[2,0]',
    '# The suffix and shape of the current word.',
    'U10:%x[0,1]',
    'U11:%x[0,2]',
    'B',
  ].join('\n'),
  bigram: [
    '# Word bigrams around the current word.',
    'U00:%x[0,0]',
    'U01:%x[-1,0]/%x[0,0]',
    'U02:%x[0,0]/%x[1,0]',
    '# Transitions, and transitions conjoined with the current word.',
    'B',
    'B01:%x[0,0]',
  ].join('\n'),
  combined: [
    '# The CRF++ documentation style: single cells and combinations.',
    'U00:%x[-1,0]',
    'U01:%x[0,0]',
    'U02:%x[1,0]',
    'U03:%x[-1,0]/%x[0,0]',
    'U04:%x[0,0]/%x[1,0]',
    'U10:%x[0,1]',
    'U11:%x[-1,1]/%x[0,1]',
    'U12:%x[0,2]',
    'B',
  ].join('\n'),
}
