/**
 * Toy text corpora for the text pipeline: the small corpora the notes work by hand (the five "cat sat on the mat"
 * documents, the thirteen pet sentences, Sennrich's "low lower newest widest", the Hugging Face WordPiece corpus), and a
 * seeded generator of short sentences about a few topics, with English inflection (plurals, -ed, -ing with doubled
 * consonants) so that stemming and subword merges have something to find.
 */

import type { DatasetInfo, DatasetMeta } from 'aifn-compute/foundation/contracts'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { gammaVariate } from 'aifn-compute/probability/samplers'
import { definer, type Entry } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A corpus: documents of raw text (one sentence or paragraph each) and what they are. */
export interface Corpus {
  readonly kind: 'corpus'
  readonly documents: readonly string[]
  readonly meta: DatasetMeta
  /** The true topic of each document, an index into `meta.labelNames` (generated corpora with topics only). */
  readonly labels?: readonly number[]
  /** The true topic of each word the generator can emit, by name (`function` for determiners and prepositions). */
  readonly wordTopics?: Readonly<Record<string, string>>
  /** The time slice of each document (corpora that change over time only). */
  readonly times?: readonly number[]
  /**
   * The generating topics, where they are known: the vocabulary and each slice's word distribution per topic,
   * `topicWord[t][k][w]` over `vocabulary`.
   */
  readonly topics?: {
    readonly vocabulary: readonly string[]
    readonly topicWord: readonly (readonly (readonly number[])[])[]
  }
}

const repeat = (word: string, n: number) => Array.from({ length: n }, () => word)

/** The named corpora of {@link namedCorpus}. */
export const NAMED_CORPORA = {
  'cat-sat-on-the-mat': {
    description: 'Five short documents about cats and dogs, the worked example of the TF-IDF and BM25 notes.',
    documents: [
      'the cat sat on the mat',
      'the dog sat on the log',
      'the cat chased the dog',
      'a dog and a cat played',
      'the bird sang',
    ],
  },
  pets: {
    description: 'Thirteen sentences about pets, the worked example of the co-occurrence and PMI note.',
    documents: [
      'the cat is a pet',
      'the dog is a pet',
      'we fed the cat',
      'we fed the dog',
      'the vet saw the cat',
      'the vet saw the dog',
      'the cat chased the mouse',
      'the dog chased the cat',
      'the mouse ate the cheese',
      'the mouse ate the grain',
      'we ate the cheese',
      'the cat slept',
      'the dog slept',
    ],
  },
  'low-lower-newest-widest': {
    description: 'Sennrich et al. (2016)’s BPE corpus: low ×5, lower ×2, newest ×6, widest ×3.',
    documents: [[...repeat('low', 5), ...repeat('lower', 2), ...repeat('newest', 6), ...repeat('widest', 3)].join(' ')],
  },
  'hug-pug-pun': {
    description: 'The Hugging Face course corpus for WordPiece: hug ×10, pug ×5, pun ×12, bun ×4, hugs ×5.',
    documents: [
      [...repeat('hug', 10), ...repeat('pug', 5), ...repeat('pun', 12), ...repeat('bun', 4), ...repeat('hugs', 5)].join(
        ' ',
      ),
    ],
  },
} as const

/** A named corpus. */
export type CorpusName = keyof typeof NAMED_CORPORA

/** One of the hand-worked corpora of the notes. */
export function namedCorpus(knobs: { name?: CorpusName } = {}): Corpus {
  const name = knobs.name ?? 'cat-sat-on-the-mat'
  const c = NAMED_CORPORA[name]
  if (!c) throw new DomainError('namedCorpus', `namedCorpus: unknown corpus '${String(name)}'`)
  return { kind: 'corpus', documents: [...c.documents], meta: { name, description: c.description, task: 'text' } }
}

// ── A seeded sentence generator ──────────────────────────────────────────────────────────────────────────────────────

type Noun = readonly [singular: string, plural: string]
/** A verb: base form, third person singular, past, -ing form. */
type Verb = readonly [string, string, string, string]

interface Topic {
  readonly nouns: readonly Noun[]
  readonly verbs: readonly Verb[]
  readonly adjectives: readonly string[]
  readonly places: readonly string[]
}

const TOPICS: Readonly<Record<string, Topic>> = {
  pets: {
    nouns: [
      ['cat', 'cats'],
      ['dog', 'dogs'],
      ['mouse', 'mice'],
      ['bird', 'birds'],
      ['puppy', 'puppies'],
      ['fox', 'foxes'],
    ],
    verbs: [
      ['chase', 'chases', 'chased', 'chasing'],
      ['watch', 'watches', 'watched', 'watching'],
      ['hop', 'hops', 'hopped', 'hopping'],
      ['sleep', 'sleeps', 'slept', 'sleeping'],
      ['play', 'plays', 'played', 'playing'],
    ],
    adjectives: ['small', 'lazy', 'quick', 'hungry', 'happy'],
    places: ['garden', 'kitchen', 'mat', 'basket'],
  },
  food: {
    nouns: [
      ['apple', 'apples'],
      ['loaf', 'loaves'],
      ['cheese', 'cheeses'],
      ['berry', 'berries'],
      ['dish', 'dishes'],
      ['cook', 'cooks'],
    ],
    verbs: [
      ['bake', 'bakes', 'baked', 'baking'],
      ['eat', 'eats', 'ate', 'eating'],
      ['cut', 'cuts', 'cut', 'cutting'],
      ['taste', 'tastes', 'tasted', 'tasting'],
      ['serve', 'serves', 'served', 'serving'],
    ],
    adjectives: ['fresh', 'sweet', 'warm', 'ripe', 'salty'],
    places: ['kitchen', 'market', 'table', 'oven'],
  },
  weather: {
    nouns: [
      ['cloud', 'clouds'],
      ['storm', 'storms'],
      ['wind', 'winds'],
      ['river', 'rivers'],
      ['valley', 'valleys'],
      ['tree', 'trees'],
    ],
    verbs: [
      ['cover', 'covers', 'covered', 'covering'],
      ['flood', 'floods', 'flooded', 'flooding'],
      ['shake', 'shakes', 'shook', 'shaking'],
      ['cool', 'cools', 'cooled', 'cooling'],
      ['drift', 'drifts', 'drifted', 'drifting'],
    ],
    adjectives: ['cold', 'dark', 'heavy', 'stormy', 'gentle'],
    places: ['hills', 'valley', 'coast', 'town'],
  },
}

/** The topics of {@link toyCorpus}. */
export const CORPUS_TOPICS = Object.keys(TOPICS)

/** Options of {@link toyCorpus}. */
export interface ToyCorpusOptions {
  /** The number of sentences (default 40). */
  sentences?: number
  /** How many of the topics (pets, food, weather) to mix, from the first (default 3). */
  topics?: number
  /**
   * Zipf exponent of word choice within a topic (default 1): word k of a list is picked with probability ∝ (k + 1)^−s,
   * so a few words dominate as in real text.
   */
  exponent?: number
}

const pick = <T>(xs: readonly T[], u: number, exponent: number): T => {
  const w = xs.map((_, k) => (k + 1) ** -exponent)
  let r = u * w.reduce((a, b) => a + b, 0)
  for (let k = 0; k < xs.length; k++) if ((r -= w[k]) < 0) return xs[k]
  return xs[xs.length - 1]
}

/**
 * A seeded toy corpus: each sentence is drawn from one topic as "the [adjective] noun(s) verb(s|ed|ing) the noun(s) in
 * the place", with determiners, number and tense drawn too. Sentence k depends only on `child(s, k)`, so a longer
 * corpus extends a shorter one.
 */
export function toyCorpus(s: Stream, options: ToyCorpusOptions = {}): Corpus {
  const { sentences = 40, topics = 3, exponent = 1 } = options
  if (!(Number.isInteger(sentences) && sentences >= 1)) throw new DomainError('toyCorpus', 'toyCorpus: sentences ≥ 1')
  const names = CORPUS_TOPICS.slice(0, Math.max(1, Math.min(topics, CORPUS_TOPICS.length)))
  const documents: string[] = []
  for (let k = 0; k < sentences; k++) {
    const u = uniform(child(s, k), 0, 1, { shape: [12] }).data
    const topic = TOPICS[names[Math.min(names.length - 1, Math.floor(u[0] * names.length))]]
    const plural = u[1] < 0.35
    const subject = pick(topic.nouns, u[2], exponent)
    const object = pick(topic.nouns, u[3], exponent)
    const verb = pick(topic.verbs, u[4], exponent)
    const tense = u[5] < 0.45 ? 'past' : u[5] < 0.75 ? 'present' : 'progressive'
    const words: string[] = [u[6] < 0.7 ? 'the' : plural ? 'some' : 'a']
    if (u[7] < 0.5) words.push(pick(topic.adjectives, u[8], exponent))
    words.push(plural ? subject[1] : subject[0])
    if (tense === 'past') words.push(verb[2])
    else if (tense === 'present') words.push(plural ? verb[0] : verb[1])
    else words.push(plural ? 'are' : 'is', verb[3])
    words.push('the', u[9] < 0.5 ? object[0] : object[1])
    if (u[10] < 0.6) words.push(u[11] < 0.5 ? 'in' : 'near', 'the', pick(topic.places, u[11], exponent))
    documents.push(words.join(' '))
  }
  return {
    kind: 'corpus',
    documents,
    meta: {
      name: 'toy corpus',
      description: `${sentences} generated sentences about ${names.join(', ')}, with plurals and verb inflections.`,
      task: 'text',
    },
  }
}

// ── A topical corpus with shared frames, for word representations ────────────────────────────────────────────────────

/** A word with its two forms: a noun's singular and plural, or a verb's third-person singular and base form. */
type Forms = readonly [one: string, many: string]

interface TopicFrames {
  readonly subjects: readonly Forms[]
  readonly transitive: readonly Forms[]
  readonly objects: readonly string[]
  readonly intransitive: readonly Forms[]
}

const TOPIC_FRAMES: Readonly<Record<string, TopicFrames>> = {
  animals: {
    subjects: [
      ['cat', 'cats'],
      ['dog', 'dogs'],
      ['horse', 'horses'],
      ['cow', 'cows'],
      ['bird', 'birds'],
    ],
    transitive: [
      ['chases', 'chase'],
      ['watches', 'watch'],
      ['bites', 'bite'],
    ],
    objects: ['fish', 'bone', 'grass', 'mouse', 'seed'],
    intransitive: [
      ['sleeps', 'sleep'],
      ['runs', 'run'],
      ['hides', 'hide'],
    ],
  },
  food: {
    subjects: [
      ['cook', 'cooks'],
      ['chef', 'chefs'],
      ['baker', 'bakers'],
      ['waiter', 'waiters'],
    ],
    transitive: [
      ['bakes', 'bake'],
      ['serves', 'serve'],
      ['slices', 'slice'],
      ['stirs', 'stir'],
    ],
    objects: ['bread', 'cake', 'soup', 'pie', 'rice', 'cheese'],
    intransitive: [
      ['eats', 'eat'],
      ['smiles', 'smile'],
    ],
  },
  vehicles: {
    subjects: [
      ['car', 'cars'],
      ['bus', 'buses'],
      ['truck', 'trucks'],
      ['train', 'trains'],
      ['van', 'vans'],
    ],
    transitive: [
      ['carries', 'carry'],
      ['pulls', 'pull'],
      ['tows', 'tow'],
    ],
    objects: ['cargo', 'trailer', 'load', 'crate', 'timber'],
    intransitive: [
      ['stops', 'stop'],
      ['turns', 'turn'],
      ['speeds', 'speed'],
    ],
  },
  colours: {
    subjects: [
      ['painter', 'painters'],
      ['artist', 'artists'],
      ['decorator', 'decorators'],
    ],
    transitive: [
      ['paints', 'paint'],
      ['colours', 'colour'],
      ['dyes', 'dye'],
    ],
    objects: ['wall', 'door', 'canvas', 'fence', 'ceiling'],
    intransitive: [
      ['sketches', 'sketch'],
      ['draws', 'draw'],
    ],
  },
  places: {
    subjects: [
      ['tourist', 'tourists'],
      ['traveller', 'travellers'],
      ['visitor', 'visitors'],
    ],
    transitive: [
      ['visits', 'visit'],
      ['explores', 'explore'],
      ['leaves', 'leave'],
    ],
    objects: ['harbour', 'museum', 'castle', 'cathedral', 'bridge'],
    intransitive: [
      ['arrives', 'arrive'],
      ['wanders', 'wander'],
    ],
  },
}

/** Colour words: adjectives in every topic, and the result of the colours topic's verbs ("paints the wall red"). */
const COLOURS = ['red', 'blue', 'green', 'yellow', 'white', 'black']
/** Places: where any topic's intransitive sentences happen ("the cat sleeps in the town"). */
const PLACES = ['city', 'town', 'village', 'park', 'market']
const FUNCTION_WORDS = ['the', 'a', 'some', 'in', 'near', 'to']

/** The topics of {@link topicCorpus}, in label order. */
export const TOPIC_CORPUS_TOPICS = Object.keys(TOPIC_FRAMES)

/** Options of {@link topicCorpus}. */
export interface TopicCorpusOptions {
  /** The number of sentences (default 300). */
  sentences?: number
  /** How many of the topics (animals, food, vehicles, colours, places) to use, from the first (default 5). */
  topics?: number
  /** The chance that a noun takes a colour adjective, in any topic (default 0.2). */
  colourRate?: number
  /** The chance that an intransitive sentence ends at a place, in any topic (default 0.7). */
  placeRate?: number
}

/**
 * A seeded corpus for word representations: short sentences about animals, food, vehicles, colours and places, built
 * from the same frames ("the cat chases the mouse", "the buses stop near the market", "the painter paints the wall
 * red"), so that words cluster both by topic (the nouns and verbs of one topic share contexts) and by syntactic role
 * (colour words follow determiners in every topic, place words follow prepositions, plural nouns take base-form verbs).
 * The document labels are each sentence's topic; `wordTopics` gives every word's. Sentence k depends only on
 * `child(s, k)`, so a longer corpus extends a shorter one.
 */
export function topicCorpus(s: Stream, options: TopicCorpusOptions = {}): Corpus {
  const { sentences = 300, topics = TOPIC_CORPUS_TOPICS.length, colourRate = 0.2, placeRate = 0.7 } = options
  if (!(Number.isInteger(sentences) && sentences >= 1))
    throw new DomainError('topicCorpus', 'topicCorpus: sentences ≥ 1')
  const names = TOPIC_CORPUS_TOPICS.slice(0, Math.max(1, Math.min(topics, TOPIC_CORPUS_TOPICS.length)))
  const documents: string[] = []
  const labels: number[] = []
  const at = <T>(xs: readonly T[], u: number) => xs[Math.min(xs.length - 1, Math.floor(u * xs.length))]
  for (let k = 0; k < sentences; k++) {
    const u = uniform(child(s, k), 0, 1, { shape: [16] }).data
    const t = Math.min(names.length - 1, Math.floor(u[0] * names.length))
    const name = names[t]
    const f = TOPIC_FRAMES[name]
    const plural = u[1] < 0.35
    const subject = at(f.subjects, u[2])
    const words: string[] = [u[3] < 0.7 ? 'the' : plural ? 'some' : 'a']
    if (u[4] < colourRate) words.push(at(COLOURS, u[5]))
    words.push(plural ? subject[1] : subject[0])
    if (u[6] < 0.6) {
      words.push(at(f.transitive, u[7])[plural ? 1 : 0], 'the')
      if (u[8] < colourRate && name !== 'colours') words.push(at(COLOURS, u[9]))
      words.push(at(f.objects, u[10]))
      if (name === 'colours') words.push(at(COLOURS, u[9]))
    } else {
      words.push(at(f.intransitive, u[11])[plural ? 1 : 0])
      if (u[12] < placeRate) words.push(at(['in', 'near', 'to'], u[13]), 'the', at(PLACES, u[14]))
    }
    documents.push(words.join(' '))
    labels.push(t)
  }
  const wordTopics: Record<string, string> = {}
  for (const w of FUNCTION_WORDS) wordTopics[w] = 'function'
  for (const n of names) {
    const f = TOPIC_FRAMES[n]
    for (const w of [...f.subjects.flat(), ...f.transitive.flat(), ...f.objects, ...f.intransitive.flat()])
      wordTopics[w] = n
  }
  for (const w of COLOURS) wordTopics[w] = 'colours'
  for (const w of PLACES) wordTopics[w] = 'places'
  return {
    kind: 'corpus',
    documents,
    labels,
    wordTopics,
    meta: {
      name: 'topic corpus',
      description: `${sentences} generated sentences about ${names.join(', ')}, built from shared frames; each sentence is labelled with its topic.`,
      task: 'text',
      labelNames: names,
    },
  }
}

/** The themes of `driftingTopicCorpus`, each a list of words ordered from the oldest usage to the newest. */
export const DRIFTING_TOPICS: Readonly<Record<string, readonly string[]>> = {
  travel: [
    'horse',
    'carriage',
    'coach',
    'canal',
    'steamship',
    'railway',
    'tram',
    'bicycle',
    'motorcar',
    'airliner',
    'motorway',
    'jet',
  ],
  messages: [
    'letter',
    'courier',
    'post',
    'telegraph',
    'telegram',
    'wireless',
    'telephone',
    'radio',
    'television',
    'fax',
    'email',
    'internet',
  ],
  medicine: [
    'herbs',
    'leeches',
    'bleeding',
    'tonic',
    'quinine',
    'ether',
    'antiseptic',
    'aspirin',
    'insulin',
    'penicillin',
    'vaccine',
    'genome',
  ],
  work: [
    'field',
    'plough',
    'loom',
    'mill',
    'forge',
    'factory',
    'foundry',
    'assembly',
    'office',
    'typewriter',
    'computer',
    'software',
  ],
}

/** Options of `driftingTopicCorpus`. */
export interface DriftingTopicCorpusOptions {
  /** Time slices (default 6) and documents per slice (default 30). */
  slices?: number
  documentsPerSlice?: number
  /** Words per document (default 40) and the Dirichlet concentration of the documents' topic proportions (0.1). */
  length?: number
  alpha?: number
  /** Width of a theme's usage window, in words of its list (default 2.5). */
  width?: number
}

/**
 * Documents whose themes' vocabularies change over time, for dynamic topic models (Blei & Lafferty, 2006): four themes
 * (travel, messages, medicine, work), each a list of twelve words from old usage to new; in slice t of T a theme's
 * word distribution is a Gaussian window over its list, centred at position 11·t/(T − 1), so "horse" gives way to
 * "jet" and "letter" to "email". Each document draws proportions θ ~ Dir(α) over the themes, then each word's theme
 * and the word. The true topics of every slice are returned with the documents. Document (t, i) depends only on
 * `child(s, t, i)`.
 */
export function driftingTopicCorpus(s: Stream, options: DriftingTopicCorpusOptions = {}): Corpus {
  const { slices = 6, documentsPerSlice = 30, length = 40, alpha = 0.1, width = 2.5 } = options
  if (!(Number.isInteger(slices) && slices >= 1 && Number.isInteger(documentsPerSlice) && documentsPerSlice >= 1))
    throw new DomainError('driftingTopicCorpus', 'driftingTopicCorpus: slices and documentsPerSlice ≥ 1')
  if (!(Number.isInteger(length) && length >= 1 && alpha > 0 && width > 0))
    throw new DomainError('driftingTopicCorpus', 'driftingTopicCorpus: length ≥ 1, α > 0 and width > 0')
  const names = Object.keys(DRIFTING_TOPICS)
  const vocabulary = names.flatMap((n) => DRIFTING_TOPICS[n])
  const K = names.length
  const V = vocabulary.length
  const topicWord = Array.from({ length: slices }, (_, t) => {
    const centre = slices > 1 ? (11 * t) / (slices - 1) : 5.5
    return names.map((n, k) => {
      const row = new Array<number>(V).fill(0)
      let z = 0
      DRIFTING_TOPICS[n].forEach(
        (_, i) => (z += row[k * 12 + i] = Math.exp(-((i - centre) ** 2) / (2 * width * width))),
      )
      return row.map((v) => v / z)
    })
  })
  const documents: string[] = []
  const labels: number[] = []
  const times: number[] = []
  const pick = (weights: readonly number[], u: number) => {
    let acc = 0
    for (let i = 0; i < weights.length; i++) if ((acc += weights[i]) > u) return i
    return weights.length - 1
  }
  for (let t = 0; t < slices; t++)
    for (let i = 0; i < documentsPerSlice; i++) {
      const r = child(s, t, i)
      // θ ~ Dir(α) by normalised Gamma draws.
      const g = toFlat(gammaVariate(child(r, 'theta'), alpha, 1, { shape: [K] }) as Tensor)
      const total = g.reduce((a, b) => a + b, 0)
      const theta = Array.from(g, (v) => v / total)
      const u = uniform(child(r, 'words'), 0, 1, { shape: [2 * length] }).data
      const words: string[] = []
      for (let n = 0; n < length; n++) {
        const k = pick(theta, u[2 * n])
        words.push(vocabulary[pick(topicWord[t][k], u[2 * n + 1])])
      }
      documents.push(words.join(' '))
      labels.push(theta.indexOf(Math.max(...theta)))
      times.push(t)
    }
  const wordTopics: Record<string, string> = {}
  names.forEach((n) => DRIFTING_TOPICS[n].forEach((w) => (wordTopics[w] = n)))
  return {
    kind: 'corpus',
    documents,
    labels,
    wordTopics,
    times,
    topics: { vocabulary, topicWord },
    meta: {
      name: 'drifting topic corpus',
      description: `${slices} time slices of ${documentsPerSlice} documents mixing ${names.join(', ')}, whose words change over time; labelled by each document's main theme.`,
      task: 'text',
      labelNames: names,
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'text/corpora')

dataset(
  {
    key: 'namedCorpus',
    name: 'Worked-example corpora',
    summary: 'The small corpora the text notes work by hand.',
    task: 'text',
    output: 'corpus',
    knobs: space({ name: oneOf(Object.keys(NAMED_CORPORA) as CorpusName[]) }),
    truth: false,
    notes: [
      'term-frequency-inverse-document-frequency',
      'co-occurrence-matrices-and-pointwise-mutual-information',
      'byte-pair-encoding',
      'wordpiece-and-unigram-tokenisation',
    ],
    cite: ['sennrich2016', 'huggingface2024wordpiece'],
  },
  namedCorpus,
)

dataset(
  {
    key: 'toyCorpus',
    name: 'Toy corpus',
    summary: 'Seeded sentences about pets, food and weather, with plurals and verb inflections.',
    task: 'text',
    output: 'corpus',
    knobs: space({
      sentences: int(1, 2000, { default: 40 }),
      topics: int(1, 3, { default: 3 }),
      exponent: real(0, 3, { default: 1 }),
    }),
    truth: false,
    random: true,
    notes: ['text-representation-pipeline', 'tokenisation', 'bag-of-words'],
  },
  toyCorpus,
)

dataset(
  {
    key: 'topicCorpus',
    name: 'Topic corpus',
    summary: 'Seeded sentences about animals, food, vehicles, colours and places in shared frames, labelled by topic.',
    task: 'text',
    output: 'corpus',
    knobs: space({
      sentences: int(1, 5000, { default: 300 }),
      topics: int(1, 5, { default: 5 }),
      colourRate: real(0, 1, { default: 0.2 }),
      placeRate: real(0, 1, { default: 0.7 }),
    }),
    truth: false,
    random: true,
    notes: ['distributional-semantics', 'latent-semantic-analysis', 'word-embeddings'],
    cite: ['harris1954', 'firth1957'],
  },
  topicCorpus,
)

dataset(
  {
    key: 'driftingTopicCorpus',
    name: 'Drifting topic corpus',
    summary: 'Documents over time slices mixing four themes whose words change from old usage to new.',
    task: 'text',
    output: 'corpus',
    knobs: space({
      slices: int(1, 20, { default: 6 }),
      documentsPerSlice: int(1, 500, { default: 30 }),
      length: int(1, 500, { default: 40 }),
      alpha: real(0.01, 10, { default: 0.1 }),
      width: real(0.1, 6, { default: 2.5 }),
    }),
    truth: true,
    random: true,
    notes: ['dynamic-topic-model', 'latent-dirichlet-allocation'],
    cite: ['blei2006dtm'],
  },
  driftingTopicCorpus,
)

/** The corpus generators, keyed by name. */
export const corpusDatasets = { namedCorpus, toyCorpus, topicCorpus, driftingTopicCorpus } as unknown as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, DatasetInfo>>
>
