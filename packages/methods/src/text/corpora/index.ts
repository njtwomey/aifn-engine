/**
 * `aifn-methods/text/corpora`: toy corpora for the text pipeline, from the hand-worked corpora of the notes to seeded
 * generators with known topics.
 *
 * - Worked examples: `namedCorpus` returns one of `NAMED_CORPORA` by name (`CorpusName`): the TF-IDF and BM25
 *   documents, the PMI sentences, Sennrich's BPE corpus and the Hugging Face WordPiece corpus.
 * - Inflected sentences: `toyCorpus` writes sentences about the `CORPUS_TOPICS` (pets, food, weather) with plurals and
 *   verb forms, words picked by Zipf's law, for tokenising, stemming and bags of words.
 * - Word representations: `topicCorpus` builds sentences about the `TOPIC_CORPUS_TOPICS` from shared frames, labelled
 *   by topic, so words cluster by topic and by syntactic role.
 * - Topics over time: `driftingTopicCorpus` mixes the `DRIFTING_TOPICS`, whose words change from slice to slice, and
 *   returns the generating topics, for dynamic topic models.
 * - `corpusDatasets`: the registered generators, keyed by name.
 *
 * Every function returns a `Corpus` of raw-text documents; the generators are seeded by a stream, one child stream per
 * document, and throw `DomainError` for a size that is not a positive integer.
 */

export {
  CORPUS_TOPICS,
  DRIFTING_TOPICS,
  driftingTopicCorpus,
  type DriftingTopicCorpusOptions,
  corpusDatasets,
  NAMED_CORPORA,
  namedCorpus,
  toyCorpus,
  topicCorpus,
  TOPIC_CORPUS_TOPICS,
  type TopicCorpusOptions,
  type Corpus,
  type CorpusName,
  type ToyCorpusOptions,
} from './corpora'
