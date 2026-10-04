/**
 * A toy corpus for character-level language models: a few traditional English nursery rhymes (public domain),
 * lower-cased, about 1,300 characters over 30 symbols. Small enough to fit in a browser in seconds, repetitive enough
 * that a model can learn spelling, word boundaries and recurring phrases.
 */

import { characterTokenise } from 'aifn-compute/text/tokenise'
import { vocabularyOf, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** The toy corpus: nursery rhymes, one per paragraph. */
export const NURSERY_RHYMES = `twinkle, twinkle, little star, how i wonder what you are. up above the world so high, like a diamond in the sky. twinkle, twinkle, little star, how i wonder what you are.

mary had a little lamb, its fleece was white as snow. and everywhere that mary went, the lamb was sure to go. it followed her to school one day, which was against the rule. it made the children laugh and play to see a lamb at school.

humpty dumpty sat on a wall. humpty dumpty had a great fall. all the king's horses and all the king's men couldn't put humpty together again.

jack and jill went up the hill to fetch a pail of water. jack fell down and broke his crown, and jill came tumbling after.

hey diddle diddle, the cat and the fiddle, the cow jumped over the moon. the little dog laughed to see such sport, and the dish ran away with the spoon.

baa, baa, black sheep, have you any wool? yes sir, yes sir, three bags full. one for the master, one for the dame, and one for the little boy who lives down the lane.

hickory dickory dock, the mouse ran up the clock. the clock struck one, the mouse ran down, hickory dickory dock.

little bo peep has lost her sheep, and doesn't know where to find them. leave them alone, and they'll come home, wagging their tails behind them.`

/** A corpus encoded as character ids. */
export type CharCorpus = {
  /** The characters in order of their ids (sorted by code point). */
  readonly vocabulary: Vocabulary
  /** The text as ids. */
  readonly ids: readonly number[]
}

/** Encode `text` character by character over its own sorted alphabet (no special tokens). */
export function charCorpus(text: string = NURSERY_RHYMES): CharCorpus {
  const chars = characterTokenise(text).tokens
  const alphabet = [...new Set(chars)].sort()
  const vocabulary = vocabularyOf(alphabet)
  const index = new Map(alphabet.map((c, i) => [c, i]))
  return { vocabulary, ids: chars.map((c) => index.get(c)!) }
}

/** Ids of `text` in a corpus's vocabulary (unknown characters are dropped). */
export function encodeChars(corpus: CharCorpus, text: string): number[] {
  const index = new Map(corpus.vocabulary.tokens.map((c, i) => [c, i]))
  return characterTokenise(text)
    .tokens.map((c) => index.get(c))
    .filter((i): i is number => i !== undefined)
}

/** The text of a list of ids. */
export function decodeChars(corpus: CharCorpus, ids: readonly number[]): string {
  return ids.map((i) => corpus.vocabulary.tokens[i] ?? '').join('')
}
