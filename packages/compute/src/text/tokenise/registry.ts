/** The functions of `aifn-compute/text/tokenise`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as casual from './casual'
import * as sentences from './sentences'
import * as t from './tokenise'
import * as treebank from './treebank'

const fn = definer<FunctionInfo>('function', 'text/tokenise')
const notes = ['tokenisation']

fn(
  {
    key: 'tokenise',
    name: 'Regular-expression tokeniser',
    role: 'transform',
    returns: 'tokens',
    summary:
      'Tokens and their offsets by a regular expression: words, words and punctuation, white space, BERT, and the GPT-2, cl100k and o200k pre-tokenisers.',
    notes,
    cite: ['jurafsky2025', 'radford2019', 'devlin2019'],
  },
  t.tokenise,
)
fn(
  {
    key: 'whitespaceTokenise',
    name: 'White-space tokeniser',
    role: 'transform',
    returns: 'tokens',
    notes,
  },
  t.whitespaceTokenise,
)
fn(
  {
    key: 'characterTokenise',
    name: 'Character tokeniser',
    role: 'transform',
    returns: 'tokens',
    summary: 'One token per code point or per grapheme cluster, with offsets.',
    notes: ['tokenisation', 'character-n-grams-and-shingles'],
  },
  t.characterTokenise,
)
fn(
  {
    key: 'detokenise',
    name: 'Detokeniser',
    role: 'transform',
    summary: 'Tokens back to text: exactly from offsets, or by spacing rules for punctuation and clitics.',
    notes,
  },
  t.detokenise,
)

fn(
  {
    key: 'treebankTokens',
    name: 'Penn Treebank word tokeniser',
    role: 'transform',
    summary:
      'Penn Treebank conventions by regular-expression substitution, as NLTK: clitics split ("don\'t" → "do n\'t"), quotes to `` and \'\'.',
    notes,
    cite: ['bird2009', 'jurafsky2025'],
  },
  treebank.treebankTokens,
)
fn(
  {
    key: 'treebankTokenise',
    name: 'Penn Treebank tokeniser with offsets',
    role: 'transform',
    returns: 'tokens',
    summary: 'Treebank tokens located in the text, as NLTK’s span_tokenize.',
    notes,
    cite: ['bird2009'],
  },
  treebank.treebankTokenise,
)
fn(
  {
    key: 'casualTokenise',
    name: 'Casual (tweet) tokeniser',
    role: 'transform',
    returns: 'tokens',
    summary:
      'URLs, emoticons, @-handles, hashtags, e-mail addresses, emoji sequences and words, as NLTK’s TweetTokenizer.',
    notes,
    cite: ['bird2009'],
  },
  casual.casualTokenise,
)
fn(
  {
    key: 'sentenceSplit',
    name: 'Sentence splitter (Punkt rules)',
    role: 'transform',
    returns: 'tokens',
    summary: 'Sentence boundaries at ?, ! and periods that do not end an abbreviation, initial or number.',
    notes,
    cite: ['bird2009', 'jurafsky2025'],
  },
  sentences.sentenceSplit,
)

/** The functions of the module, keyed by name. */
export const tokeniseFunctions = entries<FunctionInfo>('function', t, treebank, casual, sentences) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
