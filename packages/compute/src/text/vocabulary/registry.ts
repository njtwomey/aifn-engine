/** The functions of `aifn-compute/text/vocabulary`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as v from './vocabulary'

const fn = definer<FunctionInfo>('function', 'text/vocabulary')
const notes = ['one-hot-encoding-of-words', 'bag-of-words', 'tokenisation']

fn(
  {
    key: 'buildVocabulary',
    name: 'Vocabulary',
    role: 'construction',
    returns: 'vocabulary',
    summary: 'Token ↔ id map with special tokens first, a minimum count and a maximum size.',
    notes,
    cite: ['jurafsky2025'],
  },
  v.buildVocabulary,
)
fn(
  { key: 'vocabularyOf', name: 'Vocabulary from a token list', role: 'construction', returns: 'vocabulary', notes },
  v.vocabularyOf,
)
fn(
  {
    key: 'tokenCounts',
    name: 'Token and document counts',
    role: 'estimator',
    summary: 'Distinct tokens in order of appearance with their counts and document frequencies.',
    notes: ['bag-of-words'],
  },
  v.tokenCounts,
)
fn({ key: 'tokenId', name: 'Token id', role: 'property', notes }, v.tokenId)
fn(
  {
    key: 'encodeTokens',
    name: 'Encode tokens to ids',
    role: 'transform',
    summary: 'Token ids, with out-of-vocabulary tokens mapped to the unknown id.',
    notes,
  },
  v.encodeTokens,
)
fn({ key: 'decodeTokens', name: 'Decode ids to tokens', role: 'transform', notes }, v.decodeTokens)

/** The functions of the module, keyed by name. */
export const vocabularyFunctions = entries<FunctionInfo>('function', v) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
