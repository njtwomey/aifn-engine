/** The functions of `aifn-compute/text/stem`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as porter from './porter'
import * as stop from './stop-words'

const fn = definer<FunctionInfo>('function', 'text/stem')

fn(
  {
    key: 'porterStem',
    name: 'Porter stemmer',
    role: 'transform',
    summary: "Porter's 1980 suffix-stripping stemmer: five steps of rules conditioned on the stem's measure.",
    notes: ['stemming'],
    cite: ['porter1980'],
  },
  porter.porterStem,
)
fn(
  {
    key: 'porterStemTrace',
    name: 'Porter stemmer, rule by rule',
    role: 'transform',
    summary: 'The Porter stem with every rule that fired, step by step.',
    notes: ['stemming'],
    cite: ['porter1980'],
  },
  porter.porterStemTrace,
)
fn(
  {
    key: 'porterMeasure',
    name: 'Porter measure',
    tex: 'm',
    role: 'property',
    summary: 'The number of vowel–consonant pairs m in the form [C](VC)^m[V].',
    notes: ['stemming'],
    cite: ['porter1980'],
  },
  porter.porterMeasure,
)
fn(
  {
    key: 'removeStopWords',
    name: 'Stop-word removal',
    role: 'transform',
    summary: "Drop the tokens of a stop list (NLTK's or scikit-learn's English list, or any list).",
    notes: ['stop-words'],
    cite: ['manning2008', 'bird2009'],
  },
  stop.removeStopWords,
)

/** The functions of the module, keyed by name. */
export const stemFunctions = entries<FunctionInfo>('function', porter, stop) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
