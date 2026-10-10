/**
 * `aifn-compute/text/stem`: the Porter stemmer as published in 1980, and English stop-word lists.
 *
 * - Stemming: `porterStem` for the stem, `porterStemTrace` for the rules that fired step by step, and the pieces the
 *   rules are conditioned on, `consonantVowelForm` and `porterMeasure` ($m$ in $[C](VC)^m[V]$). The later revisions
 *   (Porter2, Snowball) are not applied.
 * - Stop words: `STOP_WORDS` holds NLTK's and scikit-learn's English lists as published, and `removeStopWords` drops
 *   them (or any list) from a token list, comparing in lower case.
 *
 * Stems are lower case; words of one or two letters, or with characters outside a to z, are only lower-cased.
 */

export {
  consonantVowelForm,
  porterMeasure,
  porterStem,
  porterStemTrace,
  type PorterStep,
  type PorterTrace,
} from './porter'
export { removeStopWords, STOP_WORDS, type StopList } from './stop-words'
export { stemFunctions } from './registry'
