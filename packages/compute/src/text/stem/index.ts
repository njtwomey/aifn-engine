/**
 * `aifn-compute/text/stem`: the Porter stemmer as published in 1980, with a rule-by-rule trace and the measure, and English
 * stop-word lists (NLTK's and scikit-learn's) with their removal.
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
