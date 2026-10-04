/**
 * `aifn-methods/text`: applications of text processing on `aifn-compute/text`. Children: corpora (toy and worked-example
 * corpora) and tokenisers (a trained suite of named tokenisers to compare).
 */

export { namedCorpus, toyCorpus, type Corpus } from './corpora'
export { tokeniserSuite, TOKENISER_KINDS, type TokeniserKind } from './tokenisers'
