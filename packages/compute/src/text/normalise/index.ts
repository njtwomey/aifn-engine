/**
 * `aifn-compute/text/normalise`: text normalisation for matching, string to string.
 *
 * - The steps: Unicode normal forms (NFC, NFD, NFKC, NFKD, as `String.prototype.normalize`), `caseFold` (full Unicode
 *   case folding, as Python's `str.casefold`), `stripAccents` (NFKD, then combining marks dropped, as scikit-learn's
 *   `strip_accents='unicode'`) and `collapseWhitespace`.
 * - `normalise` composes them in that order; its default (NFKC, then case folding, then white space) is the caseless
 *   matching key of UAX #15.
 *
 * Every function is pure and keeps no offsets; the offset-keeping normalisers of a tokeniser are in
 * `aifn-compute/text/pipeline`.
 */

export {
  caseFold,
  collapseWhitespace,
  normalise,
  stripAccents,
  type NormaliseOptions,
  type UnicodeForm,
} from './normalise'
export { normaliseFunctions } from './registry'
