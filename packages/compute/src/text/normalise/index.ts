/**
 * `aifn-compute/text/normalise`: Unicode normal forms, full case folding, accent stripping and white space, composed by
 * `normalise`.
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
