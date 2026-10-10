/**
 * `aifn-compute/text/statistics`: how tokenisers compare on a corpus.
 *
 * - `tokenisationStatistics` measures tokenisations already made: fertility (tokens per word, Rust et al., 2021),
 *   UTF-8 bytes and code points per token (compression), the unknown-token rate, word coverage and the share of the
 *   vocabulary used.
 * - `tokeniserStatistics` encodes a corpus with a pipeline tokeniser (no special tokens, truncation or padding) and
 *   measures the result, with unknown tokens and usage read from its ids.
 *
 * Words are runs of non-space characters of the original texts, so every tokeniser is measured against the same
 * denominator. A rate with a zero denominator is NaN.
 */

export {
  tokenisationStatistics,
  tokeniserStatistics,
  type TokenStatistics,
  type TokenStatisticsOptions,
} from './statistics'
export { statisticsFunctions } from './registry'
