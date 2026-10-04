/**
 * `aifn-compute/text/statistics`: how tokenisers compare on a corpus: fertility (tokens per word), bytes and characters per
 * token, the unknown rate and word coverage, and the share of the vocabulary used.
 */

export {
  tokenisationStatistics,
  tokeniserStatistics,
  type TokenStatistics,
  type TokenStatisticsOptions,
} from './statistics'
export { statisticsFunctions } from './registry'
