/**
 * `aifn-methods/evaluation/fairness`: group fairness metrics of a binary classifier, comparing its rates across the
 * groups of a protected attribute.
 *
 * - Per-group rates: `groupRates` gives each group's base rate, selection rate, true- and false-positive rates and
 *   precision.
 * - Independence: `demographicParityDifference` and `demographicParityRatio` compare selection rates, the ratio being
 *   the disparate impact ratio.
 * - Separation: `equalOpportunityDifference` compares true-positive rates, and `equalisedOddsDifference` takes the
 *   larger of the true- and false-positive-rate gaps.
 * - Sufficiency: `predictiveParityDifference` compares precisions.
 *
 * Every metric takes the true labels, the predictions and `FairnessOptions` (the group of each case, and the positive
 * class); a difference is the largest group value minus the smallest. `fairnessFunctions` registers `groupRates`, the
 * module's one function that is not a metric.
 */

export {
  demographicParityDifference,
  demographicParityRatio,
  equalOpportunityDifference,
  equalisedOddsDifference,
  groupRates,
  predictiveParityDifference,
  type FairnessOptions,
  type GroupRates,
} from './fairness'
export { fairnessFunctions } from './registry'
