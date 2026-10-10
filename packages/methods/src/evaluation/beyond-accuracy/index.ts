/**
 * `aifn-methods/evaluation/beyond-accuracy`: beyond-accuracy metrics of recommendation lists, which score what is
 * recommended rather than whether it was right.
 *
 * - Within a list: `intraListDiversity`, the mean pairwise dissimilarity of its items, and `novelty`, the mean
 *   self-information of their popularities, in bits.
 * - Across lists: `catalogueCoverage`, the fraction of the catalogue recommended at all, and the concentration of
 *   exposure over items, `giniCoefficient` (0 for equal exposure) and `herfindahlIndex` ($1/M$ for equal exposure).
 *
 * Each is a metric of the `aifn-compute/learning/metrics` registry, with `info` giving its range and whether higher is
 * better, and is collected in `evaluationMetricRegistry` of `aifn-methods/evaluation`.
 */
export { intraListDiversity, catalogueCoverage, giniCoefficient, herfindahlIndex, novelty } from './beyondAccuracy'
