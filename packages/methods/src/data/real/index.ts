/**
 * `aifn-methods/data/real`: small embedded real datasets.
 *
 * - Classic tables as datasets: `iris` (classification), `oldFaithful` (clustering), `anscombe` (regression, four
 *   sets) and `coalMining` (a count series with a changepoint), each with its source in `meta`.
 * - A graph: `karateClub`, Zachary's friendship network as a `GraphDataset`, the nodes labelled by club.
 * - A table of nominal columns: `titanic`, the 2201 people aboard by class, sex, age and survival.
 *
 * All are built from values embedded in the source, take no arguments and draw nothing at random. Larger real data
 * live in their own modules, so that only the pages that use them load them: `aifn-methods/data/real/ecg` (an
 * annotated electrocardiogram), `aifn-methods/data/real/fonts` (glyph outlines of 66 fonts) and
 * `aifn-methods/data/real/hyphenation` (English words with dictionary hyphenation points).
 */

export { anscombe, coalMining, iris, karateClub, oldFaithful, type GraphDataset } from './real'
export { titanic } from './titanic'
