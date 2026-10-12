/**
 * `aifn-methods/data/real`: small embedded real datasets.
 *
 * - Classic tables as datasets: `iris` (classification), `oldFaithful` (clustering), `anscombe` (regression, four
 *   sets) and `coalMining` (a count series with a changepoint), each with its source in `meta`.
 * - A graph: `karateClub`, Zachary's friendship network as a `GraphDataset`, the nodes labelled by club.
 * - A table of nominal columns: `titanic`, the 2201 people aboard by class, sex, age and survival.
 * - Vendored from a pinned source by a script: `diabetes`, the 442 patients of Efron, Hastie, Johnstone and Tibshirani
 *   (2004) with ten baseline measurements and disease progression a year later (regression), the data of the lasso,
 *   elastic-net and least-angle regression paths. `scripts/diabetes.py` downloads the source, checks its sha256 and
 *   writes it verbatim; the dataset's `meta.sha256` records the hash.
 *
 * All are built from values embedded in the source, take no arguments and draw nothing at random. Larger real data
 * live in their own modules, so that only the pages that use them load them: `aifn-methods/data/real/ecg` (an
 * annotated electrocardiogram), `aifn-methods/data/real/fonts` (glyph outlines of 66 fonts) and
 * `aifn-methods/data/real/hyphenation` (English words with dictionary hyphenation points).
 */

export { anscombe, coalMining, iris, karateClub, oldFaithful, type GraphDataset } from './real'
export { diabetes } from './diabetes'
export { titanic } from './titanic'
