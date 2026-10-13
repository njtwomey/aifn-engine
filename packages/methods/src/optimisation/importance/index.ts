/**
 * `aifn-methods/optimisation/importance`: hyperparameter importance by the functional ANOVA of a surface tabulated on a
 * full grid (Hutter, Hoos and Leyton-Brown, 2014), exact on the grid, with no random forest.
 *
 * - Decomposition: `fanova` splits the table into its mean, main effects $f_i$, pairwise terms $f_{ij}$ and higher
 *   interactions up to a chosen order, each with its marginal $\expect[f \mid \lambda_U]$, its variance and its
 *   fraction of $\var f$ (`Fanova`, `FanovaTerm`, `FanovaOptions`).
 * - Reconstruction: `fanovaReconstruct` adds the terms back up to an order, so a figure can set the additive part
 *   beside the table.
 *
 * The decomposition is taken under a product measure over the grid (equal weights unless given per axis), under which
 * the terms are orthogonal and their variances sum to the total. For the functional ANOVA of a fitted model on a grid
 * see `functionalAnova` in `aifn-compute/learning/explain`. `importanceFunctions` registers the functions.
 */

export { fanova, fanovaReconstruct, type Fanova, type FanovaOptions, type FanovaTerm } from './fanova'
export { importanceFunctions } from './registry'
