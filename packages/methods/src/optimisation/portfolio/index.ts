/**
 * `aifn-methods/optimisation/portfolio`: portfolios of configurations chosen on a table of losses over earlier tasks,
 * for warm-starting a hyperparameter search.
 *
 * - Construction: `greedyPortfolio` adds, $k$ times, the configuration that most lowers the mean over tasks of the
 *   best member's loss (within $1 - 1/e$ of the best reduction, by submodularity), and finds the optimal subset by
 *   exhaustive search when the table is small enough (`Portfolio`, `PortfolioOptions`).
 * - Assessment: `portfolioLoss` scores a given portfolio, and `leaveOneTaskOut` builds the portfolio without each
 *   task and scores it on the task left out.
 * - Baseline: `averageRank`, the configurations ordered by their mean rank over the tasks.
 *
 * Tables are $n \times T$ matrices, one row per candidate configuration and one column per task, with losses to
 * minimise; `tuningTables` in `aifn-methods/data/objectives` gives one for an RBF SVM on seven tasks.
 * `portfolioFunctions` registers the functions.
 */

export {
  averageRank,
  greedyPortfolio,
  leaveOneTaskOut,
  portfolioLoss,
  type Portfolio,
  type PortfolioOptions,
} from './portfolio'
export { portfolioFunctions } from './registry'
