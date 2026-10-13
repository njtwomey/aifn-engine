/**
 * Random forests of CART regression trees (Breiman, 2001, "Random forests", Machine Learning 45), as scikit-learn's
 * `RandomForestRegressor`, with the per-tree predictions that make a forest a surrogate model in sequential
 * model-based optimisation (SMAC; Hutter, Hoos and Leyton-Brown, 2011).
 *
 * Each of the $B$ trees is grown on a bootstrap sample of the rows (integer sample weights) and searches a random
 * subset of the features at every node; the forest predicts the mean of the trees' leaf means. The spread of the trees'
 * predictions at a point is SMAC's measure of uncertainty:
 *
 * $$ \hat\mu(\xvec) = \frac{1}{B}\sum_{b=1}^B \hat f_b(\xvec), \qquad
 * \hat\sigma^2(\xvec) = \frac{1}{B}\sum_{b=1}^B \bigl(\hat f_b(\xvec) - \hat\mu(\xvec)\bigr)^2, $$
 *
 * and, by the law of total variance, adding the mean of the leaves' own variances $v_b(\xvec)$ (the weighted variance
 * of the training targets in the leaf $\xvec$ reaches in tree $b$) gives the variance of a mixture of the leaves'
 * empirical laws, $\hat\sigma^2 + \frac{1}{B}\sum_b v_b(\xvec)$ (Hutter et al., 2014).
 */

import type { Decides, Estimator, FitOptions, Fitted, Supervised, Trained } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { bool, int, space } from 'aifn-compute/foundation/space'
import { applyTree, featureImportances, predictTree, type DecisionTree, type TreeParams } from '../tree'
import { inputs, matrix, targets, values } from '../../util'
import { forestGrowth, type ForestState } from './forest'

/** The hyperparameters of `randomForestRegressor`. */
export type RandomForestRegressorParams = Omit<TreeParams, 'criterion' | 'maxFeatures'> & {
  /** The number $B$ of trees (default 100). */
  trees?: number
  /** Grow each tree on a bootstrap sample (default true); otherwise on every row. */
  bootstrap?: boolean
  /**
   * The features searched at each node: all of them (`'all'`, the default, as scikit-learn's regressor), a count, a
   * fraction below 1, or $\lfloor\sqrt{d}\rfloor$ (`'sqrt'`) or $\lfloor\log_2 d\rfloor$ (`'log2'`).
   */
  maxFeatures?: number | 'sqrt' | 'log2' | 'all'
}

/** Options of `RandomForestRegressionModel.moments`. */
export type ForestMomentsOptions = {
  /**
   * Add the mean of the leaves' variances to the spread of the trees' predictions (the law of total variance), instead
   * of the spread alone (default false, SMAC's original measure).
   */
  total?: boolean
}

/** A fitted random forest regressor. */
export interface RandomForestRegressionModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Trained<ForestState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'random-forest-regressor'
  /** The $B$ regression trees, in the order grown. */
  readonly trees: DecisionTree[]
  /** The bootstrap count of each training row in each tree's sample, $B \times n$ (all ones without bootstrap). */
  readonly inBag: Tensor
  /** The mean of the trees' impurity importances, $d$ values summing to 1 (all 0 when no tree splits). */
  readonly featureImportances: Tensor
  /**
   * The out-of-bag predictions, $n$ values (each row averaged over the trees whose bag missed it; NaN for a row in
   * every bag), and their mean squared error and $R^2$ over the rows that have one (NaN when none has).
   */
  readonly outOfBag: { predictions: Tensor; mse: number; r2: number }
  /** Every tree's prediction at the rows of `x` ($m \times d$): a $B \times m$ tensor, row $b$ for tree $b$. */
  treePredictions(x: Tensor): Tensor
  /**
   * The forest's mean $\hat\mu$ and variance $\hat\sigma^2$ at the rows of `x`, $m$ values each: the mean and the
   * population variance (divided by $B$) of the trees' predictions, plus the mean leaf variance with `total`.
   */
  moments(x: Tensor, options?: ForestMomentsOptions): { mean: Tensor; variance: Tensor }
}

/**
 * A random forest of regression trees (Breiman, 2001), as scikit-learn's `RandomForestRegressor`: `trees` CART trees on
 * squared error, each grown on a bootstrap sample (as integer sample weights) searching `maxFeatures` features (default
 * all) at every node. `forward` and `decide` give the mean of the trees' predictions; `treePredictions` gives each
 * tree's, and `moments` their mean and variance, the surrogate of SMAC (Hutter et al., 2011). Without bootstrap and
 * with every feature searched, every tree is the single CART tree of `regressionTree`, and the variance is 0. The
 * growth (`forestGrowth`, one tree per step) is kept in `training`; tree $b$ depends only on the stream and $b$. Throws
 * `ShapeError` when `x` is not a matrix or `y` does not have one target per row.
 *
 * @param params The tree hyperparameters, the number of trees, bootstrapping and the features searched, as
 *   `RandomForestRegressorParams`.
 * @returns An estimator whose `fit({ x, y }, { stream? })` grows the forest (from `stream(0)` when no stream is given)
 *   and returns the fitted `RandomForestRegressionModel`.
 *
 * @example The forest's mean and the spread of its trees on a noisy sine
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [4.5]])
 * const y = tensor([0.1, 0.4, 0.9, 1.0, 0.9, 0.6, 0.1, -0.4, -0.7, -1.0])
 * const model = randomForestRegressor({ trees: 20 }).fit({ x, y }, { stream: stream(1) })
 * const q = tensor([[1.2], [3.2], [6]])
 * const { mean, variance } = model.moments(q)
 * print('mean:', mean, ' sd over trees:', variance.data.map(Math.sqrt))
 * print('first three trees at the queries:', model.treePredictions(q).data.slice(0, 9))
 *
 * @example Without bootstrap every tree is the same CART tree, so the spread vanishes
 * const x = tensor([[1], [2], [3], [4], [5], [6]])
 * const y = tensor([1, 1.2, 0.8, 5, 5.4, 4.6])
 * const model = randomForestRegressor({ trees: 5, bootstrap: false, maxDepth: 1 }).fit({ x, y })
 * const q = tensor([[0], [10]])
 * print('forest:', model.decide(q), ' every tree splits at', model.trees.map((t) => t.nodes[0].threshold))
 * print('variance over trees:', model.moments(q).variance)
 * print('with the leaves:', model.moments(q, { total: true }).variance)
 */
export function randomForestRegressor(
  params: RandomForestRegressorParams = {},
): Estimator<Supervised<Tensor, Tensor>, RandomForestRegressionModel> {
  const { trees: B = 100, bootstrap = true, maxFeatures = 'all', ...rest } = params
  const where = 'randomForestRegressor'
  return {
    name: 'random-forest-regressor',
    params: { trees: B, bootstrap, maxFeatures, ...rest },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, where)
      const yv = targets(y, n, where)
      const treeParams: TreeParams = { ...rest, ...(maxFeatures === 'all' ? {} : { maxFeatures }) }
      const training = trace(
        forestGrowth({ x, y, classes: 1, task: 'regression', params: treeParams, bootstrap }),
        undefined,
        B,
        {
          stream: options.stream,
          every: options.trace?.every ?? 1,
          checkpointEvery: options.trace?.checkpointEvery,
          keep: 'checkpoints',
        },
      )
      const trees = [...training.final.trees]
      const inBag = values(training.final.inBag)
      const treePredictions = (q: Tensor): Tensor => {
        const { n: m } = inputs(q, d, where)
        const out = new Float64Array(B * m)
        for (let b = 0; b < B; b++) out.set(values(predictTree(trees[b], q)), b * m)
        return fromData(out, [B, m])
      }
      const moments = (q: Tensor, { total = false }: ForestMomentsOptions = {}) => {
        const { n: m } = inputs(q, d, where)
        const p = values(treePredictions(q))
        const mean = new Float64Array(m)
        const variance = new Float64Array(m)
        for (let b = 0; b < B; b++) for (let i = 0; i < m; i++) mean[i] += p[b * m + i] / B
        for (let b = 0; b < B; b++) for (let i = 0; i < m; i++) variance[i] += (p[b * m + i] - mean[i]) ** 2 / B
        if (total)
          for (let b = 0; b < B; b++) {
            const leaves = applyTree(trees[b], q).data as Int32Array
            for (let i = 0; i < m; i++) variance[i] += trees[b].nodes[leaves[i]].impurity / B
          }
        return { mean: fromData(mean, [m]), variance: fromData(variance, [m]) }
      }
      const predict = (q: Tensor) => moments(q).mean
      // Out-of-bag: average only the trees whose bag missed the row.
      const oob = new Float64Array(n)
      const votes = new Float64Array(n)
      for (let b = 0; b < B; b++) {
        const p = values(predictTree(trees[b], x))
        for (let i = 0; i < n; i++) {
          if (inBag[b * n + i] > 0) continue
          votes[i]++
          oob[i] += p[i]
        }
      }
      let counted = 0
      let sse = 0
      let ybar = 0
      for (let i = 0; i < n; i++) {
        if (votes[i] === 0) {
          oob[i] = NaN
          continue
        }
        oob[i] /= votes[i]
        counted++
        ybar += yv[i]
        sse += (oob[i] - yv[i]) ** 2
      }
      ybar /= counted
      let sst = 0
      for (let i = 0; i < n; i++) if (votes[i] > 0) sst += (yv[i] - ybar) ** 2
      const importance = new Float64Array(d)
      for (const tree of trees) {
        const imp = values(featureImportances(tree))
        for (let j = 0; j < d; j++) importance[j] += (Number.isFinite(imp[j]) ? imp[j] : 0) / B
      }
      return {
        kind: 'model',
        name: 'random-forest-regressor',
        trees,
        inBag: training.final.inBag,
        featureImportances: fromData(importance, [d]),
        outOfBag: {
          predictions: fromData(oob, [n]),
          mse: counted ? sse / counted : NaN,
          r2: counted && sst > 0 ? 1 - sse / sst : NaN,
        },
        training,
        treePredictions,
        moments,
        forward: predict,
        decide: predict,
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'randomForestRegressor',
    module: 'learning/trees-and-ensembles/bagging',
    name: 'Random forest regressor',
    summary: 'Bagged regression trees; the spread of the trees’ predictions is the surrogate variance of SMAC.',
    task: 'regression',
    capabilities: ['forward', 'decide'],
    hyper: space({
      trees: int(1, 500, { default: 100 }),
      bootstrap: bool({ default: true }),
      maxDepth: int(1, 32, { default: 32, doc: 'The factory default is unlimited.' }),
      minSamplesSplit: int(2, 100, { default: 2 }),
      minSamplesLeaf: int(1, 100, { default: 1 }),
    }),
    cite: ['breiman2001', 'hutter2011smac', 'hutter2014'],
  },
  randomForestRegressor,
)
