/**
 * Bagging and random forests of CART classification trees (Breiman, 1996, "Bagging predictors"; Breiman, 2001,
 * "Random forests", Machine Learning 45), as scikit-learn's `RandomForestClassifier`.
 *
 * Each tree is grown on a bootstrap sample of the rows, drawn as integer sample weights (a row drawn twice weighs 2),
 * and searches a random subset of the features at every node; the forest averages the trees' leaf class shares. With
 * every feature searched it is bagging. The rows a tree's bootstrap missed give the out-of-bag estimate of accuracy.
 * The growth is a traceable algorithm, one tree per step, and each tree depends only on the root stream and its index.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type {
  AnyUnivariate,
  Decides,
  Estimator,
  FitOptions,
  Fitted,
  Predicts,
  Scores,
  Supervised,
  Trained,
} from 'aifn-compute/learning/estimators'
import { child, integers } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { featureImportances, growTree, predictTree, type DecisionTree, type TreeParams } from '../tree'
import { classLabels, classPredictive, inputs, matrix, probabilityModel, values } from '../../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, space } from 'aifn-compute/foundation/space'

/**
 * The problem a forest grows on: inputs ($n \times d$), labels $0, \dots, K - 1$, and the tree and bagging settings.
 */
export interface ForestProblem {
  /** The inputs, $n \times d$. */
  x: Tensor
  /** The class labels, $n$ integers from 0. */
  y: Tensor
  /** The number of classes $K$. */
  classes: number
  /** The growth hyperparameters of every tree (`maxFeatures` sets the features searched at each node). */
  params: TreeParams
  /** Grow each tree on a bootstrap sample (default true); otherwise on every row. */
  bootstrap?: boolean
}

/** A state of forest growth: the trees so far and their bootstrap counts. */
export interface ForestState extends Status {
  /** The number of trees grown. */
  t: number
  /** The trees grown, in order. */
  trees: readonly DecisionTree[]
  /** The bootstrap count of each training row in each tree's sample, $t \times n$ (all ones without bootstrap). */
  inBag: Tensor
}

/**
 * Random-forest growth as a traceable algorithm (Breiman, 2001, Machine Learning 45): each step grows one CART tree
 * on a bootstrap sample ($n$ rows drawn uniformly with replacement, as integer sample weights) drawn from
 * `child(ctx.stream, 'bootstrap')`, its feature subsets from the step's child stream `'features'`, so tree $t$ depends
 * only on the root key and $t$, and adding trees keeps the existing ones. No start.
 *
 * @param problem The data, the number of classes, the tree parameters and whether to bootstrap.
 * @returns The algorithm, whose state holds the trees grown so far and their bootstrap counts.
 *
 * @example Three trees, each on its own bootstrap sample
 * const x = tensor([[0, 0], [1, 1], [2, 0], [5, 5], [6, 4], [7, 5]])
 * const y = tensor([0, 0, 0, 1, 1, 1])
 * const alg = forestGrowth({ x, y, classes: 2, params: { maxFeatures: 1 } })
 * const state = run(alg, undefined, 3, { stream: stream(1) })
 * print('trees:', state.t)
 * print('bootstrap counts:', state.inBag)
 * print('root splits:', state.trees.map((tree) => tree.nodes[0].label))
 */
export function forestGrowth(problem: ForestProblem): Algorithm<void, ForestState> {
  const { x, y, classes: K, params, bootstrap = true } = problem
  const n = x.shape[0]
  return {
    name: 'random-forest-growth',
    init: () => ({ t: 0, trees: [], inBag: fromData(new Float64Array(0), [0, n]) }),
    step: (s, ctx) => {
      const counts = new Float64Array(n)
      if (bootstrap) {
        const b = child(ctx.stream, 'bootstrap')
        for (let r = 0; r < n; r++) counts[integers(b, n)]++
      } else counts.fill(1)
      const tree = growTree(child(ctx.stream, 'features'), {
        x,
        y,
        weights: fromData(Float64Array.from(counts), [n]),
        task: 'classification',
        classes: K,
        params,
      })
      const inBag = new Float64Array((s.t + 1) * n)
      inBag.set(values(s.inBag))
      inBag.set(counts, s.t * n)
      return { t: s.t + 1, trees: [...s.trees, tree], inBag: fromData(inBag, [s.t + 1, n]) }
    },
  }
}

// ── Random forests ───────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted random forest classifier. */
export interface RandomForestModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Trained<ForestState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'random-forest'
  /** The $T$ trees, in the order grown. */
  readonly trees: DecisionTree[]
  /** The bootstrap count of each training row in each tree's sample, $T \times n$ (all ones without bootstrap). */
  readonly inBag: Tensor
  /** The number of classes $K$. */
  readonly classes: number
  /** The mean of the trees' impurity importances, $d$ values summing to 1. */
  readonly featureImportances: Tensor
  /**
   * The out-of-bag class shares, $n \times K$ (each row averaged over the trees whose bag missed it; NaN for a row in
   * every bag), and the out-of-bag accuracy over the rows that have them (NaN when none has; always so without
   * bootstrap).
   */
  readonly outOfBag: { probabilities: Tensor; accuracy: number }
  /** The predictive law of the first `t` trees only (clamped to 1 to $T$), e.g. to show the forest growing. */
  predictiveUpTo(x: Tensor, t: number): AnyUnivariate
}

/**
 * A random forest (Breiman, 2001), as scikit-learn's `RandomForestClassifier`: `trees` CART trees, each grown on a
 * bootstrap sample (as integer sample weights) searching a random subset of `maxFeatures` features (default `'sqrt'`)
 * at every node. `forward`, `score` and the predictive average the trees' leaf class shares; `decide` takes the
 * largest. The growth (`forestGrowth`, one tree per step) is kept in `training`. If a node's feature subset has no
 * valid split, further features are drawn until one has (as scikit-learn), so a node is a leaf for want of a split
 * only when no feature splits it.
 *
 * @param params The tree hyperparameters (`TreeParams`, with `maxFeatures` defaulting to `'sqrt'`), `trees`, the number
 *   $T$ of trees (default 100), and `bootstrap`, whether each tree is grown on a bootstrap sample (default true) or on
 *   every row.
 * @returns An estimator whose `fit({ x, y }, { stream? })` grows the forest (from `stream(0)` when no stream is given)
 *   and returns the fitted `RandomForestModel`.
 *
 * @example Accuracy on two separable clusters
 * const s = stream(1)
 * const x = concat([normal(s, -2, 1, { shape: [30, 2] }), normal(stream(2), 2, 1, { shape: [30, 2] })], 0)
 * const y = tensor([...Array(30).fill(0), ...Array(30).fill(1)])
 * const model = randomForest({ trees: 20 }).fit({ x, y }, { stream: stream(3) })
 * const right = Array.from(toFlat(model.decide(x)), (c, i) => (c === toFlat(y)[i] ? 1 : 0))
 * print('training accuracy:', right.reduce((a, b) => a + b) / 60)
 * print('out-of-bag accuracy:', model.outOfBag.accuracy)
 * print('importances:', model.featureImportances)
 *
 * @example The predicted probability as trees are added
 * const x = tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * const model = randomForest({ trees: 30 }).fit({ x, y }, { stream: stream(1) })
 * for (const t of [1, 5, 30]) print(`P(class 1 at 3.5), ${t} trees:`, model.predictiveUpTo(tensor([[3.5]]), t).mean())
 */
export function randomForest(
  params: TreeParams & { trees?: number; bootstrap?: boolean } = {},
): Estimator<Supervised<Tensor, Tensor>, RandomForestModel> {
  const { trees: T = 100, bootstrap = true, maxFeatures = 'sqrt', ...rest } = params
  return {
    name: 'random-forest',
    params: { trees: T, bootstrap, maxFeatures, ...rest },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'randomForest')
      const { y: labels, k: K } = classLabels(y, n, 'randomForest')
      const training = trace(
        forestGrowth({ x, y, classes: K, params: { ...rest, maxFeatures }, bootstrap }),
        undefined,
        T,
        {
          stream: options.stream,
          every: options.trace?.every ?? 1,
          checkpointEvery: options.trace?.checkpointEvery,
          keep: 'checkpoints',
        },
      )
      const trees = training.final.trees
      const inBag = values(training.final.inBag)
      const average = (q: Tensor, upTo: number) => {
        const { n: m } = inputs(q, d, 'randomForest')
        const out = new Float64Array(m * K)
        for (let t = 0; t < upTo; t++) {
          const p = values(predictTree(trees[t], q))
          for (let j = 0; j < out.length; j++) out[j] += p[j] / upTo
        }
        return out
      }
      // Out-of-bag: average only the trees whose bag missed the row.
      const oob = new Float64Array(n * K)
      const votes = new Float64Array(n)
      for (let t = 0; t < T; t++) {
        const p = values(predictTree(trees[t], x))
        for (let i = 0; i < n; i++) {
          if (inBag[t * n + i] > 0) continue
          votes[i]++
          for (let c = 0; c < K; c++) oob[i * K + c] += p[i * K + c]
        }
      }
      let correct = 0
      let counted = 0
      for (let i = 0; i < n; i++) {
        if (votes[i] === 0) {
          for (let c = 0; c < K; c++) oob[i * K + c] = NaN
          continue
        }
        let best = 0
        for (let c = 0; c < K; c++) {
          oob[i * K + c] /= votes[i]
          if (oob[i * K + c] > oob[i * K + best]) best = c
        }
        counted++
        if (best === labels[i]) correct++
      }
      const importance = new Float64Array(d)
      for (const tree of trees) {
        const imp = values(featureImportances(tree))
        for (let j = 0; j < d; j++) importance[j] += imp[j] / T
      }
      return {
        kind: 'model',
        name: 'random-forest',
        trees: [...trees],
        inBag: training.final.inBag,
        classes: K,
        featureImportances: fromData(importance, [d]),
        outOfBag: { probabilities: fromData(oob, [n, K]), accuracy: counted ? correct / counted : NaN },
        training,
        predictiveUpTo: (q: Tensor, t: number) =>
          classPredictive(average(q, Math.max(1, Math.min(t, T))), q.shape[0], K),
        ...probabilityModel(
          (q) => average(q, T),
          (h) => h,
          K,
        ),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'randomForest',
    module: 'learning/trees-and-ensembles/bagging',
    name: 'Random forest',
    summary: 'Bagged classification trees with random feature subsets at each split.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({
      trees: int(1, 500, { default: 100 }),
      bootstrap: bool({ default: true }),
      maxDepth: int(1, 32, { default: 32, doc: 'The factory default is unlimited.' }),
      minSamplesLeaf: int(1, 100, { default: 1 }),
    }),
    notes: ['random-forest'],
    cite: ['breiman2001'],
  },
  randomForest,
)
