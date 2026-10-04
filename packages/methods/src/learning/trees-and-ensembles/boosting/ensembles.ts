/**
 * Tree estimators and tree ensembles:
 *
 * - `decisionTree`, `regressionTree`: CART as estimators (growth traced, optional cost-complexity pruning).
 * - `randomForest`: bagged trees with random feature subsets per node (Breiman, 2001, "Random forests").
 * - `adaBoostSteps`, `adaBoost`: multiclass AdaBoost by SAMME (Zhu, Zou, Rosset and Hastie, 2009), which is
 *   Freund and Schapire's (1997) AdaBoost.M1 for two classes.
 * - `gradientBoostingSteps`, `gradientBoosting`: gradient tree boosting (Friedman, 2001, "Greedy function
 *   approximation: a gradient boosting machine") for squared error and the logistic (binomial and multinomial)
 *   deviance, with Newton leaf values as scikit-learn's `GradientBoostingClassifier`.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import {
  withExpectation,
  type AnyUnivariate,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Scores,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { integers } from 'aifn-compute/foundation/random'
import { dense, fromData, logsumexp, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { sigmoid } from 'aifn-compute/numerics/special'
import { Normal } from 'aifn-compute/probability/distributions'
import { applyTree, growTree, nodeLabel, predictTree, type DecisionTree, type TreeParams } from '../tree'
import { classLabels, inputs, matrix, probabilityModel, softmaxRows, targets, values } from '../../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'

// ── AdaBoost (SAMME) ─────────────────────────────────────────────────────────────────────────────────────────────

/** The problem an AdaBoost run solves. */
export interface AdaBoostProblem {
  x: Tensor
  /** Labels 0 … K−1. */
  y: Tensor
  /** The weak learner's tree parameters (default stumps: `{ maxDepth: 1 }`). */
  base?: TreeParams
  learningRate?: number
}

/** One AdaBoost state: the learners so far, their weights and the current sample weights. */
export interface AdaBoostState extends Status {
  /** Rounds done. */
  t: number
  learners: DecisionTree[]
  /** The weight α_m of each learner. */
  alphas: number[]
  /** The weighted training error of each learner. */
  errors: number[]
  /** Sample weights for the next round [n] (sum 1). */
  sampleWeights: Tensor
  /** The ensemble's votes Σ α_m 1[h_m(x) = k] on the training rows [n, K], and its training error. */
  votes: Tensor
  trainingError: number
  /** A learner had zero error (it is kept) or no better than chance (the run stops without it). */
  stopped: 'perfect' | 'chance' | null
}

/**
 * AdaBoost by SAMME as a traceable algorithm: each round fits a weighted tree, weights it by
 * α = η (log((1 − err)/err) + log(K − 1)), multiplies the weights of misclassified rows by e^α and renormalises. It
 * is done when a learner is perfect or no better than chance (err ≥ 1 − 1/K).
 */
export function adaBoostSteps(problem: AdaBoostProblem): Algorithm<void, AdaBoostState> {
  const { n } = matrix(problem.x, 'adaBoostSteps')
  const { y, k: K } = classLabels(problem.y, n, 'adaBoostSteps')
  const eta = problem.learningRate ?? 1
  const base = problem.base ?? { maxDepth: 1 }
  return {
    name: 'adaboost-samme',
    init: () => ({
      t: 0,
      learners: [],
      alphas: [],
      errors: [],
      sampleWeights: fromData(new Float64Array(n).fill(1 / n), [n]),
      votes: fromData(new Float64Array(n * K), [n, K]),
      trainingError: NaN,
      stopped: null,
    }),
    step: (state) => {
      const w = values(state.sampleWeights)
      const tree = growTree(undefined, {
        x: problem.x,
        y: problem.y,
        weights: state.sampleWeights,
        task: 'classification',
        classes: K,
        params: base,
      })
      const probs = values(predictTree(tree, problem.x))
      const pred = new Int32Array(n)
      let err = 0
      let total = 0
      for (let i = 0; i < n; i++) {
        let best = 0
        for (let c = 1; c < K; c++) if (probs[i * K + c] > probs[i * K + best]) best = c
        pred[i] = best
        total += w[i]
        if (best !== y[i]) err += w[i]
      }
      err /= total
      if (err >= 1 - 1 / K) return { ...state, t: state.t + 1, stopped: 'chance' }
      const perfect = err <= 0
      // A perfect learner gets weight 1 and ends the run, as in scikit-learn.
      const alpha = perfect ? 1 : eta * (Math.log((1 - err) / err) + Math.log(K - 1))
      const next = new Float64Array(n)
      let z = 0
      for (let i = 0; i < n; i++) z += next[i] = w[i] * (!perfect && pred[i] !== y[i] ? Math.exp(alpha) : 1)
      for (let i = 0; i < n; i++) next[i] /= z
      const votes = Float64Array.from(values(state.votes))
      let wrong = 0
      for (let i = 0; i < n; i++) {
        votes[i * K + pred[i]] += alpha
        let best = 0
        for (let c = 1; c < K; c++) if (votes[i * K + c] > votes[i * K + best]) best = c
        if (best !== y[i]) wrong++
      }
      return {
        t: state.t + 1,
        learners: [...state.learners, tree],
        alphas: [...state.alphas, alpha],
        errors: [...state.errors, err],
        sampleWeights: fromData(next, [n]),
        votes: fromData(votes, [n, K]),
        trainingError: wrong / n,
        stopped: perfect ? 'perfect' : null,
      }
    },
    done: (state) => state.stopped !== null,
  }
}

/** A fitted AdaBoost classifier. */
export interface AdaBoostModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<AdaBoostState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'adaboost'
  readonly learners: DecisionTree[]
  readonly alphas: Tensor
  readonly errors: Tensor
  readonly classes: number
  /** Normalised votes Σ α_m 1[h_m(x) = k] / Σ α_m from the first `rounds` learners [m, K]. */
  votesUpTo(x: Tensor, rounds: number): Tensor
}

/**
 * AdaBoost (SAMME) with `rounds` weak learners (default 50 stumps). `forward` and `score` are the normalised votes
 * [m, K]; `decide` is the class with the most weight.
 */
export function adaBoost(
  params: { rounds?: number; learningRate?: number; base?: TreeParams } = {},
): Estimator<Supervised<Tensor, Tensor>, AdaBoostModel> {
  const { rounds = 50, learningRate = 1, base = { maxDepth: 1 } } = params
  return {
    name: 'adaboost',
    params: { rounds, learningRate, base },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'adaBoost')
      const { k: K } = classLabels(y, n, 'adaBoost')
      const training = trace(adaBoostSteps({ x, y, base, learningRate }), undefined, rounds, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        stopOnNonFinite: false,
        record: {
          trainingError: (s) => s.trainingError,
          ...(options.trace?.record as Record<string, (s: AdaBoostState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const votesUpTo = (q: Tensor, upTo: number) => {
        const { n: m } = inputs(q, d, 'adaBoost')
        const out = new Float64Array(m * K)
        const R = Math.min(upTo, final.learners.length)
        let total = 0
        for (let r = 0; r < R; r++) {
          total += final.alphas[r]
          const p = values(predictTree(final.learners[r], q))
          for (let i = 0; i < m; i++) {
            let best = 0
            for (let c = 1; c < K; c++) if (p[i * K + c] > p[i * K + best]) best = c
            out[i * K + best] += final.alphas[r]
          }
        }
        if (total > 0) for (let j = 0; j < out.length; j++) out[j] /= total
        return out
      }
      const model = probabilityModel(
        (q) => votesUpTo(q, Infinity),
        (h) => h,
        K,
      )
      return {
        kind: 'model',
        name: 'adaboost',
        learners: final.learners,
        alphas: fromData(Float64Array.from(final.alphas), [final.alphas.length]),
        errors: fromData(Float64Array.from(final.errors), [final.errors.length]),
        classes: K,
        training,
        votesUpTo: (q: Tensor, r: number) => fromData(votesUpTo(q, r), [q.shape[0], K]),
        forward: model.forward,
        score: model.score,
        decide: model.decide,
      }
    },
  }
}

// ── Gradient boosting ────────────────────────────────────────────────────────────────────────────────────────────

/** The loss gradient boosting minimises. */
export type BoostingLoss = 'squared' | 'logistic'

/** The problem a gradient boosting run solves. */
export interface GradientBoostingProblem {
  x: Tensor
  /** Real targets (squared) or labels 0 … K−1 (logistic). */
  y: Tensor
  loss: BoostingLoss
  learningRate?: number
  /** Tree parameters for each stage (default `{ maxDepth: 3 }`). */
  tree?: Omit<TreeParams, 'criterion'>
  /** Fraction of rows each stage fits on, drawn without replacement (stochastic gradient boosting; default 1). */
  subsample?: number
}

/** One state of gradient boosting. */
export interface GradientBoostingState extends Status {
  /** Stages done. */
  t: number
  /** The initial raw prediction F₀ ([1] for squared or binary logistic, [K] for multinomial). */
  initial: number[]
  /** Trees per stage: one, or K for the multinomial loss. Leaf values are the stage's Newton steps (before η). */
  stages: DecisionTree[][]
  /** The raw prediction F on the training rows: [n] or [n, K]. */
  raw: Tensor
  /** The negative gradients the latest stage fitted (the pseudo-residuals): [n] or [n, K]. */
  residuals: Tensor
  /** The training loss: half the mean squared error, or the mean negative log-likelihood. */
  loss: number
}

/**
 * Gradient tree boosting as a traceable algorithm. Each stage computes the pseudo-residuals rᵢ = −∂L/∂F(xᵢ) (yᵢ − Fᵢ
 * for squared error, yᵢ − pᵢ for the logistic deviance), fits a regression tree to them (squared-error splits), sets
 * each leaf to a Newton step (the mean residual for squared error; Σr / Σp(1 − p) for the binomial loss, times
 * (K − 1)/K per class for the multinomial one), and adds η times the tree to F. With `subsample` < 1 the stage's
 * rows are drawn from the step's stream. No start.
 */
export function gradientBoostingSteps(problem: GradientBoostingProblem): Algorithm<void, GradientBoostingState> {
  const { n } = matrix(problem.x, 'gradientBoostingSteps')
  const eta = problem.learningRate ?? 0.1
  const treeParams = problem.tree ?? { maxDepth: 3 }
  const subsample = problem.subsample ?? 1
  const logistic = problem.loss === 'logistic'
  const lab = logistic ? classLabels(problem.y, n, 'gradientBoostingSteps') : null
  const K = lab ? lab.k : 1
  const C = logistic && K > 2 ? K : 1 // raw columns
  const y = logistic ? Float64Array.from(lab!.y) : targets(problem.y, n, 'gradientBoostingSteps')
  const probs = (F: Float64Array): Float64Array =>
    C === 1 ? Float64Array.from(F, (f) => sigmoid(f)) : softmaxRows(F, n, C)
  const lossOf = (F: Float64Array): number => {
    let s = 0
    if (!logistic) {
      for (let i = 0; i < n; i++) s += 0.5 * (y[i] - F[i]) ** 2
      return s / n
    }
    if (C === 1) {
      for (let i = 0; i < n; i++) s += Math.max(F[i], 0) + Math.log1p(Math.exp(-Math.abs(F[i]))) - y[i] * F[i]
      return s / n
    }
    const lse = dense.data(logsumexp(fromData(F, [n, C]), 1))
    for (let i = 0; i < n; i++) s += lse[i] - F[i * C + y[i]]
    return s / n
  }
  const residualsOf = (F: Float64Array): Float64Array => {
    if (!logistic) return Float64Array.from(F, (f, i) => y[i] - f)
    const p = probs(F)
    if (C === 1) return Float64Array.from(p, (q, i) => y[i] - q)
    return Float64Array.from(p, (q, j) => (y[Math.floor(j / C)] === j % C ? 1 : 0) - q)
  }
  return {
    name: 'gradient-boosting',
    init: () => {
      let initial: number[]
      if (!logistic) initial = [y.reduce((a, b) => a + b, 0) / n]
      else if (C === 1) {
        const p = y.reduce((a, b) => a + b, 0) / n
        initial = [Math.log(p / (1 - p))]
      } else {
        const counts = new Float64Array(C)
        for (const c of y) counts[c]++
        const logs = Array.from(counts, (c) => Math.log(c / n))
        const mean = logs.reduce((a, b) => a + b, 0) / C
        initial = logs.map((l) => l - mean)
      }
      const F = new Float64Array(n * C)
      for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) F[i * C + c] = initial[c]
      const shape = C === 1 ? [n] : [n, C]
      return {
        t: 0,
        initial,
        stages: [],
        raw: fromData(F, shape),
        residuals: fromData(residualsOf(F), shape),
        loss: lossOf(F),
      }
    },
    step: (state, ctx) => {
      const F = Float64Array.from(values(state.raw))
      const r = residualsOf(F)
      const p = logistic ? probs(F) : null
      // Rows this stage fits on.
      let rowWeights: Float64Array | undefined
      if (subsample < 1) {
        const s = ctx.stream
        const order = Array.from({ length: n }, (_, i) => i)
        const take = Math.max(1, Math.floor(subsample * n))
        for (let a = 0; a < take; a++) {
          const b = a + integers(s, n - a)
          ;[order[a], order[b]] = [order[b], order[a]]
        }
        rowWeights = new Float64Array(n)
        for (let a = 0; a < take; a++) rowWeights[order[a]] = 1
      }
      const trees: DecisionTree[] = []
      for (let c = 0; c < C; c++) {
        const rc = C === 1 ? r : Float64Array.from({ length: n }, (_, i) => r[i * C + c])
        const tree = growTree(undefined, {
          x: problem.x,
          y: fromData(rc, [n]),
          weights: rowWeights ? fromData(rowWeights, [n]) : undefined,
          task: 'regression',
          params: { ...treeParams, criterion: 'squared' },
        })
        const leaves = values(applyTree(tree, problem.x))
        if (logistic) {
          // Newton leaf values: Σr / Σ p(1 − p) over the leaf's (in-sample) rows, times (K − 1)/K for K classes.
          const num = new Float64Array(tree.nodes.length)
          const den = new Float64Array(tree.nodes.length)
          for (let i = 0; i < n; i++) {
            if (rowWeights && rowWeights[i] === 0) continue
            const q = p![i * C + c]
            num[leaves[i]] += rc[i]
            den[leaves[i]] += q * (1 - q)
          }
          const factor = C === 1 ? 1 : (C - 1) / C
          tree.nodes = tree.nodes.map((node) =>
            node.children.length
              ? node
              : (() => {
                  const value = [Math.abs(den[node.id]) < 1e-150 ? 0 : (factor * num[node.id]) / den[node.id]]
                  return { ...node, value, label: nodeLabel({ ...node, value }, 'regression') }
                })(),
          )
        }
        for (let i = 0; i < n; i++) F[i * C + c] += eta * tree.nodes[leaves[i]].value[0]
        trees.push(tree)
      }
      const shape = C === 1 ? [n] : [n, C]
      const loss = lossOf(F)
      return {
        t: state.t + 1,
        initial: state.initial,
        stages: [...state.stages, trees],
        raw: fromData(F, shape),
        residuals: fromData(r, shape),
        loss,
        diverged: !Number.isFinite(loss),
      }
    },
  }
}

/** A fitted gradient boosting model. */
export interface GradientBoostingModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Trained<GradientBoostingState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gradient-boosting'
  readonly loss: BoostingLoss
  readonly learningRate: number
  readonly initial: number[]
  readonly stages: DecisionTree[][]
  /** Classes for the logistic loss (1 for squared error). */
  readonly classes: number
  /** The raw prediction F(x) after the first `stages` stages: [m] or [m, K]. */
  rawUpTo(x: Tensor, stages: number): Tensor
  /**
   * Logistic loss: the class law (Bernoulli or Categorical). Squared error: the plug-in Gaussian N(F(x), σ̂²), σ̂² the
   * training mean squared error.
   */
  predictive(x: Tensor): AnyUnivariate
  /** E[f(y) | x] under the predictive; F(x) for squared error without f. */
  expect: Expects<Tensor>['expect']
  score?(x: Tensor): Tensor
}

/**
 * Gradient boosting (default 100 stages of depth-3 trees, η = 0.1). Squared error: `forward`, `decide` and `expect`
 * give F(x), `predictive` the plug-in Gaussian. Logistic: `forward`/`score` give F(x) (the log-odds [m] for two
 * classes, [m, K] scores otherwise), `predictive` the class law, `decide` the most probable class.
 */
export function gradientBoosting(
  params: {
    loss?: BoostingLoss
    stages?: number
    learningRate?: number
    tree?: Omit<TreeParams, 'criterion'>
    subsample?: number
  } = {},
): Estimator<Supervised<Tensor, Tensor>, GradientBoostingModel> {
  const { loss = 'squared', stages = 100, learningRate = 0.1, tree = { maxDepth: 3 }, subsample = 1 } = params
  return {
    name: 'gradient-boosting',
    params: { loss, stages, learningRate, tree, subsample },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'gradientBoosting')
      const K = loss === 'logistic' ? classLabels(y, n, 'gradientBoosting').k : 1
      const C = K > 2 ? K : 1
      const training = trace(gradientBoostingSteps({ x, y, loss, learningRate, tree, subsample }), undefined, stages, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          loss: (s) => s.loss,
          ...(options.trace?.record as Record<string, (s: GradientBoostingState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const rawUpTo = (q: Tensor, upTo: number): Float64Array => {
        const { n: m } = inputs(q, d, 'gradientBoosting')
        const F = new Float64Array(m * C)
        for (let i = 0; i < m; i++) for (let c = 0; c < C; c++) F[i * C + c] = final.initial[c]
        const S = Math.min(upTo, final.stages.length)
        for (let s = 0; s < S; s++) {
          for (let c = 0; c < C; c++) {
            const v = values(predictTree(final.stages[s][c], q))
            for (let i = 0; i < m; i++) F[i * C + c] += learningRate * v[i]
          }
        }
        return F
      }
      const shape = (m: number) => (C === 1 ? [m] : [m, C])
      const common = {
        kind: 'model' as const,
        name: 'gradient-boosting' as const,
        loss,
        learningRate,
        initial: final.initial,
        stages: final.stages,
        classes: K,
        training,
        rawUpTo: (q: Tensor, s: number) => fromData(rawUpTo(q, s), shape(q.shape[0])),
      }
      const forward = (q: Tensor) => fromData(rawUpTo(q, Infinity), shape(q.shape[0]))
      if (loss === 'squared') {
        // The training loss is half the mean squared error.
        const sd = Math.sqrt(2 * final.loss)
        return withExpectation({
          ...common,
          forward,
          decide: forward,
          predictive: (q: Tensor) => {
            const mean = forward(q)
            return Normal(mean, fromData(new Float64Array(mean.shape[0]).fill(sd), mean.shape)) as AnyUnivariate
          },
        })
      }
      const toProbs = (F: Float64Array, m: number): Float64Array => {
        if (C > 1) return softmaxRows(F, m, C)
        const out = new Float64Array(2 * m)
        for (let i = 0; i < m; i++) {
          out[2 * i + 1] = sigmoid(F[i]) as number
          out[2 * i] = 1 - out[2 * i + 1]
        }
        return out
      }
      const model = probabilityModel((q) => rawUpTo(q, Infinity), toProbs, K)
      return withExpectation({
        ...common,
        forward,
        score: forward,
        decide: model.decide,
        predictive: model.predictive,
      })
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'adaBoost',
    module: 'learning/trees-and-ensembles/boosting',
    name: 'AdaBoost',
    summary: 'SAMME boosting of shallow trees (stumps by default).',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({ rounds: int(1, 1000, { default: 50 }), learningRate: real(1e-3, 2, { default: 1, scale: 'log' }) }),
    notes: ['adaboost'],
    cite: ['freund1997'],
  },
  adaBoost,
)

defineModel(
  {
    key: 'gradientBoosting',
    module: 'learning/trees-and-ensembles/boosting',
    name: 'Gradient boosting',
    summary: 'Stagewise fitting of regression trees to the negative gradient of a squared or logistic loss.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect'],
    hyper: space({
      loss: oneOf(['squared', 'logistic']),
      stages: int(1, 1000, { default: 100 }),
      learningRate: real(1e-3, 1, { default: 0.1, scale: 'log' }),
      subsample: real(0.1, 1, { default: 1 }),
    }),
    notes: ['gradient-boosting'],
    cite: ['friedman2001'],
  },
  gradientBoosting,
)
