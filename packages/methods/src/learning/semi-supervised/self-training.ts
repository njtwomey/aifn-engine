/**
 * Self-training as a traceable algorithm (Scudder, 1965; Yarowsky, 1995; Lee, 2013 for the deep, thresholded form;
 * Xie et al., 2020 for the noisy student), and the row helpers the module's other wrappers share.
 *
 * Round 0 fits a probabilistic classifier to the labelled points alone. Every round then predicts all points, proposes
 * a pseudo-label (the arg max class) for each unlabelled point whose largest class probability reaches the threshold
 * $\tau$, optionally keeping only the most confident `perClass` of each class, and the next round refits on the
 * labelled points plus the proposed pseudo-labels. With `keep: 'recompute'` (default) the pseudo-labels are proposed
 * afresh each round, so a point can change class or drop out; with `keep: 'accumulate'` a pseudo-label, once given,
 * stays fixed (the classic wrapper of Yarowsky and of Blum and Mitchell). From round 1 on an optional `student`
 * estimator replaces the base learner and an optional `augment` function adds noise to the training set before each
 * fit: with both, the loop is Noisy Student training (Xie et al., 2020, Algorithm 1), whose student is fitted on noised
 * data and becomes the next teacher. True labels, when given, are used only for diagnostics: the accuracy of the
 * pseudo-labels and of the model on the unlabelled points. States are plain data (probabilities and label arrays), so a
 * trace can be scrubbed with a Player; the probabilities on extra `evaluate` inputs (a grid or a test set) are kept in
 * every state.
 *
 * Class probabilities are row-major $n \times K$ `Float64Array`s throughout: row $i$ is entries $iK$ to $iK + K - 1$.
 */

import type { Status, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child } from 'aifn-compute/foundation/random'
import { shapeOf, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  classProbabilities,
  dataset,
  takeRows,
  type Estimator,
  type Predicts,
  type Supervised,
} from 'aifn-compute/learning/estimators'

/** A labelled training set, as the estimators fit it. */
export type TrainingSet = Supervised<Tensor, Tensor>

/** The data of a self-training run. */
export interface SelfTrainingProblem {
  /** All inputs, labelled and unlabelled, $n \times d$. */
  x: Tensor
  /** One entry per row of `x`: a class $0, \dots, K - 1$ for a labelled point, $-1$ for an unlabelled one. */
  labels: ArrayLike<number>
  /** The number of classes $K \ge 2$. */
  classes: number
  /** The true classes of all $n$ points, for diagnostics only (never used to fit). */
  truth?: ArrayLike<number>
  /** Further inputs ($m \times d$, a grid or a test set) whose class probabilities are recorded every round. */
  evaluate?: Tensor
  /** The true classes of the `evaluate` inputs, for the recorded test accuracy. */
  evaluateTruth?: ArrayLike<number>
}

/** Controls of `selfTrainingSteps`. */
export interface SelfTrainingOptions {
  /** The base learner, fitted in round 0 (and every round without a `student`); its models give class laws. */
  base: Estimator<TrainingSet, Predicts<Tensor>>
  /** The learner of rounds 1 and later (Noisy Student's student); default `base`. */
  student?: Estimator<TrainingSet, Predicts<Tensor>>
  /** Confidence threshold $\tau \in [0, 1]$ on the largest class probability. Default 0.9. */
  threshold?: number
  /** At most this many pseudo-labels per class per round, the most confident first. Default no cap. */
  perClass?: number
  /** `recompute` (default): pseudo-labels proposed afresh every round; `accumulate`: once given, kept. */
  keep?: 'recompute' | 'accumulate'
  /**
   * Noise for the student: maps the training set of a round 1 or later fit to the set actually fitted (augmented
   * copies, jittered inputs). Gets the round's child stream `'augment'`. Default none.
   */
  augment?: (data: TrainingSet, s: Stream) => TrainingSet
}

/** One round of self-training. */
export interface SelfTrainingState extends Status {
  /** The round: 0 is the fit on the labelled points alone. */
  t: number
  /** Class probabilities of the round's model on all $n$ points, row-major $n \times K$. */
  probs: Float64Array
  /** The label each point was fitted with this round: its label, its pseudo-label, or $-1$ when left out. */
  fitted: Int32Array
  /** The pseudo-label this round's model proposes for the next fit ($-1$: none, always for labelled points). */
  proposed: Int32Array
  /** Number of proposed pseudo-labels. */
  proposedCount: number
  /** Number of proposed pseudo-labels per class. */
  proposedPerClass: number[]
  /** Fraction of the proposed pseudo-labels that are wrong (needs `truth`; NaN otherwise or when none). */
  pseudoLabelError: number
  /** Accuracy of the round's arg max decisions on the unlabelled points (needs `truth`; NaN otherwise). */
  unlabelledAccuracy: number
  /** Class probabilities on the `evaluate` inputs, row-major $m \times K$ (empty without them). */
  evaluateProbs: Float64Array
  /** Accuracy on the `evaluate` inputs (needs `evaluateTruth`; NaN otherwise). */
  evaluateAccuracy: number
  /** The proposed pseudo-labels equal those fitted this round: a further round would refit the same data. */
  converged: boolean
}

/**
 * The arg max class and the largest probability of each row of a row-major $n \times K$ matrix (ties go to the lower
 * class).
 *
 * @param p The probabilities, $nK$ values row-major; not modified.
 * @param K The number of classes (columns).
 * @returns `cls`, the arg max of each row, and `conf`, its probability.
 *
 * @example The most probable class of each row
 * const { cls, conf } = rowArgmax([0.2, 0.8, 0.6, 0.4, 0.5, 0.5], 2)
 * print('classes:', cls)
 * print('confidences:', conf)
 */
export function rowArgmax(p: ArrayLike<number>, K: number): { cls: Int32Array; conf: Float64Array } {
  const n = p.length / K
  const cls = new Int32Array(n)
  const conf = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let k = 1; k < K; k++) if (p[i * K + k] > p[i * K + best]) best = k
    cls[i] = best
    conf[i] = p[i * K + best]
  }
  return { cls, conf }
}

/**
 * The class probabilities of a fitted model on `x`, flattened row-major: `classProbabilities` of its predictive law.
 * Throws `ShapeError` when the model does not give $K$ probabilities per row.
 *
 * @param model A fitted model whose predictive distribution is a Bernoulli or categorical law.
 * @param x The inputs, $n \times d$.
 * @param K The number of classes the model should give.
 * @returns $nK$ probabilities, row $i$ at entries $iK$ to $iK + K - 1$.
 *
 * @example Probabilities of a model whose predictive is a Bernoulli law
 * import { bernoulliPredictive } from 'aifn-compute/learning/estimators'
 * const model = { predictive: (q) => bernoulliPredictive(tensor(toArray(q).map((r) => 1 / (1 + Math.exp(-r[0]))))) }
 * print('rows of [P(0), P(1)]:', probabilities(model, tensor([[-1], [0], [1]]), 2))
 */
export function probabilities(model: Predicts<Tensor>, x: Tensor, K: number): Float64Array {
  const p = Float64Array.from(toFlat(classProbabilities(model.predictive(x))))
  if (p.length !== shapeOf(x)[0] * K)
    throw new ShapeError('probabilities', 'probabilities: the model does not give K class probabilities per row')
  return p
}

/**
 * The accuracy of the arg max decisions of row-major class probabilities against true classes, over the rows listed
 * (all rows when left out). NaN when no row is listed.
 *
 * @param probs The class probabilities, row-major $n \times K$; not modified.
 * @param K The number of classes.
 * @param truth The true class of every row (indexed by row, not by position in `rows`).
 * @param rows The rows to score; all $n$ when left out.
 * @returns The share of the listed rows whose arg max equals the truth.
 *
 * @example Two of three rows right
 * print('accuracy:', accuracyOf(Float64Array.from([0.9, 0.1, 0.3, 0.7, 0.6, 0.4]), 2, [0, 1, 1]))
 * print('on rows 0 and 1:', accuracyOf(Float64Array.from([0.9, 0.1, 0.3, 0.7, 0.6, 0.4]), 2, [0, 1, 1], [0, 1]))
 */
export function accuracyOf(probs: Float64Array, K: number, truth: ArrayLike<number>, rows?: ArrayLike<number>): number {
  const { cls } = rowArgmax(probs, K)
  const index = rows ?? Array.from({ length: cls.length }, (_, i) => i)
  if (index.length === 0) return NaN
  let right = 0
  for (let a = 0; a < index.length; a++) if (cls[index[a]] === truth[index[a]]) right++
  return right / index.length
}

/**
 * The training set of the rows that hold a label: the rows of `x` whose entry in `label` is a class (not $-1$), in
 * increasing order, with those classes as the targets.
 *
 * @param x The inputs, $n \times d$.
 * @param label One entry per row: a class $0, \dots, K - 1$, or $-1$ to leave the row out.
 * @returns The labelled rows and their classes, ready for an estimator's `fit`.
 *
 * @example Rows 0 and 2 hold labels
 * const d = labelledSet(tensor([[1, 2], [3, 4], [5, 6]]), [1, -1, 0])
 * print('x:', d.x)
 * print('y:', d.y)
 */
export function labelledSet(x: Tensor, label: ArrayLike<number>): TrainingSet {
  const rows: number[] = []
  const ys: number[] = []
  for (let i = 0; i < label.length; i++)
    if (label[i] >= 0) {
      rows.push(i)
      ys.push(label[i])
    }
  return dataset(takeRows(x, rows), tensor(ys))
}

/**
 * Self-training (and pseudo-labelling, and Noisy Student) as an algorithm over rounds; see the file comment. Each
 * round's fit draws from `child(stream, 'fit')` and the student noise from `child(stream, 'augment')` of the round's
 * stream. The run converges when the proposed pseudo-labels equal those just fitted. Throws `ShapeError` when `x` and
 * `labels` differ in rows and `DomainError` for a threshold outside $[0, 1]$. Options hold estimators and a function,
 * so the algorithm runs on the page's thread (a few milliseconds per round on a thousand points).
 *
 * @param problem All inputs, the labels with $-1$ for unlabelled points, the number of classes and optional truth.
 * @param options The base learner, the threshold, the per-class cap, how pseudo-labels are kept, and student noise.
 * @returns The algorithm; `trace(alg, undefined, rounds)` records one state per round.
 *
 * @example Two labelled points and many unlabelled ones
 * import { bernoulliPredictive } from 'aifn-compute/learning/estimators'
 * // A tiny learner on the first feature: a logistic curve through the midpoint of the two class means.
 * const mean = (u, c, k) => u.filter((_, i) => c[i] === k).reduce((a, b) => a + b, 0) / c.filter((v) => v === k).length
 * const midpoint = {
 *   name: 'midpoint',
 *   fit: ({ x, y }) => {
 *     const u = toArray(x).map((r) => r[0])
 *     const t = (mean(u, toArray(y), 0) + mean(u, toArray(y), 1)) / 2
 *     const p = (q) => tensor(toArray(q).map((r) => 1 / (1 + Math.exp(4 * (t - r[0])))))
 *     return { predictive: (q) => bernoulliPredictive(p(q)) }
 *   },
 * }
 * const u = [-0.5, 2.5, -3.1, -2.6, -2.2, -1.7, -1.3, -0.9, 0.2, 1.1, 1.6, 2.1, 2.4, 2.9, 3.3, 3.8]
 * const truth = u.map((v, i) => (i < 2 ? i : v < 0.6 ? 0 : 1))
 * const labels = truth.map((v, i) => (i < 2 ? v : -1))
 * const x = tensor(u.map((v) => [v]))
 * const alg = selfTrainingSteps({ x, labels, classes: 2, truth }, { base: midpoint })
 * const t = trace(alg, undefined, 30, { keep: 'all' })
 * print('pseudo-labels per round:', t.steps.map((s) => s.proposedCount))
 * print('final round:', t.final.t, ' accuracy on the unlabelled points:', t.final.unlabelledAccuracy)
 */
export function selfTrainingSteps(
  problem: SelfTrainingProblem,
  options: SelfTrainingOptions,
): Algorithm<void, SelfTrainingState> {
  const { x, labels, classes: K, truth, evaluate, evaluateTruth } = problem
  const tau = options.threshold ?? 0.9
  const cap = options.perClass ?? Infinity
  const keep = options.keep ?? 'recompute'
  const n = labels.length
  if (shapeOf(x)[0] !== n) throw new ShapeError('selfTrainingSteps', 'selfTrainingSteps: x and labels differ in rows')
  if (!(tau >= 0 && tau <= 1))
    throw new DomainError('selfTrainingSteps', 'selfTrainingSteps: threshold must be in [0, 1]')
  const unlabelled: number[] = []
  for (let i = 0; i < n; i++) if (labels[i] < 0) unlabelled.push(i)

  // Fit the round's learner on the points with a label (given or pseudo), after the student noise from round 1 on.
  const fitRound = (fitted: Int32Array, round: number, s: Stream) => {
    let data = labelledSet(x, fitted)
    const learner = round > 0 && options.student ? options.student : options.base
    if (round > 0 && options.augment) data = options.augment(data, child(s, 'augment'))
    return learner.fit(data, { stream: child(s, 'fit') })
  }

  // The state of a round from its fitted model: probabilities, the proposals for the next fit and the diagnostics.
  const describe = (
    t: number,
    fitted: Int32Array,
    model: Predicts<Tensor>,
    previous: Int32Array | null,
  ): SelfTrainingState => {
    const probs = probabilities(model, x, K)
    const { cls, conf } = rowArgmax(probs, K)
    const proposed = new Int32Array(n).fill(-1)
    // Candidates by class, most confident first; accumulated pseudo-labels stay as they were.
    const byClass: number[][] = Array.from({ length: K }, () => [])
    for (const i of unlabelled) {
      if (keep === 'accumulate' && previous && previous[i] >= 0) proposed[i] = previous[i]
      else if (conf[i] >= tau) byClass[cls[i]].push(i)
    }
    for (let k = 0; k < K; k++) {
      byClass[k].sort((a, b) => conf[b] - conf[a] || a - b)
      for (const i of byClass[k].slice(0, cap)) proposed[i] = k
    }
    const proposedPerClass = new Array<number>(K).fill(0)
    let proposedCount = 0
    let wrong = 0
    for (const i of unlabelled)
      if (proposed[i] >= 0) {
        proposedCount++
        proposedPerClass[proposed[i]]++
        if (truth && proposed[i] !== truth[i]) wrong++
      }
    let same = true
    for (const i of unlabelled) if (proposed[i] !== fitted[i]) same = false
    const evaluateProbs = evaluate ? probabilities(model, evaluate, K) : new Float64Array(0)
    return {
      t,
      probs,
      fitted,
      proposed,
      proposedCount,
      proposedPerClass,
      pseudoLabelError: truth && proposedCount > 0 ? wrong / proposedCount : NaN,
      unlabelledAccuracy: truth ? accuracyOf(probs, K, truth, unlabelled) : NaN,
      evaluateProbs,
      evaluateAccuracy: evaluate && evaluateTruth ? accuracyOf(evaluateProbs, K, evaluateTruth) : NaN,
      converged: t > 0 && same,
    }
  }

  return {
    name: 'self-training',
    init(_start: void, s: Stream) {
      const fitted = Int32Array.from(labels, (v) => (v >= 0 ? v : -1))
      return describe(0, fitted, fitRound(fitted, 0, s), null)
    },
    step(state, ctx) {
      const fitted = Int32Array.from(labels, (v) => (v >= 0 ? v : -1))
      for (const i of unlabelled) fitted[i] = state.proposed[i]
      const t = state.t + 1
      return describe(t, fitted, fitRound(fitted, t, ctx.stream), state.proposed)
    },
  }
}
