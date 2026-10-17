/**
 * Learners that label unlabelled data for each other, co-training and tri-training, as traceable algorithms whose
 * states are plain data.
 *
 * `coTrainingSteps` is the co-training loop of Blum and Mitchell (1998, §5, Table 1): two classifiers, one per view,
 * are fitted on the shared labelled set; from a pool $U'$ of unlabelled points each one labels its most confident
 * `perRound[k]` points of each class $k$, which join the shared labelled set with those labels for good; the pool is
 * refilled from the remaining unlabelled points. The paper used naive Bayes, one positive and three negatives per
 * classifier per round, a pool of 75 and 30 rounds. Here the base learners are any probabilistic classifiers, the pool
 * defaults to all unlabelled points, and when both classifiers pick the same point in one round the first view's label
 * stands. The combined prediction multiplies the two views' class probabilities and renormalises
 * (`productOfExperts`; Blum and Mitchell's combined classifier for naive Bayes, exact when the views are independent
 * given the class).
 *
 * `triTrainingSteps` is tri-training (Zhou and Li, 2005, Table 1): three classifiers of one learner, fitted on
 * bootstrap samples of the labelled set, need no views. In each round learner $i$ receives the unlabelled points on
 * which the other two agree, labelled with their common class, when the other two's estimated joint error $e_i$
 * (labelled points where both agree and are wrong, over labelled points where both agree) has fallen and
 * $e_i \lvert L_i \rvert < e'_i \lvert L'_i \rvert$ (`triTrainingRule`, with the paper's subsampling of $L_i$ when only
 * the size condition fails); learner $i$ is then refitted on the labelled set plus $L_i$. Pseudo-labels are re-derived
 * every round. The final prediction is the majority vote (`majority`). The error estimate on the labelled set is the
 * paper's; with few labels it is often zero.
 */

import type { Status, Stream } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { child, choice, permutation } from 'aifn-compute/foundation/random'
import { shapeOf, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { dataset, takeRows, type Estimator, type Predicts } from 'aifn-compute/learning/estimators'
import { accuracyOf, labelledSet, probabilities, rowArgmax, type TrainingSet } from './self-training'

/** A learner whose fitted models give class laws, as the wrappers of this module take. */
type ClassLearner = Estimator<TrainingSet, Predicts<Tensor>>

/** The data of a co-training run: two views of the same $n$ points. */
export interface CoTrainingProblem {
  /** The two views, $n \times d_1$ and $n \times d_2$. */
  views: readonly [Tensor, Tensor]
  /** A class $0, \dots, K - 1$ per labelled point, $-1$ per unlabelled point. */
  labels: ArrayLike<number>
  /** The number of classes $K$. */
  classes: number
  /** True classes of all points, for diagnostics only. */
  truth?: ArrayLike<number>
  /** Test inputs in both views, with their classes, for the recorded test accuracies. */
  test?: { views: readonly [Tensor, Tensor]; truth: ArrayLike<number> }
  /** Further inputs for each view's classifier (grids), whose class probabilities are recorded every round. */
  evaluate?: readonly [Tensor, Tensor]
}

/** Controls of `coTrainingSteps`. */
export interface CoTrainingOptions {
  /** The learner of each view (one estimator for both, or one per view). */
  base: ClassLearner | readonly [ClassLearner, ClassLearner]
  /**
   * How many points of each class every classifier labels per round (Blum and Mitchell: one positive, three
   * negatives). Default one per class.
   */
  perRound?: readonly number[]
  /** Size of the pool $U'$ the classifiers label from, refilled each round. Default all unlabelled points. */
  pool?: number
}

/** One round of co-training. */
export interface CoTrainingState extends Status {
  /** The round: 0 is the fit on the labelled points alone. */
  t: number
  /** The label each point holds in the shared labelled set ($-1$: unlabelled). */
  assigned: Int32Array
  /** Who labelled each point: 0 nobody, 1 the original labels, 2 the view-a classifier, 3 the view-b classifier. */
  source: Int8Array
  /** The round in which each point was labelled ($-1$: not labelled; 0 for the original labels). */
  labelledAt: Int32Array
  /** 1 for a point in the pool $U'$ the classifiers label from next. */
  inPool: Uint8Array
  /** Class probabilities of the view-a and view-b classifiers on all points, row-major $n \times K$. */
  probs: readonly [Float64Array, Float64Array]
  /** Points labelled by the classifiers so far. */
  added: number
  /** How many of the classifiers' labels are wrong (needs `truth`; else 0). */
  addedWrong: number
  /** Test accuracy of the view-a classifier, the view-b classifier and their product (needs `test`; else NaN). */
  testAccuracy: readonly [number, number, number]
  /** Class probabilities of the two classifiers on their `evaluate` inputs, row-major (empty without them). */
  evaluateProbs: readonly [Float64Array, Float64Array]
  /** The pool is empty: no point is left to label. */
  terminated: boolean
}

/**
 * The elementwise product of two row-major class-probability matrices, each row renormalised (a row whose product is
 * all zero becomes uniform): the combined prediction of two experts independent given the class.
 *
 * @param p The first expert's probabilities, row-major $n \times K$; not modified.
 * @param q The second expert's probabilities, of the same shape; not modified.
 * @param K The number of classes.
 * @returns The renormalised products, row-major $n \times K$.
 *
 * @example Two unsure experts that lean the same way agree more strongly together
 * print(productOfExperts(Float64Array.from([0.6, 0.4]), Float64Array.from([0.7, 0.3]), 2))
 */
export function productOfExperts(p: Float64Array, q: Float64Array, K: number): Float64Array {
  const out = new Float64Array(p.length)
  for (let i = 0; i < p.length; i += K) {
    let z = 0
    for (let k = 0; k < K; k++) z += out[i + k] = p[i + k] * q[i + k]
    for (let k = 0; k < K; k++) out[i + k] = z > 0 ? out[i + k] / z : 1 / K
  }
  return out
}

/**
 * Co-training (Blum and Mitchell, 1998); see the file comment. Round $t$ draws the pool refill from
 * `child(stream, 'pool')` and each view's fit from `child(stream, 'fit', v)`. The run terminates when the pool is
 * empty. Throws `ShapeError` when a view and the labels differ in rows. Options hold estimators, so the algorithm runs
 * on the page's thread (about 80 ms for 25 rounds on 200 points).
 *
 * @param problem The two views, the labels with $-1$ for unlabelled points, the number of classes and diagnostics.
 * @param options The learners, the points labelled per class per round, and the pool size.
 * @returns The algorithm; each step is one round in which both classifiers label points for the shared set.
 *
 * @example Each round both classifiers label one point of each class
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
 * // Two views of twelve points: view b is view a with a little noise. One label per class.
 * const a = [-2.4, -1.9, -1.5, -1.1, -0.6, -0.2, 0.3, 0.7, 1.2, 1.6, 2.1, 2.5]
 * const b = a.map((v, i) => v + ((i % 3) - 1) * 0.4)
 * const truth = a.map((v) => (v > 0 ? 1 : 0))
 * const labels = truth.map((c, i) => (i === 0 || i === 11 ? c : -1))
 * const views = [tensor(a.map((v) => [v])), tensor(b.map((v) => [v]))]
 * const alg = coTrainingSteps({ views, labels, classes: 2, truth }, { base: midpoint })
 * const t = trace(alg, undefined, 5, { keep: 'all' })
 * print('points added:', t.steps.map((s) => s.added))
 * print('of which wrong:', t.steps.map((s) => s.addedWrong))
 */
export function coTrainingSteps(
  problem: CoTrainingProblem,
  options: CoTrainingOptions,
): Algorithm<void, CoTrainingState> {
  const { views, labels, classes: K, truth, test, evaluate } = problem
  const n = labels.length
  if (shapeOf(views[0])[0] !== n || shapeOf(views[1])[0] !== n)
    throw new ShapeError('coTrainingSteps', 'coTrainingSteps: views and labels differ in rows')
  const learners =
    'fit' in options.base
      ? ([options.base, options.base] as const)
      : (options.base as readonly [ClassLearner, ClassLearner])
  const perRound = options.perRound ?? new Array<number>(K).fill(1)
  const poolSize = options.pool ?? Infinity

  // Fit both views' classifiers on the shared labelled set and record the round.
  const describe = (
    t: number,
    assigned: Int32Array,
    source: Int8Array,
    labelledAt: Int32Array,
    inPool: Uint8Array,
    s: Stream,
  ): CoTrainingState => {
    const models = [0, 1].map((v) => learners[v].fit(labelledSet(views[v], assigned), { stream: child(s, 'fit', v) }))
    const probs = [probabilities(models[0], views[0], K), probabilities(models[1], views[1], K)] as const
    let added = 0
    let addedWrong = 0
    for (let i = 0; i < n; i++)
      if (source[i] >= 2) {
        added++
        if (truth && assigned[i] !== truth[i]) addedWrong++
      }
    let testAccuracy: readonly [number, number, number] = [NaN, NaN, NaN]
    if (test) {
      const pa = probabilities(models[0], test.views[0], K)
      const pb = probabilities(models[1], test.views[1], K)
      testAccuracy = [
        accuracyOf(pa, K, test.truth),
        accuracyOf(pb, K, test.truth),
        accuracyOf(productOfExperts(pa, pb, K), K, test.truth),
      ]
    }
    const evaluateProbs = evaluate
      ? ([probabilities(models[0], evaluate[0], K), probabilities(models[1], evaluate[1], K)] as const)
      : ([new Float64Array(0), new Float64Array(0)] as const)
    let poolLeft = 0
    for (let i = 0; i < n; i++) poolLeft += inPool[i]
    return {
      t,
      assigned,
      source,
      labelledAt,
      inPool,
      probs,
      added,
      addedWrong,
      testAccuracy,
      evaluateProbs,
      terminated: poolLeft === 0,
    }
  }

  // Move random unlabelled points outside the pool into it until it holds `poolSize` (or none are left).
  const refill = (assigned: Int32Array, inPool: Uint8Array, s: Stream) => {
    let count = 0
    for (let i = 0; i < n; i++) count += inPool[i]
    const order = toFlat(permutation(s, n))
    for (const i of order) {
      if (count >= poolSize) break
      if (assigned[i] < 0 && !inPool[i]) {
        inPool[i] = 1
        count++
      }
    }
  }

  return {
    name: 'co-training',
    init(_start: void, s: Stream) {
      const assigned = Int32Array.from(labels, (v) => (v >= 0 ? v : -1))
      const source = Int8Array.from(labels, (v) => (v >= 0 ? 1 : 0))
      const labelledAt = Int32Array.from(labels, (v) => (v >= 0 ? 0 : -1))
      const inPool = new Uint8Array(n)
      refill(assigned, inPool, child(s, 'pool'))
      return describe(0, assigned, source, labelledAt, inPool, s)
    },
    step(state, ctx) {
      const t = state.t + 1
      const assigned = Int32Array.from(state.assigned)
      const source = Int8Array.from(state.source)
      const labelledAt = Int32Array.from(state.labelledAt)
      const inPool = Uint8Array.from(state.inPool)
      for (let v = 0; v < 2; v++) {
        const p = state.probs[v]
        for (let k = 0; k < K; k++) {
          const candidates: number[] = []
          for (let i = 0; i < n; i++) if (inPool[i] && assigned[i] < 0) candidates.push(i)
          // Most confident for class k first: the probability of k, ties to the lower index.
          candidates.sort((i, j) => p[j * K + k] - p[i * K + k] || i - j)
          for (const i of candidates.slice(0, perRound[k] ?? 0)) {
            assigned[i] = k
            source[i] = v === 0 ? 2 : 3
            labelledAt[i] = t
            inPool[i] = 0
          }
        }
      }
      refill(assigned, inPool, child(ctx.stream, 'pool'))
      return describe(t, assigned, source, labelledAt, inPool, ctx.stream)
    },
  }
}

/** The data of a tri-training run: one view. */
export interface TriTrainingProblem {
  /** All inputs, $n \times d$. */
  x: Tensor
  /** A class per labelled point, $-1$ per unlabelled point. */
  labels: ArrayLike<number>
  /** The number of classes $K$. */
  classes: number
  /** True classes of all points, for diagnostics only. */
  truth?: ArrayLike<number>
  /** Further inputs ($m \times d$, a grid or a test set) whose vote is recorded each round. */
  evaluate?: Tensor
  /** The classes of the `evaluate` inputs, for the recorded accuracy of the vote. */
  evaluateTruth?: ArrayLike<number>
}

/** One round of tri-training. */
export interface TriTrainingState extends Status {
  /** The round: 0 is the three fits on bootstrap samples of the labelled set. */
  t: number
  /** Decisions of the three learners on all points (three arrays of length $n$). */
  decisions: readonly [Int32Array, Int32Array, Int32Array]
  /** Decisions of the three learners on the `evaluate` inputs (empty without them). */
  evaluateDecisions: readonly [Int32Array, Int32Array, Int32Array]
  /** The pseudo-labels each learner was last fitted with ($-1$: none; three arrays of length $n$). */
  pseudo: readonly [Int32Array, Int32Array, Int32Array]
  /** Whether each learner was refitted in this round. */
  updated: readonly [boolean, boolean, boolean]
  /** Each learner's $e'_i$ (the joint error of the other two when it was last refitted; 0.5 at the start). */
  errorBound: readonly [number, number, number]
  /** Each learner's $l'_i$, the size of the pseudo-labelled set it was last refitted with. */
  lastSize: readonly [number, number, number]
  /** Majority-vote decisions on the `evaluate` inputs. */
  evaluateVote: Int32Array
  /** Accuracy of the vote on the `evaluate` inputs (needs `evaluateTruth`; else NaN). */
  evaluateAccuracy: number
  /** Accuracy of the majority vote on the unlabelled points (needs `truth`; else NaN). */
  unlabelledAccuracy: number
  /** The fraction of the three learners' pseudo-labels that are wrong (needs `truth`; else NaN). */
  pseudoLabelError: number
  /** No learner changed in this round. */
  converged: boolean
}

/**
 * The majority vote of three decision arrays (a three-way tie, possible with three or more classes, goes to the first
 * learner).
 *
 * @param d The three learners' decisions, arrays of the same length; not modified.
 * @returns The vote for each entry.
 *
 * @example Two of three agree
 * print(majority([Int32Array.from([0, 1, 2]), Int32Array.from([1, 1, 0]), Int32Array.from([1, 0, 1])]))
 */
export function majority(d: readonly [Int32Array, Int32Array, Int32Array]): Int32Array {
  return Int32Array.from(d[0], (a, i) => (d[1][i] === d[2][i] ? d[1][i] : a))
}

/** The decision of one tri-training round for one learner (Zhou and Li, 2005, Table 1 and eqs. 9 to 11). */
export interface TriTrainingDecision {
  /** Refit the learner on the labelled set plus `keep` of the agreed points. */
  update: boolean
  /** How many of the agreed points to use: all of them, or the subsample size of eq. 10. */
  keep: number
  /** The learner's $l'_i$ after this round's bookkeeping (set on the first candidate update, before any refit). */
  last: number
}

/**
 * Whether learner $i$ takes the points the other two agree on (Zhou and Li, 2005, eqs. 9 to 11). The update must keep
 * the noise term $e \lvert L_i \rvert$ below $e'_i l'_i$ while the set grows, or else a random subsample of size
 * $\lceil e'_i l'_i / e - 1 \rceil$ is taken when that is still larger than $l'_i$. Before the first update,
 * $l'_i = \lfloor e / (e'_i - e) + 1 \rfloor$.
 *
 * @param e The other two learners' joint error $e_i$ now.
 * @param bound Their joint error $e'_i$ when learner $i$ was last refitted (0.5 at the start).
 * @param last The number $l'_i$ of pseudo-labels learner $i$ was last refitted with (0 before its first update).
 * @param available The number $\lvert L_i \rvert$ of unlabelled points the other two agree on now.
 * @returns Whether to refit, how many agreed points to keep, and the updated $l'_i$.
 *
 * @example The first round of the tri-training note's worked example, then a refusal
 * print(triTrainingRule(0.125, 0.5, 0, 60))
 * print(triTrainingRule(0.1, 0.125, 3, 70))
 */
export function triTrainingRule(e: number, bound: number, last: number, available: number): TriTrainingDecision {
  if (!(e < bound)) return { update: false, keep: 0, last }
  const l = last === 0 ? Math.floor(e / (bound - e) + 1) : last
  if (!(l < available)) return { update: false, keep: 0, last: l }
  if (e * available < bound * l) return { update: true, keep: available, last: l }
  if (l > e / (bound - e)) return { update: true, keep: Math.ceil((bound * l) / e - 1), last: l }
  return { update: false, keep: 0, last: l }
}

/**
 * Tri-training (Zhou and Li, 2005); see the file comment. Step 0 fits each learner $j$ on a bootstrap sample of the
 * labelled set from `child(stream, 'bootstrap', j, attempt)`; a sample that misses a class of the labelled set is
 * redrawn, at most 20 times, so every learner can be fitted (the paper's plain bootstrap does not need this with its
 * larger labelled sets). Each round's subsamples come from `child(stream, 'subsample', j)` and the fits from
 * `child(stream, 'fit', j)`. The run converges when no learner is refitted. Throws `ShapeError` when `x` and `labels`
 * differ in rows.
 *
 * @param problem The inputs, the labels with $-1$ for unlabelled points, the number of classes and diagnostics.
 * @param options The base learner (one estimator, fitted three times).
 * @param options.base The learner; its fitted models must give class laws.
 * @returns The algorithm; each step is one round of agreement labelling for all three learners.
 *
 * @example Tri-training on a line with two labels per class
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
 * const u = [-2.4, -1.9, -1.5, -1.1, -0.6, -0.2, 0.3, 0.7, 1.2, 1.6, 2.1, 2.5]
 * const y = u.map((v) => (v > 0 ? 1 : 0))
 * const labels = y.map((c, i) => (i < 2 || i > 9 ? c : -1))
 * const alg = triTrainingSteps({ x: tensor(u.map((v) => [v])), labels, classes: 2, truth: y }, { base: midpoint })
 * const t = trace(alg, undefined, 10)
 * print('rounds:', t.final.t, ' converged:', t.final.converged)
 * print('accuracy of the vote on the unlabelled points:', t.final.unlabelledAccuracy)
 */
export function triTrainingSteps(
  problem: TriTrainingProblem,
  options: { base: ClassLearner },
): Algorithm<void, TriTrainingState> {
  const { x, labels, classes: K, truth, evaluate, evaluateTruth } = problem
  const n = labels.length
  if (shapeOf(x)[0] !== n) throw new ShapeError('triTrainingSteps', 'triTrainingSteps: x and labels differ in rows')
  const L: number[] = []
  const U: number[] = []
  for (let i = 0; i < n; i++) (labels[i] >= 0 ? L : U).push(i)
  type Three<T> = readonly [T, T, T]

  // Fit one learner and return its decisions on all points and on the `evaluate` inputs.
  const fitDecide = (rows: number[], ys: number[], s: Stream) => {
    const model = options.base.fit(dataset(takeRows(x, rows), tensor(ys)), { stream: s })
    return {
      on: rowArgmax(probabilities(model, x, K), K).cls,
      grid: evaluate ? rowArgmax(probabilities(model, evaluate, K), K).cls : new Int32Array(0),
    }
  }

  // The state of a round from the three learners' decisions and bookkeeping.
  const describe = (
    t: number,
    decisions: Three<Int32Array>,
    evaluateDecisions: Three<Int32Array>,
    pseudo: Three<Int32Array>,
    updated: Three<boolean>,
    errorBound: Three<number>,
    lastSize: Three<number>,
  ): TriTrainingState => {
    const vote = majority(decisions)
    let right = 0
    let wrong = 0
    let given = 0
    if (truth) {
      for (const i of U) if (vote[i] === truth[i]) right++
      for (let j = 0; j < 3; j++)
        for (const i of U)
          if (pseudo[j][i] >= 0) {
            given++
            if (pseudo[j][i] !== truth[i]) wrong++
          }
    }
    const evaluateVote = evaluate ? majority(evaluateDecisions) : new Int32Array(0)
    let evaluateAccuracy = NaN
    if (evaluate && evaluateTruth) {
      let r = 0
      for (let i = 0; i < evaluateVote.length; i++) if (evaluateVote[i] === evaluateTruth[i]) r++
      evaluateAccuracy = r / evaluateVote.length
    }
    return {
      t,
      decisions,
      evaluateDecisions,
      pseudo,
      updated,
      errorBound,
      lastSize,
      evaluateVote,
      evaluateAccuracy,
      unlabelledAccuracy: truth && U.length ? right / U.length : NaN,
      pseudoLabelError: truth && given ? wrong / given : NaN,
      converged: t > 0 && !updated.some(Boolean),
    }
  }

  return {
    name: 'tri-training',
    init(_start: void, s: Stream) {
      const present = new Set(L.map((i) => labels[i])).size
      const fits = [0, 1, 2].map((j) => {
        let rows: number[] = []
        for (let attempt = 0; attempt < 20; attempt++) {
          rows = toFlat(choice(child(s, 'bootstrap', j, attempt), L.length, L.length)).map((a) => L[a])
          if (new Set(rows.map((i) => labels[i])).size === present) break
        }
        return fitDecide(
          rows,
          rows.map((i) => labels[i]),
          child(s, 'fit', j),
        )
      })
      const none = () => new Int32Array(n).fill(-1)
      return describe(
        0,
        [fits[0].on, fits[1].on, fits[2].on],
        [fits[0].grid, fits[1].grid, fits[2].grid],
        [none(), none(), none()],
        [false, false, false],
        [0.5, 0.5, 0.5],
        [0, 0, 0],
      )
    },
    step(state, ctx) {
      const d = state.decisions
      const decisions = [...d] as [Int32Array, Int32Array, Int32Array]
      const evaluateDecisions = [...state.evaluateDecisions] as [Int32Array, Int32Array, Int32Array]
      const pseudo = state.pseudo.map((p) => Int32Array.from(p)) as [Int32Array, Int32Array, Int32Array]
      const updated: [boolean, boolean, boolean] = [false, false, false]
      const bound = [...state.errorBound] as [number, number, number]
      const size = [...state.lastSize] as [number, number, number]
      // Every learner's update is decided from the previous round's three learners, then the chosen ones are refitted.
      const proposals: { j: number; set: number[]; e: number }[] = []
      for (let j = 0; j < 3; j++) {
        const [a, b] = [0, 1, 2].filter((v) => v !== j)
        // e_j: labelled points where the other two agree and are wrong, over those where they agree.
        let agree = 0
        let bothWrong = 0
        for (const i of L)
          if (d[a][i] === d[b][i]) {
            agree++
            if (d[a][i] !== labels[i]) bothWrong++
          }
        const e = agree ? bothWrong / agree : 0.5
        const all = U.filter((i) => d[a][i] === d[b][i])
        const rule = triTrainingRule(e, bound[j], size[j], all.length)
        size[j] = rule.last
        if (!rule.update) continue
        let set = all
        if (rule.keep < all.length) {
          const order = toFlat(permutation(child(ctx.stream, 'subsample', j), all.length))
          set = order.slice(0, rule.keep).map((r) => all[r])
        }
        proposals.push({ j, set, e })
      }
      for (const { j, set, e } of proposals) {
        const [a] = [0, 1, 2].filter((v) => v !== j)
        pseudo[j].fill(-1)
        for (const i of set) pseudo[j][i] = d[a][i]
        const rows = [...L, ...set]
        const fit = fitDecide(
          rows,
          rows.map((i) => (labels[i] >= 0 ? labels[i] : pseudo[j][i])),
          child(ctx.stream, 'fit', j),
        )
        decisions[j] = fit.on
        evaluateDecisions[j] = fit.grid
        updated[j] = true
        bound[j] = e
        size[j] = set.length
      }
      return describe(state.t + 1, decisions, evaluateDecisions, pseudo, updated, bound, size)
    },
  }
}
