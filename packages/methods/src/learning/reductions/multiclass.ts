/**
 * Multiclass classification from binary classifiers, and one natively multiclass SVM.
 *
 * - `oneVersusRest`, `oneVersusOne`: the classical reductions (Rifkin and Klautau, 2004, "In defense of one-vs-all
 *   classification"); one-versus-one breaks vote ties with confidences as scikit-learn's `OneVsOneClassifier`.
 * - `outputCode` with code matrices `oneVersusRestCode`, `oneVersusOneCode`, `exhaustiveCode`, `randomCode`: error-
 *   correcting output codes (Dietterich and Bakiri, 1995) with ternary entries and Hamming or loss-based decoding
 *   (Allwein, Schapire and Singer, 2000); `codeDistance` gives the minimum row distance.
 * - `nestedDichotomies`: a binary tree over the classes, with class probabilities as products of the binary
 *   probabilities along each path (Frank and Kramer, 2004).
 * - `crammerSingerSteps`, `crammerSinger`: the Crammer–Singer (2001) multiclass linear SVM by sequential dual
 *   coordinate ascent (Keerthi, Sundararajan, Chang, Hsieh and Lin, 2008, "A sequential dual method for large scale
 *   multi-class linear SVMs", KDD), as LIBLINEAR's `MCSVM_CS`.
 *
 * A binary base estimator takes labels 0/1 and must have `score` or `forward` giving a real margin [m] (positive for
 * class 1), such as `logisticRegression`, `supportVectorMachine`, `linearSvm` or `perceptron`.
 */

import {
  type AnyUnivariate,
  type Decides,
  type Estimator,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Scores,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import type { Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { argmax } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { sigmoid } from 'aifn-compute/numerics/special'
import { classLabels, classPredictive, matrix, softmaxRows, values } from '../util'
import { child, integers, uniform } from 'aifn-compute/foundation/random'
import { defineModel } from 'aifn-compute/learning/estimators'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A fitted binary model with a real-valued margin. */
export type BinaryModel = { score?(x: Tensor): Tensor; forward?(x: Tensor): Tensor; predictive?(x: Tensor): unknown }

/** A binary estimator: fits labels 0/1. */
export type BinaryEstimator<M extends BinaryModel = BinaryModel> = {
  fit(data: Supervised<Tensor, Tensor>, options?: FitOptions): M
}

/** The margin [m] of a binary model (from `score`, else `forward`). */
function margin(model: BinaryModel, x: Tensor): Float64Array {
  const f = model.score ?? model.forward
  if (!f) throw new DomainError('multiclass', 'multiclass: the binary model needs score or forward')
  const s = f.call(model, x)
  if (s.shape.length !== 1)
    throw new ShapeError('multiclass', 'multiclass: the binary model must give one margin per row, [m]')
  return values(s)
}

/** Rows of x whose labels are in `keep`, with binary targets 1 where `positive(label)`. */
function subproblem(x: Tensor, y: Int32Array, keep: (c: number) => boolean, positive: (c: number) => boolean) {
  const [, d] = x.shape
  const v = values(x)
  const rows: number[] = []
  for (let i = 0; i < y.length; i++) if (keep(y[i])) rows.push(i)
  const sub = new Float64Array(rows.length * d)
  rows.forEach((i, r) => sub.set(v.subarray(i * d, (i + 1) * d), r * d))
  return {
    x: fromData(sub, [rows.length, d]),
    y: fromData(
      Float64Array.from(rows, (i) => (positive(y[i]) ? 1 : 0)),
      [rows.length],
    ),
    rows,
  }
}

/** A fitted multiclass reduction. */
export interface ReductionModel<M extends BinaryModel = BinaryModel>
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'one-versus-rest' | 'one-versus-one' | 'output-code'
  readonly classes: number
  /** The binary models, one per column of `code`. */
  readonly models: M[]
  /** The K × L code matrix with entries −1, 0 (not trained on) and +1 (the class is positive for that model). */
  readonly code: Tensor
  /** The binary margins of every model [m, L]. */
  margins(x: Tensor): Tensor
}

/** Fit one binary model per column of a ternary code. */
function fitCode<M extends BinaryModel>(
  base: BinaryEstimator<M>,
  x: Tensor,
  y: Int32Array,
  code: Float64Array,
  L: number,
  options: FitOptions,
) {
  const models: M[] = []
  for (let l = 0; l < L; l++) {
    const sub = subproblem(
      x,
      y,
      (c) => code[c * L + l] !== 0,
      (c) => code[c * L + l] > 0,
    )
    models.push(
      base.fit(dataset(sub.x, sub.y), { ...options, stream: options.stream && child(options.stream, 'model', l) }),
    )
  }
  return models
}

function marginsOf(models: BinaryModel[], x: Tensor): Float64Array {
  const m = x.shape[0]
  const L = models.length
  const out = new Float64Array(m * L)
  models.forEach((model, l) => {
    const s = margin(model, x)
    for (let i = 0; i < m; i++) out[i * L + l] = s[i]
  })
  return out
}

/**
 * One-versus-rest: K binary models, model k separating class k from the others. `score` gives the K margins [m, K]
 * and `decide` their argmax. When the base model has probabilities, `predictive` normalises the K positive-class
 * probabilities (scikit-learn's `OneVsRestClassifier.predict_proba`); otherwise it is absent.
 */
export function oneVersusRest<M extends BinaryModel>(
  base: BinaryEstimator<M>,
): Estimator<Supervised<Tensor, Tensor>, ReductionModel<M> & Partial<Predicts<Tensor, AnyUnivariate>>> {
  return {
    name: 'one-versus-rest',
    fit({ x, y }, options: FitOptions = {}) {
      const { n } = matrix(x, 'oneVersusRest')
      const { y: labels, k: K } = classLabels(y, n, 'oneVersusRest')
      const code = oneVersusRestCode(K)
      const c = values(code)
      const models = fitCode(base, x, labels, c, K, options)
      const score = (q: Tensor) => fromData(marginsOf(models, q), [q.shape[0], K])
      const out: ReductionModel<M> & Partial<Predicts<Tensor, AnyUnivariate>> = {
        kind: 'model',
        name: 'one-versus-rest',
        classes: K,
        models,
        code,
        margins: score,
        forward: score,
        score,
        decide: (q: Tensor) => argmax(score(q), -1),
      }
      if (models.every((mo) => typeof mo.predictive === 'function')) {
        out.predictive = (q: Tensor) => {
          const m = q.shape[0]
          const p = new Float64Array(m * K)
          const s = marginsOf(models, q)
          // The positive-class probability of a binary model with a logit margin is σ(margin).
          for (let i = 0; i < m; i++) {
            let z = 0
            for (let k = 0; k < K; k++) z += p[i * K + k] = sigmoid(s[i * K + k])
            for (let k = 0; k < K; k++) p[i * K + k] /= z
          }
          return classPredictive(p, m, K)
        }
      }
      return out
    },
  }
}

/**
 * One-versus-one: K(K − 1)/2 binary models, one per pair (j, k) with j < k trained on those two classes only (class j
 * positive). Each model votes; `score` [m, K] is the votes plus the summed confidences squashed into (−⅓, ⅓), so ties
 * go to the more confident class (scikit-learn's `_ovr_decision_function`).
 */
export function oneVersusOne<M extends BinaryModel>(
  base: BinaryEstimator<M>,
): Estimator<Supervised<Tensor, Tensor>, ReductionModel<M>> {
  return {
    name: 'one-versus-one',
    fit({ x, y }, options: FitOptions = {}) {
      const { n } = matrix(x, 'oneVersusOne')
      const { y: labels, k: K } = classLabels(y, n, 'oneVersusOne')
      const code = oneVersusOneCode(K)
      const L = code.shape[1]
      const models = fitCode(base, x, labels, values(code), L, options)
      const pairs: [number, number][] = []
      for (let j = 0; j < K; j++) for (let k = j + 1; k < K; k++) pairs.push([j, k])
      const score = (q: Tensor) => {
        const m = q.shape[0]
        const s = marginsOf(models, q)
        const votes = new Float64Array(m * K)
        const conf = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          pairs.forEach(([j, k], l) => {
            const v = s[i * L + l]
            votes[i * K + (v > 0 ? j : k)] += 1
            conf[i * K + j] += v
            conf[i * K + k] -= v
          })
          for (let c = 0; c < K; c++) {
            const t = conf[i * K + c]
            votes[i * K + c] += t / (3 * (Math.abs(t) + 1))
          }
        }
        return fromData(votes, [m, K])
      }
      return {
        kind: 'model',
        name: 'one-versus-one',
        classes: K,
        models,
        code,
        margins: (q: Tensor) => fromData(marginsOf(models, q), [q.shape[0], L]),
        forward: score,
        score,
        decide: (q: Tensor) => argmax(score(q), -1),
      }
    },
  }
}

/**
 * Error-correcting output codes: one binary model per column of the K × L code (entries +1, −1, and 0 for classes a
 * model is not trained on). Decoding picks the class whose row is nearest the vector of margins:
 *
 * - `hamming` (default): Σₗ (1 − sign(sₗ)·C_kl)/2, so a zero entry costs ½ whatever the model says;
 * - `loss`: Σₗ ℓ(C_kl sₗ) with the logistic loss ℓ(z) = log(1 + e^(−z)) (zero entries cost nothing).
 *
 * `score` returns minus the decoding distances [m, K], so larger is better and `decide` is its argmax.
 */
export function outputCode<M extends BinaryModel>(
  base: BinaryEstimator<M>,
  code: Tensor | readonly (readonly number[])[],
  params: { decoding?: 'hamming' | 'loss' } = {},
): Estimator<Supervised<Tensor, Tensor>, ReductionModel<M> & { distances(x: Tensor): Tensor }> {
  const C = Array.isArray(code)
    ? fromData(Float64Array.from((code as number[][]).flat()), [code.length, (code as number[][])[0].length])
    : (code as Tensor)
  const decoding = params.decoding ?? 'hamming'
  const [K, L] = C.shape
  const c = values(C)
  for (let l = 0; l < L; l++) {
    let pos = false
    let neg = false
    for (let k = 0; k < K; k++) {
      if (c[k * L + l] > 0) pos = true
      if (c[k * L + l] < 0) neg = true
    }
    if (!pos || !neg) throw new DomainError('outputCode', `outputCode: column ${l} needs both a +1 and a −1 class`)
  }
  return {
    name: 'output-code',
    params: { decoding },
    fit({ x, y }, options: FitOptions = {}) {
      const { n } = matrix(x, 'outputCode')
      const { y: labels, k } = classLabels(y, n, 'outputCode')
      if (k > K) throw new DomainError('outputCode', `outputCode: labels reach ${k - 1} but the code has ${K} rows`)
      const models = fitCode(base, x, labels, c, L, options)
      const distances = (q: Tensor) => {
        const m = q.shape[0]
        const s = marginsOf(models, q)
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          for (let r = 0; r < K; r++) {
            let dist = 0
            for (let l = 0; l < L; l++) {
              const e = c[r * L + l]
              const v = s[i * L + l]
              if (decoding === 'hamming') dist += (1 - (v > 0 ? 1 : v < 0 ? -1 : 0) * e) / 2
              else if (e !== 0) dist += Math.max(-e * v, 0) + Math.log1p(Math.exp(-Math.abs(e * v)))
            }
            out[i * K + r] = dist
          }
        }
        return out
      }
      const score = (q: Tensor) =>
        fromData(
          Float64Array.from(distances(q), (v) => -v),
          [q.shape[0], K],
        )
      return {
        kind: 'model',
        name: 'output-code',
        classes: K,
        models,
        code: C,
        margins: (q: Tensor) => fromData(marginsOf(models, q), [q.shape[0], L]),
        distances: (q: Tensor) => fromData(distances(q), [q.shape[0], K]),
        forward: score,
        score,
        decide: (q: Tensor) => argmax(score(q), -1),
      }
    },
  }
}

// ── Code matrices ────────────────────────────────────────────────────────────────────────────────────────────────

/** The one-versus-rest code: K × K, +1 on the diagonal and −1 elsewhere. */
export function oneVersusRestCode(K: number): Tensor {
  return fromData(
    Float64Array.from({ length: K * K }, (_, j) => (Math.floor(j / K) === j % K ? 1 : -1)),
    [K, K],
  )
}

/** The one-versus-one code: K × K(K − 1)/2, column (i, j) is +1 for class i, −1 for class j and 0 for the rest. */
export function oneVersusOneCode(K: number): Tensor {
  const L = (K * (K - 1)) / 2
  const out = new Float64Array(K * L)
  let l = 0
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      out[i * L + l] = 1
      out[j * L + l] = -1
      l++
    }
  }
  return fromData(out, [K, L])
}

/**
 * Dietterich and Bakiri's (1995) exhaustive code: every split of the classes into two non-empty groups once,
 * 2^(K−1) − 1 columns, with class 0 always +1. Any two rows differ in 2^(K−2) columns.
 */
export function exhaustiveCode(K: number): Tensor {
  if (K < 2 || K > 16) throw new DomainError('exhaustiveCode', 'exhaustiveCode: K must be between 2 and 16')
  const L = 2 ** (K - 1) - 1
  const out = new Float64Array(K * L)
  for (let r = 0; r < K; r++) {
    for (let c = 0; c < L; c++) out[r * L + c] = r === 0 ? 1 : Math.floor(c / 2 ** (K - 1 - r)) % 2 === 1 ? 1 : -1
  }
  return fromData(out, [K, L])
}

/**
 * A random dense (±1 with probability ½) or sparse (0 with probability ½, else ±1 equally) code with L columns; a
 * column without both signs is redrawn. Column l draws from `child(s, 'column', l)`.
 */
export function randomCode(s: Stream, K: number, L: number, params: { sparse?: boolean } = {}): Tensor {
  const out = new Float64Array(K * L)
  for (let l = 0; l < L; l++) {
    const cs = child(s, 'column', l)
    for (;;) {
      let pos = false
      let neg = false
      for (let k = 0; k < K; k++) {
        const u = uniform(cs)
        const e = params.sparse ? (u < 0.5 ? 0 : u < 0.75 ? 1 : -1) : u < 0.5 ? 1 : -1
        out[k * L + l] = e
        if (e > 0) pos = true
        if (e < 0) neg = true
      }
      if (pos && neg) break
    }
  }
  return fromData(out, [K, L])
}

/**
 * The minimum distance between two rows of a code, counting a column only when both rows are non-zero there (a model
 * says nothing reliable about a class it was not trained on). A code corrects ⌊(distance − 1)/2⌋ binary errors.
 */
export function codeDistance(code: Tensor): number {
  const [K, L] = code.shape
  const c = values(code)
  let best = Infinity
  for (let a = 0; a < K; a++) {
    for (let b = a + 1; b < K; b++) {
      let dist = 0
      for (let l = 0; l < L; l++) if (c[a * L + l] !== 0 && c[b * L + l] !== 0 && c[a * L + l] !== c[b * L + l]) dist++
      best = Math.min(best, dist)
    }
  }
  return best
}

// ── Nested dichotomies ───────────────────────────────────────────────────────────────────────────────────────────

/** A binary tree over the classes: a leaf is a class, an internal node splits its classes into two children. */
export type Dichotomy = number | [Dichotomy, Dichotomy]

/**
 * A class tree: `balanced` (halve the sorted classes recursively) or `chain` (peel off one class at a time: 0 vs rest,
 * then 1 vs rest, …). `randomDichotomyTree` draws a random one.
 */
export function dichotomyTree(K: number, shape: 'balanced' | 'chain' = 'balanced'): Dichotomy {
  return buildDichotomy(K, shape, undefined)
}

/** A random class tree: a random split of the classes at every node, node `path` drawing from `child(s, 'node', path)`. */
export function randomDichotomyTree(s: Stream, K: number): Dichotomy {
  return buildDichotomy(K, 'random', s)
}

function buildDichotomy(K: number, shape: 'balanced' | 'chain' | 'random', s: Stream | undefined): Dichotomy {
  const build = (classes: number[], path: string): Dichotomy => {
    if (classes.length === 1) return classes[0]
    let cut = Math.floor(classes.length / 2)
    let order = classes
    if (shape === 'chain') cut = 1
    if (shape === 'random' && s) {
      const sub = child(s, 'node', path)
      order = classes.slice()
      for (let a = order.length - 1; a > 0; a--) {
        const b = integers(sub, a + 1)
        ;[order[a], order[b]] = [order[b], order[a]]
      }
      cut = 1 + integers(sub, classes.length - 1)
    }
    const left = order.slice(0, cut).sort((a, b) => a - b)
    const right = order.slice(cut).sort((a, b) => a - b)
    return [build(left, path + 'L'), build(right, path + 'R')]
  }
  return build(
    Array.from({ length: K }, (_, k) => k),
    '',
  )
}

const classesOf = (t: Dichotomy): number[] => (typeof t === 'number' ? [t] : [...classesOf(t[0]), ...classesOf(t[1])])

/** A fitted nested dichotomy. */
export interface NestedDichotomyModel<M extends BinaryModel = BinaryModel>
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Predicts<Tensor, AnyUnivariate> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'nested-dichotomies'
  readonly classes: number
  readonly tree: Dichotomy
  /** One binary model per internal node, in preorder; model i gives P(left child | node i's classes). */
  readonly models: M[]
}

/**
 * Nested dichotomies: each internal node of a class tree has a binary model for "left group versus right group",
 * trained only on its classes; P(y = k | x) is the product of the branch probabilities on the path to k. The binary
 * probability of the left group is σ(margin) (the base's margin read as a logit).
 */
export function nestedDichotomies<M extends BinaryModel>(
  base: BinaryEstimator<M>,
  tree: Dichotomy | ((K: number) => Dichotomy) = (K) => dichotomyTree(K),
): Estimator<Supervised<Tensor, Tensor>, NestedDichotomyModel<M>> {
  return {
    name: 'nested-dichotomies',
    fit({ x, y }, options: FitOptions = {}) {
      const { n } = matrix(x, 'nestedDichotomies')
      const { y: labels, k: K } = classLabels(y, n, 'nestedDichotomies')
      const t = typeof tree === 'function' ? tree(K) : tree
      if (
        classesOf(t)
          .sort((a, b) => a - b)
          .join() !== Array.from({ length: K }, (_, k) => k).join()
      ) {
        throw new DomainError('nestedDichotomies', 'nestedDichotomies: the tree must contain every class exactly once')
      }
      const models: M[] = []
      const fitNode = (node: Dichotomy) => {
        if (typeof node === 'number') return
        const left = new Set(classesOf(node[0]))
        const all = new Set(classesOf(node))
        const sub = subproblem(
          x,
          labels,
          (c) => all.has(c),
          (c) => left.has(c),
        )
        models.push(
          base.fit(dataset(sub.x, sub.y), {
            ...options,
            stream: options.stream && child(options.stream, 'node', models.length),
          }),
        )
        fitNode(node[0])
        fitNode(node[1])
      }
      fitNode(t)
      const probabilities = (q: Tensor): Float64Array => {
        const m = q.shape[0]
        const out = new Float64Array(m * K)
        const margins = models.map((mo) => margin(mo, q))
        let index = 0
        const walk = (node: Dichotomy, mass: Float64Array) => {
          if (typeof node === 'number') {
            for (let i = 0; i < m; i++) out[i * K + node] = mass[i]
            return
          }
          const s = margins[index++]
          walk(
            node[0],
            Float64Array.from(mass, (w, i) => w * sigmoid(s[i])),
          )
          walk(
            node[1],
            Float64Array.from(mass, (w, i) => w * sigmoid(-s[i])),
          )
        }
        walk(t, new Float64Array(m).fill(1))
        return out
      }
      const forward = (q: Tensor) => fromData(Float64Array.from(probabilities(q), Math.log), [q.shape[0], K])
      return {
        kind: 'model',
        name: 'nested-dichotomies',
        classes: K,
        tree: t,
        models,
        forward,
        score: forward,
        predictive: (q: Tensor) => classPredictive(probabilities(q), q.shape[0], K),
        decide: (q: Tensor) => argmax(fromData(probabilities(q), [q.shape[0], K]), -1),
      }
    },
  }
}

/** Softmax of scores [m, K] as class probabilities, for reductions whose scores are logits. */
export function softmaxScores(scores: Tensor): Tensor {
  const [m, K] = scores.shape
  return fromData(softmaxRows(values(scores), m, K), [m, K])
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'oneVersusRest',
    module: 'learning/reductions',
    name: 'One versus rest',
    summary:
      'K binary problems, each class against the rest; a predictive when the binary base has one. Capabilities declared for a logistic-regression base.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({}),
    notes: ['one-versus-rest'],
    cite: ['allwein2000reducing'],
  },
  oneVersusRest,
)

defineModel(
  {
    key: 'oneVersusOne',
    module: 'learning/reductions',
    name: 'One versus one',
    summary: 'K(K−1)/2 pairwise binary problems combined by voting.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({}),
    notes: ['one-versus-one'],
    cite: ['hastie1998pairwise'],
  },
  oneVersusOne,
)

defineModel(
  {
    key: 'outputCode',
    module: 'learning/reductions',
    name: 'Error-correcting output codes',
    summary:
      'One binary problem per column of a code matrix, decoded by Hamming or loss distance; the code is a required argument.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({ decoding: oneOf(['hamming', 'loss']) }),
    notes: ['error-correcting-output-codes'],
    cite: ['dietterich1995ecoc'],
  },
  outputCode,
)

defineModel(
  {
    key: 'nestedDichotomies',
    module: 'learning/reductions',
    name: 'Nested dichotomies',
    summary: 'A binary tree of class splits whose branch probabilities multiply to class probabilities.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({}),
    notes: ['nested-dichotomies-and-tree-reductions'],
    cite: ['frank2004nested'],
  },
  nestedDichotomies,
)
