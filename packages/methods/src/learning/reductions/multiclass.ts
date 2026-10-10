/**
 * Multiclass classification from binary classifiers: $K$ classes reduced to binary problems whose margins are combined.
 *
 * The classical reductions are one-versus-rest and one-versus-one (Rifkin and Klautau, 2004, "In defense of one-vs-all
 * classification"); one-versus-one breaks vote ties with confidences as scikit-learn's `OneVsOneClassifier`. Both are
 * error-correcting output codes (Dietterich and Bakiri, 1995): a $K \times L$ code matrix $\Cmat$ with entries
 * $+1$, $-1$ and $0$ trains one binary model per column, and a point goes to the class whose row is nearest its $L$
 * margins, in Hamming or loss-based distance (Allwein, Schapire and Singer, 2000). Nested dichotomies instead arrange
 * the classes in a binary tree and multiply the binary probabilities along each path (Frank and Kramer, 2004).
 *
 * A binary base estimator is fitted on labels 0 and 1, and its model must have `score` or `forward` giving one real
 * margin per row (positive for class 1), such as `logisticRegression`, `supportVectorMachine`, `linearSvm` or
 * `perceptron`. Labels are the integers $0, \dots, K - 1$. Binary model $l$ is fitted with the stream
 * `child(stream, 'model', l)` (`'node'` for a nested dichotomy) of the fit options.
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

/**
 * A fitted binary model with a real-valued margin: `score` (or, without it, `forward`) gives one margin per query row,
 * positive for class 1; `predictive`, when present, marks the model as probabilistic (see `oneVersusRest`).
 */
export type BinaryModel = { score?(x: Tensor): Tensor; forward?(x: Tensor): Tensor; predictive?(x: Tensor): unknown }

/** A binary estimator: `fit` takes an $n \times d$ matrix `x` and labels `y` of 0 and 1, and returns the model. */
export type BinaryEstimator<M extends BinaryModel = BinaryModel> = {
  fit(data: Supervised<Tensor, Tensor>, options?: FitOptions): M
}

/**
 * The margins of a binary model on $m$ query rows (from `score`, else `forward`). Throws `DomainError` when the model
 * has neither, and `ShapeError` when it does not give a vector.
 *
 * @param model The fitted binary model.
 * @param x The query matrix, $m \times d$.
 * @returns The $m$ margins.
 */
function margin(model: BinaryModel, x: Tensor): Float64Array {
  const f = model.score ?? model.forward
  if (!f) throw new DomainError('multiclass', 'multiclass: the binary model needs score or forward')
  const s = f.call(model, x)
  if (s.shape.length !== 1)
    throw new ShapeError('multiclass', 'multiclass: the binary model must give one margin per row, [m]')
  return values(s)
}

/**
 * The binary subproblem of one model: the rows of `x` whose labels are kept, with target 1 where the label is
 * positive and 0 elsewhere.
 *
 * @param x The training matrix, $n \times d$.
 * @param y The integer label of each row.
 * @param keep Whether rows of a class are trained on.
 * @param positive Whether a class is the positive one (target 1).
 * @returns The kept rows `x` and their 0/1 targets `y`, as tensors, and `rows`, their indices in the full data.
 */
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

/** A fitted multiclass reduction. `score` gives $m \times K$ class scores and `decide` their argmax. */
export interface ReductionModel<M extends BinaryModel = BinaryModel>
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'one-versus-rest' | 'one-versus-one' | 'output-code'
  /** The number of classes $K$. */
  readonly classes: number
  /** The binary models, one per column of `code`. */
  readonly models: M[]
  /**
   * The $K \times L$ code matrix with entries $-1$, $0$ (not trained on) and $+1$ (the class is positive for that
   * model).
   */
  readonly code: Tensor
  /** The margins of every binary model on $m$ query rows, $m \times L$. */
  margins(x: Tensor): Tensor
}

/**
 * Fit one binary model per column of a ternary code: column $l$ trains on the classes with a non-zero entry, the
 * $+1$ classes positive.
 *
 * @param base The binary estimator.
 * @param x The training matrix, $n \times d$.
 * @param y The integer label of each row.
 * @param code The $K \times L$ code, row-major.
 * @param L The number of columns of the code.
 * @param options The fit options, passed on; model $l$ gets the stream `child(stream, 'model', l)` when one is given.
 * @returns The $L$ fitted models, in column order.
 */
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

/**
 * The margins of several binary models on the same query rows.
 *
 * @param models The $L$ fitted binary models.
 * @param x The query matrix, $m \times d$.
 * @returns The margins, $m \times L$ row-major (model $l$ in column $l$).
 */
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
 * One-versus-rest: $K$ binary models, model $k$ separating class $k$ (positive) from the others. `score` gives the
 * $m \times K$ margins and `decide` their argmax. When every binary model has a `predictive`, the fitted model has
 * one too: it reads each margin $s_k$ as a logit, and normalises the $K$ positive-class probabilities
 * $\sigma(s_k)$ to sum to 1 (as scikit-learn's `OneVsRestClassifier.predict_proba`); otherwise it is absent. `fit`
 * throws `ShapeError` or `DomainError` for data that is not a matrix with integer labels.
 *
 * @param base The binary estimator fitted once per class.
 * @returns The estimator: `fit({ x, y })` returns a `ReductionModel` whose `code` is `oneVersusRestCode(K)`.
 *
 * @example One-versus-rest on three Gaussian classes
 * // A tiny binary base: the margin is how much nearer the class-1 mean a row is than the class-0 mean.
 * const nearerMean = {
 *   fit({ x, y }) {
 *     const centre = (c) => div(sum(mul(x, reshape(equalTo(y, c), [-1, 1])), 0), sum(equalTo(y, c)))
 *     const [m0, m1] = [centre(0), centre(1)]
 *     return { score: (q) => sub(sum(square(sub(q, m0)), -1), sum(square(sub(q, m1)), -1)) }
 *   },
 * }
 * // Three Gaussian classes of ten points, centred at (0, 0), (4, 0) and (0, 4).
 * const block = (s, c) => add(normals(stream(s), [10, 2]), tensor(c))
 * const x = concat([block(0, [0, 0]), block(1, [4, 0]), block(2, [0, 4])])
 * const y = tensor(Array.from({ length: 30 }, (_, i) => Math.floor(i / 10)))
 * const model = oneVersusRest(nearerMean).fit({ x, y })
 * const q = tensor([[0, 0], [4, 0], [0, 4]])
 * print('margins =', model.score(q))
 * print('classes =', model.decide(q))
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
 * One-versus-one: $K(K - 1)/2$ binary models, one per pair $(j, k)$ with $j < k$, trained on those two classes only
 * (class $j$ positive). Each model votes for $j$ when its margin $s$ is positive and for $k$ otherwise; the
 * $m \times K$ `score` is the votes plus each class's summed confidence $t$ squashed to $t / (3(\lvert t \rvert + 1))$,
 * inside $(-\tfrac13, \tfrac13)$, so ties go to the more confident class (scikit-learn's `_ovr_decision_function`).
 *
 * @param base The binary estimator fitted once per pair of classes.
 * @returns The estimator: `fit({ x, y })` returns a `ReductionModel` whose `code` is `oneVersusOneCode(K)`.
 *
 * @example Pairwise votes on three Gaussian classes
 * // A tiny binary base: the margin is how much nearer the class-1 mean a row is than the class-0 mean.
 * const nearerMean = {
 *   fit({ x, y }) {
 *     const centre = (c) => div(sum(mul(x, reshape(equalTo(y, c), [-1, 1])), 0), sum(equalTo(y, c)))
 *     const [m0, m1] = [centre(0), centre(1)]
 *     return { score: (q) => sub(sum(square(sub(q, m0)), -1), sum(square(sub(q, m1)), -1)) }
 *   },
 * }
 * // Three Gaussian classes of ten points, centred at (0, 0), (4, 0) and (0, 4).
 * const block = (s, c) => add(normals(stream(s), [10, 2]), tensor(c))
 * const x = concat([block(0, [0, 0]), block(1, [4, 0]), block(2, [0, 4])])
 * const y = tensor(Array.from({ length: 30 }, (_, i) => Math.floor(i / 10)))
 * const model = oneVersusOne(nearerMean).fit({ x, y })
 * const q = tensor([[0, 0], [4, 0], [0, 4]])
 * print('margins of (0, 1), (0, 2), (1, 2) =', model.margins(q))
 * print('votes =', model.score(q))
 * print('classes =', model.decide(q))
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
 * Error-correcting output codes (Dietterich and Bakiri, 1995; Allwein, Schapire and Singer, 2000): one binary model
 * per column of the $K \times L$ code $\Cmat$ (entries $+1$, $-1$, and $0$ for classes a model is not trained on).
 * Decoding picks the class whose row is nearest the margins $s_1, \dots, s_L$:
 *
 * - `hamming` (default): $\sum_l (1 - \sgn(s_l) C_{kl})/2$, so a zero entry costs $\tfrac12$ whatever the model says;
 * - `loss`: $\sum_l \ell(C_{kl} s_l)$ with the logistic loss $\ell(z) = \log(1 + e^{-z})$ (zero entries cost
 *   nothing).
 *
 * `score` returns minus the $m \times K$ decoding distances, so larger is better and `decide` is its argmax. Throws
 * `DomainError` at once when a column lacks a $+1$ or a $-1$, and from `fit` when the labels reach beyond the $K$ rows.
 *
 * @param base The binary estimator fitted once per column.
 * @param code The code matrix $\Cmat$, $K \times L$, as a tensor or as rows of numbers; row $k$ is the codeword of
 *   class $k$. `oneVersusRestCode`, `oneVersusOneCode`, `exhaustiveCode` and `randomCode` build one.
 * @param params `decoding`, the distance between margins and codewords: `'hamming'` (default) or `'loss'`.
 * @returns The estimator: `fit({ x, y })` returns a `ReductionModel` with `distances`, the $m \times K$ decoding
 *   distances.
 *
 * @example The exhaustive code on three Gaussian classes
 * // A tiny binary base: the margin is how much nearer the class-1 mean a row is than the class-0 mean.
 * const nearerMean = {
 *   fit({ x, y }) {
 *     const centre = (c) => div(sum(mul(x, reshape(equalTo(y, c), [-1, 1])), 0), sum(equalTo(y, c)))
 *     const [m0, m1] = [centre(0), centre(1)]
 *     return { score: (q) => sub(sum(square(sub(q, m0)), -1), sum(square(sub(q, m1)), -1)) }
 *   },
 * }
 * // Three Gaussian classes of ten points, centred at (0, 0), (4, 0) and (0, 4).
 * const block = (s, c) => add(normals(stream(s), [10, 2]), tensor(c))
 * const x = concat([block(0, [0, 0]), block(1, [4, 0]), block(2, [0, 4])])
 * const y = tensor(Array.from({ length: 30 }, (_, i) => Math.floor(i / 10)))
 * const model = outputCode(nearerMean, exhaustiveCode(3)).fit({ x, y })
 * const q = tensor([[0, 0], [4, 0], [0, 4]])
 * print('Hamming distances =', model.distances(q))
 * print('classes =', model.decide(q))
 *
 * @example A column without both signs is refused
 * try {
 *   outputCode({ fit: () => ({}) }, [[1, 1], [-1, 1], [-1, 1]])
 * } catch (e) {
 *   print(e.message)
 * }
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

/**
 * The one-versus-rest code: $K \times K$, $+1$ on the diagonal and $-1$ elsewhere.
 *
 * @param K The number of classes.
 * @returns The $K \times K$ code.
 *
 * @example The code for three classes
 * print(oneVersusRestCode(3))
 */
export function oneVersusRestCode(K: number): Tensor {
  return fromData(
    Float64Array.from({ length: K * K }, (_, j) => (Math.floor(j / K) === j % K ? 1 : -1)),
    [K, K],
  )
}

/**
 * The one-versus-one code: $K \times K(K - 1)/2$, the column of pair $(i, j)$ with $i < j$ is $+1$ for class $i$,
 * $-1$ for class $j$ and $0$ for the rest. Pairs are in the order $(0, 1), (0, 2), \dots, (K - 2, K - 1)$.
 *
 * @param K The number of classes.
 * @returns The $K \times K(K - 1)/2$ code.
 *
 * @example The code for three classes
 * print(oneVersusOneCode(3))
 */
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
 * $2^{K-1} - 1$ columns, with class 0 always $+1$. Any two rows differ in $2^{K-2}$ columns. Throws `DomainError`
 * unless $2 \le K \le 16$.
 *
 * @param K The number of classes.
 * @returns The $K \times (2^{K-1} - 1)$ code.
 *
 * @example Four classes: seven columns, any two rows four apart
 * const C = exhaustiveCode(4)
 * print(C)
 * print('distance =', codeDistance(C))
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
 * A random dense ($\pm 1$ with probability $\tfrac12$ each) or sparse ($0$ with probability $\tfrac12$, else
 * $\pm 1$ equally) code with $L$ columns (Allwein, Schapire and Singer, 2000); a column without both signs is
 * redrawn, so $K$ must be at least 2. Column $l$ draws from `child(s, 'column', l)`. Throws `DomainError` unless $K$
 * is an integer of at least 2 and $L$ a non-negative integer.
 *
 * @param s The random stream.
 * @param K The number of classes (rows).
 * @param L The number of columns (binary models).
 * @param params `sparse`: draw a ternary code with zeros (default false, a dense binary code).
 * @returns The $K \times L$ code.
 *
 * @example A dense and a sparse code for four classes
 * print(randomCode(stream(0), 4, 5))
 * print(randomCode(stream(0), 4, 5, { sparse: true }))
 */
export function randomCode(s: Stream, K: number, L: number, params: { sparse?: boolean } = {}): Tensor {
  if (!Number.isInteger(K) || K < 2)
    throw new DomainError('randomCode', `randomCode: K must be an integer of at least 2, got ${K}`)
  if (!Number.isInteger(L) || L < 0)
    throw new DomainError('randomCode', `randomCode: L must be a non-negative integer, got ${L}`)
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
 * says nothing reliable about a class it was not trained on). A code corrects $\lfloor (d - 1)/2 \rfloor$ binary
 * errors, $d$ the distance.
 *
 * @param code The $K \times L$ code matrix.
 * @returns The smallest number of columns in which two rows have opposite non-zero entries (Infinity when $K < 2$).
 *
 * @example The distances of three codes for four classes
 * print('one-versus-rest:', codeDistance(oneVersusRestCode(4)))
 * print('one-versus-one:', codeDistance(oneVersusOneCode(4)))
 * print('exhaustive:', codeDistance(exhaustiveCode(4)))
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
 * A class tree: `balanced` (halve the sorted classes recursively, the smaller half on the left) or `chain` (peel off
 * one class at a time: 0 versus the rest, then 1 versus the rest, and so on). `randomDichotomyTree` draws a random one.
 * Throws `DomainError` unless $K$ is an integer of at least 1.
 *
 * @param K The number of classes, at least 1.
 * @param shape The shape of the tree: `'balanced'` or `'chain'`.
 * @returns The tree over the classes $0, \dots, K - 1$.
 *
 * @example Balanced and chain trees over five classes
 * print('balanced:', dichotomyTree(5))
 * print('chain:', dichotomyTree(5, 'chain'))
 */
export function dichotomyTree(K: number, shape: 'balanced' | 'chain' = 'balanced'): Dichotomy {
  return buildDichotomy(K, shape, undefined, 'dichotomyTree')
}

/**
 * A random class tree: at every node the classes are shuffled and cut at a uniform point into two non-empty groups.
 * The node at `path` (`''` for the root, then `'L'` and `'R'` for each step down) draws from `child(s, 'node', path)`.
 * Throws `DomainError` unless $K$ is an integer of at least 1.
 *
 * @param s The random stream.
 * @param K The number of classes, at least 1.
 * @returns The tree over the classes $0, \dots, K - 1$.
 *
 * @example Two random trees over five classes
 * print(randomDichotomyTree(stream(0), 5))
 * print(randomDichotomyTree(stream(1), 5))
 */
export function randomDichotomyTree(s: Stream, K: number): Dichotomy {
  return buildDichotomy(K, 'random', s, 'randomDichotomyTree')
}

/**
 * Build a class tree by splitting the classes recursively: in half, one class off the front, or at random.
 *
 * @param K The number of classes.
 * @param shape `'balanced'`, `'chain'` or `'random'`.
 * @param s The random stream of a `'random'` tree (unused otherwise).
 * @param where The public function, named in the `DomainError` thrown unless $K$ is an integer of at least 1.
 * @returns The tree over the classes $0, \dots, K - 1$.
 */
function buildDichotomy(
  K: number,
  shape: 'balanced' | 'chain' | 'random',
  s: Stream | undefined,
  where: string,
): Dichotomy {
  if (!Number.isInteger(K) || K < 1)
    throw new DomainError(where, `${where}: K must be an integer of at least 1, got ${K}`)
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

/**
 * The classes at the leaves of a tree, left to right.
 *
 * @param t The tree or subtree.
 * @returns Its leaf classes.
 */
const classesOf = (t: Dichotomy): number[] => (typeof t === 'number' ? [t] : [...classesOf(t[0]), ...classesOf(t[1])])

/** A fitted nested dichotomy. */
export interface NestedDichotomyModel<M extends BinaryModel = BinaryModel>
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Predicts<Tensor, AnyUnivariate> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'nested-dichotomies'
  /** The number of classes $K$. */
  readonly classes: number
  /** The class tree the models were fitted on. */
  readonly tree: Dichotomy
  /**
   * One binary model per internal node, in preorder; model $i$ gives the probability of the left child given node
   * $i$'s classes.
   */
  readonly models: M[]
}

/**
 * Nested dichotomies (Frank and Kramer, 2004): each internal node of a class tree has a binary model for "left group
 * versus right group", trained only on its classes; $p(y = k \mid \xvec)$ is the product of the branch probabilities
 * on the path to $k$. The probability of the left group is $\sigma(s)$, the base's margin $s$ read as a logit.
 * `forward` and `score` are the $m \times K$ log-probabilities, `predictive` the class law and `decide` the most
 * probable class. `fit` throws `DomainError` unless the tree holds every class $0, \dots, K - 1$ exactly once.
 *
 * @param base The binary estimator fitted once per internal node.
 * @param tree The class tree, or a function of $K$ that builds it (default `dichotomyTree(K)`, balanced).
 * @returns The estimator: `fit({ x, y })` returns a `NestedDichotomyModel`.
 *
 * @example A balanced tree over three Gaussian classes
 * // A tiny binary base: the margin is how much nearer the class-1 mean a row is than the class-0 mean.
 * const nearerMean = {
 *   fit({ x, y }) {
 *     const centre = (c) => div(sum(mul(x, reshape(equalTo(y, c), [-1, 1])), 0), sum(equalTo(y, c)))
 *     const [m0, m1] = [centre(0), centre(1)]
 *     return { score: (q) => sub(sum(square(sub(q, m0)), -1), sum(square(sub(q, m1)), -1)) }
 *   },
 * }
 * // Three Gaussian classes of ten points, centred at (0, 0), (4, 0) and (0, 4).
 * const block = (s, c) => add(normals(stream(s), [10, 2]), tensor(c))
 * const x = concat([block(0, [0, 0]), block(1, [4, 0]), block(2, [0, 4])])
 * const y = tensor(Array.from({ length: 30 }, (_, i) => Math.floor(i / 10)))
 * const model = nestedDichotomies(nearerMean).fit({ x, y })
 * const q = tensor([[0, 0], [4, 0], [0, 4]])
 * print('tree =', model.tree)
 * print('log-probabilities =', model.forward(q))
 * print('classes =', model.decide(q))
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

/**
 * Row-wise softmax of class scores as class probabilities, for reductions whose scores are logits.
 *
 * @param scores The $m \times K$ scores.
 * @returns The $m \times K$ probabilities; each row sums to 1.
 *
 * @example Scores to probabilities
 * print(softmaxScores(tensor([[2, 0, 0], [0, 0, 0]])))
 */
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
