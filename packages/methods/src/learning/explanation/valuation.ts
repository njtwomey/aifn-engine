/**
 * Which training points matter: a study of data attribution on a binary task with planted label noise, for a worker.
 * One L2 logistic regression (with an intercept feature) is explained five ways by `aifn-compute/learning/explain`:
 *
 * - influence functions on the validation loss (exact Hessian; Koh and Liang, 2017): a point is harmful when removing
 *   it lowers that loss;
 * - self-influence $\nabla\ell^\top \Hmat^{-1} \nabla\ell$, large for points the fit bends towards;
 * - TracIn self-influence (Pruthi, Liu, Kale and Sundararajan, 2020) over the checkpoints of full-batch gradient
 *   descent;
 * - exact KNN-Shapley (Jia et al., 2019) on the validation set, in the raw features;
 * - TMC data Shapley (Ghorbani and Zou, 2019) with the given estimator's validation accuracy as the utility (retrained
 *   on every prefix of every permutation; an empty subset scores $\tfrac12$, a one-class subset predicts its class).
 *
 * Each method's `suspicion` orders the points from most to least suspect (high harm, high self-influence, low value),
 * so the page can ask how many mislabelled points each finds among the first it inspects.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { add, fromData, mean, mul, sub, sum, square, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Estimator, type Supervised } from 'aifn-compute/learning/estimators'
import { dataShapley, influenceFunctions, knnShapley, tracIn, type ExampleLoss } from 'aifn-compute/learning/explain'
import { softplus } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The attribution methods compared, in their fixed order (and colour slots). */
export const VALUATION_METHODS = ['influence', 'self-influence', 'tracin', 'knn-shapley', 'data-shapley'] as const
/** One of the attribution methods of `VALUATION_METHODS`. */
export type ValuationMethod = (typeof VALUATION_METHODS)[number]

/** Options of `dataValuationStudy`. */
export type ValuationOptions = {
  /**
   * The L2 strength $\lambda$ of the explained logistic regression, whose objective is the mean loss plus
   * $(\lambda/2)\lVert \thetavec \rVert^2$ (default 0.01).
   */
  l2?: number
  /** Neighbours of KNN-Shapley (default 5). */
  K?: Size
  /** TMC permutations (default 40). */
  permutations?: Size
  /** TMC's truncation tolerance on the validation accuracy (default 0.01). */
  tolerance?: number
  /** Steps of the full-batch gradient descent whose checkpoints TracIn reads (default 200). */
  steps?: Size
  /** The step size of that gradient descent (default 0.5). */
  rate?: number
  /**
   * About how many checkpoints TracIn reads: one every $\lfloor \text{steps} / \text{checkpoints} \rfloor$ steps
   * (default 20).
   */
  checkpoints?: Size
  /** Seeds the TMC permutations (default 0). */
  seed?: number
}

/** A snapshot of the study: scores so far (data Shapley grows with the permutations). */
export type ValuationSnapshot = {
  /** The last stage finished: the fit, then each method in turn, then `'done'`. */
  stage: 'fit' | 'influence' | 'tracin' | 'knn-shapley' | 'data-shapley' | 'done'
  /** TMC permutations done. */
  done: Size
  /** TMC permutations asked for. */
  total: Size
  /** The explained fit $(\theta_1, \dots, \theta_d, \text{intercept})$. */
  theta: Float64Array
  /** Its validation accuracy. */
  accuracy: number
  /** Raw values per method (influence: predicted change of the validation loss on removal). */
  values: Partial<Record<ValuationMethod, Float64Array>>
  /** One score per training point and method: the higher, the more suspect. */
  suspicion: Partial<Record<ValuationMethod, Float64Array>>
  /** Utility calls made by TMC. */
  calls: Size
  /** The share of TMC's marginal contributions that were truncated. */
  truncated: number
}

/**
 * $\ell(\thetavec, \xvec, y) = \operatorname{softplus}(\thetavec^\top\xvec) - y \thetavec^\top\xvec$: the logistic loss
 * on the logit $\thetavec^\top\xvec$ ($\xvec$ carries a trailing 1).
 *
 * @param theta The parameters $\thetavec$, the intercept last.
 * @param x One example's features with the trailing 1.
 * @param y Its label, 0 or 1.
 * @returns The loss of the example.
 */
const logisticLoss: ExampleLoss = (theta, x, y) => {
  const z = sum(mul(theta, x))
  return sub(softplus(z), mul(y, z))
}

/**
 * The design with a column of ones appended, for the intercept.
 *
 * @param x The features, $n \times d$.
 * @returns The $n \times (d + 1)$ design.
 */
const withIntercept = (x: Tensor): Tensor => {
  const [n, d] = x.shape
  const v = toFlat(x)
  const out = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) out[i * (d + 1) + j] = v[i * d + j]
    out[i * (d + 1) + d] = 1
  }
  return fromData(out, [n, d + 1])
}

/**
 * Run the study on `train` and `valid` (binary labels) with `estimator` as the TMC utility's model (such as an
 * `aifn-methods` logistic regression), yielding after the fit, after each method and after every five permutations.
 * The explained logistic regression itself is fitted by L-BFGS, whatever `estimator` is. Throws `DomainError` when
 * either set has no labels.
 *
 * @param train The training points: features $n \times d$ and labels 0 or 1, the points that are valued.
 * @param valid The validation points the utilities and losses are measured on, with the same features.
 * @param estimator The model data Shapley retrains on each subset; its fitted model must have `decide`.
 * @param options The regularisation, KNN-Shapley's neighbours, TMC's length and tolerance, the TracIn run and the
 *   seed, as `ValuationOptions`.
 * @returns A generator of snapshots: each holds every method's values so far; the last has stage `'done'`.
 *
 * @example Which methods find two flipped labels
 * // A tiny estimator for the data-Shapley utility: decide by the nearer class mean.
 * const nearerMean = {
 *   fit({ x, y }) {
 *     const centre = (c) => div(sum(mul(x, reshape(equalTo(y, c), [-1, 1])), 0), sum(equalTo(y, c)))
 *     const [m0, m1] = [centre(0), centre(1)]
 *     return { decide: (q) => greater(sum(square(sub(q, m0)), -1), sum(square(sub(q, m1)), -1)) }
 *   },
 * }
 * // Sixteen training points, class 1 to the right; the labels of points 0 and 8 are flipped.
 * const x = add(normals(stream(0), [16, 2]), tensor(Array.from({ length: 16 }, (_, i) => [i < 8 ? -1.5 : 1.5, 0])))
 * const y = tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
 * const vx = add(normals(stream(1), [20, 2]), tensor(Array.from({ length: 20 }, (_, i) => [i < 10 ? -1.5 : 1.5, 0])))
 * const vy = tensor(Array.from({ length: 20 }, (_, i) => (i < 10 ? 0 : 1)))
 * const study = [...dataValuationStudy({ x, y }, { x: vx, y: vy }, nearerMean, { permutations: 10, steps: 50 })]
 * const last = study[study.length - 1]
 * print('validation accuracy =', last.accuracy)
 * for (const method of VALUATION_METHODS) {
 *   const order = Array.from(last.suspicion[method], (s, i) => [s, i]).sort((a, b) => b[0] - a[0])
 *   print(method, 'suspects most:', order.slice(0, 2).map(([, i]) => i))
 * }
 */
export function* dataValuationStudy(
  train: { readonly x: Tensor; readonly y?: Tensor },
  valid: { readonly x: Tensor; readonly y?: Tensor },
  estimator: Estimator<Supervised<Tensor, Tensor>, unknown>,
  options: ValuationOptions = {},
): Generator<ValuationSnapshot> {
  const {
    l2 = 0.01,
    K = 5,
    permutations = 40,
    tolerance = 0.01,
    steps = 200,
    rate = 0.5,
    checkpoints = 20,
    seed = 0,
  } = options
  if (!train.y || !valid.y) throw new DomainError('dataValuationStudy', 'dataValuationStudy: the data need labels')
  const Xa = withIntercept(train.x)
  const Va = withIntercept(valid.x)
  const [n, p] = Xa.shape
  const y = Float64Array.from(toFlat(train.y))
  const yv = Float64Array.from(toFlat(valid.y))
  const yt = fromData(y, [n])
  // mean ℓ over the rows + (λ/2)‖θ‖², the objective `influenceFunctions` differentiates.
  const objective = (theta: Tensor) => {
    const z = sum(mul(Xa, theta), -1)
    return add(mean(sub(softplus(z), mul(yt, z))), mul(l2 / 2, sum(square(theta))))
  }
  const fitted = minimize({ kind: 'objective', name: 'logistic', dim: p, value: objective }, new Float64Array(p), {
    method: 'lbfgs',
    maxSteps: 500,
  } as never)
  const theta = Float64Array.from(toFlat(fitted.x))
  const V = toFlat(Va)
  let right = 0
  for (let i = 0; i < yv.length; i++) {
    let z = 0
    for (let j = 0; j < p; j++) z += V[i * p + j] * theta[j]
    if ((z > 0 ? 1 : 0) === yv[i]) right++
  }
  const snap: ValuationSnapshot = {
    stage: 'fit',
    done: 0,
    total: permutations,
    theta,
    accuracy: right / yv.length,
    values: {},
    suspicion: {},
    calls: 0,
    truncated: 0,
  }
  const copy = (): ValuationSnapshot => ({ ...snap, values: { ...snap.values }, suspicion: { ...snap.suspicion } })
  yield copy()

  const inf = influenceFunctions(logisticLoss, theta, { x: Xa, y }, { x: Va, y: yv }, { l2 })
  snap.values.influence = inf.removal
  snap.suspicion.influence = Float64Array.from(inf.removal, (r) => -r)
  snap.values['self-influence'] = inf.selfInfluence
  snap.suspicion['self-influence'] = Float64Array.from(inf.selfInfluence)
  snap.stage = 'influence'
  yield copy()

  // Full-batch gradient descent from zero, keeping checkpoints for TracIn.
  const g = grad(objective)
  let th = new Float64Array(p)
  const saved: Float64Array[] = []
  const every = Math.max(1, Math.floor(steps / checkpoints))
  for (let t = 1; t <= steps; t++) {
    const gt = toFlat(g(fromData(Float64Array.from(th), [p])) as Tensor)
    th = Float64Array.from(th, (v, j) => v - rate * gt[j])
    if (t % every === 0) saved.push(Float64Array.from(th))
  }
  const tr = tracIn(logisticLoss, saved, rate * every, { x: Xa, y })
  snap.values.tracin = tr.selfInfluence
  snap.suspicion.tracin = Float64Array.from(tr.selfInfluence)
  snap.stage = 'tracin'
  yield copy()

  const knn = knnShapley({ x: train.x, y }, { x: valid.x, y: yv }, K)
  snap.values['knn-shapley'] = knn.values
  snap.suspicion['knn-shapley'] = Float64Array.from(knn.values, (v) => -v)
  snap.stage = 'knn-shapley'
  yield copy()

  // TMC data Shapley, a block of permutations at a time so the page sees it converge.
  const X = Float64Array.from(toFlat(train.x))
  const d = train.x.shape[1]
  const utility = (indices: readonly number[]): number => {
    if (indices.length === 0) return 0.5
    const labels = indices.map((i) => y[i])
    const first = labels[0]
    if (labels.every((l) => l === first)) return yv.reduce((a, v) => a + (v === first ? 1 : 0), 0) / yv.length
    const xs = new Float64Array(indices.length * d)
    indices.forEach((i, k) => xs.set(X.subarray(i * d, (i + 1) * d), k * d))
    const model = estimator.fit(
      dataset(fromData(xs, [indices.length, d]), fromData(Int32Array.from(labels), [labels.length])),
    )
    const decided = toFlat((model as { decide: (x: Tensor) => Tensor }).decide(valid.x))
    let ok = 0
    for (let i = 0; i < yv.length; i++) if (decided[i] === yv[i]) ok++
    return ok / yv.length
  }
  const block = 5
  const sums = new Float64Array(n)
  let calls = 0
  let truncated = 0
  for (let start = 0; start < permutations; start += block) {
    const m = Math.min(block, permutations - start)
    const r = dataShapley(utility, n, stream(`valuation-${seed}-${start}`), { permutations: m, tolerance })
    for (let i = 0; i < n; i++) sums[i] += r.values[i] * m
    calls += r.calls
    truncated += r.truncated * m
    const done = start + m
    const values = Float64Array.from(sums, (s) => s / done)
    snap.values['data-shapley'] = values
    snap.suspicion['data-shapley'] = Float64Array.from(values, (v) => -v)
    snap.done = done
    snap.calls = calls
    snap.truncated = truncated / done
    snap.stage = 'data-shapley'
    yield copy()
  }
  snap.stage = 'done'
  yield copy()
}
