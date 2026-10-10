/**
 * Exact t-SNE (van der Maaten and Hinton, 2008, "Visualizing data using t-SNE", JMLR 9), for small $n$: Gaussian input
 * affinities calibrated to a perplexity, a Student-t (one degree of freedom) output kernel, and gradient descent on
 * $\KL(\Pmat \Vert \Qmat)$ with momentum, per-parameter gains and early exaggeration, as scikit-learn's
 * `TSNE(method='exact', init='random')`. Every iteration costs $O(n^2)$.
 *
 * The joint affinities are $p_{ij} = (p_{j \mid i} + p_{i \mid j}) / 2n$ with
 * $p_{j \mid i} \propto \exp(-\beta_i \lVert \xvec_i - \xvec_j \rVert^2)$, each $\beta_i$ set so the row's entropy is
 * $\log(\text{perplexity})$. The output affinities are $q_{ij} = w_{ij} / \sum_{k \ne l} w_{kl}$ with
 * $w_{ij} = (1 + \lVert \yvec_i - \yvec_j \rVert^2)^{-1}$, and the gradient is
 * $\partial \KL / \partial \yvec_i = 4 \sum_j (p_{ij} - q_{ij}) w_{ij} (\yvec_i - \yvec_j)$.
 */

import type { Dataset, Estimator, FitOptions, Trained } from 'aifn-compute/learning/estimators'
import type { Status } from 'aifn-compute/foundation/contracts'
import { normals } from 'aifn-compute/foundation/random'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { squaredDistances } from '../neighbourhoods'
import { mat, matrix, square, values, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The per-point calibration of the input affinities. */
export interface PerplexityCalibration {
  /** Conditional probabilities $p_{j \mid i}$ ($n \times n$; row $i$ sums to 1, zero diagonal). */
  conditional: Tensor
  /** Precisions $\beta_i = 1/(2\sigma_i^2)$ ($n$ values) found by the search. */
  precisions: Tensor
  /** Bandwidths $\sigma_i$ ($n$ values). */
  sigmas: Tensor
  /** Entropy of each row in nats ($n$ values); the target is $\log(\text{perplexity})$. */
  entropies: Tensor
  /** Bisection steps each row used ($n$ values; 101 when the 100-step limit ran out). */
  steps: Tensor
}

/**
 * Finds, for each row of the squared distances, the precision $\beta_i$ with
 * $H(p_{\cdot \mid i}) = \log(\text{perplexity})$, $p_{j \mid i} \propto \exp(-\beta_i D_{ij}^2)$, by bisection from
 * $\beta = 1$ (doubling or halving until bracketed; tolerance $10^{-5}$ nats, at most 100 steps), as scikit-learn's
 * `_binary_search_perplexity`. A row that does not converge keeps its last $\beta$, and its entropy shows how far off
 * it is. Throws `ShapeError` when `squared` is not square, and `DomainError` unless $0 <$ perplexity $< n$.
 *
 * @param squared The squared distances $D_{ij}^2$ ($n \times n$); the diagonal is ignored.
 * @param perplexity The target perplexity, the effective number of neighbours of each point.
 * @returns The conditional probabilities and, per row, the precision, bandwidth, entropy and steps the search ended on.
 *
 * @example Each row's entropy is matched to log 2
 * // Four points at 0, 1, 2 and 3 on a line: the end points need wider kernels.
 * const D2 = tensor([[0, 1, 4, 9], [1, 0, 1, 4], [4, 1, 0, 1], [9, 4, 1, 0]])
 * const c = perplexityCalibration(D2, 2)
 * print('entropies =', c.entropies, 'target =', Math.log(2))
 * print('sigmas =', c.sigmas)
 * print('p(j | 0) =', toArray(c.conditional)[0])
 */
export function perplexityCalibration(squared: Tensor, perplexity: number): PerplexityCalibration {
  const { n, v: D2 } = square(squared, 'perplexityCalibration')
  if (!(perplexity > 0 && perplexity < n))
    throw new DomainError('perplexityCalibration', 'perplexityCalibration: perplexity must lie in (0, n)')
  const target = Math.log(perplexity)
  const P = new Float64Array(n * n)
  const betas = new Float64Array(n)
  const entropies = new Float64Array(n)
  const steps = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    let beta = 1
    let lo = -Infinity
    let hi = Infinity
    let h = 0
    let t = 0
    for (; t < 100; t++) {
      let sum = 0
      for (let j = 0; j < n; j++) {
        const p = j === i ? 0 : Math.exp(-D2[i * n + j] * beta)
        P[i * n + j] = p
        sum += p
      }
      if (sum === 0) sum = 1e-8
      let sumDP = 0
      for (let j = 0; j < n; j++) {
        P[i * n + j] /= sum
        sumDP += D2[i * n + j] * P[i * n + j]
      }
      h = Math.log(sum) + beta * sumDP
      if (Math.abs(h - target) <= 1e-5) break
      if (h > target) {
        lo = beta
        beta = hi === Infinity ? beta * 2 : (beta + hi) / 2
      } else {
        hi = beta
        beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2
      }
    }
    betas[i] = beta
    entropies[i] = h
    steps[i] = t + 1
  }
  return {
    conditional: mat(P, n, n),
    precisions: vec(betas),
    sigmas: vec(Array.from(betas, (b) => Math.sqrt(1 / (2 * b)))),
    entropies: vec(entropies),
    steps: vec(steps),
  }
}

/**
 * The symmetric joint affinities of t-SNE, $p_{ij} = (p_{j \mid i} + p_{i \mid j}) / 2n$ from the perplexity
 * calibration of the squared Euclidean distances between rows, with the off-diagonal entries floored at machine
 * $\varepsilon$ as scikit-learn's `_joint_probabilities`. Throws as `perplexityCalibration`, and `ShapeError` when `x`
 * is not a matrix.
 *
 * @param x The points ($n \times d$), one per row.
 * @param perplexity The target perplexity, in $(0, n)$.
 * @returns $\Pmat$ ($n \times n$, symmetric, zero diagonal, summing to 1).
 *
 * @example Four points on a line: neighbours share the most mass
 * const P = jointProbabilities(tensor([[0], [1], [2], [3]]), 2)
 * print('P =', P)
 * print('sum =', sum(P))
 */
export function jointProbabilities(x: Tensor, perplexity: number): Tensor {
  const { n, d, v } = matrix(x, 'jointProbabilities')
  const cond = values(perplexityCalibration(mat(squaredDistances(v, n, d), n, n), perplexity).conditional)
  const P = new Float64Array(n * n)
  let total = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) total += P[i * n + j] = cond[i * n + j] + cond[j * n + i]
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j) P[i * n + j] = Math.max(P[i * n + j] / total, Number.EPSILON)
  return mat(P, n, n)
}

/** One t-SNE state. */
export interface TsneState extends Status {
  /** The embedding ($n \times$ `dims`). */
  embedding: Tensor
  /** The last update, carried as the momentum term ($n \times$ `dims`). */
  update: Tensor
  /** The per-coordinate gains that scale the learning rate ($n \times$ `dims`, at least `minGain`). */
  gains: Tensor
  /**
   * $\KL(\Pmat \Vert \Qmat)$ at this embedding, with $\Pmat$ multiplied by `earlyExaggeration` while the next step is
   * still in the exaggeration phase (but not at the start, $t = 0$).
   */
  kl: number
  /** $\lVert \partial \KL / \partial \Ymat \rVert$ at the previous embedding (NaN at the start). */
  gradientNorm: number
  /** The exaggeration applied to $\Pmat$ in the step that produced this state (`earlyExaggeration` at the start). */
  exaggeration: number
  /** Gradient iterations done. */
  t: number
}

/** Options of t-SNE's optimisation (scikit-learn's defaults). */
export interface TsneParams {
  /** The dimension of the embedding (default 2). */
  dims?: number
  /** The step size $\eta$. Default `'auto'`: $\max(n / \text{earlyExaggeration} / 4, 50)$. */
  learningRate?: number | 'auto'
  /** The factor $\Pmat$ is multiplied by in the first iterations (default 12). */
  earlyExaggeration?: number
  /** Iterations of early exaggeration and low momentum (default 250). */
  exaggerationSteps?: number
  /** The momentum during and after early exaggeration (default `[0.5, 0.8]`). */
  momentum?: [number, number]
  /** The floor of the per-coordinate gains (default 0.01). */
  minGain?: number
}

/**
 * $\KL(\Pmat \Vert \Qmat)$ and its gradient for an embedding, with the Student-t output kernel; $q_{ij}$ and the
 * logarithm's argument are floored at machine $\varepsilon$.
 *
 * @param P The joint affinities $\Pmat$ as a row-major array of $n^2$ values.
 * @param Y The embedding as a row-major array of $n \times$ `dims` values.
 * @param n The number of points.
 * @param dims The dimension of the embedding.
 * @param exaggeration The factor $\Pmat$ is multiplied by (1 for the plain divergence).
 * @returns `kl`, the divergence, and `grad`, its gradient with respect to `Y` (row-major, $n \times$ `dims`).
 */
function klAndGradient(P: Float64Array, Y: Float64Array, n: number, dims: number, exaggeration: number) {
  const W = new Float64Array(n * n)
  let z = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0
      for (let c = 0; c < dims; c++) s += (Y[i * dims + c] - Y[j * dims + c]) ** 2
      const w = 1 / (1 + s)
      W[i * n + j] = W[j * n + i] = w
      z += 2 * w
    }
  }
  const grad = new Float64Array(n * dims)
  let kl = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const p = exaggeration * P[i * n + j]
      const q = Math.max(W[i * n + j] / z, Number.EPSILON)
      kl += p * Math.log(Math.max(p, Number.EPSILON) / q)
      const f = 4 * (p - q) * W[i * n + j]
      for (let c = 0; c < dims; c++) grad[i * dims + c] += f * (Y[i * dims + c] - Y[j * dims + c])
    }
  }
  return { kl, grad }
}

/**
 * Exact t-SNE on joint affinities as a traceable algorithm; each step is one gradient iteration (van der Maaten and
 * Hinton, 2008), with scikit-learn's gains (Jacobs, 1988), momentum schedule and learning rate. Unlike scikit-learn,
 * which restarts them, the update and gains carry over from the exaggeration phase into the rest of the run. `init`
 * takes an embedding, or draws one from the run's stream, normal with standard deviation $10^{-4}$ (scikit-learn's
 * `init='random'`). There is no convergence test: the run goes on for as many steps as it is given. Throws
 * `ShapeError` when `joint` is not square.
 *
 * @param joint The joint affinities $\Pmat$ ($n \times n$, symmetric, summing to 1), as `jointProbabilities` returns.
 * @param params The optimisation settings.
 * @returns The algorithm, for `run` or `trace`; its states are `TsneState`s.
 *
 * @example The divergence falls once the exaggeration ends
 * const x = concat([normals(stream(1), [5, 3]), add(normals(stream(2), [5, 3]), 10)], 0)
 * const steps = tsneSteps(jointProbabilities(x, 3), { exaggerationSteps: 50 })
 * print('KL at the start =', run(steps, {}, 0, { stream: stream(3) }).kl)
 * print('after 100 iterations =', run(steps, {}, 100, { stream: stream(3) }).kl)
 * print('after 200 iterations =', run(steps, {}, 200, { stream: stream(3) }).kl)
 */
export function tsneSteps(joint: Tensor, params: TsneParams = {}): Algorithm<{ embedding?: Tensor }, TsneState> {
  const { n, v: P } = square(joint, 'tsneSteps')
  const { dims = 2, earlyExaggeration = 12, exaggerationSteps = 250, momentum = [0.5, 0.8], minGain = 0.01 } = params
  const eta =
    params.learningRate === undefined || params.learningRate === 'auto'
      ? Math.max(n / earlyExaggeration / 4, 50)
      : params.learningRate
  return {
    name: 'tsne',
    init: ({ embedding } = {}, s) => {
      const Y = embedding
        ? Float64Array.from(values(embedding))
        : Float64Array.from(values(normals(s, [n, dims], 0, 1e-4)))
      return {
        embedding: mat(Y, n, dims),
        update: mat(new Float64Array(n * dims), n, dims),
        gains: mat(new Float64Array(n * dims).fill(1), n, dims),
        kl: klAndGradient(P, Y, n, dims, 1).kl,
        gradientNorm: NaN,
        exaggeration: earlyExaggeration,
        t: 0,
      }
    },
    step: (state) => {
      const early = state.t < exaggerationSteps
      const ex = early ? earlyExaggeration : 1
      const mom = early ? momentum[0] : momentum[1]
      const Y = Float64Array.from(values(state.embedding))
      const update = Float64Array.from(values(state.update))
      const gains = Float64Array.from(values(state.gains))
      const { grad } = klAndGradient(P, Y, n, dims, ex)
      let norm = 0
      for (let t = 0; t < Y.length; t++) {
        norm += grad[t] * grad[t]
        // Gains grow where the gradient reverses the last update and shrink where it continues it (Jacobs, 1988).
        gains[t] = update[t] * grad[t] < 0 ? gains[t] + 0.2 : gains[t] * 0.8
        if (gains[t] < minGain) gains[t] = minGain
        update[t] = mom * update[t] - eta * gains[t] * grad[t]
        Y[t] += update[t]
      }
      const kl = klAndGradient(P, Y, n, dims, state.t + 1 < exaggerationSteps ? earlyExaggeration : 1).kl
      return {
        embedding: mat(Y, n, dims),
        update: mat(update, n, dims),
        gains: mat(gains, n, dims),
        kl,
        gradientNorm: Math.sqrt(norm),
        exaggeration: ex,
        t: state.t + 1,
        diverged: !Number.isFinite(kl),
      }
    },
  }
}

/** A fitted t-SNE embedding, with its run (`training`). */
export interface TsneModel extends Trained<TsneState> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** t-SNE places the training rows only: it has no out-of-sample map. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'tsne'
  /** The coordinates of the training rows ($n \times$ `dims`). */
  readonly embedding: Tensor
  /** The joint affinities $\Pmat$ ($n \times n$). */
  readonly joint: Tensor
  /** The perplexity calibration of the input affinities. */
  readonly calibration: PerplexityCalibration
  /** $\KL(\Pmat \Vert \Qmat)$ of the final embedding (without exaggeration). */
  readonly kl: number
}

/**
 * Exact t-SNE of the training rows (van der Maaten and Hinton, 2008): the joint affinities at the given perplexity,
 * then `iterations` gradient steps of `tsneSteps` from a random start drawn from the fit options' `stream`. The run is
 * traced every 10 iterations (or every `trace.every` of the fit options). Distances between clusters in the result are
 * not meaningful, only which points are together.
 *
 * @param params The optimisation settings of `TsneParams`, and:
 * @param params.perplexity The effective number of neighbours (default 30), capped at $(n - 1)/3$.
 * @param params.iterations The number of gradient iterations, early exaggeration included (default 1000).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `TsneModel`.
 *
 * @example Two well-separated blobs stay apart
 * // Two blobs of 10 points in three dimensions, their centres 17 apart.
 * const x = concat([normals(stream(1), [10, 3]), add(normals(stream(2), [10, 3]), 10)], 0)
 * const model = tsne({ perplexity: 5, iterations: 300, exaggerationSteps: 100 }).fit({ x }, { stream: stream(3) })
 * const Y = toArray(model.embedding)
 * const centre = (rows) => [0, 1].map((c) => rows.reduce((s, r) => s + r[c], 0) / rows.length)
 * const blobs = [Y.slice(0, 10), Y.slice(10)]
 * const [a, b] = blobs.map(centre)
 * const radius = (rows, m) => Math.max(...rows.map((r) => Math.hypot(r[0] - m[0], r[1] - m[1])))
 * print('distance between the blob centres =', Math.hypot(a[0] - b[0], a[1] - b[1]))
 * print('largest distance of a point from its centre =', Math.max(radius(blobs[0], a), radius(blobs[1], b)))
 * print('KL =', model.kl)
 */
export function tsne(
  params: TsneParams & { perplexity?: number; iterations?: number } = {},
): Estimator<Dataset<Tensor>, TsneModel> {
  const { perplexity = 30, iterations = 1000, ...rest } = params
  return {
    name: 'tsne',
    params: { perplexity, iterations, ...rest },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'tsne')
      const perp = Math.min(perplexity, (n - 1) / 3)
      const calibration = perplexityCalibration(mat(squaredDistances(v, n, d), n, n), perp)
      const joint = jointProbabilities(x, perp)
      const training = trace(tsneSteps(joint, rest), {}, iterations, {
        stream: options.stream,
        every: options.trace?.every ?? 10,
        record: { kl: (s) => s.kl },
      })
      const final = training.final
      const Y = values(final.embedding)
      return {
        kind: 'model',
        transductive: true,
        name: 'tsne',
        embedding: final.embedding,
        joint,
        calibration,
        kl: klAndGradient(values(joint), Y, n, final.embedding.shape[1], 1).kl,
        training,
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'tsne',
    module: 'unsupervised/embedding/neighbour',
    name: 't-SNE',
    summary:
      'Matching Gaussian neighbour probabilities with Student-t ones in the embedding, by gradient descent on their KL divergence.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      perplexity: real(2, 100, { default: 30 }),
      iterations: int(10, 5000, { default: 1000 }),
      dims: int(1, 3, { default: 2 }),
      earlyExaggeration: real(1, 50, { default: 12 }),
    }),
    notes: ['t-distributed-stochastic-neighbour-embedding'],
    cite: ['vandermaaten2008'],
  },
  tsne,
)
