/**
 * Exact t-SNE (van der Maaten and Hinton, 2008, "Visualizing data using t-SNE", JMLR 9), for small n: Gaussian input
 * affinities calibrated to a perplexity, a Student-t (one degree of freedom) output kernel, and gradient descent on
 * KL(P ‖ Q) with momentum, per-parameter gains and early exaggeration, as scikit-learn's `TSNE(method='exact')`.
 * Every iteration costs O(n²).
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
  /** Conditional probabilities p_{j|i} [n, n] (rows sum to 1, zero diagonal). */
  conditional: Tensor
  /** Precisions βᵢ = 1/(2σᵢ²) [n] found by the search. */
  precisions: Tensor
  /** Bandwidths σᵢ [n]. */
  sigmas: Tensor
  /** Entropy of each row in nats [n]; the target is log(perplexity). */
  entropies: Tensor
  /** Bisection steps each row used [n]. */
  steps: Tensor
}

/**
 * Finds, for each row of the squared distances D² [n, n], the precision β with
 * H(p_{·|i}) = log(perplexity), p_{j|i} ∝ exp(−β D²ᵢⱼ), by bisection (tolerance 1e-5, at most 100 steps), as
 * scikit-learn's `_binary_search_perplexity`.
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

/** The symmetric joint affinities pᵢⱼ = (p_{j|i} + p_{i|j}) / Σ, floored at machine ε as scikit-learn, [n, n]. */
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
  /** The embedding [n, dims]. */
  embedding: Tensor
  /** The last update (the momentum term) and the per-coordinate gains [n, dims]. */
  update: Tensor
  gains: Tensor
  /** KL(P ‖ Q) at this embedding (with the exaggerated P during early exaggeration). */
  kl: number
  /** ‖∂KL/∂Y‖ at the previous embedding (NaN at the start). */
  gradientNorm: number
  /** The exaggeration applied to P in the step that produced this state. */
  exaggeration: number
  /** Gradient iterations done. */
  t: number
}

/** Options of t-SNE's optimisation (scikit-learn's defaults). */
export interface TsneParams {
  dims?: number
  /** Default `auto`: max(n / exaggeration / 4, 50). */
  learningRate?: number | 'auto'
  earlyExaggeration?: number
  /** Iterations of early exaggeration and low momentum (default 250). */
  exaggerationSteps?: number
  momentum?: [number, number]
  minGain?: number
}

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
 * Exact t-SNE on joint affinities P [n, n] as a traceable algorithm; each step is one gradient iteration. `init` takes
 * an embedding, or draws one from N(0, 10⁻⁴) with the `init` stream (scikit-learn's random initialisation). Van der
 * Maaten and Hinton (2008), JMLR 9, with scikit-learn's gains and momentum schedule.
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

/** A fitted t-SNE embedding. */
export interface TsneModel extends Trained<TsneState> {
  readonly kind: 'model'
  /** t-SNE places the training rows only: it has no out-of-sample map. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'tsne'
  readonly embedding: Tensor
  readonly joint: Tensor
  readonly calibration: PerplexityCalibration
  /** KL(P ‖ Q) of the final embedding (without exaggeration). */
  readonly kl: number
}

/** Exact t-SNE of the rows of x: perplexity (default 30, capped below n), `iterations` (default 1000). */
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
