/**
 * The linear-chain conditional random field (Lafferty, McCallum & Pereira 2001, "Conditional random fields:
 * probabilistic models for segmenting and labeling sequence data", ICML; Sutton & McCallum 2012, "An introduction to
 * conditional random fields", §4–5):
 *
 *   p(y | x) = (1/Z(x)) exp( Σ_n s_{y_0} [n = 0] + Σ_n w_{y_n} · x_n + Σ_{n ≥ 1} T(y_{n−1}, y_n) ),
 *
 * with unary weights W (K × F) on the features x_n (length F), a transition matrix T (K × K) and start weights s (K).
 * Inference is log-space sum-product and max-product on the chain; the gradient of the log-likelihood is the
 * empirical feature counts minus their expectation under the model (Sutton & McCallum, eq. 5.6).
 */

import { fromData, toRows, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { chainTemplate, type StructuredGraph } from 'aifn-compute/graph/structured'
import {
  chainForwardBackward,
  chainViterbi,
  type ChainMarginals,
  type ViterbiResult,
} from 'aifn-compute/inference/exact'
import {
  discreteFactor,
  discreteFactorGraph,
  type DiscreteFactor,
  type DiscreteFactorGraph,
} from 'aifn-compute/inference/model'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The structure of a linear-chain CRF (`aifn-compute/graph/structured`): an undirected chain of labels y_n of length `N`,
 * each with its observed features x_n. Its shape is a chain, which is why the chain engines (forward–backward,
 * Viterbi) are exact for it.
 */
export function crfStructure(): StructuredGraph {
  return chainTemplate('N', { name: 'y', observed: 'x', directed: false, group: 'positions' })
}

/** Parameters of a linear-chain CRF. */
export interface LinearChainCrf {
  /** W, K × F. */
  weights: Matrix
  /** T, K × K: T(u, v) scores label u followed by v. */
  transitions: Matrix
  /** s, length K: scores the first label. */
  start: Vector
}

type Rows = readonly (readonly number[])[]
const asMatrix = (m: Tensor | Rows): Matrix =>
  'shape' in m
    ? (m as Tensor)
    : fromData(Float64Array.from((m as Rows).flat()), [(m as Rows).length, (m as Rows)[0]?.length ?? 0])

/** Build a CRF from arrays or tensors; `start` defaults to zeros. */
export function linearChainCrf(
  weights: Matrix | Rows,
  transitions: Matrix | Rows,
  start?: Vector | readonly number[],
): LinearChainCrf {
  const W = asMatrix(weights)
  const T = asMatrix(transitions)
  const K = W.shape[0]
  if (T.shape[0] !== K || T.shape[1] !== K)
    throw new ShapeError('linearChainCrf', 'linearChainCrf: transitions must be K × K')
  const s =
    start === undefined
      ? new Float64Array(K)
      : 'shape' in start
        ? Float64Array.from(start.data)
        : Float64Array.from(start)
  if (s.length !== K) throw new ShapeError('linearChainCrf', 'linearChainCrf: start must have length K')
  return { weights: W, transitions: T, start: fromData(s, [K]) }
}

/** The log-potentials of a sequence: unary u_n(k) = w_k · x_n (+ s_k at n = 0), N × K, and the pairwise T. */
export function crfPotentials(crf: LinearChainCrf, features: Matrix | Rows): { logUnary: Matrix; logPairwise: Matrix } {
  const X = 'shape' in features ? toRows(features as Tensor) : (features as Rows)
  const W = toRows(crf.weights)
  const K = W.length
  const out = new Float64Array(X.length * K)
  X.forEach((x, n) => {
    for (let k = 0; k < K; k++) {
      let u = n === 0 ? crf.start.data[k] : 0
      for (let f = 0; f < x.length; f++) u += W[k][f] * x[f]
      out[n * K + k] = u
    }
  })
  return { logUnary: fromData(out, [X.length, K]), logPairwise: crf.transitions }
}

/**
 * A CRF on one sequence tabulated as a discrete factor graph: a unary factor exp u_n per label and a pairwise factor
 * exp T per neighbouring pair. Its bipartite graph has the shape of a chain, so `aifn-compute/inference/exact`'s
 * `factorChain` reads the chain potentials back and `chainSumProduct` runs on it.
 */
export function crfFactorGraph(crf: LinearChainCrf, features: Matrix | Rows): DiscreteFactorGraph {
  const { logUnary } = crfPotentials(crf, features)
  const [N, K] = logUnary.shape
  const U = toRows(logUnary)
  const T = toRows(crf.transitions)
  const cards = new Array<number>(N).fill(K)
  const factors: DiscreteFactor[] = []
  for (let n = 0; n < N; n++) factors.push(discreteFactor([n], cards, (a) => Math.exp(U[n][a[0]]), `u${n}`))
  for (let n = 1; n < N; n++)
    factors.push(discreteFactor([n - 1, n], cards, (a) => Math.exp(T[a[0]][a[1]]), `ψ${n - 1},${n}`))
  return discreteFactorGraph(
    cards,
    factors,
    Array.from({ length: N }, (_, n) => `y${n}`),
  )
}

/** Posterior marginals, pairwise marginals and log Z(x) of a CRF on one sequence (the chain engine). */
export function crfMarginals(crf: LinearChainCrf, features: Matrix | Rows): ChainMarginals {
  const { logUnary, logPairwise } = crfPotentials(crf, features)
  return chainForwardBackward(logUnary, logPairwise)
}

/** The most probable labelling of one sequence; `logProbability` is its unnormalised score. */
export function crfViterbi(crf: LinearChainCrf, features: Matrix | Rows): ViterbiResult {
  const { logUnary, logPairwise } = crfPotentials(crf, features)
  return chainViterbi(logUnary, logPairwise)
}

/** The unnormalised score of a labelling: s_{y_0} + Σ_n w_{y_n} · x_n + Σ T(y_{n−1}, y_n). */
export function crfScore(crf: LinearChainCrf, features: Matrix | Rows, labels: ArrayLike<number>): number {
  const { logUnary } = crfPotentials(crf, features)
  const K = logUnary.shape[1]
  const T = crf.transitions.data
  let score = 0
  for (let n = 0; n < labels.length; n++) {
    score += logUnary.data[n * K + labels[n]]
    if (n > 0) score += T[labels[n - 1] * K + labels[n]]
  }
  return score
}

/** log p(y | x) = score(x, y) − log Z(x). */
export function crfLogLikelihood(crf: LinearChainCrf, features: Matrix | Rows, labels: ArrayLike<number>): number {
  return crfScore(crf, features, labels) - crfMarginals(crf, features).logZ
}

/** The gradient of log p(y | x) with respect to each parameter block, and the log-likelihood. */
export interface CrfGradient {
  weights: Matrix
  transitions: Matrix
  start: Vector
  logLikelihood: number
}

/**
 * ∇ log p(y | x): observed minus expected feature counts. ∂/∂W_{kf} = Σ_n ([y_n = k] − p(y_n = k | x)) x_{nf};
 * ∂/∂T_{uv} = Σ_n ([y_{n−1} = u, y_n = v] − p(y_{n−1} = u, y_n = v | x)); ∂/∂s_k = [y_0 = k] − p(y_0 = k | x).
 */
export function crfGradient(crf: LinearChainCrf, features: Matrix | Rows, labels: ArrayLike<number>): CrfGradient {
  const X = 'shape' in features ? toRows(features as Tensor) : (features as Rows)
  const m = crfMarginals(crf, X)
  const [K, F] = crf.weights.shape
  const N = X.length
  const gW = new Float64Array(K * F)
  const gT = new Float64Array(K * K)
  const gS = new Float64Array(K)
  for (let n = 0; n < N; n++)
    for (let k = 0; k < K; k++) {
      const r = (labels[n] === k ? 1 : 0) - m.marginals.data[n * K + k]
      for (let f = 0; f < F; f++) gW[k * F + f] += r * X[n][f]
      if (n === 0) gS[k] = r
    }
  for (let n = 1; n < N; n++) {
    gT[labels[n - 1] * K + labels[n]] += 1
    for (let i = 0; i < K * K; i++) gT[i] -= m.pairwise.data[(n - 1) * K * K + i]
  }
  return {
    weights: fromData(gW, [K, F]),
    transitions: fromData(gT, [K, K]),
    start: fromData(gS, [K]),
    logLikelihood: crfScore(crf, X, labels) - m.logZ,
  }
}
