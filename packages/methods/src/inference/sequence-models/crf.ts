/**
 * The linear-chain conditional random field over dense feature vectors: its parameters, potentials, inference,
 * likelihood and gradient.
 *
 * The model (Lafferty, McCallum & Pereira 2001, "Conditional random fields: probabilistic models for segmenting and
 * labeling sequence data", ICML; Sutton & McCallum 2012, "An introduction to conditional random fields", §4–5) labels
 * a sequence of feature vectors $\xvec_0, \dots, \xvec_{N-1}$ (each of length $F$) with labels
 * $y_0, \dots, y_{N-1} \in \{0, \dots, K - 1\}$:
 *
 * $p(\yvec \mid \xvec) = \exp(s_{y_0} + \sum_n \wvec_{y_n}^\top \xvec_n + \sum_{n \ge 1} T_{y_{n-1} y_n}) / Z(\xvec)$,
 *
 * with unary weights $\Wmat$ ($K \times F$, row $\wvec_k$ for label $k$), a transition matrix $\Tmat$ ($K \times K$)
 * and start weights $\svec$ (length $K$). Inference is log-space sum-product and max-product on the chain
 * (`chainForwardBackward` and `chainViterbi` of `aifn-compute/inference/exact`); the gradient of the log-likelihood is
 * the empirical feature counts minus their expectation under the model (Sutton & McCallum, eq. 5.6). A sequence's
 * features are an $N \times F$ matrix, as a tensor or as rows.
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
 * The structure of a linear-chain CRF (`aifn-compute/graph/structured`): an undirected chain of labels $y_n$ of length
 * `N`, each with its observed features $\xvec_n$, in a group `positions`. Its shape is a chain, which is why the chain
 * engines (forward–backward, Viterbi) are exact for it.
 *
 * @returns The compact structured graph, with the length `N` left as a size to bind.
 *
 * @example The compact chain
 * const g = crfStructure()
 * print('nodes:', g.attributes.map((n) => n.name))
 * print('sizes:', g.sizes)
 */
export function crfStructure(): StructuredGraph {
  return chainTemplate('N', { name: 'y', observed: 'x', directed: false, group: 'positions' })
}

/** Parameters of a linear-chain CRF with $K$ labels and $F$ features, as `linearChainCrf` builds them. */
export interface LinearChainCrf {
  /** $\Wmat$ ($K \times F$): row $k$ is $\wvec_k$, the weights of the features for label $k$. */
  weights: Matrix
  /** $\Tmat$ ($K \times K$): $T_{uv}$ scores label $u$ followed by label $v$. */
  transitions: Matrix
  /** $\svec$ (length $K$): $s_k$ scores label $k$ at the first position. */
  start: Vector
}

type Rows = readonly (readonly number[])[]
/**
 * A matrix as given, or an array of rows packed into one (its column count from the first row).
 *
 * @param m A matrix tensor (returned as it is) or an array of equal-length rows.
 * @returns The matrix.
 */
const asMatrix = (m: Tensor | Rows): Matrix =>
  'shape' in m
    ? (m as Tensor)
    : fromData(Float64Array.from((m as Rows).flat()), [(m as Rows).length, (m as Rows)[0]?.length ?? 0])

/**
 * Build a CRF from arrays or tensors. Throws `ShapeError` unless `transitions` is $K \times K$ and `start` has length
 * $K$, with $K$ the number of rows of `weights`.
 *
 * @param weights The unary weights $\Wmat$ ($K \times F$, as rows or a tensor): row $k$ weighs the features for label
 *   $k$.
 * @param transitions The transition scores $\Tmat$ ($K \times K$): $T_{uv}$ for label $u$ followed by $v$.
 * @param start The start scores $\svec$ (length $K$); zeros when left out.
 * @returns The CRF, with its arrays as tensors (`start` copied).
 *
 * @example Two labels that like to repeat
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]])
 * print('W =', crf.weights)
 * print('T =', crf.transitions)
 * print('s =', crf.start)
 */
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

/**
 * The log-potentials of a sequence: the unary $u_n(k) = \wvec_k^\top \xvec_n$, plus $s_k$ at $n = 0$ ($N \times K$),
 * and the pairwise $\Tmat$, shared by every step.
 *
 * @param crf The CRF.
 * @param features The features $\xvec_n$ of the sequence, $N \times F$ (as rows or a tensor): row $n$ is position $n$.
 * @returns `logUnary` ($N \times K$) and `logPairwise`, the CRF's `transitions` itself.
 *
 * @example Three positions
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]], [0.2, 0])
 * const { logUnary, logPairwise } = crfPotentials(crf, [[1, 0], [0.2, 0.8], [0, 1]])
 * print('log unary =', logUnary)
 * print('log pairwise =', logPairwise)
 */
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
 * A CRF on one sequence tabulated as a discrete factor graph: a unary factor $\exp u_n(y_n)$ per position (named
 * `u0`, `u1`, ...) and a pairwise factor $\exp T_{y_{n-1} y_n}$ per neighbouring pair, over variables `y0`, `y1`, ...
 * Its bipartite graph has the shape of a chain, so `aifn-compute/inference/exact`'s `factorChain` reads the chain
 * potentials back and `chainSumProduct` runs on it.
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @returns The factor graph: $N$ variables of $K$ values, $N$ unary factors, then $N - 1$ pairwise ones.
 *
 * @example Three positions, five factors
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]])
 * const g = crfFactorGraph(crf, [[1, 0], [0.2, 0.8], [0, 1]])
 * print('variables:', g.names, 'cardinalities:', g.cardinalities)
 * print('factors:', g.factors.map((f) => `${f.name} on [${f.scope}]`))
 * print('pairwise table:', g.factors[3].table)
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

/**
 * Posterior marginals $p(y_n = k \mid \xvec)$, pairwise marginals and $\log Z(\xvec)$ of a CRF on one sequence, by
 * log-space forward–backward (`chainForwardBackward`).
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @returns The log messages, the marginals ($N \times K$), the pairwise marginals ($(N - 1) \times K \times K$) and
 *   $\log Z(\xvec)$.
 *
 * @example The middle position leans towards its neighbours
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]])
 * const m = crfMarginals(crf, [[1, 0], [0.5, 0.5], [1, 0]])
 * print('p(y_n | x) =', m.marginals)
 * print('log Z =', m.logZ)
 */
export function crfMarginals(crf: LinearChainCrf, features: Matrix | Rows): ChainMarginals {
  const { logUnary, logPairwise } = crfPotentials(crf, features)
  return chainForwardBackward(logUnary, logPairwise)
}

/**
 * The most probable labelling of one sequence, by max-product in log space (`chainViterbi`).
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @returns The path (label ids, length $N$), its `logProbability`, which here is the unnormalised score
 *   (`crfScore` of the path, not $\log p(\yvec \mid \xvec)$), and the Viterbi tables.
 *
 * @example Strong transitions overrule a weak feature
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[2, -2], [-2, 2]])
 * const x = [[1, 0], [0.4, 0.6], [1, 0]]
 * const v = crfViterbi(crf, x)
 * print('path =', v.path, 'score =', v.logProbability)
 * print('crfScore of the path =', crfScore(crf, x, toFlat(v.path)))
 */
export function crfViterbi(crf: LinearChainCrf, features: Matrix | Rows): ViterbiResult {
  const { logUnary, logPairwise } = crfPotentials(crf, features)
  return chainViterbi(logUnary, logPairwise)
}

/**
 * The unnormalised score of a labelling, $s_{y_0} + \sum_n \wvec_{y_n}^\top \xvec_n + \sum_{n \ge 1} T_{y_{n-1} y_n}$.
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @param labels The labelling $\yvec$: one label id in $0, \dots, K - 1$ per position.
 * @returns The score, the log of the labelling's unnormalised probability.
 *
 * @example Agreeing labels score higher
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]])
 * const x = [[1, 0], [0.2, 0.8], [0, 1]]
 * print('[0, 1, 1]:', crfScore(crf, x, [0, 1, 1]))
 * print('[1, 0, 0]:', crfScore(crf, x, [1, 0, 0]))
 */
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

/**
 * The conditional log-likelihood $\log p(\yvec \mid \xvec) = \operatorname{score}(\xvec, \yvec) - \log Z(\xvec)$, the
 * score by `crfScore` and $\log Z$ by `crfMarginals`.
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @param labels The labelling $\yvec$: one label id per position.
 * @returns $\log p(\yvec \mid \xvec)$, at most 0.
 *
 * @example The probabilities of all four labellings of two positions sum to one
 * const crf = linearChainCrf([[1, 0], [0, 1]], [[0.5, -0.5], [-0.5, 0.5]])
 * const x = [[1, 0], [0, 1]]
 * const all = [[0, 0], [0, 1], [1, 0], [1, 1]].map((y) => Math.exp(crfLogLikelihood(crf, x, y)))
 * print('p(y | x) =', all)
 * print('sum =', all.reduce((a, b) => a + b))
 */
export function crfLogLikelihood(crf: LinearChainCrf, features: Matrix | Rows, labels: ArrayLike<number>): number {
  return crfScore(crf, features, labels) - crfMarginals(crf, features).logZ
}

/** The gradient of $\log p(\yvec \mid \xvec)$ with respect to each parameter block, and the log-likelihood. */
export interface CrfGradient {
  /** With respect to $\Wmat$ ($K \times F$). */
  weights: Matrix
  /** With respect to $\Tmat$ ($K \times K$). */
  transitions: Matrix
  /** With respect to $\svec$ (length $K$). */
  start: Vector
  /** $\log p(\yvec \mid \xvec)$ at the current parameters. */
  logLikelihood: number
}

/**
 * The gradient of $\log p(\yvec \mid \xvec)$: observed minus expected feature counts (Sutton & McCallum 2012, eq. 5.6),
 * with the marginals from `crfMarginals`.
 *
 * - $\partial / \partial W_{kf} = \sum_n ([y_n = k] - p(y_n = k \mid \xvec)) x_{nf}$,
 * - $\partial / \partial T_{uv} = \sum_{n \ge 1} ([y_{n-1} = u, y_n = v] - p(y_{n-1} = u, y_n = v \mid \xvec))$,
 * - $\partial / \partial s_k = [y_0 = k] - p(y_0 = k \mid \xvec)$.
 *
 * @param crf The CRF.
 * @param features The features of the sequence, $N \times F$ (as rows or a tensor).
 * @param labels The observed labelling $\yvec$: one label id per position.
 * @returns The gradient of each parameter block, and the log-likelihood.
 *
 * @example Gradient ascent raises the log-likelihood
 * const crf = linearChainCrf([[0, 0], [0, 0]], [[0, 0], [0, 0]])
 * const x = [[1, 0], [0, 1], [0, 1]]
 * const y = [0, 1, 1]
 * const g = crfGradient(crf, x, y)
 * print('dW =', g.weights)
 * print('dT =', g.transitions)
 * const W = add(crf.weights, g.weights)
 * const step = linearChainCrf(W, add(crf.transitions, g.transitions), add(crf.start, g.start))
 * print('log p before:', g.logLikelihood, 'after one step:', crfLogLikelihood(step, x, y))
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
