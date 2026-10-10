/**
 * Chain engines: scaled forward–backward, Viterbi and forward-filtering backward-sampling on a chain of potentials
 * (Rabiner 1989, "A tutorial on hidden Markov models", Proc. IEEE 77(2), §III; Durbin, Eddy, Krogh & Mitchison 1998,
 * "Biological Sequence Analysis", ch. 3), log-space sum-product and max-product on a chain of log-potentials (which
 * the linear-chain CRF runs), and chain-shaped discrete factor graphs. Named chain models (the HMM, the CRF) are
 * applications in `aifn-methods/inference/sequence-models`; they build the potentials these engines take.
 *
 * Notation (Twomey, Diethe & Flach 2016): positions $n = 0, \dots, N - 1$, states $k = 0, \dots, K - 1$. $\alpha_n$
 * is the forward message into position $n$ and $\beta_n$ the backward message; the posterior marginal is
 * $\propto \alpha_n \odot \psi_n \odot \beta_n$ with $\psi_n$ the node potentials (for an HMM the emission
 * likelihoods, with the initial distribution $\pi$ folded into $\psi_0$). The engines on log-potentials write
 * $\phi_n(k)$ for the unary log-potential of state $k$ at position $n$ and $\phi_{n,n+1}(u, v)$ for the pairwise one
 * between positions $n$ and $n + 1$.
 *
 * State paths are int32 vectors of length $N$, and every matrix is $N \times K$ with row $n$ for position $n$. Ties
 * between states go to the smaller one.
 */

import type { Index, Size, Status } from 'aifn-compute/foundation/contracts'
import { categorical, child, type Stream } from 'aifn-compute/foundation/random'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, log, toFlat, toRows, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { chainOrder } from 'aifn-compute/graph/structured'
import { bipartiteGraph, valuesOf, type DiscreteFactorGraph } from 'aifn-compute/inference/model'

/**
 * A float64 matrix from rows of equal length.
 *
 * @param rows The rows, each of `k` values; they are copied, not kept.
 * @param k The number of columns (default: the length of the first row, or 0 when there are no rows).
 * @returns The `rows.length` $\times$ `k` matrix.
 */
const matrix = (rows: readonly ArrayLike<number>[], k = rows[0]?.length ?? 0): Matrix => {
  const out = new Float64Array(rows.length * k)
  rows.forEach((r, i) => out.set(r, i * k))
  return fromData(out, [rows.length, k])
}

/**
 * A chain of non-negative potentials: $p(\yvec) \propto \prod_n \psi_n(y_n) \prod_n A(y_n, y_{n+1})$ over state paths
 * $\yvec$ of length $N$ with $K$ states. For a hidden Markov model $\psi_n(k) = B_{k, x_n}$ (times $\pi_k$ at $n = 0$)
 * and $\Amat$ is its transition matrix, so the normaliser is the likelihood $p(\xvec)$.
 */
export interface ChainPotentials {
  /** The node potentials $\psi$ ($N \times K$): row $n$ holds $\psi_n(k)$, with any initial distribution in row 0. */
  nodePotentials: Matrix
  /** The transition potentials $\Amat$ ($K \times K$), $A_{uv}$ from state $u$ to state $v$, shared by every step. */
  transition: Matrix
}

/**
 * Check that $\psi$ is $N \times K$ and $\Amat$ is $K \times K$; throws `ShapeError` otherwise.
 *
 * @param chain The chain whose shapes are checked.
 * @param where The caller's name for the error message.
 */
function checkChain(chain: ChainPotentials, where: string): void {
  const [, K] = chain.nodePotentials.shape
  const [r, c] = chain.transition.shape
  if (chain.nodePotentials.shape.length !== 2 || r !== K || c !== K)
    throw new ShapeError(
      where,
      `${where}: node potentials must be N × K and the transition K × K, got [${chain.nodePotentials.shape.join(', ')}] and [${chain.transition.shape.join(', ')}]`,
    )
}

/** The result of the scaled forward–backward recursions. */
export interface ForwardBackwardResult {
  /**
   * $\alpha_n = \Amat^\top \gamma_{n-1}$ ($N \times K$), with $\gamma$ the filtered distribution, so rows $n \ge 1$ sum
   * to one when the rows of $\Amat$ do; $\alpha_0 = \ones$ ($\pi$ is in $\psi_0$).
   */
  alpha: Matrix
  /**
   * $\beta_n$ ($N \times K$), scaled by the same constants, so that $\gamma_n \odot \beta_n$ (that is,
   * $\alpha_n \odot \psi_n \odot \beta_n / c_n$) sums to one; $\beta_{N-1} = \ones$.
   */
  beta: Matrix
  /** $\psi_n$ ($N \times K$), the node potentials as given. */
  psi: Matrix
  /** Filtering distributions $\gamma_n = p(y_n \mid x_0, \dots, x_n)$ ($N \times K$). */
  filtered: Matrix
  /** Smoothing marginals $p(y_n \mid \xvec)$ ($N \times K$). */
  marginals: Matrix
  /** Pairwise marginals $\xi_n(u, v) = p(y_n = u, y_{n+1} = v \mid \xvec)$, $(N - 1) \times K \times K$. */
  pairwise: Tensor
  /** $c_n$, the normaliser of $\alpha_n \odot \psi_n$ (length $N$); $\log Z = \sum_n \log c_n$. */
  scale: Vector
  /** $\log Z = \sum_n \log c_n$, the log normaliser of the chain ($\log p(\xvec)$ for an HMM). */
  logLikelihood: number
}

/**
 * Forward–backward with per-step scaling (Rabiner 1989, §V.A) on a chain of potentials:
 * $\alpha_n = \Amat^\top \gamma_{n-1}$ with $\gamma_{n-1}$ the filtered distribution,
 * $c_n = \sum_k \alpha_n(k) \psi_n(k)$, $\gamma_n = \alpha_n \odot \psi_n / c_n$, and
 * $\beta_n = \Amat (\beta_{n+1} \odot \psi_{n+1}) / c_{n+1}$; $\log Z = \sum_n \log c_n$. Scaling keeps every
 * message in range however long the chain. Throws `ShapeError` when the shapes do not match. Not differentiable (it
 * works on plain arrays).
 *
 * @param chain The node potentials $\psi$ ($N \times K$) and the transition $\Amat$ ($K \times K$).
 * @returns The messages, the filtered, smoothed and pairwise marginals, the scale factors and $\log Z$.
 *
 * @example Two steps, checked by hand
 * // A uniform start, then an observation that pins y₁ = 0. By hand, p(y₀ | y₁ = 0) ∝ 0.5 A(·, 0) = [0.45, 0.1] and
 * // Z = 0.45 + 0.1.
 * const chain = {
 *   nodePotentials: tensor([[0.5, 0.5], [1, 0]]),
 *   transition: tensor([[0.9, 0.1], [0.2, 0.8]]),
 * }
 * const fb = forwardBackward(chain)
 * print('marginals =', fb.marginals)
 * print('Z =', Math.exp(fb.logLikelihood))
 *
 * @example Filtering looks back, smoothing uses the whole sequence
 * // An HMM with a uniform start: emissions B = [[0.9, 0.1], [0.2, 0.8]] and observations x = [0, 1, 0].
 * const B = [[0.9, 0.1], [0.2, 0.8]]
 * const x = [0, 1, 0]
 * const psi = x.map((xn, n) => [0, 1].map((k) => (n === 0 ? 0.5 : 1) * B[k][xn]))
 * const fb = forwardBackward({ nodePotentials: tensor(psi), transition: tensor([[0.8, 0.2], [0.2, 0.8]]) })
 * print('filtered =', fb.filtered)
 * print('smoothed =', fb.marginals)
 * print('log p(x) =', fb.logLikelihood)
 */
export function forwardBackward(chain: ChainPotentials): ForwardBackwardResult {
  checkChain(chain, 'forwardBackward')
  return scaledForwardBackward(toRows(chain.nodePotentials), toRows(chain.transition))
}

/**
 * The scaled forward–backward recursions of `forwardBackward` on plain rows, without the shape check.
 *
 * @param psi The node potentials $\psi$ as $N$ rows of $K$ values; read, not modified.
 * @param A The transition $\Amat$ as $K$ rows of $K$ values; read, not modified.
 * @returns The full result of `forwardBackward`.
 */
function scaledForwardBackward(psi: number[][], A: number[][]): ForwardBackwardResult {
  const N = psi.length
  const K = A.length
  const alpha: number[][] = []
  const filtered: number[][] = []
  const scale: number[] = []
  for (let n = 0; n < N; n++) {
    const a =
      n === 0
        ? Array(K).fill(1)
        : Array.from({ length: K }, (_, v) => filtered[n - 1].reduce((s, g, u) => s + g * A[u][v], 0))
    const g = a.map((ai, v) => ai * psi[n][v])
    const c = g.reduce((s, x) => s + x, 0)
    scale.push(c)
    alpha.push(a)
    filtered.push(g.map((x) => x / c))
  }
  const beta: number[][] = Array.from({ length: N }, () => Array(K).fill(1))
  for (let n = N - 2; n >= 0; n--) {
    const d = beta[n + 1].map((b, v) => b * psi[n + 1][v])
    beta[n] = Array.from({ length: K }, (_, u) => A[u].reduce((s, auv, v) => s + auv * d[v], 0) / scale[n + 1])
  }
  const marginals = filtered.map((f, n) => {
    const p = f.map((fi, v) => fi * beta[n][v])
    const z = p.reduce((s, x) => s + x, 0)
    return p.map((x) => x / z)
  })
  const pairwise = new Float64Array(Math.max(N - 1, 0) * K * K)
  for (let n = 0; n + 1 < N; n++) {
    let z = 0
    for (let u = 0; u < K; u++)
      for (let v = 0; v < K; v++) {
        const x = filtered[n][u] * A[u][v] * psi[n + 1][v] * beta[n + 1][v]
        pairwise[(n * K + u) * K + v] = x
        z += x
      }
    for (let i = 0; i < K * K; i++) pairwise[n * K * K + i] /= z
  }
  return {
    alpha: matrix(alpha, K),
    beta: matrix(beta, K),
    psi: matrix(psi, K),
    filtered: matrix(filtered, K),
    marginals: matrix(marginals, K),
    pairwise: fromData(pairwise, [Math.max(N - 1, 0), K, K]),
    scale: fromData(Float64Array.from(scale), [N]),
    logLikelihood: scale.reduce((s, c) => s + Math.log(c), 0),
  }
}

/** The result of Viterbi decoding. */
export interface ViterbiResult {
  /** The most probable state path (int32, length $N$). */
  path: Vector
  /**
   * Its log score: the log joint $\log p(\xvec, \yvec)$ for an HMM, the unnormalised log score for a chain of
   * potentials or of log-potentials.
   */
  logProbability: number
  /** $\delta_n(k)$: the best log score of a path ending in state $k$ at position $n$ ($N \times K$). */
  delta: Matrix
  /** Back-pointers: the best predecessor of state $k$ at position $n$ (int32, $N \times K$; row 0 is $-1$). */
  backpointers: Tensor
}

/**
 * Max-product (Viterbi) on a chain in log space: $\delta_0 = \phi_0$,
 * $\delta_n(v) = \max_u [\delta_{n-1}(u) + \phi_{n-1,n}(u, v)] + \phi_n(v)$, then a back-trace from the best final
 * state. Ties go to the smaller state. The engine a linear-chain CRF decodes with; `viterbi` calls it on the logs of a
 * chain of potentials. A log-potential of $-\infty$ forbids a state or a transition.
 *
 * @param logUnary The unary log-potentials $\phi_n(k)$ ($N \times K$, $N \ge 1$).
 * @param logPairwise The pairwise log-potentials: one $K \times K$ matrix $\phi(u, v)$ shared by every step, or a
 *   $(N - 1) \times K \times K$ tensor whose slice $n$ is $\phi_{n,n+1}$.
 * @returns The best path, its log score, the $\delta$ table and the back-pointers.
 *
 * @example A path that the marginals would not choose
 * // The joint over two binary positions is the pairwise table: p(0, 0) = 0.4, p(0, 1) = 0.3, p(1, 1) = 0.3.
 * const v = chainViterbi(tensor([[0, 0], [0, 0]]), log(tensor([[0.4, 0.3], [0, 0.3]])))
 * print('path =', v.path)
 * print('probability =', Math.exp(v.logProbability))
 * print('delta =', v.delta)
 *
 * @example A potential per step
 * // Three positions, three states; the second step forbids staying put.
 * const stay = [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]
 * const move = [[-Infinity, 0, 0], [0, -Infinity, 0], [0, 0, -Infinity]]
 * const v = chainViterbi(tensor([[2, 0, 0], [0, 0, 0], [0, 0, 0]]), tensor([stay, move]))
 * print('path =', v.path)
 * print('log score =', v.logProbability)
 */
export function chainViterbi(logUnary: Matrix, logPairwise: Tensor): ViterbiResult {
  const [N, K] = logUnary.shape
  checkPositions(N, 'chainViterbi')
  const U = toRows(logUnary)
  const pair = pairwiseAt(logPairwise, K)
  const delta: number[][] = [U[0]]
  const back = new Int32Array(N * K).fill(-1)
  for (let n = 1; n < N; n++) {
    const prev = delta[n - 1]
    const P = pair(n - 1)
    delta.push(
      Array.from({ length: K }, (_, v) => {
        let best = 0
        for (let u = 1; u < K; u++) if (prev[u] + P[u][v] > prev[best] + P[best][v]) best = u
        back[n * K + v] = best
        return prev[best] + P[best][v] + U[n][v]
      }),
    )
  }
  let last = 0
  for (let k = 1; k < K; k++) if (delta[N - 1][k] > delta[N - 1][last]) last = k
  const path = new Int32Array(N)
  path[N - 1] = last
  for (let n = N - 1; n > 0; n--) path[n - 1] = back[n * K + path[n]]
  return {
    path: fromData(path, [N]),
    logProbability: delta[N - 1][last],
    delta: matrix(delta, K),
    backpointers: fromData(back, [N, K]),
  }
}

/**
 * Throw `ShapeError` unless a chain has at least one position.
 *
 * @param N The number of positions (rows of the unary log-potentials).
 * @param where The caller's name for the error message.
 */
function checkPositions(N: number, where: string): void {
  if (!(N >= 1)) throw new ShapeError(where, `${where}: the chain needs at least one position (N >= 1), got ${N}`)
}

/**
 * The pairwise log-potential matrix between positions $n$ and $n + 1$, as a lookup by $n$.
 *
 * @param logPairwise One shared $K \times K$ matrix, or an $(N - 1) \times K \times K$ tensor with a matrix per step
 *   (read in row-major order, whatever its strides).
 * @param K The number of states.
 * @returns A function from $n$ to the $K \times K$ matrix (as rows) between positions $n$ and $n + 1$; for a shared
 *   matrix it returns the same rows every time.
 */
function pairwiseAt(logPairwise: Tensor, K: number): (n: number) => number[][] {
  if (logPairwise.shape.length === 2) {
    const P = toRows(logPairwise)
    return () => P
  }
  const all = toFlat(logPairwise)
  return (n) => Array.from({ length: K }, (_, u) => all.slice((n * K + u) * K, (n * K + u + 1) * K))
}

/**
 * Viterbi decoding of a chain of potentials (Rabiner 1989, §III.B): the most probable state path $\yvec^*$ and its
 * unnormalised log score $\sum_n \log \psi_n(y_n) + \sum_n \log A(y_n, y_{n+1})$ ($\log p(\xvec, \yvec^*)$ for an HMM),
 * by `chainViterbi` on the logs of the potentials. Ties go to the smaller state. Throws `ShapeError` when the shapes do
 * not match.
 *
 * @param chain The node potentials $\psi$ ($N \times K$) and the transition $\Amat$ ($K \times K$); zeros are allowed
 *   and forbid a state or a transition.
 * @returns The best path, its log score, the $\delta$ table and the back-pointers.
 *
 * @example The most probable weather
 * // States 0 (dry) and 1 (wet), a uniform start, sticky weather, and emissions B = [[0.9, 0.1], [0.2, 0.8]] for the
 * // observations x = [0, 1, 1, 0].
 * const chain = {
 *   nodePotentials: tensor([[0.45, 0.1], [0.1, 0.8], [0.1, 0.8], [0.9, 0.2]]),
 *   transition: tensor([[0.8, 0.2], [0.2, 0.8]]),
 * }
 * const v = viterbi(chain)
 * print('path =', v.path)
 * print('p(x, path) =', Math.exp(v.logProbability))
 */
export function viterbi(chain: ChainPotentials): ViterbiResult {
  checkChain(chain, 'viterbi')
  return chainViterbi(log(chain.nodePotentials), log(chain.transition))
}

/**
 * $\log \sum_i \exp x_i$ over a plain array, shifted by the maximum ($-\infty$ for an empty or all $-\infty$ input;
 * $+\infty$ if any is $+\infty$).
 *
 * @param xs The values $x_i$ (logs); read, not modified.
 * @returns The log of the sum of their exponentials.
 */
function lse(xs: readonly number[]): number {
  let m = -Infinity
  for (const x of xs) if (x > m) m = x
  if (m === -Infinity || m === Infinity || Number.isNaN(m)) return m
  let s = 0
  for (const x of xs) s += Math.exp(x - m)
  return m + Math.log(s)
}

/** The result of log-space forward–backward on a chain of log-potentials. */
export interface ChainMarginals {
  /**
   * $\log \alpha_n$ ($N \times K$): the log of the sum over the paths into position $n$ of their potentials, excluding
   * $\phi_n$.
   */
  logAlpha: Matrix
  /**
   * $\log \beta_n$ ($N \times K$): the log of the sum over the paths out of position $n$ of their potentials, excluding
   * $\phi_n$.
   */
  logBeta: Matrix
  /** The marginals $p(y_n = k)$ ($N \times K$; each row sums to one). */
  marginals: Matrix
  /** The pairwise marginals $p(y_n = u, y_{n+1} = v)$, $(N - 1) \times K \times K$. */
  pairwise: Tensor
  /** The log normaliser $\log Z$ of the chain. */
  logZ: number
}

/**
 * Sum-product (forward–backward) on a chain in log space, the engine a linear-chain CRF runs: $\log \alpha_0 = 0$,
 * $\log \alpha_n(v) = \operatorname{logsumexp}_u [\log \alpha_{n-1}(u) + \phi_{n-1}(u) + \phi_{n-1,n}(u, v)]$,
 * $\log \beta_{N-1} = 0$,
 * $\log \beta_n(u) = \operatorname{logsumexp}_v [\phi_{n,n+1}(u, v) + \phi_{n+1}(v) + \log \beta_{n+1}(v)]$, and
 * $\log Z = \operatorname{logsumexp}_k [\log \alpha_0(k) + \phi_0(k) + \log \beta_0(k)]$.
 * Working in logs needs no scaling. A log-potential of $-\infty$ forbids a state or a transition; when every path is
 * forbidden, $\log Z = -\infty$ and the marginals are NaN.
 *
 * @param logUnary The unary log-potentials $\phi_n(k)$ ($N \times K$, $N \ge 1$).
 * @param logPairwise The pairwise log-potentials: one $K \times K$ matrix shared by every step, or a
 *   $(N - 1) \times K \times K$ tensor whose slice $n$ is $\phi_{n,n+1}$.
 * @returns The log messages, the marginals, the pairwise marginals and $\log Z$.
 *
 * @example Two binary positions, checked by hand
 * // With zero unary terms the joint is the pairwise table: p(0, 0) = 0.4, p(0, 1) = 0.3, p(1, 1) = 0.3. Its rows
 * // sum to p(y₀) and its columns to p(y₁).
 * const m = chainForwardBackward(tensor([[0, 0], [0, 0]]), log(tensor([[0.4, 0.3], [0, 0.3]])))
 * print('marginals =', m.marginals)
 * print('pairwise =', m.pairwise)
 * print('Z =', Math.exp(m.logZ))
 */
export function chainForwardBackward(logUnary: Matrix, logPairwise: Tensor): ChainMarginals {
  const [N, K] = logUnary.shape
  checkPositions(N, 'chainForwardBackward')
  const U = toRows(logUnary)
  const pair = pairwiseAt(logPairwise, K)
  const la: number[][] = [Array(K).fill(0)]
  for (let n = 1; n < N; n++) {
    const P = pair(n - 1)
    la.push(Array.from({ length: K }, (_, v) => lse(la[n - 1].map((a, u) => a + U[n - 1][u] + P[u][v]))))
  }
  const lb: number[][] = Array.from({ length: N }, () => Array(K).fill(0))
  for (let n = N - 2; n >= 0; n--) {
    const P = pair(n)
    lb[n] = Array.from({ length: K }, (_, u) => lse(lb[n + 1].map((b, v) => P[u][v] + U[n + 1][v] + b)))
  }
  const logZ = lse(la[0].map((a, k) => a + U[0][k] + lb[0][k]))
  const marginals = la.map((row, n) => row.map((a, k) => Math.exp(a + U[n][k] + lb[n][k] - logZ)))
  const pairwise = new Float64Array(Math.max(N - 1, 0) * K * K)
  for (let n = 0; n + 1 < N; n++) {
    const P = pair(n)
    for (let u = 0; u < K; u++)
      for (let v = 0; v < K; v++)
        pairwise[(n * K + u) * K + v] = Math.exp(la[n][u] + U[n][u] + P[u][v] + U[n + 1][v] + lb[n + 1][v] - logZ)
  }
  return {
    logAlpha: matrix(la, K),
    logBeta: matrix(lb, K),
    marginals: matrix(marginals, K),
    pairwise: fromData(pairwise, [Math.max(N - 1, 0), K, K]),
    logZ,
  }
}

/** The result of posterior (max-marginal) decoding. */
export interface PosteriorDecoding {
  /** $\hat{y}_n = \argmax_k P(y_n = k \mid \xvec)$ (int32, length $N$); ties go to the smaller state. */
  path: Vector
  /** $P(y_n = \hat{y}_n \mid \xvec)$ per position (length $N$). */
  confidence: Vector
  /** $\sum_n P(y_n = \hat{y}_n \mid \xvec)$: the expected number of correct labels, the largest any labelling has. */
  expectedCorrect: number
}

/**
 * Posterior (marginal, max-marginal) decoding (Rabiner 1989, §III.B, "individually most likely states"): each position
 * takes its most probable label under the marginals, $\hat{y}_n = \argmax_k P(y_n = k \mid \xvec)$. It maximises the
 * expected number of correct labels $\sum_n P(y_n = \hat{y}_n \mid \xvec)$, where Viterbi maximises the probability
 * of the whole sequence; the two can differ, and the posterior path may even contain a transition of zero probability
 * (a log-potential of $-\infty$).
 *
 * @param marginals The marginals $P(y_n = k \mid \xvec)$ ($N \times K$, row $n$ for position $n$), as
 *   `forwardBackward` or `chainForwardBackward` return them.
 * @returns The decoded path, the probability of each of its labels, and their sum.
 *
 * @example Posterior decoding against Viterbi
 * // p(0, 0) = 0.4, p(0, 1) = 0.3, p(1, 1) = 0.3: Viterbi takes the single best pair (0, 0), while position by
 * // position y₀ = 0 and y₁ = 1 are each most probable.
 * const unary = tensor([[0, 0], [0, 0]])
 * const pair = log(tensor([[0.4, 0.3], [0, 0.3]]))
 * const d = posteriorDecode(chainForwardBackward(unary, pair).marginals)
 * print('posterior path =', d.path)
 * print('confidence =', d.confidence)
 * print('expected correct =', d.expectedCorrect)
 * print('Viterbi path =', chainViterbi(unary, pair).path)
 *
 * @example The posterior path can be impossible
 * // Three states: the pair (0, 0) has probability 0, yet 0 is the most probable label at both positions.
 * const pair = tensor([[0, 0.2, 0.2], [0.2, 0.1, 0], [0.2, 0, 0.1]])
 * const m = chainForwardBackward(tensor([[0, 0, 0], [0, 0, 0]]), log(pair))
 * print('marginals =', m.marginals)
 * print('posterior path =', posteriorDecode(m.marginals).path)
 */
export function posteriorDecode(marginals: Matrix): PosteriorDecoding {
  const [N, K] = marginals.shape
  const P = toFlat(marginals)
  const path = new Int32Array(N)
  const confidence = new Float64Array(N)
  let expectedCorrect = 0
  for (let n = 0; n < N; n++) {
    let best = 0
    for (let k = 1; k < K; k++) if (P[n * K + k] > P[n * K + best]) best = k
    path[n] = best
    confidence[n] = P[n * K + best]
    expectedCorrect += confidence[n]
  }
  return { path: fromData(path, [N]), confidence: fromData(confidence, [N]), expectedCorrect }
}

// ── Stepping ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Forward–backward stepped one position at a time: steps $1, \dots, N$ fill the forward rows ($\alpha_n$ and the
 * filtered distribution), steps $N + 1, \dots, 2N - 1$ the backward rows from the end; then the marginals are
 * complete. Rows not yet computed are NaN.
 */
export interface ForwardBackwardState extends Status {
  /** The chain being smoothed. */
  chain: ChainPotentials
  /** Which pass the last step belonged to; `'done'` once every row is known. */
  phase: 'forward' | 'backward' | 'done'
  /** The position computed by the last step ($-1$ before the first). */
  position: number
  /** The forward messages $\alpha_n$ computed so far ($N \times K$). */
  alpha: Matrix
  /** The filtered distributions $\gamma_n$ computed so far ($N \times K$). */
  filtered: Matrix
  /** The backward messages $\beta_n$ computed so far, from the end ($N \times K$). */
  beta: Matrix
  /** The smoothing marginals of the positions whose backward message is known ($N \times K$). */
  marginals: Matrix
  /** $\sum \log c_n$ up to the last forward position ($\log p(x_0, \dots, x_n)$ for an HMM). */
  logLikelihood: number
  /** The full result, computed once at init; the partial matrices above reveal it row by row. */
  full: ForwardBackwardResult
}

/**
 * A copy of a matrix with the rows not yet computed set to NaN.
 *
 * @param full The complete matrix ($N \times K$); not modified.
 * @param rows Whether row $n$ is known.
 * @returns The matrix with every unknown row NaN.
 */
function reveal(full: Matrix, rows: (n: number) => boolean): Matrix {
  const [N, K] = full.shape
  const out = Float64Array.from(full.data)
  for (let n = 0; n < N; n++) if (!rows(n)) out.fill(NaN, n * K, (n + 1) * K)
  return fromData(out, [N, K])
}

/**
 * The forward–backward state at a phase and position: the rows known by then revealed from the full result, and the
 * log-likelihood of the forward rows.
 *
 * @param base The state without its partial matrices: the step count, the chain, the phase, the position and the
 *   full result to reveal.
 * @returns The complete state.
 */
function fbState(
  base: Omit<ForwardBackwardState, 'alpha' | 'filtered' | 'beta' | 'marginals' | 'logLikelihood'>,
): ForwardBackwardState {
  const N = base.chain.nodePotentials.shape[0]
  const f = base.full
  const fwd = base.phase === 'forward' ? base.position : N - 1
  const bwdFrom = base.phase === 'forward' ? N : base.phase === 'backward' ? base.position : 0
  let ll = 0
  for (let n = 0; n <= fwd; n++) ll += Math.log(f.scale.data[n])
  return {
    ...base,
    alpha: reveal(f.alpha, (n) => n <= fwd),
    filtered: reveal(f.filtered, (n) => n <= fwd),
    beta: reveal(f.beta, (n) => n >= bwdFrom),
    marginals: reveal(f.marginals, (n) => n >= bwdFrom),
    logLikelihood: ll,
  }
}

/**
 * Forward–backward on a chain of potentials as a traceable algorithm (see {@link ForwardBackwardState}): $N$ forward
 * steps, then $N - 1$ backward steps. The result is computed in full by `forwardBackward` at `init` (which throws
 * `ShapeError` for mismatched shapes), and the steps reveal it row by row, for display.
 *
 * @param chain The node potentials $\psi$ ($N \times K$) and the transition $\Amat$ ($K \times K$).
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; it is `done` after $2N - 1$ steps.
 *
 * @example Halfway through the forward pass
 * const chain = {
 *   nodePotentials: tensor([[0.45, 0.1], [0.1, 0.8], [0.9, 0.2]]),
 *   transition: tensor([[0.8, 0.2], [0.2, 0.8]]),
 * }
 * const s = run(forwardBackwardSteps(chain), undefined, 2)
 * print('phase =', s.phase, 'at position', s.position)
 * print('filtered so far =', s.filtered)
 * print('log p(x0, x1) =', s.logLikelihood)
 *
 * @example Run to the end, it matches forwardBackward
 * const chain = {
 *   nodePotentials: tensor([[0.45, 0.1], [0.1, 0.8], [0.9, 0.2]]),
 *   transition: tensor([[0.8, 0.2], [0.2, 0.8]]),
 * }
 * const s = run(forwardBackwardSteps(chain), undefined, 100)
 * print('steps =', s.t, s.phase)
 * print('stepped marginals =', s.marginals)
 * print('forwardBackward =', forwardBackward(chain).marginals)
 */
export function forwardBackwardSteps(chain: ChainPotentials): Algorithm<void, ForwardBackwardState> {
  return {
    name: 'forward-backward',
    init: () => fbState({ t: 0, chain, phase: 'forward', position: -1, full: forwardBackward(chain) }),
    step: (s) => {
      const N = s.chain.nodePotentials.shape[0]
      const t = s.t + 1
      if (s.phase === 'forward') {
        const position = s.position + 1
        if (position < N - 1) return fbState({ ...s, t, position })
        return fbState({ ...s, t, phase: N > 1 ? 'backward' : 'done', position: N - 1 })
      }
      const position = s.position - 1
      return fbState({ ...s, t, position, phase: position <= 0 ? 'done' : 'backward' })
    },
    done: (s) => s.phase === 'done',
  }
}

/**
 * Viterbi stepped one position at a time: $\delta$ rows fill forwards, then the back-trace fills the path backwards.
 */
export interface ViterbiState extends Status {
  /** The chain being decoded. */
  chain: ChainPotentials
  /** Which pass the last step belonged to; `'done'` once the whole path is known. */
  phase: 'forward' | 'backtrack' | 'done'
  /** The position of the last $\delta$ row or path entry filled (0 at `init`, where row 0 is known). */
  position: number
  /** $\delta$ rows computed so far (NaN elsewhere). */
  delta: Matrix
  /** The path entries recovered so far ($-1$ elsewhere). */
  path: Vector
  /** The full result, computed once at init; `delta` and `path` reveal it step by step. */
  full: ViterbiResult
}

/**
 * Viterbi on a chain of potentials as a traceable algorithm (see {@link ViterbiState}): $N - 1$ forward steps fill the
 * $\delta$ rows, one step picks the best final state and $N - 1$ steps trace the path back. The result is computed in
 * full by `viterbi` at `init` (which throws `ShapeError` for mismatched shapes), and the steps reveal it, for display.
 *
 * @param chain The node potentials $\psi$ ($N \times K$) and the transition $\Amat$ ($K \times K$).
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; it is `done` after $2N - 1$ steps.
 *
 * @example The back-trace fills the path from the end
 * const chain = {
 *   nodePotentials: tensor([[0.45, 0.1], [0.1, 0.8], [0.9, 0.2]]),
 *   transition: tensor([[0.8, 0.2], [0.2, 0.8]]),
 * }
 * for (const steps of [2, 3, 4, 5]) {
 *   const s = run(viterbiSteps(chain), undefined, steps)
 *   print(`after ${steps} steps:`, s.phase, s.path)
 * }
 */
export function viterbiSteps(chain: ChainPotentials): Algorithm<void, ViterbiState> {
  return {
    name: 'viterbi',
    init: () => {
      const full = viterbi(chain)
      const N = full.path.shape[0]
      return {
        t: 0,
        chain,
        phase: 'forward',
        position: 0,
        delta: reveal(full.delta, (n) => n === 0),
        path: fromData(new Int32Array(N).fill(-1), [N]),
        full,
      }
    },
    step: (s) => {
      const N = s.chain.nodePotentials.shape[0]
      const t = s.t + 1
      if (s.phase === 'forward') {
        const position = s.position + 1
        if (position < N) return { ...s, t, position, delta: reveal(s.full.delta, (n) => n <= position) }
        const path = Int32Array.from(s.path.data)
        path[N - 1] = s.full.path.data[N - 1]
        return { ...s, t, phase: N > 1 ? 'backtrack' : 'done', position: N - 1, path: fromData(path, [N]) }
      }
      const position = s.position - 1
      const path = Int32Array.from(s.path.data)
      path[position] = s.full.path.data[position]
      return { ...s, t, position, path: fromData(path, [N]), phase: position === 0 ? 'done' : 'backtrack' }
    },
    done: (s) => s.phase === 'done',
  }
}

/**
 * Forward-filtering backward-sampling (Carter & Kohn 1994; Frühwirth-Schnatter 1994): an exact draw of the state path
 * from the chain's distribution ($p(\yvec \mid \xvec)$ for an HMM). The last state is drawn from the filtered
 * distribution $\gamma_{N-1}$, then each earlier one from $p(y_n \mid y_{n+1}) \propto \gamma_n(y_n) A(y_n, y_{n+1})$.
 * Throws `ShapeError` when the shapes do not match.
 *
 * @param s The random stream. Position $n$ draws from its child stream $n$, so the same stream gives the same path:
 *   pass a different stream for each draw.
 * @param chain The node potentials $\psi$ ($N \times K$) and the transition $\Amat$ ($K \times K$).
 * @returns The sampled state path (int32, length $N$).
 *
 * @example Draws agree with the marginals
 * // A uniform start and an observation that pins y₁ = 0.
 * const chain = {
 *   nodePotentials: tensor([[0.5, 0.5], [1, 0]]),
 *   transition: tensor([[0.9, 0.1], [0.2, 0.8]]),
 * }
 * print('one path =', sampleHiddenPath(stream(0), chain))
 * const draws = Array.from({ length: 1000 }, (_, i) => sampleHiddenPath(stream(i), chain))
 * print('fraction with y0 = 0:', draws.filter((y) => y.data[0] === 0).length / 1000)
 * print('p(y0 = 0) =', forwardBackward(chain).marginals.data[0])
 */
export function sampleHiddenPath(s: Stream, chain: ChainPotentials): Vector {
  const fb = forwardBackward(chain)
  const F = toRows(fb.filtered)
  const A = toRows(chain.transition)
  const N = F.length
  const path = new Int32Array(N)
  path[N - 1] = categorical(child(s, N - 1), F[N - 1])
  for (let n = N - 2; n >= 0; n--)
    path[n] = categorical(
      child(s, n),
      F[n].map((g, u) => g * A[u][path[n + 1]]),
    )
  return fromData(path, [N])
}

// ── Chains of factors ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A chain-shaped discrete factor graph as chain log-potentials, with $N$ the number of variables and $K$ the largest
 * cardinality. Values beyond a variable's own cardinality get $-\infty$, so they are never chosen.
 */
export interface FactorChain {
  /** The variables of the graph along the chain (length $N$), starting from the end with the smaller index. */
  order: Index[]
  /** The cardinality of each variable along the chain (length $N$). */
  cardinalities: Size[]
  /** The sum of the logs of the unary factors of each variable along the chain ($N \times K$). */
  logUnary: Matrix
  /**
   * The sum of the logs of the pairwise factors between neighbours along the chain ($(N - 1) \times K \times K$,
   * oriented along `order`).
   */
  logPairwise: Tensor
  /** The sum of the logs of the factors with no variable. */
  logConstant: number
}

/**
 * The chain log-potentials of a discrete factor graph whose shape is a chain (every factor over at most two variables,
 * which are neighbours along one path; see `shape` in `aifn-compute/graph/structured`), or null when it is not a
 * chain. An HMM or a linear-chain CRF tabulated as a factor graph reads back as its chain. Factors on the same
 * variables are multiplied (their logs added).
 *
 * @param graph The factor graph; its factors are read, not modified.
 * @returns The chain's order and log-potentials, or null when the graph is not a chain or has no variable.
 *
 * @example A three-variable chain, and a triangle that is not one
 * const pairTable = tensor([[0.9, 0.1], [0.2, 0.8]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.5, 0.5]) },
 *     { scope: [0, 1], table: pairTable },
 *     { scope: [1, 2], table: pairTable },
 *   ],
 * }
 * const c = factorChain(graph)
 * print('order =', c.order)
 * print('unary =', c.logUnary)
 * print('pairwise =', exp(c.logPairwise))
 * const triangle = { ...graph, factors: [...graph.factors, { scope: [0, 2], table: pairTable }] }
 * print('triangle:', factorChain(triangle))
 */
export function factorChain(graph: DiscreteFactorGraph): FactorChain | null {
  const order = chainOrder(bipartiteGraph(graph))
  if (order === null || order.length === 0) return null
  const N = order.length
  const at = new Map(order.map((v, n) => [v, n]))
  const cardinalities = order.map((v) => graph.cardinalities[v])
  const K = Math.max(...cardinalities)
  const unary = new Float64Array(N * K).fill(-Infinity)
  order.forEach((v, n) => unary.fill(0, n * K, n * K + graph.cardinalities[v]))
  const pairwise = new Float64Array(Math.max(N - 1, 0) * K * K)
  let logConstant = 0
  for (const f of graph.factors) {
    const table = valuesOf(f.table)
    if (f.scope.length === 0) logConstant += Math.log(table[0])
    else if (f.scope.length === 1) {
      const n = at.get(f.scope[0])!
      for (let k = 0; k < table.length; k++) unary[n * K + k] += Math.log(table[k])
    } else {
      const [p, q] = [at.get(f.scope[0])!, at.get(f.scope[1])!]
      const n = Math.min(p, q)
      const [cu, cv] = [f.table.shape[0], f.table.shape[1]]
      for (let a = 0; a < cu; a++)
        for (let b = 0; b < cv; b++) {
          const [u, v] = p < q ? [a, b] : [b, a]
          pairwise[(n * K + u) * K + v] += Math.log(table[a * cv + b])
        }
    }
  }
  // Pairs beyond either variable's cardinality are impossible.
  for (let n = 0; n + 1 < N; n++)
    for (let u = 0; u < K; u++)
      for (let v = 0; v < K; v++)
        if (u >= cardinalities[n] || v >= cardinalities[n + 1]) pairwise[(n * K + u) * K + v] = -Infinity
  return {
    order,
    cardinalities,
    logUnary: fromData(unary, [N, K]),
    logPairwise: fromData(pairwise, [Math.max(N - 1, 0), K, K]),
    logConstant,
  }
}

/**
 * The state of {@link chainSumProduct}: forward messages fill position by position, then backward ones; the
 * marginals of a position appear once both of its messages are known (NaN before). Variables are in the graph's
 * order; `map` is the most probable joint assignment (by max-product), set once done.
 */
export interface ChainSumProductState extends Status {
  /** The graph as chain log-potentials (see `factorChain`). */
  chain: FactorChain
  /** Which pass the last step belonged to; `'done'` once every marginal is known. */
  phase: 'forward' | 'backward' | 'done'
  /** The position computed by the last step ($-1$ before the first). */
  position: Index
  /** $\log \alpha$ by chain position ($N \times K$; NaN rows not yet computed). */
  logAlpha: Matrix
  /** $\log \beta$ by chain position ($N \times K$; NaN rows not yet computed). */
  logBeta: Matrix
  /** $p(x_v)$ for every variable $v$ of the graph (float64 vectors of its cardinality; NaN until known). */
  marginals: Tensor[]
  /** $\log Z$ of the graph (the chain's plus the constant factors), set once done (NaN before). */
  logZ: number
  /** The MAP assignment (int32, one entry per variable), set once done (null before). */
  map: Tensor | null
  /** Whether both passes are complete. */
  done: boolean
}

/**
 * Sum-product on a chain-shaped discrete factor graph (exact, in log space; the forward–backward recursions of
 * Rabiner 1989 on the chain's potentials), stepped one position at a time: $N$ forward steps, then $N - 1$ backward
 * steps. The messages are computed in full by `chainForwardBackward` when the algorithm is made and revealed step by
 * step; the last step adds $\log Z$ and the MAP assignment (by `chainViterbi`). Throws `DomainError` when the graph is
 * not a chain.
 *
 * @param graph The discrete factor graph, chain-shaped (see `factorChain`).
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; it is `done` after $2N - 1$ steps.
 *
 * @example Marginals, log Z and MAP of a three-variable chain
 * // x₀ is uniform and each variable copies the one before with probability 0.9.
 * const copy = tensor([[0.9, 0.1], [0.1, 0.9]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.5, 0.5]) },
 *     { scope: [0, 1], table: copy },
 *     { scope: [1, 2], table: copy },
 *     { scope: [2], table: tensor([0.2, 0.8]) },
 *   ],
 * }
 * const s = run(chainSumProduct(graph), undefined, 10)
 * print('steps =', s.t)
 * print('marginals =', s.marginals)
 * print('log Z =', s.logZ, 'and by enumeration', enumerate(graph).logZ)
 * print('MAP =', s.map)
 */
export function chainSumProduct(graph: DiscreteFactorGraph): Algorithm<void, ChainSumProductState> {
  const chain = factorChain(graph)
  if (!chain) throw new DomainError('chainSumProduct', 'chainSumProduct: the factor graph is not a chain')
  const N = chain.order.length
  const full = chainForwardBackward(chain.logUnary, chain.logPairwise)
  const known = (fwd: number, bwdFrom: number) => {
    const rows = toRows(full.marginals)
    const marginals = graph.cardinalities.map((k) => fromData(new Float64Array(k).fill(NaN), [k]))
    chain.order.forEach((v, n) => {
      if (n <= fwd && n >= bwdFrom)
        marginals[v] = fromData(Float64Array.from(rows[n].slice(0, chain.cardinalities[n])), [chain.cardinalities[n]])
    })
    return {
      logAlpha: reveal(full.logAlpha, (n) => n <= fwd),
      logBeta: reveal(full.logBeta, (n) => n >= bwdFrom),
      marginals,
    }
  }
  const finish = (): Pick<ChainSumProductState, 'logZ' | 'map'> => {
    const best = chainViterbi(chain.logUnary, chain.logPairwise).path
    const map = new Int32Array(graph.cardinalities.length)
    chain.order.forEach((v, n) => (map[v] = best.data[n]))
    return { logZ: full.logZ + chain.logConstant, map: fromData(map, [map.length]) }
  }
  const state = (t: Size, phase: ChainSumProductState['phase'], position: Index): ChainSumProductState => {
    const fwd = phase === 'forward' ? position : N - 1
    const bwdFrom = phase === 'forward' ? (position === N - 1 ? N - 1 : N) : phase === 'backward' ? position : 0
    const done = phase === 'done'
    return {
      t,
      chain,
      phase,
      position,
      ...known(fwd, bwdFrom),
      ...(done ? finish() : { logZ: NaN, map: null }),
      done,
    }
  }
  return {
    name: 'chain-sum-product',
    init: () => state(0, 'forward', -1),
    step: (s) => {
      if (s.phase === 'forward') {
        const position = s.position + 1
        if (position < N - 1) return state(s.t + 1, 'forward', position)
        return state(s.t + 1, N > 1 ? 'backward' : 'done', N - 1)
      }
      const position = s.position - 1
      return state(s.t + 1, position <= 0 ? 'done' : 'backward', position)
    },
    done: (s) => s.done,
  }
}
