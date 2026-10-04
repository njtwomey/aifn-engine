/**
 * Chain engines: scaled forward–backward, Viterbi and forward-filtering backward-sampling on a chain of potentials
 * (Rabiner 1989, "A tutorial on hidden Markov models", Proc. IEEE 77(2), §III; Durbin, Eddy, Krogh & Mitchison 1998,
 * "Biological Sequence Analysis", ch. 3), log-space sum-product and max-product on a chain of log-potentials (which
 * the linear-chain CRF runs), and chain-shaped discrete factor graphs. Named chain models (the HMM, the CRF) are
 * applications in `aifn-methods/inference/sequence-models`; they build the potentials these engines take.
 *
 * Notation (Twomey, Diethe & Flach 2016): positions n = 0 … N − 1, states k = 0 … K − 1. α_n is the forward message
 * into position n and β_n the backward message; the posterior marginal is ∝ α_n ⊙ ψ_n ⊙ β_n with ψ_n the node
 * potentials (for an HMM the emission likelihoods, with π folded into ψ_0).
 */

import type { Index, Size, Status } from 'aifn-compute/foundation/contracts'
import { categorical, child, type Stream } from 'aifn-compute/foundation/random'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, log, toRows, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { chainOrder } from 'aifn-compute/graph/structured'
import { bipartiteGraph, valuesOf, type DiscreteFactorGraph } from 'aifn-compute/inference/model'

const matrix = (rows: readonly ArrayLike<number>[], k = rows[0]?.length ?? 0): Matrix => {
  const out = new Float64Array(rows.length * k)
  rows.forEach((r, i) => out.set(r, i * k))
  return fromData(out, [rows.length, k])
}

/**
 * A chain of non-negative potentials: p(y) ∝ Π_n ψ_n(y_n) Π_n A(y_n, y_{n+1}) over state paths y of length N with K
 * states. `nodePotentials` ψ is N × K (any initial distribution folded into ψ_0) and `transition` A is K × K, shared by
 * every step. For a hidden Markov model ψ_n(k) = B[k, x_n] (times π_k at n = 0) and A is its transition matrix, so the
 * normaliser is the likelihood p(x).
 */
export interface ChainPotentials {
  nodePotentials: Matrix
  transition: Matrix
}

/** Check that ψ is N × K and A is K × K. */
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
  /** α_n = Aᵀ γ_{n−1} (N × K), with γ the filtered distribution, so rows n ≥ 1 sum to one; α_0 = 1 (π is in ψ_0). */
  alpha: Matrix
  /** β_n (N × K), scaled by the same constants so that α_n ⊙ ψ_n ⊙ β_n sums to one; β_{N−1} = 1. */
  beta: Matrix
  /** ψ_n (N × K). */
  psi: Matrix
  /** Filtering distributions p(y_n | x_{0..n}) (N × K). */
  filtered: Matrix
  /** Smoothing marginals p(y_n | x) (N × K). */
  marginals: Matrix
  /** Pairwise marginals ξ_n(u, v) = p(y_n = u, y_{n+1} = v | x), (N − 1) × K × K. */
  pairwise: Tensor
  /** c_n, the normaliser of α_n ⊙ ψ_n (length N); log Z = Σ log c_n. */
  scale: Vector
  /** log Z = Σ log c_n, the log normaliser of the chain (log p(x) for an HMM). */
  logLikelihood: number
}

/**
 * Forward–backward with per-step scaling (Rabiner 1989, §V.A) on a chain of potentials: α_n = Aᵀ γ_{n−1} with γ_{n−1}
 * the filtered distribution, c_n = Σ α_n ⊙ ψ_n, and β_n = A (β_{n+1} ⊙ ψ_{n+1}) / c_{n+1}. log Z = Σ log c_n.
 */
export function forwardBackward(chain: ChainPotentials): ForwardBackwardResult {
  checkChain(chain, 'forwardBackward')
  return scaledForwardBackward(toRows(chain.nodePotentials), toRows(chain.transition))
}

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
  /** The most probable state path (int32, length N). */
  path: Vector
  /** Its log-probability (log joint p(x, y) for an HMM; the unnormalised log score for a chain of potentials). */
  logProbability: number
  /** δ_n(k): the best log score of a path ending in k at n (N × K). */
  delta: Matrix
  /** Back-pointers: the best predecessor of k at n (int32, N × K; row 0 is −1). */
  backpointers: Tensor
}

/**
 * Max-product on a chain in log space: logUnary (N × K) and logPairwise (K × K shared, or (N − 1) × K × K).
 * δ_0 = u_0, δ_n(v) = max_u δ_{n−1}(u) + P(u, v) + u_n(v); ties go to the smaller state.
 */
export function chainViterbi(logUnary: Matrix, logPairwise: Tensor): ViterbiResult {
  const [N, K] = logUnary.shape
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

/** The pairwise log-potential matrix between positions n and n + 1. */
function pairwiseAt(logPairwise: Tensor, K: number): (n: number) => number[][] {
  if (logPairwise.shape.length === 2) {
    const P = toRows(logPairwise)
    return () => P
  }
  const all = Array.from(logPairwise.data)
  return (n) => Array.from({ length: K }, (_, u) => all.slice((n * K + u) * K, (n * K + u + 1) * K))
}

/**
 * Viterbi decoding of a chain of potentials: the most probable state path and its unnormalised log score
 * Σ log ψ_n(y_n) + Σ log A(y_n, y_{n+1}) (log p(x, y*) for an HMM).
 */
export function viterbi(chain: ChainPotentials): ViterbiResult {
  checkChain(chain, 'viterbi')
  return chainViterbi(log(chain.nodePotentials), log(chain.transition))
}

/** log Σ exp xᵢ over a plain array, shifted by the maximum (−∞ for an empty or all −∞ input; +∞ if any is +∞). */
function lse(xs: readonly number[]): number {
  let m = -Infinity
  for (const x of xs) if (x > m) m = x
  if (m === -Infinity || m === Infinity || Number.isNaN(m)) return m
  let s = 0
  for (const x of xs) s += Math.exp(x - m)
  return m + Math.log(s)
}

/** The result of log-space forward–backward on a chain of potentials. */
export interface ChainMarginals {
  /** log α_n (N × K): log Σ over paths to n, excluding u_n. */
  logAlpha: Matrix
  /** log β_n (N × K): log Σ over paths from n, excluding u_n. */
  logBeta: Matrix
  marginals: Matrix
  /** (N − 1) × K × K. */
  pairwise: Tensor
  logZ: number
}

/**
 * Sum-product on a chain in log space: log α_0 = 0, log α_n(v) = logsumexp_u [log α_{n−1}(u) + u_{n−1}(u) + P(u, v)],
 * log β_{N−1} = 0, log β_n(u) = logsumexp_v [P(u, v) + u_{n+1}(v) + log β_{n+1}(v)], log Z = logsumexp(α + u + β).
 */
export function chainForwardBackward(logUnary: Matrix, logPairwise: Tensor): ChainMarginals {
  const [N, K] = logUnary.shape
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
  /** ŷ_n = argmax_k P(y_n = k | x) (int32, length N); ties go to the smaller state. */
  path: Vector
  /** P(y_n = ŷ_n | x) per position. */
  confidence: Vector
  /** Σ_n P(y_n = ŷ_n | x): the expected number of correct labels, the largest any labelling has. */
  expectedCorrect: number
}

/**
 * Posterior (marginal, max-marginal) decoding (Rabiner 1989, §III.B, "individually most likely states"): each position
 * takes its most probable label under the marginals, ŷ_n = argmax_k P(y_n = k | x). It maximises the expected number of
 * correct labels Σ_n P(y_n = ŷ_n | x), where Viterbi maximises the probability of the whole sequence; the two can
 * differ, and the posterior path may even contain a transition of zero probability (−∞ log-potential). `marginals` is
 * N × K, e.g. `chainForwardBackward(…).marginals`.
 */
export function posteriorDecode(marginals: Matrix): PosteriorDecoding {
  const [N, K] = marginals.shape
  const P = marginals.data
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
 * Forward–backward stepped one position at a time: steps 1 … N fill the forward rows (α_n and the filtered
 * distribution), steps N + 1 … 2N − 1 the backward rows from the end; then the marginals are complete. Rows not yet
 * computed are NaN.
 */
export interface ForwardBackwardState extends Status {
  chain: ChainPotentials
  phase: 'forward' | 'backward' | 'done'
  /** The position computed by the last step (−1 before the first). */
  position: number
  alpha: Matrix
  filtered: Matrix
  beta: Matrix
  marginals: Matrix
  /** Σ log c up to the last forward position (log p(x_0 … x_n) for an HMM). */
  logLikelihood: number
  /** The full result, computed once at init; the partial matrices above reveal it row by row. */
  full: ForwardBackwardResult
}

function reveal(full: Matrix, rows: (n: number) => boolean): Matrix {
  const [N, K] = full.shape
  const out = Float64Array.from(full.data)
  for (let n = 0; n < N; n++) if (!rows(n)) out.fill(NaN, n * K, (n + 1) * K)
  return fromData(out, [N, K])
}

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

/** Forward–backward on a chain of potentials as a traceable algorithm (see {@link ForwardBackwardState}). */
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

/** Viterbi stepped one position at a time: δ rows fill forwards, then the back-trace fills the path backwards. */
export interface ViterbiState extends Status {
  chain: ChainPotentials
  phase: 'forward' | 'backtrack' | 'done'
  position: number
  /** δ rows computed so far (NaN elsewhere). */
  delta: Matrix
  /** The path entries recovered so far (−1 elsewhere). */
  path: Vector
  full: ViterbiResult
}

/** Viterbi on a chain of potentials as a traceable algorithm (see {@link ViterbiState}). */
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
 * Forward-filtering backward-sampling (Carter & Kohn 1994; Frühwirth-Schnatter 1994): a draw of the state path from
 * the chain's distribution (p(y | x) for an HMM), using the filtered distributions and
 * y_n | y_{n+1} ∝ γ_n(y_n) A(y_n, y_{n+1}).
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
 * A chain-shaped discrete factor graph as chain log-potentials: `order` lists the variables along the chain,
 * `logUnary` (N × K) and `logPairwise` ((N − 1) × K × K, oriented along `order`) sum the logs of its unary and pairwise
 * factors, with K the largest cardinality and −∞ for values beyond a variable's own; `logConstant` sums the factors
 * with no variable.
 */
export interface FactorChain {
  order: Index[]
  cardinalities: Size[]
  logUnary: Matrix
  logPairwise: Tensor
  logConstant: number
}

/**
 * The chain log-potentials of a discrete factor graph whose shape is a chain (every factor over at most two variables,
 * which are neighbours along one path; see `shape` in `aifn-compute/graph/structured`), or null when it is not a chain. An HMM
 * or a linear-chain CRF tabulated as a factor graph reads back as its chain.
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
  chain: FactorChain
  phase: 'forward' | 'backward' | 'done'
  /** The position computed by the last step (−1 before the first). */
  position: Index
  /** log α and log β by chain position (N × K; NaN rows not yet computed). */
  logAlpha: Matrix
  logBeta: Matrix
  /** p(x_v) for every variable v of the graph (float64 vectors of its cardinality). */
  marginals: Tensor[]
  /** log Z of the graph (the chain's plus the constant factors), set once done (NaN before). */
  logZ: number
  /** The MAP assignment (int32, one entry per variable), set once done. */
  map: Tensor | null
  done: boolean
}

/**
 * Sum-product on a chain-shaped discrete factor graph (exact, in log space; the forward–backward recursions of
 * Rabiner 1989 on the chain's potentials), stepped one position at a time: N forward steps, then N − 1 backward steps.
 * Throws when the graph is not a chain.
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
