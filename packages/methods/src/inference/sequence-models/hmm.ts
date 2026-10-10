/**
 * Discrete hidden Markov models: the `Hmm` type and its constructor, its chain of potentials, the occasionally
 * dishonest casino, sampling, and the HMM in the model language.
 *
 * An HMM with $K$ hidden states and $M$ symbols (Rabiner 1989, "A tutorial on hidden Markov models", Proc. IEEE
 * 77(2)) has initial probabilities $\pivec$ (length $K$), a transition matrix $\Amat$ ($K \times K$, with
 * $A_{uv} = p(y_{n+1} = v \mid y_n = u)$) and an emission matrix $\Bmat$ ($K \times M$, with
 * $B_{km} = p(x_n = m \mid y_n = k)$). States and symbols are integer ids from 0. The HMM carries no inference of its
 * own: `hmmChain` turns it and an observation sequence into the chain of potentials that the generic engines of
 * `aifn-compute/inference/exact` take (`forwardBackward(hmmChain(h, x))`, `viterbi`, `sampleHiddenPath`), and
 * `hmmModel` writes it in the model language as a chain group, so `aifn-compute/inference/engines`' `infer` runs
 * forward–backward by its shape.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { categorical, type Stream, child } from 'aifn-compute/foundation/random'
import {
  fromData,
  tensor,
  toArray,
  toFlat,
  toRows,
  type Matrix,
  type NestedArray,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { ChainPotentials } from 'aifn-compute/inference/exact'
import { dist, model, type Model } from 'aifn-compute/inference/model'

/** A discrete-emission HMM with $K$ states and $M$ symbols, as `hmm` builds it (plain data). */
export interface Hmm {
  /** The initial probabilities $\pi_k = p(y_0 = k)$ (length $K$). */
  initial: Vector
  /** The transition matrix $\Amat$, $A_{uv} = p(y_{n+1} = v \mid y_n = u)$ ($K \times K$; each row sums to one). */
  transition: Matrix
  /** The emission matrix $\Bmat$, $B_{km} = p(x_n = m \mid y_n = k)$ ($K \times M$; each row sums to one). */
  emission: Matrix
  /** Names of the states, for display. */
  stateNames?: readonly string[]
  /** Names of the symbols, for display. */
  symbolNames?: readonly string[]
}

type Rows = readonly (readonly number[])[]

/**
 * A tensor as given, or a nested array made into one.
 *
 * @param v A tensor (returned as it is), a vector as an array, or a matrix as an array of rows.
 * @returns The tensor.
 */
const asTensor = (v: Tensor | readonly number[] | Rows): Tensor =>
  'shape' in v ? v : tensor(v as number[] | number[][])

/**
 * Build an HMM from arrays or tensors. Throws `ShapeError` unless $\pivec$ has length $K$, $\Amat$ is $K \times K$ and
 * $\Bmat$ has $K$ rows, and `DomainError` unless $\pivec$ and every row of $\Amat$ and $\Bmat$ sums to one (to within
 * $10^{-9}$). The entries themselves are not checked to be non-negative.
 *
 * @param initial The initial probabilities $\pivec$ (length $K$).
 * @param transition The transition matrix $\Amat$ ($K \times K$, as rows or a tensor): row $u$ is the distribution of
 *   the next state after state $u$.
 * @param emission The emission matrix $\Bmat$ ($K \times M$, as rows or a tensor): row $k$ is the distribution of the
 *   symbol emitted in state $k$.
 * @param names Optional display names: `stateNames` (one per state) and `symbolNames` (one per symbol), copied onto
 *   the result as given.
 * @returns The HMM, with its arrays as tensors.
 *
 * @example A two-state weather model
 * const h = hmm([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]], {
 *   stateNames: ['rainy', 'sunny'],
 *   symbolNames: ['walk', 'shop', 'clean'],
 * })
 * print('π =', h.initial)
 * print('A =', h.transition)
 * print('B =', h.emission)
 *
 * @example Rows that do not sum to one are refused
 * try {
 *   hmm([0.5, 0.5], [[0.9, 0.2], [0.5, 0.5]], [[1, 0], [0, 1]])
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function hmm(
  initial: Tensor | readonly number[],
  transition: Tensor | Rows,
  emission: Tensor | Rows,
  names: { stateNames?: readonly string[]; symbolNames?: readonly string[] } = {},
): Hmm {
  const pi = asTensor(initial)
  const A = asTensor(transition)
  const B = asTensor(emission)
  const K = pi.shape[0]
  if (A.shape[0] !== K || A.shape[1] !== K || B.shape[0] !== K)
    throw new ShapeError('hmm', 'hmm: π must be [K], A [K, K] and B [K, M]', [pi.shape, A.shape, B.shape])
  const sums = [toFlat(pi), ...toRows(A), ...toRows(B)].map((r) => r.reduce((a, b) => a + b, 0))
  if (sums.some((s) => Math.abs(s - 1) > 1e-9))
    throw new DomainError('hmm', 'hmm: π and the rows of A and B must sum to 1')
  return { initial: pi, transition: A, emission: B, ...names }
}

/**
 * The chain of potentials of an HMM given observed symbols $\xvec$ (length $N$): node potentials
 * $\psi_n(k) = B_{k x_n}$ with $\pivec$ folded into the first, $\psi_0(k) = \pi_k B_{k x_0}$ ($N \times K$), and the
 * transition matrix $\Amat$ as the pairwise potential. Its normaliser is the likelihood $p(\xvec)$, so
 * `forwardBackward(hmmChain(h, x)).logLikelihood` is $\log p(\xvec)$. Throws `DomainError` for a symbol that is not an
 * integer in $0, \dots, M - 1$.
 *
 * @param h The HMM.
 * @param observations The observed symbol ids $x_0, \dots, x_{N-1}$, each an integer in $0, \dots, M - 1$.
 * @returns `nodePotentials` ($N \times K$, row $n$ holding $\psi_n$) and `transition`, the HMM's $\Amat$ itself.
 *
 * @example Four rolls of the casino's dice: three sixes, then a one
 * const { nodePotentials, transition } = hmmChain(dishonestCasino(), [5, 5, 5, 0])
 * print('ψ (fair, loaded) per roll:', nodePotentials)
 * print('A =', transition)
 *
 * @example Filtered probability of the loaded die, by the forward recursion
 * const { nodePotentials, transition } = hmmChain(dishonestCasino(), [5, 5, 5, 0])
 * const A = toRows(transition)
 * let f = [1, 1]
 * toRows(nodePotentials).forEach((psi, n) => {
 *   const prior = n === 0 ? [1, 1] : [0, 1].map((v) => f[0] * A[0][v] + f[1] * A[1][v])
 *   const u = psi.map((p, k) => p * prior[k])
 *   f = u.map((p) => p / (u[0] + u[1]))
 *   print(`after roll ${n + 1}: p(loaded) =`, f[1])
 * })
 */
export function hmmChain(h: Hmm, observations: ArrayLike<number>): ChainPotentials {
  const K = h.initial.shape[0]
  const M = h.emission.shape[1]
  const B = toRows(h.emission)
  const pi = toFlat(h.initial)
  const N = observations.length
  const psi = new Float64Array(N * K)
  for (let n = 0; n < N; n++) {
    const x = observations[n]
    if (!(Number.isInteger(x) && x >= 0 && x < M))
      throw new DomainError('hmmChain', `hmmChain: symbol ${x} not in 0 … ${M - 1}`)
    for (let k = 0; k < K; k++) psi[n * K + k] = (n === 0 ? pi[k] : 1) * B[k][x]
  }
  return { nodePotentials: fromData(psi, [N, K]), transition: h.transition }
}

/**
 * An HMM in the model language (Rabiner, 1989): a chain `time` of length `T` with $z_0 \sim \Cat(\pivec)$,
 * $z_t \mid z_{t-1} \sim \Cat(\Amat_{z_{t-1}})$ and $x_t \mid z_t \sim \Cat(\Bmat_{z_t})$ ($\Amat_u$ the row $u$ of
 * $\Amat$), the parameters as constants `π`, `A` and `B`. Its latent structure is a chain, so
 * `infer(hmmModel(h), { sizes: { T }, data: { x } })` picks forward–backward.
 *
 * @param h The HMM whose $\pivec$, $\Amat$ and $\Bmat$ become the model's constants.
 * @returns The model, named `hidden Markov model`, with the length `T` left as a size to bind.
 *
 * @example The casino as a model
 * const m = hmmModel(dishonestCasino())
 * print('name:', m.name)
 * print('nodes:', m.attributes.map((n) => n.name))
 * print('sizes:', m.sizes)
 */
export function hmmModel(h: Hmm): Model {
  return model('hidden Markov model', (m) => {
    const pi = m.constant('π', toArray(h.initial) as NestedArray, { label: '\\boldsymbol{\\pi}' })
    const A = m.constant('A', toArray(h.transition) as NestedArray, { label: 'A' })
    const B = m.constant('B', toArray(h.emission) as NestedArray, { label: 'B' })
    const time = m.chain('time', 'T', { label: 'T' })
    const z = time.variable('z', dist.Categorical(pi), {
      label: 'z_t',
      next: (previous) => dist.Categorical(A.at(previous)),
    })
    time.observed('x', dist.Categorical(B.at(z)), { label: 'x_t' })
  })
}

/**
 * The occasionally dishonest casino (Durbin et al. 1998, §3.2): a fair die (state 0) and a loaded die (state 1) that
 * rolls a six half the time and each other face with probability 0.1; the casino switches from fair to loaded with
 * probability 0.05 and back with 0.1, and starts with either die with probability 0.5. Symbols $0, \dots, 5$ are the
 * faces $1, \dots, 6$.
 *
 * @returns The HMM, with state names `fair` and `loaded` and symbol names `1` to `6`.
 *
 * @example The fair and the loaded die
 * const casino = dishonestCasino()
 * print('states:', casino.stateNames)
 * print('A =', casino.transition)
 * print('B =', casino.emission)
 */
export function dishonestCasino(): Hmm {
  return hmm(
    [0.5, 0.5],
    [
      [0.95, 0.05],
      [0.1, 0.9],
    ],
    [Array(6).fill(1 / 6), [...Array(5).fill(0.1), 0.5]],
    { stateNames: ['fair', 'loaded'], symbolNames: ['1', '2', '3', '4', '5', '6'] },
  )
}

// ── Sampling ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Draw a state path and its observations from an HMM by ancestral sampling: $y_0 \sim \Cat(\pivec)$,
 * $y_t \sim \Cat(\Amat_{y_{t-1}})$, $x_t \sim \Cat(\Bmat_{y_t})$. Deterministic in the stream.
 *
 * @param s The random stream; step $t$ draws from its children `step`, $t$.
 * @param model The HMM to sample.
 * @param n The length of the sequence.
 * @returns `states` and `observations`, int32 vectors of length `n`.
 *
 * @example Twenty rolls at the casino
 * const { states, observations } = sampleHmm(stream(1), dishonestCasino(), 20)
 * print('die (1 loaded):', states)
 * print('faces - 1:     ', observations)
 */
export function sampleHmm(s: Stream, model: Hmm, n: number): { states: Vector; observations: Vector } {
  const A = toRows(model.transition)
  const B = toRows(model.emission)
  const states = new Int32Array(n)
  const obs = new Int32Array(n)
  for (let t = 0; t < n; t++) {
    const st = child(s, 'step', t)
    states[t] =
      t === 0
        ? categorical(child(st, 'state'), toFlat(model.initial))
        : categorical(child(st, 'state'), A[states[t - 1]])
    obs[t] = categorical(child(st, 'symbol'), B[states[t]])
  }
  return { states: fromData(states, [n]), observations: fromData(obs, [n]) }
}
