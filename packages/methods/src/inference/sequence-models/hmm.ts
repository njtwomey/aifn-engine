/**
 * Hidden Markov models (Rabiner 1989, "A tutorial on hidden Markov models", Proc. IEEE 77(2)), part of
 * `aifn-methods/inference/sequence-models`: the `Hmm` type and its constructor, its chain of potentials for the
 * generic engines of `aifn-compute/inference/exact` (`forwardBackward(hmmChain(h, x))`, `viterbi`, `sampleHiddenPath`), the
 * occasionally dishonest casino, sampling, and the HMM in the model language (a chain group, so
 * `aifn-compute/inference/engines`' `infer` runs forward–backward by its shape).
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

/** A discrete-emission HMM: π (K), A[u, v] = p(y_{n+1} = v | y_n = u) (K × K), B[k, m] = p(x = m | y = k) (K × M). */
export interface Hmm {
  initial: Vector
  transition: Matrix
  emission: Matrix
  /** Names for display. */
  stateNames?: readonly string[]
  symbolNames?: readonly string[]
}

type Rows = readonly (readonly number[])[]

const asTensor = (v: Tensor | readonly number[] | Rows): Tensor =>
  'shape' in v ? v : tensor(v as number[] | number[][])

/** Build an HMM from arrays or tensors, checking that π and each row of A and B sum to one (to 1e-9). */
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
 * The chain of potentials of an HMM given observed symbols x (length N): node potentials ψ_n(k) = B[k, x_n] with π
 * folded into ψ_0 (N × K), and the transition matrix A. Its normaliser is the likelihood p(x), so
 * `forwardBackward(hmmChain(h, x)).logLikelihood` is log p(x).
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
 * An HMM in the model language (Rabiner, 1989): a chain `time` of length `T` with z₀ ~ Cat(π), z_t | z_{t−1} ~
 * Cat(A[z_{t−1}]) and x_t | z_t ~ Cat(B[z_t]), the parameters as constants. Its latent structure is a chain, so
 * `infer(hmmModel(h), { sizes: { T }, data: { x } })` picks forward–backward.
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
 * rolls a six half the time; the casino switches from fair to loaded with probability 0.05 and back with 0.1. Symbols
 * 0 … 5 are the faces 1 … 6.
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

/** Draw a state path and observations of length n from an HMM (int32 vectors). */
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
