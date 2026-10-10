/**
 * Named Markov chains for demonstrations: the weather chain, gambler's ruin, a lazy random walk on a cycle or a path,
 * and the Ehrenfest urn. Each returns its row-stochastic transition matrix $\Pmat$ ($n \times n$, $P_{ij}$ the
 * probability of a step from state $i$ to state $j$) with state labels; the chain computations are in
 * `aifn-compute/probability/markov`.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { matrix } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A named chain: its transition matrix and a label for each state. */
export interface NamedChain {
  /** The transition matrix, $n \times n$: row $i$ holds the probabilities of moving from state $i$ to each state. */
  readonly P: Tensor
  /** The label of each state, in the order of the rows of `P`. */
  readonly states: readonly string[]
}

/**
 * An $n \times n$ matrix from a function of its indices.
 *
 * @param n The number of rows and columns.
 * @param entry The entry at row $i$ and column $j$.
 * @returns The matrix, with `entry(i, j)` at $(i, j)$.
 */
const build = (n: number, entry: (i: number, j: number) => number): Tensor => {
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = entry(i, j)
  return matrix(P, n, n)
}

/**
 * The weather chain on sunny, cloudy and rainy days, with
 * $\Pmat = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.2 & 0.3 & 0.5 \end{pmatrix}$ and stationary
 * law $\pivec = (21, 13, 12)/46$.
 *
 * @returns The chain, with states `'sunny'`, `'cloudy'` and `'rainy'`.
 *
 * @example The stationary law is left unchanged by a step
 * const { P, states } = weatherChain()
 * print('states:', states)
 * print('P =', P)
 * const pi = tensor([[21 / 46, 13 / 46, 12 / 46]])
 * print('pi =', pi)
 * print('pi P =', matmul(pi, P))
 */
export function weatherChain(): NamedChain {
  const rows = [
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
  ]
  return { P: build(3, (i, j) => rows[i][j]), states: ['sunny', 'cloudy', 'rainy'] }
}

/**
 * Gambler's ruin on $\{0, \dots, N\}$: from $0 < i < N$, win one unit with probability $p$ or lose one with
 * probability $q = 1 - p$; 0 and $N$ absorb. The chance of reaching $N$ from $i$ is $i/N$ in a fair game and
 * $(1 - (q/p)^i)/(1 - (q/p)^N)$ otherwise. Throws `DomainError` when $N$ is not an integer of at least 2 or $p$ is not
 * in $(0, 1)$.
 *
 * @param target The target fortune $N$, at which the gambler stops; the chain has $N + 1$ states.
 * @param p The probability $p$ of winning each round.
 * @returns The chain, with states labelled by the fortune, `'0'`, `'1'`, and so on up to $N$.
 *
 * @example Many steps on, the chance of reaching the target
 * const { P } = gamblersRuinChain(4, 0.6)
 * print('P =', P)
 * let Q = P
 * for (let k = 0; k < 8; k++) Q = matmul(Q, Q)
 * // From 2 the target is reached with probability (1 - (2/3)^2) / (1 - (2/3)^4) = 0.6923.
 * print('row 2 of P^256:', toArray(Q)[2])
 */
export function gamblersRuinChain(target = 10, p = 0.5): NamedChain {
  if (!(Number.isInteger(target) && target >= 2))
    throw new DomainError('gamblersRuinChain', 'gamblersRuinChain: the target must be ≥ 2')
  if (!(p > 0 && p < 1)) throw new DomainError('gamblersRuinChain', 'gamblersRuinChain: p must be in (0, 1)')
  const P = build(target + 1, (i, j) =>
    i === 0 || i === target ? (i === j ? 1 : 0) : j === i + 1 ? p : j === i - 1 ? 1 - p : 0,
  )
  return { P, states: Array.from({ length: target + 1 }, (_, i) => `${i}`) }
}

/**
 * A random walk on $n$ states that holds with probability $h$ (`hold`) and otherwise steps to a neighbour, on a cycle
 * (wrapping) or a path (a step off an end holds instead). It steps right with probability $(1 - h)(1 + d)/2$ and left
 * with $(1 - h)(1 - d)/2$, $d$ the `drift`. $h = 0$ on an even cycle is periodic; $h = 1/2$ is the lazy walk. Throws
 * `DomainError` when $n$ is not an integer of at least 2, $h$ is not in $[0, 1)$ or $d$ is not in $[-1, 1]$.
 *
 * @param n The number of states.
 * @param options `hold` (default 0.5), the probability $h$ of staying put; `topology` (default `'cycle'`), whether the
 *   ends join up (`'cycle'`) or not (`'path'`); `drift` (default 0), the bias $d$ to the right ($-1$ always left,
 *   1 always right).
 * @returns The chain, with states labelled by their index, `'0'`, `'1'`, and so on.
 *
 * @example A drifting walk on a path piles up at the right end
 * const { P } = randomWalkChain(4, { topology: 'path', drift: 0.5 })
 * print('P =', P)
 * let Q = P
 * for (let k = 0; k < 7; k++) Q = matmul(Q, Q)
 * print('row 0 of P^128:', toArray(Q)[0])
 */
export function randomWalkChain(
  n = 8,
  options: { hold?: number; topology?: 'cycle' | 'path'; drift?: number } = {},
): NamedChain {
  const { hold = 0.5, topology = 'cycle', drift = 0 } = options
  if (!(Number.isInteger(n) && n >= 2)) throw new DomainError('randomWalkChain', 'randomWalkChain: needs n ≥ 2')
  if (!(hold >= 0 && hold < 1)) throw new DomainError('randomWalkChain', 'randomWalkChain: hold must be in [0, 1)')
  if (!(Math.abs(drift) <= 1)) throw new DomainError('randomWalkChain', 'randomWalkChain: drift must be in [−1, 1]')
  const right = ((1 - hold) * (1 + drift)) / 2
  const left = ((1 - hold) * (1 - drift)) / 2
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    P[i * n + i] += hold
    const r = i + 1
    const l = i - 1
    if (topology === 'cycle') {
      P[i * n + ((r + n) % n)] += right
      P[i * n + ((l + n) % n)] += left
    } else {
      P[i * n + (r < n ? r : i)] += right
      P[i * n + (l >= 0 ? l : i)] += left
    }
  }
  return { P: matrix(P, n, n), states: Array.from({ length: n }, (_, i) => `${i}`) }
}

/**
 * The Ehrenfest urn with $N$ balls in two urns: state $i$ is the number in the first urn, and each step moves a ball
 * chosen uniformly to the other urn (with probability `hold` nothing moves). Its stationary law is
 * $\Binom(N, 1/2)$. Without holding it is periodic, alternating between odd and even states. Throws `DomainError` when
 * $N$ is not a positive integer.
 *
 * @param balls The number of balls $N$; the chain has $N + 1$ states.
 * @param hold The probability that a step moves nothing; 0 for the classic urn.
 * @returns The chain, with states labelled by the number of balls in the first urn, `'0'` up to $N$.
 *
 * @example The binomial law is stationary
 * const { P } = ehrenfestChain(4)
 * print('P =', P)
 * const pi = tensor([[1, 4, 6, 4, 1].map((c) => c / 16)])
 * print('pi P =', matmul(pi, P))
 */
export function ehrenfestChain(balls = 10, hold = 0): NamedChain {
  if (!(Number.isInteger(balls) && balls >= 1))
    throw new DomainError('ehrenfestChain', 'ehrenfestChain: needs at least one ball')
  const n = balls + 1
  const P = build(n, (i, j) =>
    j === i ? hold : j === i - 1 ? ((1 - hold) * i) / balls : j === i + 1 ? ((1 - hold) * (balls - i)) / balls : 0,
  )
  return { P, states: Array.from({ length: n }, (_, i) => `${i}`) }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'weatherChain',
    name: 'Weather chain',
    summary: 'A three-state sunny, cloudy, rainy chain with π = (21, 13, 12)/46.',
    role: 'construction',
    notes: ['markov-chain'],
    cite: ['norris1997'],
  },
  weatherChain,
)
fn(
  {
    key: 'gamblersRuinChain',
    name: "Gambler's ruin chain",
    summary: 'Win or lose one unit with probability p until reaching 0 or N.',
    role: 'construction',
    notes: ['gamblers-ruin'],
    cite: ['feller1968'],
  },
  gamblersRuinChain,
)
fn(
  {
    key: 'randomWalkChain',
    name: 'Random walk on a cycle or a path',
    summary: 'Hold with some probability, otherwise step to a neighbour, with an optional drift.',
    role: 'construction',
    notes: ['random-walk', 'markov-chain'],
    cite: ['levin2017'],
  },
  randomWalkChain,
)
fn(
  {
    key: 'ehrenfestChain',
    name: 'Ehrenfest urn',
    summary: 'Move a uniformly chosen ball to the other urn; stationary law Binomial(N, ½).',
    role: 'construction',
    notes: ['markov-chain'],
    cite: ['levin2017'],
  },
  ehrenfestChain,
)
