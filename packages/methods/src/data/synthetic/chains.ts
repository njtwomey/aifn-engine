/**
 * Named Markov chains for demonstrations: the weather chain, gambler's ruin, a lazy random walk on a cycle or a path,
 * and the Ehrenfest urn. Each returns its transition matrix [n, n] with state labels; the chain computations are in
 * `aifn-compute/probability/markov`.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { matrix } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A named chain: its transition matrix and a label for each state. */
export interface NamedChain {
  readonly P: Tensor
  readonly states: readonly string[]
}

const build = (n: number, entry: (i: number, j: number) => number): Tensor => {
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = entry(i, j)
  return matrix(P, n, n)
}

/** Sunny, cloudy, rainy: P = [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]], with π = (21, 13, 12)/46. */
export function weatherChain(): NamedChain {
  const rows = [
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
  ]
  return { P: build(3, (i, j) => rows[i][j]), states: ['sunny', 'cloudy', 'rainy'] }
}

/**
 * Gambler's ruin on {0, …, N}: from 0 < i < N, win one unit with probability p or lose one; 0 and N absorb. The chance
 * of reaching N from i is i/N in a fair game and (1 − (q/p)ⁱ)/(1 − (q/p)ᴺ) otherwise.
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
 * A random walk on n states that holds with probability `hold` and otherwise steps to a neighbour, on a cycle (wrapping)
 * or a path (a step off an end holds instead). hold = 0 on an even cycle is periodic; hold = ½ is the lazy walk.
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
 * The Ehrenfest urn with N balls in two urns: state i is the number in the first urn, and each step moves a ball
 * chosen uniformly to the other urn (with probability `hold` nothing moves). Its stationary law is Binomial(N, ½).
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
