/**
 * Small discrete models for the inference tests, built from compute pieces only: the sprinkler network in the model
 * language, a random tree-shaped factor graph, an Ising grid, and the occasionally dishonest casino as chain
 * potentials (with brute-force enumeration of its hidden paths).
 */
import {
  discreteFactor,
  discreteFactorGraph,
  dist,
  model,
  type DiscreteFactorGraph,
} from 'aifn-compute/inference/model'
import type { ChainPotentials } from 'aifn-compute/inference/exact'
import { gridGraph } from 'aifn-compute/graph/structures'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { fromRows } from 'aifn-compute/foundation/tensor'

/** The sprinkler network (Russell & Norvig), written in the description language with CPTs indexed by parents. */
export const sprinkler = model('Sprinkler', (m) => {
  const cloudy = m.variable('cloudy', dist.Bernoulli(0.5))
  const pS = m.constant('pS', [0.5, 0.1])
  const pR = m.constant('pR', [0.2, 0.8])
  const pW = m.constant('pW', [
    [0.01, 0.9],
    [0.9, 0.99],
  ])
  const s = m.variable('sprinkler', dist.Bernoulli(pS.at(cloudy)))
  const r = m.variable('rain', dist.Bernoulli(pR.at(cloudy)))
  const pw = m.deterministic('pw', 'index', [pW, s, r])
  m.observed('wet', dist.Bernoulli(pw))
})

/** P(rain = 1 | wet = 1), P(cloudy = 1 | wet = 1) and P(wet = 1) by summing the joint by hand. */
export function sprinklerBrute() {
  const pS = [0.5, 0.1]
  const pR = [0.2, 0.8]
  const pW = [
    [0.01, 0.9],
    [0.9, 0.99],
  ]
  const b = (p: number, x: number) => (x ? p : 1 - p)
  let evidence = 0
  let rain = 0
  let cloudy = 0
  for (const c of [0, 1])
    for (const s of [0, 1])
      for (const r of [0, 1]) {
        const p = 0.5 * b(pS[c], s) * b(pR[c], r) * pW[s][r]
        evidence += p
        if (r) rain += p
        if (c) cloudy += p
      }
  return { rain: rain / evidence, cloudy: cloudy / evidence, evidence }
}

/** A tree-shaped factor graph: a chain x0 – x1 – x2 with a branch x1 – x3, unary factors, random positive tables. */
export function randomTree(seed: number): DiscreteFactorGraph {
  const s = stream(seed)
  const cards = [2, 3, 2, 2]
  const table = (scope: number[], k: number) =>
    discreteFactor(scope, cards, (a) => 0.2 + (uniform(child(s, k, ...Array.from(a))) as number))
  return discreteFactorGraph(cards, [
    table([0], 0),
    table([0, 1], 1),
    table([1, 2], 2),
    table([1, 3], 3),
    table([3], 4),
  ])
}

/** An Ising model on an r × c grid: spins s ∈ {−1, +1}, p(s) ∝ exp(J Σ_{ij} sᵢsⱼ + h Σ sᵢ). */
export function isingGrid(rows: number, cols: number, coupling: number, field: number): DiscreteFactorGraph {
  const n = rows * cols
  const cards = new Array<number>(n).fill(2)
  const spin = (x: number) => 2 * x - 1
  const factors = [
    ...Array.from({ length: n }, (_, v) => discreteFactor([v], cards, (a) => Math.exp(field * spin(a[0])))),
    ...gridGraph(rows, cols).edges.map((e) =>
      discreteFactor([e.from, e.to], cards, (a) => Math.exp(coupling * spin(a[0]) * spin(a[1]))),
    ),
  ]
  return discreteFactorGraph(cards, factors)
}

/** The occasionally dishonest casino (Durbin et al., 1998): fair and loaded dice, sticky switching. */
export const casino = {
  initial: [0.5, 0.5],
  transition: [
    [0.95, 0.05],
    [0.1, 0.9],
  ],
  emission: [
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
  ],
}

/** The chain potentials of the casino for observations x: ψ₀(k) = π_k B_k(x₀), ψₙ(k) = B_k(xₙ). */
export function casinoChain(obs: readonly number[]): ChainPotentials {
  const psi = obs.map((x, n) => casino.emission.map((row, k) => (n === 0 ? casino.initial[k] : 1) * row[x]))
  return { nodePotentials: fromRows(psi), transition: fromRows(casino.transition) }
}

/** Every hidden path of the casino enumerated: marginals, pairwise marginals, log p(x) and the best path. */
export function casinoBrute(obs: readonly number[]) {
  const K = 2
  const N = obs.length
  const A = casino.transition
  const B = casino.emission
  const marg = Array.from({ length: N }, () => new Array<number>(K).fill(0))
  const pair = Array.from({ length: N - 1 }, () => Array.from({ length: K }, () => new Array<number>(K).fill(0)))
  let total = 0
  let best = -Infinity
  let bestPath: number[] = []
  for (let code = 0; code < K ** N; code++) {
    const y = Array.from({ length: N }, (_, n) => Math.floor(code / K ** (N - 1 - n)) % K)
    let p = casino.initial[y[0]] * B[y[0]][obs[0]]
    for (let n = 1; n < N; n++) p *= A[y[n - 1]][y[n]] * B[y[n]][obs[n]]
    total += p
    y.forEach((k, n) => (marg[n][k] += p))
    for (let n = 0; n + 1 < N; n++) pair[n][y[n]][y[n + 1]] += p
    if (p > best) [best, bestPath] = [p, y]
  }
  return {
    marginals: marg.map((r) => r.map((v) => v / total)),
    pairwise: pair.map((m) => m.map((r) => r.map((v) => v / total))),
    logLikelihood: Math.log(total),
    path: bestPath,
    logBest: Math.log(best),
  }
}
