/**
 * Hopfield networks as associative memories, on patterns of ±1.
 *
 * - **Classical** (Hopfield, 1982): the Hebbian weights W = (1/N) Σ_μ ξ^μ ξ^μᵀ with a zero diagonal store P patterns of
 *   N units; asynchronous updates sᵢ ← sign(Σⱼ Wᵢⱼ sⱼ) never raise the energy E(s) = −½ sᵀWs, so the state falls into a
 *   local minimum, ideally the stored pattern nearest the cue. Random patterns are recalled reliably only up to about
 *   0.138 N of them (Amit, Gutfreund and Sompolinsky, 1985); past that, spurious mixtures take over.
 * - **Modern** (Ramsauer et al., 2021, after Krotov and Hopfield, 2016): with the patterns as columns of X, one update
 *   ξ ← X softmax(β Xᵀξ) minimises E(ξ) = −β⁻¹ log Σ_μ exp(β ξ^μᵀξ) + ½ ξᵀξ + const. It is the attention rule, and it
 *   stores exponentially many patterns: with a large β one step retrieves the pattern the cue overlaps most.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike } from 'aifn-compute/foundation/tensor'

/** Patterns as rows [P × N] of ±1 (or reals for the modern network). */
const patternsOf = (patterns: MatrixLike, where: string) => {
  const M = dense.toMatrixF64(patterns, where)
  return { X: M.data, P: M.m, N: M.n }
}

/** The Hebbian weights W = (1/N) Σ ξξᵀ with zero diagonal, row-major [N × N]. */
export function hebbianWeights(patterns: MatrixLike): Float64Array {
  const { X, P, N } = patternsOf(patterns, 'hebbianWeights')
  const W = new Float64Array(N * N)
  for (let m = 0; m < P; m++)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) if (i !== j) W[i * N + j] += (X[m * N + i] * X[m * N + j]) / N
  return W
}

/** The energy −½ sᵀWs. */
export function hopfieldEnergy(W: ArrayLike<number>, s: ArrayLike<number>): number {
  const N = s.length
  let e = 0
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) e -= 0.5 * W[i * N + j] * s[i] * s[j]
  return e
}

/** A recall: the state after each sweep of asynchronous updates (random order), and the energy after each. */
export interface HopfieldRecall {
  readonly states: Float64Array[]
  readonly energies: number[]
  /** True when a sweep changed nothing. */
  readonly converged: boolean
}

/** Asynchronous recall from a cue, at most `sweeps` sweeps over the units in random order. */
export function hopfieldRecall(W: ArrayLike<number>, cue: ArrayLike<number>, s: Stream, sweeps = 20): HopfieldRecall {
  const N = cue.length
  let state = Float64Array.from(cue)
  const states = [state]
  const energies = [hopfieldEnergy(W, state)]
  for (let t = 0; t < sweeps; t++) {
    const u = units(child(s, 'sweep', t), N)
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => u[a] - u[b])
    const next = Float64Array.from(state)
    let changed = false
    for (const i of order) {
      let h = 0
      for (let j = 0; j < N; j++) h += W[i * N + j] * next[j]
      const v = h > 0 ? 1 : h < 0 ? -1 : next[i]
      if (v !== next[i]) changed = true
      next[i] = v
    }
    state = next
    states.push(state)
    energies.push(hopfieldEnergy(W, state))
    if (!changed) return { states, energies, converged: true }
  }
  return { states, energies, converged: false }
}

/** One modern-Hopfield update ξ ← X softmax(β Xᵀ ξ), with the softmax weights over the patterns. */
export function modernHopfieldUpdate(
  patterns: MatrixLike,
  xi: ArrayLike<number>,
  beta: number,
): { state: Float64Array; weights: Float64Array } {
  const { X, P, N } = patternsOf(patterns, 'modernHopfieldUpdate')
  const scores = Float64Array.from({ length: P }, (_, m) => {
    let d = 0
    for (let i = 0; i < N; i++) d += X[m * N + i] * xi[i]
    return beta * d
  })
  const top = Math.max(...scores)
  const w = scores.map((v) => Math.exp(v - top))
  const z = w.reduce((a, b) => a + b, 0)
  const weights = w.map((v) => v / z)
  const state = new Float64Array(N)
  for (let m = 0; m < P; m++) for (let i = 0; i < N; i++) state[i] += weights[m] * X[m * N + i]
  return { state, weights }
}

/** The modern network's energy −β⁻¹ log Σ exp(β ξ^μᵀξ) + ½ ξᵀξ. */
export function modernHopfieldEnergy(patterns: MatrixLike, xi: ArrayLike<number>, beta: number): number {
  const { X, P, N } = patternsOf(patterns, 'modernHopfieldEnergy')
  const scores = Array.from({ length: P }, (_, m) => {
    let d = 0
    for (let i = 0; i < N; i++) d += X[m * N + i] * xi[i]
    return beta * d
  })
  const top = Math.max(...scores)
  const lse = top + Math.log(scores.reduce((a, v) => a + Math.exp(v - top), 0))
  let half = 0
  for (let i = 0; i < N; i++) half += 0.5 * xi[i] * xi[i]
  return -lse / beta + half
}

/** A cue: a pattern with each unit flipped with probability `flip` (and the first `mask` units set to 0). */
export function corruptPattern(pattern: ArrayLike<number>, s: Stream, flip = 0.2, mask = 0): Float64Array {
  const u = units(s, pattern.length)
  return Float64Array.from(pattern, (v, i) => (i < mask ? 0 : u[i] < flip ? -v : v))
}

/** The overlap (1/N) ξᵀs of a state with each pattern. */
export function overlaps(patterns: MatrixLike, s: ArrayLike<number>): Float64Array {
  const { X, P, N } = patternsOf(patterns, 'overlaps')
  return Float64Array.from({ length: P }, (_, m) => {
    let d = 0
    for (let i = 0; i < N; i++) d += X[m * N + i] * s[i]
    return d / N
  })
}

/**
 * Recall quality against load: for P = 1 … maxPatterns random ±1 patterns of N units, the mean final overlap with the
 * cued pattern after classical recall and after one modern update, from cues with a fraction `flip` of units flipped.
 */
export function capacityCurve(
  s: Stream,
  options: { units?: number; maxPatterns?: number; trials?: number; flip?: number; beta?: number } = {},
): { patterns: number[]; classical: Float64Array; modern: Float64Array } {
  const { units: N = 64, maxPatterns = 24, trials = 10, flip = 0.1, beta = 1 } = options
  const patterns: number[] = []
  const classical: number[] = []
  const modern: number[] = []
  for (let P = 1; P <= maxPatterns; P++) {
    let c = 0
    let m = 0
    for (let t = 0; t < trials; t++) {
      const r = child(s, 'load', P, t)
      const u = units(child(r, 'patterns'), P * N)
      const X = Float64Array.from(u, (v) => (v < 0.5 ? -1 : 1))
      const stored = asMatrix(X, P, N)
      const target = X.subarray(0, N)
      const cue = corruptPattern(target, child(r, 'cue'), flip)
      const W = hebbianWeights(stored)
      const recalled = hopfieldRecall(W, cue, child(r, 'recall')).states.at(-1)!
      c += overlaps(asMatrix(target, 1, N), recalled)[0] / trials
      const step = modernHopfieldUpdate(stored, cue, beta).state
      m += overlaps(asMatrix(target, 1, N), step.map(Math.sign))[0] / trials
    }
    patterns.push(P)
    classical.push(c)
    modern.push(m)
  }
  return { patterns, classical: Float64Array.from(classical), modern: Float64Array.from(modern) }
}

const asMatrix = (x: ArrayLike<number>, m: number, n: number) => fromData(Float64Array.from(x), [m, n])
