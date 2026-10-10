/**
 * Hopfield networks as associative memories, on patterns of $\pm 1$.
 *
 * - **Classical** (Hopfield, 1982): the Hebbian weights
 *   $\Wmat = \frac{1}{N} \sum_\mu \xivec^{(\mu)} \xivec^{(\mu)\top}$ with a zero diagonal store $P$ patterns of
 *   $N$ units; asynchronous updates $s_i \leftarrow \sgn(\sum_j W_{ij} s_j)$ never raise the energy
 *   $E(\svec) = -\frac{1}{2} \svec^\top\Wmat\svec$, so the state falls into a local minimum, ideally the stored
 *   pattern nearest the cue. Random patterns are recalled reliably only up to about $0.138 N$ of them (Amit, Gutfreund
 *   and Sompolinsky, 1985); past that, spurious mixtures take over.
 * - **Modern** (Ramsauer et al., 2021, after Krotov and Hopfield, 2016): with the patterns as columns of $\Xmat$, one
 *   update $\xivec \leftarrow \Xmat \operatorname{softmax}(\beta \Xmat^\top\xivec)$ never raises the energy
 *   $E(\xivec) = -\beta^{-1} \log \sum_\mu \exp(\beta \xivec^{(\mu)\top}\xivec) + \frac{1}{2} \xivec^\top\xivec +
 *   \text{const}$. It is the attention rule, and it stores exponentially many patterns: with a large $\beta$ one
 *   step retrieves the pattern the cue overlaps most.
 *
 * The functions take the patterns as the rows of a $P \times N$ matrix (that is, $\Xmat^\top$), and weights and
 * states as plain arrays, the weights row-major $N \times N$.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike } from 'aifn-compute/foundation/tensor'

/**
 * Patterns as rows of $\pm 1$ (or reals for the modern network), read into a dense row-major array.
 *
 * @param patterns The patterns, $P \times N$, one per row.
 * @param where The caller's name, for error messages.
 * @returns `X`, the $P N$ values row-major, with `P` and `N`.
 */
const patternsOf = (patterns: MatrixLike, where: string) => {
  const M = dense.toMatrixF64(patterns, where)
  return { X: M.data, P: M.m, N: M.n }
}

/**
 * The Hebbian weights $\Wmat = \frac{1}{N} \sum_\mu \xivec^{(\mu)} \xivec^{(\mu)\top}$ with a zero diagonal.
 *
 * @param patterns The patterns to store, $P \times N$ with entries $\pm 1$, one per row.
 * @returns $\Wmat$, row-major $N \times N$.
 *
 * @example The weights that store one pattern of three units
 * print('W =', hebbianWeights(tensor([[1, -1, 1]])))
 */
export function hebbianWeights(patterns: MatrixLike): Float64Array {
  const { X, P, N } = patternsOf(patterns, 'hebbianWeights')
  const W = new Float64Array(N * N)
  for (let m = 0; m < P; m++)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) if (i !== j) W[i * N + j] += (X[m * N + i] * X[m * N + j]) / N
  return W
}

/**
 * The classical network's energy $E(\svec) = -\frac{1}{2} \svec^\top\Wmat\svec$.
 *
 * @param W The weights $\Wmat$, row-major $N \times N$.
 * @param s The state $\svec$, $N$ values (its length sets $N$).
 * @returns $E(\svec)$.
 *
 * @example A stored pattern sits lower than a corrupted copy of it
 * const W = hebbianWeights(tensor([[1, 1, -1, -1, 1, -1]]))
 * print('E(pattern) =', hopfieldEnergy(W, [1, 1, -1, -1, 1, -1]))
 * print('E(one unit flipped) =', hopfieldEnergy(W, [-1, 1, -1, -1, 1, -1]))
 */
export function hopfieldEnergy(W: ArrayLike<number>, s: ArrayLike<number>): number {
  const N = s.length
  let e = 0
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) e -= 0.5 * W[i * N + j] * s[i] * s[j]
  return e
}

/** A recall: the state after each sweep of asynchronous updates (random order), and the energy after each. */
export interface HopfieldRecall {
  /** The cue, then the state after each sweep, $N$ values each. */
  readonly states: Float64Array[]
  /** The energy of each state in `states`. */
  readonly energies: number[]
  /** True when a sweep changed nothing. */
  readonly converged: boolean
}

/**
 * Asynchronous recall from a cue: sweeps over the units in a fresh random order, each unit set in turn to
 * $\sgn(\sum_j W_{ij} s_j)$ of the current state (kept when the sum is 0), until a sweep changes nothing or `sweeps`
 * have run. The energy never rises.
 *
 * @param W The weights $\Wmat$, row-major $N \times N$ (from `hebbianWeights`).
 * @param cue The starting state, $N$ values of $\pm 1$; not modified.
 * @param s The stream; sweep $t$ draws its order from `child(s, 'sweep', t)`.
 * @param sweeps The most sweeps to run.
 * @returns The states and their energies, sweep by sweep, and whether the recall converged.
 *
 * @example A Hopfield network recalls a stored pattern from a corrupted cue
 * const stored = tensor([
 *   [1, 1, 1, 1, -1, -1, -1, -1],
 *   [1, -1, 1, -1, 1, -1, 1, -1],
 * ])
 * const W = hebbianWeights(stored)
 * const cue = [1, 1, -1, 1, -1, -1, -1, -1]
 * const recall = hopfieldRecall(W, cue, stream(1))
 * print('recalled =', recall.states.at(-1))
 * print('overlaps with the stored patterns =', overlaps(stored, recall.states.at(-1)))
 * print('energies by sweep =', recall.energies, ' converged =', recall.converged)
 */
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

/**
 * One modern-Hopfield update $\xivec \leftarrow \Xmat \operatorname{softmax}(\beta \Xmat^\top\xivec)$: the
 * average of the patterns weighted by the softmax of their scaled overlaps with the state.
 *
 * @param patterns The stored patterns, $P \times N$, one per row ($\Xmat^\top$).
 * @param xi The state $\xivec$, $N$ values.
 * @param beta The inverse temperature $\beta$: large retrieves the best-matching pattern, small averages them.
 * @returns `state`, the new $\xivec$ ($N$ values), and `weights`, the softmax weights over the $P$ patterns.
 *
 * @example One step retrieves the pattern the cue overlaps most
 * const stored = tensor([
 *   [1, 1, 1, 1, -1, -1, -1, -1],
 *   [1, -1, 1, -1, 1, -1, 1, -1],
 * ])
 * const { state, weights } = modernHopfieldUpdate(stored, [1, 1, -1, 1, -1, -1, -1, -1], 1)
 * print('weights =', weights)
 * print('state =', state)
 */
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

/**
 * The modern network's energy
 * $E(\xivec) = -\beta^{-1} \log \sum_\mu \exp(\beta \xivec^{(\mu)\top}\xivec) + \frac{1}{2} \xivec^\top\xivec$,
 * without the constant.
 *
 * @param patterns The stored patterns, $P \times N$, one per row.
 * @param xi The state $\xivec$, $N$ values.
 * @param beta The inverse temperature $\beta$.
 * @returns $E(\xivec)$.
 *
 * @example An update never raises the energy
 * const stored = tensor([
 *   [1, 1, 1, 1, -1, -1, -1, -1],
 *   [1, -1, 1, -1, 1, -1, 1, -1],
 * ])
 * const cue = [1, 1, -1, 1, -1, -1, -1, -1]
 * const { state } = modernHopfieldUpdate(stored, cue, 1)
 * print('E(cue) =', modernHopfieldEnergy(stored, cue, 1))
 * print('E(after one update) =', modernHopfieldEnergy(stored, state, 1))
 */
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

/**
 * A cue: a pattern with each unit flipped with probability `flip`, and the first `mask` units set to 0 (unknown).
 *
 * @param pattern The pattern, $N$ values of $\pm 1$; not modified.
 * @param s The stream of the $N$ uniforms that decide the flips.
 * @param flip The probability that each unit is flipped.
 * @param mask How many leading units to set to 0, whether flipped or not.
 * @returns The cue, $N$ values.
 *
 * @example Flip a fifth of the units, and blank the first two
 * const pattern = [1, 1, 1, 1, -1, -1, -1, -1, 1, -1]
 * print('flipped =', corruptPattern(pattern, stream(1), 0.2))
 * print('flipped and masked =', corruptPattern(pattern, stream(1), 0.2, 2))
 */
export function corruptPattern(pattern: ArrayLike<number>, s: Stream, flip = 0.2, mask = 0): Float64Array {
  const u = units(s, pattern.length)
  return Float64Array.from(pattern, (v, i) => (i < mask ? 0 : u[i] < flip ? -v : v))
}

/**
 * The overlap $m_\mu = \frac{1}{N} \xivec^{(\mu)\top}\svec$ of a state with each pattern: 1 for the pattern
 * itself, $-1$ for its negation, near 0 for an unrelated one.
 *
 * @param patterns The patterns, $P \times N$, one per row.
 * @param s The state $\svec$, $N$ values.
 * @returns The $P$ overlaps.
 *
 * @example A state against two patterns
 * print(overlaps(tensor([[1, 1, -1, -1], [1, -1, 1, -1]]), [1, 1, -1, 1]))
 */
export function overlaps(patterns: MatrixLike, s: ArrayLike<number>): Float64Array {
  const { X, P, N } = patternsOf(patterns, 'overlaps')
  return Float64Array.from({ length: P }, (_, m) => {
    let d = 0
    for (let i = 0; i < N; i++) d += X[m * N + i] * s[i]
    return d / N
  })
}

/**
 * Recall quality against load: for $P = 1, \dots,$ `maxPatterns` random $\pm 1$ patterns of $N$ units, the mean
 * overlap with the cued pattern after classical recall (`hopfieldRecall` to the end) and after one modern update (its
 * signs), from cues with each unit flipped with probability `flip`. The first stored pattern is the one cued.
 *
 * @param s The stream; load $P$ and trial $t$ draw from `child(s, 'load', P, t)`.
 * @param options The network and the experiment, with defaults: `units` $N$ (64), `maxPatterns` (24), `trials` per
 *   load (10), `flip` (0.1) and the modern network's `beta` (1).
 * @returns `patterns`, the loads $P$, with `classical` and `modern`, the mean overlap at each.
 *
 * @example As the load grows, classical recall degrades well before one modern step does
 * const curve = capacityCurve(stream(1), { units: 16, maxPatterns: 8, trials: 4 })
 * print('patterns:', curve.patterns)
 * print('classical:', curve.classical)
 * print('modern:', curve.modern)
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

/**
 * A copy of values as an $m \times n$ tensor.
 *
 * @param x The values, $m n$ of them, row-major.
 * @param m The number of rows.
 * @param n The number of columns.
 * @returns The tensor.
 */
const asMatrix = (x: ArrayLike<number>, m: number, n: number) => fromData(Float64Array.from(x), [m, n])
