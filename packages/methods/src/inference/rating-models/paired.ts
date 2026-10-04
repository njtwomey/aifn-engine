/**
 * Batch models of comparisons, fitted by minorisation–maximisation (Hunter, 2004): Bradley–Terry for pairs
 * (Bradley and Terry, 1952), P(i beats j) = γᵢ/(γᵢ + γⱼ), and Plackett–Luce for rankings (Luce, 1959; Plackett, 1975),
 * where a ranking is a sequence of choices, each item chosen from those left with probability ∝ γ.
 *
 * Each MM step replaces γᵢ by wᵢ/Dᵢ: wᵢ counts i's wins (choices), and Dᵢ sums 1/(Σ γ over the choice set) over every
 * choice set that contains i. The log-likelihood never decreases. With a prior α > 0 the step is (wᵢ + α)/(Dᵢ + α),
 * the MAP update of `choix` (Maystre and Grossglauser), with γ scaled to sum to the number of items, which also keeps a player with no wins finite. Strengths are
 * reported as log γ centred to mean zero.
 */

import type { PairedResult } from './elo'

/** Options of the MM fits. */
export interface MmOptions {
  /** The prior α (default 0: maximum likelihood). */
  prior?: number
  /** Most MM steps (default 10 000). */
  maxIterations?: number
  /** Stop when the L1 change of the centred log strengths is below this (default 1e-8, as `choix`). */
  tolerance?: number
}

/** A fitted comparison model. */
export interface ComparisonFit {
  /** log γ, centred to mean zero. */
  readonly logStrength: Float64Array
  /** The log-likelihood after each MM step (entry 0 at the start, all strengths equal). */
  readonly logLikelihood: Float64Array
  readonly iterations: number
  readonly converged: boolean
}

const centred = (g: Float64Array): Float64Array => {
  const l = Float64Array.from(g, (v) => Math.log(v))
  const m = l.reduce((a, b) => a + b, 0) / l.length
  return l.map((v) => v - m)
}

/** γ = exp(s), scaled to sum to the number of items (as `choix`, which matters only with a prior). */
const weightsOf = (s: Float64Array): Float64Array => {
  const g = Float64Array.from(s, Math.exp)
  const total = g.reduce((a, b) => a + b, 0)
  return g.map((v) => (v * g.length) / total)
}

function runMm(
  items: number,
  step: (gamma: Float64Array) => { wins: Float64Array; denominators: Float64Array },
  logLikelihood: (gamma: Float64Array) => number,
  options: MmOptions,
): ComparisonFit {
  const { prior = 0, maxIterations = 10_000, tolerance = 1e-8 } = options
  let gamma: Float64Array = new Float64Array(items).fill(1)
  let params = centred(gamma)
  const ll = [logLikelihood(gamma)]
  for (let it = 1; it <= maxIterations; it++) {
    const { wins, denominators } = step(gamma)
    const next = Float64Array.from(wins, (w, i) => (w + prior) / (denominators[i] + prior))
    const nextParams = centred(next)
    let change = 0
    for (let i = 0; i < items; i++) change += Math.abs(nextParams[i] - params[i])
    gamma = weightsOf(nextParams)
    params = nextParams
    ll.push(logLikelihood(gamma))
    if (change < tolerance)
      return { logStrength: params, logLikelihood: Float64Array.from(ll), iterations: it, converged: true }
  }
  return { logStrength: params, logLikelihood: Float64Array.from(ll), iterations: maxIterations, converged: false }
}

/** P(i beats j) = 1/(1 + exp(sⱼ − sᵢ)) for log strengths s. */
export function bradleyTerryProbability(si: number, sj: number): number {
  return 1 / (1 + Math.exp(sj - si))
}

/**
 * Bradley–Terry by MM on paired results (a draw counts as half a win to each side). Converges to the maximum-likelihood
 * strengths when every player has both won and lost against a connected set of opponents; otherwise use a prior.
 */
export function bradleyTerry(
  results: readonly PairedResult[],
  options: MmOptions & { players: number },
): ComparisonFit {
  const { players } = options
  const step = (gamma: Float64Array) => {
    const wins = new Float64Array(players)
    const denominators = new Float64Array(players)
    for (const { a, b, score } of results) {
      wins[a] += score
      wins[b] += 1 - score
      // Each game is one choice set {a, b}: both players' denominators gain 1/(γa + γb).
      const v = 1 / (gamma[a] + gamma[b])
      denominators[a] += v
      denominators[b] += v
    }
    return { wins, denominators }
  }
  const logLikelihood = (gamma: Float64Array) => {
    let ll = 0
    for (const { a, b, score } of results) {
      const p = gamma[a] / (gamma[a] + gamma[b])
      ll += score * Math.log(p) + (1 - score) * Math.log(1 - p)
    }
    return ll
  }
  return runMm(players, step, logLikelihood, options)
}

/**
 * The Plackett–Luce probability of a ranking (best first) under log strengths s: Π_r exp(s_{π(r)}) / Σ_{k ≥ r}
 * exp(s_{π(k)}). A partial ranking (a top-k of a larger set) uses only the items it lists.
 */
export function plackettLuceLogProbability(ranking: readonly number[], logStrength: ArrayLike<number>): number {
  let ll = 0
  let rest = 0
  for (const i of ranking) rest += Math.exp(logStrength[i])
  for (let r = 0; r + 1 < ranking.length; r++) {
    const w = Math.exp(logStrength[ranking[r]])
    ll += Math.log(w / rest)
    rest -= w
  }
  return ll
}

/** Plackett–Luce by MM on rankings, each a list of item indices from best to worst. */
export function plackettLuce(
  rankings: readonly (readonly number[])[],
  options: MmOptions & { items: number },
): ComparisonFit {
  const { items } = options
  const step = (gamma: Float64Array) => {
    const wins = new Float64Array(items)
    const denominators = new Float64Array(items)
    for (const ranking of rankings) {
      let rest = 0
      for (const i of ranking) rest += gamma[i]
      for (let r = 0; r + 1 < ranking.length; r++) {
        wins[ranking[r]] += 1
        const v = 1 / rest
        for (let k = r; k < ranking.length; k++) denominators[ranking[k]] += v
        rest -= gamma[ranking[r]]
      }
    }
    return { wins, denominators }
  }
  const logLikelihood = (gamma: Float64Array) => {
    const s = Float64Array.from(gamma, Math.log)
    let ll = 0
    for (const ranking of rankings) ll += plackettLuceLogProbability(ranking, s)
    return ll
  }
  return runMm(items, step, logLikelihood, options)
}

/** Options of `prequentialBradleyTerry`. */
export interface PrequentialOptions {
  players: number
  /** Refit after every this many results (default 10). */
  refitEvery?: number
  /** The MM prior α (default 0.1, so early fits with unbeaten players stay finite). */
  prior?: number
}

/**
 * Bradley–Terry evaluated prequentially: each result is predicted from a fit on the results before it (refitted every
 * `refitEvery` results), so its predictions can be compared with an online rating's. The history holds the fitted
 * strengths on Elo's scale, 1500 + (400/ln 10) log γ, row-major [(results + 1) × players].
 */
export function prequentialBradleyTerry(
  results: readonly PairedResult[],
  options: PrequentialOptions,
): { predicted: Float64Array; history: Float64Array } {
  const { players, refitEvery = 10, prior = 0.1 } = options
  const scale = 400 / Math.LN10
  const predicted = new Float64Array(results.length)
  const history = new Float64Array((results.length + 1) * players).fill(1500)
  let s: Float64Array = new Float64Array(players)
  for (let g = 0; g < results.length; g++) {
    if (g > 0 && g % refitEvery === 0)
      s = bradleyTerry(results.slice(0, g), { players, prior, tolerance: 1e-6, maxIterations: 500 }).logStrength
    const { a, b } = results[g]
    predicted[g] = bradleyTerryProbability(s[a], s[b])
    for (let p = 0; p < players; p++) history[(g + 1) * players + p] = 1500 + scale * s[p]
  }
  return { predicted, history }
}
