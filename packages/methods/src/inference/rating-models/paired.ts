/**
 * Batch models of comparisons, fitted by minorisation–maximisation (Hunter, 2004): Bradley–Terry for pairs
 * (Bradley and Terry, 1952), $P(i \text{ beats } j) = \gamma_i / (\gamma_i + \gamma_j)$, and Plackett–Luce for
 * rankings (Luce, 1959; Plackett, 1975), where a ranking is a sequence of choices, each item chosen from those left
 * with probability proportional to its $\gamma$.
 *
 * Each MM step replaces $\gamma_i$ by $w_i / D_i$: $w_i$ counts $i$'s wins (choices), and $D_i$ sums
 * $1 / \sum_{k \in C} \gamma_k$ over every choice set $C$ that contains $i$. The log-likelihood never decreases. With
 * a prior $\alpha > 0$ the step is $(w_i + \alpha) / (D_i + \alpha)$, the MAP update of `choix` (Maystre and
 * Grossglauser), with $\gamma$ scaled to sum to the number of items, which also keeps a player with no wins finite.
 * Strengths are reported as $\log\gamma$ centred to mean zero.
 */

import type { PairedResult } from './elo'

/** Options of the MM fits. */
export interface MmOptions {
  /** The prior $\alpha$ (default 0: maximum likelihood). */
  prior?: number
  /** Most MM steps (default 10000). */
  maxIterations?: number
  /** Stop when the L1 change of the centred log strengths is below this (default 1e-8, as `choix`). */
  tolerance?: number
}

/** A fitted comparison model. */
export interface ComparisonFit {
  /** $\log\gamma$, one per item, centred to mean zero. */
  readonly logStrength: Float64Array
  /** The log-likelihood after each MM step (entry 0 at the start, all strengths equal), `iterations + 1` entries. */
  readonly logLikelihood: Float64Array
  /** The MM steps taken. */
  readonly iterations: number
  /** Whether the change fell below `tolerance` before `maxIterations` steps. */
  readonly converged: boolean
}

/**
 * Log strengths from strengths: $\log\gamma$, shifted to mean zero.
 *
 * @param g The strengths $\gamma$, positive (a zero gives $-\infty$ and then NaN).
 * @returns A new array of the centred logs.
 */
const centred = (g: Float64Array): Float64Array => {
  const l = Float64Array.from(g, (v) => Math.log(v))
  const m = l.reduce((a, b) => a + b, 0) / l.length
  return l.map((v) => v - m)
}

/**
 * Strengths from log strengths: $\gamma = \exp(s)$, scaled to sum to the number of items (as `choix`, which matters
 * only with a prior).
 *
 * @param s The log strengths, one per item.
 * @returns A new array of the strengths.
 */
const weightsOf = (s: Float64Array): Float64Array => {
  const g = Float64Array.from(s, Math.exp)
  const total = g.reduce((a, b) => a + b, 0)
  return g.map((v) => (v * g.length) / total)
}

/**
 * The MM loop shared by the fits: start from equal strengths, apply $\gamma_i \gets (w_i + \alpha)/(D_i + \alpha)$
 * until the L1 change of the centred log strengths is below `tolerance`, recording the log-likelihood at every step.
 *
 * @param items The number of items (players).
 * @param step The model's wins $w_i$ and denominators $D_i$ at the current strengths, one entry per item.
 * @param logLikelihood The model's log-likelihood at given strengths.
 * @param options The prior, the step limit and the tolerance.
 * @returns The centred log strengths, the log-likelihood trace and whether the loop converged.
 */
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

/**
 * The Bradley–Terry probability that $i$ beats $j$: $1 / (1 + \exp(s_j - s_i))$ for log strengths $s$.
 *
 * @param si The log strength $s_i$ of the first player.
 * @param sj The log strength $s_j$ of the second player.
 * @returns The probability, in $(0, 1)$.
 *
 * @example Equal players, and a player twice as strong
 * print('equal:', bradleyTerryProbability(0, 0))
 * print('twice as strong:', bradleyTerryProbability(Math.log(2), 0))
 */
export function bradleyTerryProbability(si: number, sj: number): number {
  return 1 / (1 + Math.exp(sj - si))
}

/**
 * Bradley–Terry by MM on paired results (a draw counts as half a win to each side). Converges to the maximum-likelihood
 * strengths when every player has both won and lost against a connected set of opponents; otherwise use a prior.
 *
 * @param results The results; a score of $\tfrac{1}{2}$ counts as half a win to each player.
 * @param options The number of `players` (indices from 0), and the prior, step limit and tolerance of `MmOptions`.
 * @returns The fitted log strengths and the log-likelihood trace.
 *
 * @example Nine games between three players
 * // Each pair [a, b] is a win of a over b: player 0 wins 3 of 5, player 1 wins 4 of 7, player 2 wins 2 of 6.
 * const wins = [[0, 1], [0, 1], [1, 0], [1, 2], [1, 2], [1, 2], [2, 1], [0, 2], [2, 0]]
 * const fit = bradleyTerry(wins.map(([a, b]) => ({ a, b, score: 1 })), { players: 3 })
 * print('log strengths:', fit.logStrength)
 * print('steps:', fit.iterations, 'converged:', fit.converged)
 * print('P(0 beats 2):', bradleyTerryProbability(fit.logStrength[0], fit.logStrength[2]))
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
 * The Plackett–Luce log probability of a ranking $\pi$ (best first) under log strengths $s$:
 * $\sum_r \log\paren{\exp(s_{\pi(r)}) / \sum_{k \ge r} \exp(s_{\pi(k)})}$. The ranking is scored over the items
 * it lists only, so a ranking of a subset is scored as a choice among that subset.
 *
 * @param ranking The item indices, best first.
 * @param logStrength The log strengths $s$, indexed by item.
 * @returns The log probability of the ranking.
 *
 * @example Every order of three equal items has probability one in six
 * print('p:', Math.exp(plackettLuceLogProbability([0, 1, 2], [0, 0, 0])))
 * print('p, with item 2 stronger:', Math.exp(plackettLuceLogProbability([2, 0, 1], [0, 0, Math.log(4)])))
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

/**
 * Plackett–Luce by MM on rankings, each a list of item indices from best to worst (Hunter, 2004). A ranking may list
 * only some of the items, and is then a sequence of choices among those.
 *
 * @param rankings The rankings, each best first.
 * @param options The number of `items` (indices from 0), and the prior, step limit and tolerance of `MmOptions`.
 * @returns The fitted log strengths and the log-likelihood trace.
 *
 * @example Four races between three runners
 * const fit = plackettLuce([[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1]], { items: 3 })
 * print('log strengths:', fit.logStrength)
 * print('log-likelihood, start and end:', fit.logLikelihood[0], fit.logLikelihood[fit.iterations])
 */
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
  /** The number of players; results index them from 0. */
  players: number
  /** Refit after every this many results (default 10). */
  refitEvery?: number
  /** The MM prior $\alpha$ (default 0.1, so early fits with unbeaten players stay finite). */
  prior?: number
}

/**
 * Bradley–Terry evaluated prequentially: each result is predicted from a fit on the results before it (refitted every
 * `refitEvery` results, so from the latest refit; before the first, every prediction is $\tfrac{1}{2}$), so its
 * predictions can be compared with an online rating's. Each refit runs at most 500 MM steps to a tolerance of
 * $10^{-6}$. The history holds the strengths on Elo's scale, $1500 + (400 / \ln 10) \log\gamma$, row-major
 * $(\text{results} + 1) \times \text{players}$: row 0 is all 1500, and row $g + 1$ holds the strengths that predicted
 * result $g$.
 *
 * @param results The results, in order.
 * @param options The number of players, the refit interval and the prior.
 * @returns The prediction of each result (the probability that its first player scores 1), and the history.
 *
 * @example Thirty games, refitted every ten
 * // Log strengths -1, 0 and 1; each game is won with the Bradley-Terry probability, drawn from stream(2).
 * const s = [-1, 0, 1]
 * const u = toArray(uniform(stream(2), 0, 1, { shape: [30] }))
 * const results = u.map((ui, i) => {
 *   const a = i % 3
 *   const b = (i + 1) % 3
 *   return { a, b, score: ui < bradleyTerryProbability(s[a], s[b]) ? 1 : 0 }
 * })
 * const { predicted, history } = prequentialBradleyTerry(results, { players: 3 })
 * print('predictions of games 9 to 11:', predicted.slice(9, 12))
 * print('strengths used for the last game:', history.slice(-3))
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
