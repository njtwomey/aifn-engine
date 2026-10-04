/**
 * Online skill ratings from paired results: Elo (Elo, 1978), Glicko (Glickman, 1999) and Glicko-2 (Glickman, 2001).
 *
 * A result `{ a, b, score }` gives player a's score against player b: 1 for a win, ½ for a draw, 0 for a loss. Ratings
 * are on Elo's scale, where a difference of 400 points means odds of 10 : 1. Elo moves both ratings by K times the
 * surprise (score − expected score). Glicko adds a rating deviation (RD) per player: the update is larger when the
 * player's RD is large and smaller when the opponent's RD is large, and RD grows between rating periods. Glicko-2 adds
 * a volatility σ, the rate at which a player's skill is believed to drift, updated from how surprising the period was.
 */

import { spearman } from 'aifn-compute/probability/stats'

/** One paired result: player a's score against player b (1 win, ½ draw, 0 loss). */
export interface PairedResult {
  a: number
  b: number
  score: number
}

const LN10 = Math.LN10
/** Glicko's q = ln 10 / 400. */
const Q = LN10 / 400
/** Glicko-2's scale: μ = (r − 1500)/173.7178. */
const GLICKO2_SCALE = 400 / LN10

/** Elo's expected score of a rating `ra` against `rb`: 1/(1 + 10^((rb − ra)/scale)). */
export function eloExpected(ra: number, rb: number, scale = 400): number {
  return 1 / (1 + Math.pow(10, (rb - ra) / scale))
}

/** Options of Elo updates. */
export interface EloOptions {
  /** The K-factor: the most a rating moves after one game (default 32). */
  k?: number
  /** Points per factor of 10 in the odds (default 400). */
  scale?: number
}

/** One Elo update: both new ratings and a's expected score before the game. */
export function eloUpdate(
  ra: number,
  rb: number,
  score: number,
  options: EloOptions = {},
): { a: number; b: number; expected: number } {
  const { k = 32, scale = 400 } = options
  const expected = eloExpected(ra, rb, scale)
  return { a: ra + k * (score - expected), b: rb - k * (score - expected), expected }
}

/** A run of ratings over a sequence of results. */
export interface RatingRun {
  /** Ratings after each result, row-major [(results + 1) × players]; row 0 is the start. */
  readonly history: Float64Array
  /** Rating deviations in the same layout (Glicko); null for Elo. */
  readonly deviations: Float64Array | null
  /** The final ratings. */
  readonly ratings: Float64Array
  /** Predicted probability that a scores 1, before each result. */
  readonly predicted: Float64Array
  /** Mean log loss of the predictions on decisive results (draws are skipped). */
  readonly logLoss: number
  readonly players: number
}

function logLossOf(results: readonly PairedResult[], predicted: Float64Array): number {
  let sum = 0
  let n = 0
  for (let i = 0; i < results.length; i++) {
    const s = results[i].score
    if (s !== 0 && s !== 1) continue
    const p = Math.min(1 - 1e-12, Math.max(1e-12, predicted[i]))
    sum -= s === 1 ? Math.log(p) : Math.log(1 - p)
    n++
  }
  return n > 0 ? sum / n : NaN
}

/** Elo ratings over a sequence of results, each player starting at `initial` (default 1500). */
export function eloRatings(
  results: readonly PairedResult[],
  options: EloOptions & { players: number; initial?: number },
): RatingRun {
  const { players, initial = 1500 } = options
  const r = new Float64Array(players).fill(initial)
  const history = new Float64Array((results.length + 1) * players)
  history.set(r, 0)
  const predicted = new Float64Array(results.length)
  results.forEach(({ a, b, score }, i) => {
    const u = eloUpdate(r[a], r[b], score, options)
    predicted[i] = u.expected
    r[a] = u.a
    r[b] = u.b
    history.set(r, (i + 1) * players)
  })
  return { history, deviations: null, ratings: r, predicted, logLoss: logLossOf(results, predicted), players }
}

// ── Glicko ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A Glicko rating: r and its deviation RD (and the Glicko-2 volatility σ). */
export interface GlickoRating {
  rating: number
  deviation: number
  volatility?: number
}

/** One game of a rating period, from the rated player's side. */
export interface GlickoGame {
  opponent: GlickoRating
  score: number
}

/** g(RD) = 1/√(1 + 3q²RD²/π²): how much an opponent's uncertainty damps the evidence of a game. */
export function glickoG(deviation: number): number {
  return 1 / Math.sqrt(1 + (3 * Q * Q * deviation * deviation) / (Math.PI * Math.PI))
}

/** Glicko's expected score of r against an opponent rated rj with deviation RDj. */
export function glickoExpected(r: number, opponent: GlickoRating): number {
  return 1 / (1 + Math.pow(10, (-glickoG(opponent.deviation) * (r - opponent.rating)) / 400))
}

/**
 * One Glicko-1 rating period for one player (Glickman, 1999): with d² = 1/(q² Σ g² E(1 − E)),
 * r′ = r + q/(1/RD² + 1/d²) Σ g (s − E) and RD′ = (1/RD² + 1/d²)^(−½). RD is not inflated here; `glickoRatings` does
 * that at the start of each period. A period with no games leaves the rating unchanged.
 */
export function glickoUpdate(player: GlickoRating, games: readonly GlickoGame[]): GlickoRating {
  if (games.length === 0) return { ...player }
  let info = 0
  let push = 0
  for (const { opponent, score } of games) {
    const g = glickoG(opponent.deviation)
    const e = glickoExpected(player.rating, opponent)
    info += g * g * e * (1 - e)
    push += g * (score - e)
  }
  const precision = 1 / (player.deviation * player.deviation) + Q * Q * info
  return { rating: player.rating + (Q / precision) * push, deviation: Math.sqrt(1 / precision) }
}

/** Options of Glicko-2 updates. */
export interface Glicko2Options {
  /** τ, which bounds how fast the volatility changes (Glickman suggests 0.3–1.2; default 0.5). */
  tau?: number
  /** Convergence tolerance of the volatility iteration (default 1e-6). */
  tolerance?: number
  /**
   * Rating periods since the player's last update (default 1). φ grows to √(φ² + elapsed·σ²) before the games, as in
   * Lichess's fractional periods; 0 for a second game inside one period.
   */
  elapsed?: number
}

/**
 * One Glicko-2 rating period for one player (Glickman, 2001, steps 2–8): on the scale μ = (r − 1500)/173.7178,
 * φ = RD/173.7178, the variance v and improvement Δ of the period, a new volatility σ′ by the Illinois iteration on
 * f(x) = eˣ(Δ² − φ² − v − eˣ)/(2(φ² + v + eˣ)²) − (x − ln σ²)/τ², then φ′ = 1/√(1/(φ² + σ′²) + 1/v) and
 * μ′ = μ + φ′² Σ g(φⱼ)(s − E). With no games φ grows to √(φ² + σ²) and μ stays.
 */
export function glicko2Update(
  player: GlickoRating,
  games: readonly GlickoGame[],
  options: Glicko2Options = {},
): GlickoRating {
  const { tau = 0.5, tolerance = 1e-6, elapsed = 1 } = options
  const sigma = player.volatility ?? 0.06
  const mu = (player.rating - 1500) / GLICKO2_SCALE
  const phi = player.deviation / GLICKO2_SCALE
  if (games.length === 0) {
    const grown = Math.sqrt(phi * phi + elapsed * sigma * sigma)
    return { rating: player.rating, deviation: grown * GLICKO2_SCALE, volatility: sigma }
  }
  let vInverse = 0
  let sum = 0
  for (const { opponent, score } of games) {
    const muJ = (opponent.rating - 1500) / GLICKO2_SCALE
    const phiJ = opponent.deviation / GLICKO2_SCALE
    const g = 1 / Math.sqrt(1 + (3 * phiJ * phiJ) / (Math.PI * Math.PI))
    const e = 1 / (1 + Math.exp(-g * (mu - muJ)))
    vInverse += g * g * e * (1 - e)
    sum += g * (score - e)
  }
  const v = 1 / vInverse
  const delta = v * sum
  const a = Math.log(sigma * sigma)
  const f = (x: number) => {
    const ex = Math.exp(x)
    const d = phi * phi + v + ex
    return (ex * (delta * delta - phi * phi - v - ex)) / (2 * d * d) - (x - a) / (tau * tau)
  }
  let A = a
  let B: number
  if (delta * delta > phi * phi + v) B = Math.log(delta * delta - phi * phi - v)
  else {
    let k = 1
    while (f(a - k * tau) < 0) k++
    B = a - k * tau
  }
  let fA = f(A)
  let fB = f(B)
  for (let it = 0; it < 200 && Math.abs(B - A) > tolerance; it++) {
    const C = A + ((A - B) * fA) / (fB - fA)
    const fC = f(C)
    if (fC * fB <= 0) {
      A = B
      fA = fB
    } else fA /= 2
    B = C
    fB = fC
  }
  const sigmaNew = Math.exp(A / 2)
  const phiStar = Math.sqrt(phi * phi + elapsed * sigmaNew * sigmaNew)
  const phiNew = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v)
  const muNew = mu + phiNew * phiNew * sum
  return { rating: 1500 + GLICKO2_SCALE * muNew, deviation: GLICKO2_SCALE * phiNew, volatility: sigmaNew }
}

/** Options of `glickoRatings`. */
export interface GlickoRatingsOptions extends Glicko2Options {
  players: number
  /** `glicko` (Glicko-1) or `glicko2` (default `glicko`). */
  version?: 'glicko' | 'glicko2'
  /** Results per rating period (default 1: every result is its own period). */
  period?: number
  /** Initial rating, deviation and volatility (1500, 350, 0.06). */
  initial?: GlickoRating
  /** Glicko-1's c: RD grows to √(RD² + c²) at the start of each period, capped at the initial RD (default 15). */
  c?: number
}

/**
 * Glicko or Glicko-2 ratings over a sequence of results, cut into rating periods of `period` results. Within a period
 * every player is updated once from the period's games against the opponents' ratings at its start. The history has one
 * row per result: a row inside a period repeats the period's starting ratings, and the period's last result shows the
 * update.
 */
export function glickoRatings(results: readonly PairedResult[], options: GlickoRatingsOptions): RatingRun {
  const { players, version = 'glicko', period = 1, c = 15 } = options
  const initial = options.initial ?? { rating: 1500, deviation: 350, volatility: 0.06 }
  let current: GlickoRating[] = Array.from({ length: players }, () => ({
    rating: initial.rating,
    deviation: initial.deviation,
    volatility: initial.volatility ?? 0.06,
  }))
  const history = new Float64Array((results.length + 1) * players)
  const deviations = new Float64Array((results.length + 1) * players)
  const write = (row: number) => {
    for (let p = 0; p < players; p++) {
      history[row * players + p] = current[p].rating
      deviations[row * players + p] = current[p].deviation
    }
  }
  write(0)
  const predicted = new Float64Array(results.length)
  for (let start = 0; start < results.length; start += period) {
    const end = Math.min(results.length, start + period)
    // Glicko-1 inflates RD at the start of a period; Glicko-2 grows φ inside its update.
    if (version === 'glicko')
      current = current.map((r) => ({
        ...r,
        deviation: Math.min(initial.deviation, Math.sqrt(r.deviation * r.deviation + c * c)),
      }))
    const games: GlickoGame[][] = Array.from({ length: players }, () => [])
    for (let i = start; i < end; i++) {
      const { a, b, score } = results[i]
      // The prediction combines both deviations, as Glickman's expected-score formula for a match.
      const ra = current[a]
      const rb = current[b]
      predicted[i] = glickoExpected(ra.rating, {
        rating: rb.rating,
        deviation: Math.sqrt(ra.deviation * ra.deviation + rb.deviation * rb.deviation),
      })
      games[a].push({ opponent: rb, score })
      games[b].push({ opponent: ra, score: 1 - score })
      if (i < end - 1) write(i + 1)
    }
    current = current.map((r, p) =>
      version === 'glicko' ? glickoUpdate(r, games[p]) : glicko2Update(r, games[p], options),
    )
    write(end)
  }
  return {
    history,
    deviations,
    ratings: Float64Array.from(current, (r) => r.rating),
    predicted,
    logLoss: logLossOf(results, predicted),
    players,
  }
}

/**
 * How well ratings order the players at each step: Spearman's rank correlation between a rating history and the true
 * skills, row by row (both row-major [rows × players]).
 */
export function ratingAgreement(history: Float64Array, truth: Float64Array, players: number): Float64Array {
  const rows = Math.min(history.length, truth.length) / players
  const out = new Float64Array(rows)
  for (let r = 0; r < rows; r++) {
    const h = history.subarray(r * players, (r + 1) * players)
    const t = truth.subarray(r * players, (r + 1) * players)
    let spread = 0
    for (let p = 1; p < players; p++) spread += Math.abs(h[p] - h[0])
    out[r] = spread > 0 ? spearman(h, t) : 0
  }
  return out
}

/** The running mean log loss of predictions of decisive results (draws skipped): entry g covers results 0 … g. */
export function runningLogLoss(results: readonly PairedResult[], predicted: Float64Array): Float64Array {
  const out = new Float64Array(results.length)
  let sum = 0
  let n = 0
  for (let g = 0; g < results.length; g++) {
    const s = results[g].score
    if (s === 0 || s === 1) {
      const p = Math.min(1 - 1e-12, Math.max(1e-12, predicted[g]))
      sum -= s === 1 ? Math.log(p) : Math.log(1 - p)
      n++
    }
    out[g] = n > 0 ? sum / n : NaN
  }
  return out
}
