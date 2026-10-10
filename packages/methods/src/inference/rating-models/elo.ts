/**
 * Online skill ratings from paired results: Elo (Elo, 1978), Glicko (Glickman, 1999) and Glicko-2 (Glickman, 2001).
 *
 * A result `{ a, b, score }` gives player `a`'s score against player `b`: 1 for a win, $\tfrac{1}{2}$ for a draw, 0 for
 * a loss. Ratings are on Elo's scale, where a difference of 400 points means odds of $10 : 1$. Elo moves both ratings
 * by $K$ times the surprise $s - E$ (score less expected score). Glicko adds a rating deviation (RD) per player: the
 * update is larger when the player's RD is large and smaller when the opponent's RD is large, and RD grows between
 * rating periods. Glicko-2 adds a volatility $\sigma$, the rate at which a player's skill is believed to drift, updated
 * from how surprising the period was. The batch runners (`eloRatings`, `glickoRatings`) return a `RatingRun`: the
 * rating history as a row-major array with one row per result, and the prediction made before each result.
 */

import { spearman } from 'aifn-compute/probability/stats'

/** One paired result: player `a`'s score against player `b` (1 win, $\tfrac{1}{2}$ draw, 0 loss). */
export interface PairedResult {
  /** The first player's index, from 0. */
  a: number
  /** The second player's index, from 0. */
  b: number
  /** The first player's score: 1, $\tfrac{1}{2}$ or 0 (the second player scores $1 - \text{score}$). */
  score: number
}

const LN10 = Math.LN10
/** Glicko's $q = \ln 10 / 400$. */
const Q = LN10 / 400
/** Glicko-2's scale $400 / \ln 10 \approx 173.7178$: $\mu = (r - 1500) / 173.7178$. */
const GLICKO2_SCALE = 400 / LN10

/**
 * Elo's expected score of a player rated $r_a$ against one rated $r_b$: $E = 1 / (1 + 10^{(r_b - r_a)/\text{scale}})$,
 * the probability of a win with a draw counted as half.
 *
 * @param ra The player's rating $r_a$.
 * @param rb The opponent's rating $r_b$.
 * @param scale The rating difference at which the odds are $10 : 1$.
 * @returns The expected score of the first player, in $(0, 1)$.
 *
 * @example Equal ratings, and a 400-point gap
 * print('1500 vs 1500:', eloExpected(1500, 1500))
 * print('1900 vs 1500:', eloExpected(1900, 1500))
 * print('1500 vs 1900:', eloExpected(1500, 1900))
 */
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

/**
 * One Elo update after a game: $r_a \gets r_a + K(s - E)$ and $r_b \gets r_b - K(s - E)$, so the points one player
 * gains the other loses.
 *
 * @param ra The first player's rating before the game.
 * @param rb The second player's rating before the game.
 * @param score The first player's score $s$: 1 for a win, $\tfrac{1}{2}$ for a draw, 0 for a loss.
 * @param options The K-factor (default 32) and the scale (default 400).
 * @returns Both new ratings, and the first player's expected score $E$ before the game.
 *
 * @example One game between two players
 * // The lower-rated player wins, and gains more than half of K.
 * print('upset:', eloUpdate(1500, 1600, 1))
 * print('draw:', eloUpdate(1500, 1600, 0.5))
 */
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
  /**
   * Ratings after each result, row-major $(\text{results} + 1) \times \text{players}$: row 0 is the start and row
   * $i + 1$ follows result $i$.
   */
  readonly history: Float64Array
  /** Rating deviations in the same layout (Glicko); null for Elo. */
  readonly deviations: Float64Array | null
  /** The final ratings. */
  readonly ratings: Float64Array
  /** Predicted probability that a scores 1, before each result. */
  readonly predicted: Float64Array
  /** Mean log loss of the predictions on decisive results (draws are skipped; NaN when there are none). */
  readonly logLoss: number
  /** The number of players, the row length of `history` and `deviations`. */
  readonly players: number
}

/**
 * The mean log loss of predictions of decisive results: $-\log p$ for a win of the first player and $-\log(1 - p)$
 * for a loss, with $p$ clipped to $[10^{-12}, 1 - 10^{-12}]$. Draws are skipped.
 *
 * @param results The results, in order.
 * @param predicted The predicted probability that the first player scores 1, one per result.
 * @returns The mean over decisive results, or NaN when there are none.
 */
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

/**
 * Elo ratings over a sequence of results, applied in order with `eloUpdate`, each player starting at `initial`. The
 * prediction of each result is the first player's expected score before it.
 *
 * @param results The results, in order; player indices run from 0 to `players - 1`.
 * @param options The number of `players`, the starting rating `initial` (default 1500), and the K-factor and scale of
 *   `EloOptions`.
 * @returns The rating history, the final ratings, the predictions and their log loss (`deviations` is null).
 *
 * @example Thirty games between three players
 * // True strengths 1300, 1500 and 1700; each game is won with Elo's probability, drawn from stream(0).
 * const skill = [1300, 1500, 1700]
 * const u = toArray(uniform(stream(0), 0, 1, { shape: [30] }))
 * const results = u.map((ui, i) => {
 *   const a = i % 3
 *   const b = (i + 1) % 3
 *   return { a, b, score: ui < eloExpected(skill[a], skill[b]) ? 1 : 0 }
 * })
 * const run = eloRatings(results, { players: 3, k: 32 })
 * print('final ratings:', run.ratings)
 * print('log loss:', run.logLoss)
 */
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

/** A Glicko rating: $r$ and its deviation RD (and the Glicko-2 volatility $\sigma$). */
export interface GlickoRating {
  /** The rating $r$, on Elo's scale. */
  rating: number
  /** The rating deviation RD: the standard deviation of the belief about the skill, in rating points. */
  deviation: number
  /** The Glicko-2 volatility $\sigma$ (default 0.06 where it is read); Glicko-1 ignores it. */
  volatility?: number
}

/** One game of a rating period, from the rated player's side. */
export interface GlickoGame {
  /** The opponent's rating and deviation at the start of the period. */
  opponent: GlickoRating
  /** The rated player's score: 1, $\tfrac{1}{2}$ or 0. */
  score: number
}

/**
 * Glicko's $g(\text{RD}) = 1 / \sqrt{1 + 3q^2\text{RD}^2/\pi^2}$, with $q = \ln 10 / 400$: how much an opponent's
 * uncertainty damps the evidence of a game (1 for a certain opponent, smaller as RD grows).
 *
 * @param deviation The opponent's rating deviation RD, in rating points.
 * @returns The damping factor, in $(0, 1]$.
 *
 * @example An uncertain opponent counts for less
 * print('g(0), g(50), g(350):', [0, 50, 350].map(glickoG))
 */
export function glickoG(deviation: number): number {
  return 1 / Math.sqrt(1 + (3 * Q * Q * deviation * deviation) / (Math.PI * Math.PI))
}

/**
 * Glicko's expected score of a player rated $r$ against an opponent rated $r_j$ with deviation $\text{RD}_j$:
 * $E = 1 / (1 + 10^{-g(\text{RD}_j)(r - r_j)/400})$, Elo's expectation pulled towards $\tfrac{1}{2}$ by the
 * opponent's uncertainty.
 *
 * @param r The player's rating.
 * @param opponent The opponent's rating and deviation (the volatility is not read).
 * @returns The expected score, in $(0, 1)$.
 *
 * @example A 100-point favourite, against a certain and an uncertain opponent
 * print('RD 30:', glickoExpected(1500, { rating: 1400, deviation: 30 }))
 * print('RD 350:', glickoExpected(1500, { rating: 1400, deviation: 350 }))
 */
export function glickoExpected(r: number, opponent: GlickoRating): number {
  return 1 / (1 + Math.pow(10, (-glickoG(opponent.deviation) * (r - opponent.rating)) / 400))
}

/**
 * One Glicko-1 rating period for one player (Glickman, 1999): with $d^2 = 1 / (q^2 \sum_j g_j^2 E_j(1 - E_j))$,
 * $r' = r + \frac{q}{1/\text{RD}^2 + 1/d^2} \sum_j g_j (s_j - E_j)$ and
 * $\text{RD}' = (1/\text{RD}^2 + 1/d^2)^{-1/2}$, where $g_j = g(\text{RD}_j)$ and $E_j$ is `glickoExpected` against
 * opponent $j$. RD is not inflated here; `glickoRatings` does that at the start of each period. A period with no games
 * leaves the rating unchanged.
 *
 * @param player The player's rating and deviation at the start of the period (the volatility is dropped).
 * @param games The period's games, each against an opponent's rating at the start of the period.
 * @returns The new rating and deviation.
 *
 * @example Glickman's worked example
 * // A 1500 player (RD 200) beats a 1400 (RD 30), and loses to a 1550 (RD 100) and a 1700 (RD 300).
 * const games = [
 *   { opponent: { rating: 1400, deviation: 30 }, score: 1 },
 *   { opponent: { rating: 1550, deviation: 100 }, score: 0 },
 *   { opponent: { rating: 1700, deviation: 300 }, score: 0 },
 * ]
 * print('after the period:', glickoUpdate({ rating: 1500, deviation: 200 }, games))
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
  /** $\tau$, which bounds how fast the volatility changes (Glickman suggests 0.3 to 1.2; default 0.5). */
  tau?: number
  /** Convergence tolerance of the volatility iteration (default 1e-6). */
  tolerance?: number
  /**
   * Rating periods since the player's last update (default 1). $\phi$ grows to
   * $\sqrt{\phi^2 + \text{elapsed} \cdot \sigma^2}$ before the games, as in Lichess's fractional periods; 0 for a
   * second game inside one period.
   */
  elapsed?: number
}

/**
 * One Glicko-2 rating period for one player (Glickman, 2001, steps 2 to 8): on the scale
 * $\mu = (r - 1500)/173.7178$, $\phi = \text{RD}/173.7178$, the variance $v$ and improvement $\Delta$ of the period,
 * a new volatility $\sigma'$ by the Illinois iteration on
 * $f(x) = \frac{e^x(\Delta^2 - \phi^2 - v - e^x)}{2(\phi^2 + v + e^x)^2} - \frac{x - \ln\sigma^2}{\tau^2}$, then
 * $\phi' = 1 / \sqrt{1/(\phi^2 + \text{elapsed} \cdot \sigma'^2) + 1/v}$ and
 * $\mu' = \mu + \phi'^2 \sum_j g(\phi_j)(s_j - E_j)$. With no games $\phi$ grows to
 * $\sqrt{\phi^2 + \text{elapsed} \cdot \sigma^2}$ and $\mu$ and $\sigma$ stay. The iteration stops after 200
 * steps if it has not met the tolerance.
 *
 * @param player The player's rating, deviation and volatility (default 0.06) at the start of the period.
 * @param games The period's games, each against an opponent's rating and deviation at the start of the period.
 * @param options The system constant $\tau$, the tolerance of the volatility iteration and the periods elapsed.
 * @returns The new rating, deviation and volatility.
 *
 * @example Glickman's worked example
 * // A 1500 player (RD 200, volatility 0.06) beats a 1400 (RD 30), and loses to a 1550 (RD 100) and a 1700 (RD 300).
 * const games = [
 *   { opponent: { rating: 1400, deviation: 30 }, score: 1 },
 *   { opponent: { rating: 1550, deviation: 100 }, score: 0 },
 *   { opponent: { rating: 1700, deviation: 300 }, score: 0 },
 * ]
 * print('after the period:', glicko2Update({ rating: 1500, deviation: 200, volatility: 0.06 }, games, { tau: 0.5 }))
 * print('a period without games:', glicko2Update({ rating: 1500, deviation: 200, volatility: 0.06 }, []))
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
  /** The number of players; results index them from 0. */
  players: number
  /** `glicko` (Glicko-1) or `glicko2` (default `glicko`). */
  version?: 'glicko' | 'glicko2'
  /** Results per rating period (default 1: every result is its own period). */
  period?: number
  /** Initial rating, deviation and volatility (1500, 350, 0.06). */
  initial?: GlickoRating
  /**
   * Glicko-1's $c$: RD grows to $\sqrt{\text{RD}^2 + c^2}$ at the start of each period, capped at the initial RD
   * (default 15).
   */
  c?: number
}

/**
 * Glicko or Glicko-2 ratings over a sequence of results, cut into rating periods of `period` results. Within a period
 * every player is updated once from the period's games against the opponents' ratings at its start. The history has one
 * row per result after the starting row: a row inside a period repeats the period's starting ratings (with Glicko-1's
 * inflated deviations), and the period's last result shows the update. The prediction of each result is
 * `glickoExpected` with the two players' deviations combined, $\sqrt{\text{RD}_a^2 + \text{RD}_b^2}$.
 *
 * @param results The results, in order; player indices run from 0 to `players - 1`.
 * @param options The number of players, the version, the period length, the initial rating and the Glicko-2 options
 *   (passed to every `glicko2Update`).
 * @returns The rating and deviation histories, the final ratings, the predictions and their log loss.
 *
 * @example Six games in rating periods of three
 * const results = [
 *   { a: 0, b: 1, score: 1 },
 *   { a: 1, b: 2, score: 1 },
 *   { a: 0, b: 2, score: 0.5 },
 *   { a: 0, b: 1, score: 1 },
 *   { a: 2, b: 1, score: 0 },
 *   { a: 0, b: 2, score: 1 },
 * ]
 * for (const version of ['glicko', 'glicko2']) {
 *   const run = glickoRatings(results, { players: 3, version, period: 3 })
 *   print(version, 'ratings:', run.ratings)
 *   print(version, 'deviations:', run.deviations.slice(-3))
 * }
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
 * skills, row by row (both row-major $\text{rows} \times \text{players}$). A row whose ratings are all equal (the
 * start) scores 0.
 *
 * @param history The ratings, row-major, one row per step (a `RatingRun`'s `history`).
 * @param truth The true skills in the same layout; only the rows both arrays have are compared.
 * @param players The number of players, the length of a row.
 * @returns One correlation per row, in $[-1, 1]$.
 *
 * @example Ratings that learn the order
 * const truth = [1, 2, 3, 1, 2, 3, 1, 2, 3]
 * const history = [1500, 1500, 1500, 1510, 1490, 1500, 1480, 1500, 1520]
 * print('agreement per row:', ratingAgreement(new Float64Array(history), new Float64Array(truth), 3))
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

/**
 * The running mean log loss of predictions of decisive results (draws skipped): entry $g$ covers results 0 to $g$, and
 * is NaN until the first decisive result. Probabilities are clipped to $[10^{-12}, 1 - 10^{-12}]$.
 *
 * @param results The results, in order.
 * @param predicted The predicted probability that the first player scores 1, one per result (a `RatingRun`'s
 *   `predicted`).
 * @returns The running mean, one entry per result.
 *
 * @example A draw leaves the running mean as it was
 * const results = [{ a: 0, b: 1, score: 1 }, { a: 0, b: 1, score: 0.5 }, { a: 0, b: 1, score: 0 }]
 * print('running log loss:', runningLogLoss(results, new Float64Array([0.5, 0.6, 0.7])))
 */
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
