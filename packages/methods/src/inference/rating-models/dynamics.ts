/**
 * Rating dynamics: every online rating system behind one interface, run on one game stream, each reporting a mean and
 * an uncertainty per player after every round; TrueSkill Through Time as the smoother over the whole stream; and the
 * settings of real chess sites.
 *
 * Everything is on Elo's scale: a difference of 400 points means odds of $10 : 1$ under the logistic model. A round is
 * the unit of time (a day, a rating period): systems with dynamics widen a player's uncertainty by the rounds elapsed
 * since their last game, and every game is rated on its own, in order, as chess sites do.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import { eloExpected, glicko2Update, glickoExpected, glickoUpdate, type GlickoRating, type PairedResult } from './elo'
import { drawMargin, trueSkillUpdate, type Rating } from './examples'

/** Points per unit of log-odds: $400 / \ln 10 \approx 173.7$. */
export const ELO_SCALE = 400 / Math.LN10

/**
 * The performance noise $\beta$ (Elo points) at which Thurstone's $P(a \text{ beats } b) = \Phi((s_a - s_b) /
 * (\sqrt{2}\,\beta))$ has the logistic's slope at an even game: $\phi(0) / (\sqrt{2}\,\beta) = \ln 10 / 1600$, so
 * $\beta = 1600\,\phi(0) / (\sqrt{2} \ln 10) \approx 196$.
 */
export const THURSTONE_BETA = 1600 / Math.sqrt(2 * Math.PI) / (Math.SQRT2 * Math.LN10)

/** A game stream: the games of each round, in the order played. Scores are player `a`'s: 1, $\tfrac{1}{2}$ or 0. */
export interface GameStream {
  /** The number of players; games index them from 0. */
  readonly players: number
  /** The games of each round, in order; a round may be empty. */
  readonly rounds: readonly (readonly PairedResult[])[]
}

/** A belief about one player's skill on Elo's scale: `sd` is NaN for a system without one, 0 for a fixed player. */
export interface SkillEstimate {
  /** The rating (NaN for a FIDE newcomer not yet rated). */
  mean: number
  /** Its standard deviation, in rating points. */
  sd: number
}

/** An online rating system over a fixed set of players. */
export interface OnlineRater {
  /** The system's display name, such as `Glicko-2`. */
  readonly name: string
  /** Player a's expected score against b before the game. */
  predict(a: number, b: number): number
  /** Rate one game played in `round` (rounds never decrease). */
  game(result: PairedResult, round: number): void
  /** The current belief about a player's skill. */
  estimate(player: number): SkillEstimate
}

/** Starting ratings: one for everyone, or one per player. */
type Starts = number | ArrayLike<number>
/**
 * The starting rating of one player.
 *
 * @param s The starting ratings: one number for everyone, or one per player.
 * @param p The player's index.
 */
const startOf = (s: Starts, p: number) => (typeof s === 'number' ? s : s[p])

/** Elo with a fixed K. */
export interface EloSpec {
  /** Selects Elo. */
  kind: 'elo'
  /** The K-factor (default 32). */
  k?: number
  /** The starting rating, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** The lowest rating a player can fall to (default none). */
  floor?: number
}

/**
 * FIDE's Elo (FIDE Rating Regulations from 1 March 2024, https://handbook.fide.com/chapter/B022024). A newcomer is
 * unrated for `entryGames` games; the initial rating is $R_a + dp$, $R_a$ the mean rating of the opponents and two
 * hypothetical opponents rated 1800 with draws against them, $dp$ from the normal table (8.1.a) of the score fraction,
 * rounded and capped at 2200. K is 40 for the first 30 rated games, 20 below 2400, 10 once 2400 has been reached; a
 * rating difference beyond 400 counts as 400. Unrated opponents count at `placeholder` (an assumption of a closed
 * simulated pool, where nobody is rated at the start).
 */
export interface FideSpec {
  /** Selects FIDE's Elo. */
  kind: 'fide'
  /** Games a newcomer plays before receiving a rating (default 5). */
  entryGames?: number
  /** The rating at which an unrated player counts, as an opponent and in predictions (default 1500). */
  placeholder?: number
  /** Rated games played with K = 40 (default 30). */
  newcomerGames?: number
}

/** Glicko-1 with every game its own rating period and the variance $\text{RD}^2$ growing by $c^2$ per round. */
export interface GlickoSpec {
  /** Selects Glicko-1. */
  kind: 'glicko'
  /** The starting rating, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** RD of a new player, also the cap (default 350). */
  deviation?: number
  /** RD growth per round: $\text{RD} \gets \sqrt{\text{RD}^2 + c^2 t}$ after $t$ rounds (default 15). */
  c?: number
  /** The lowest RD (default 0; Glickman suggests about 30). */
  minDeviation?: number
  /** The lowest rating a player can fall to (default none). */
  floor?: number
}

/** Glicko-2 with every game rated on its own and $\phi$ grown by the elapsed (possibly fractional) rating periods. */
export interface Glicko2Spec {
  /** Selects Glicko-2. */
  kind: 'glicko2'
  /** The starting rating, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** RD of a new player, also the cap (default 350). */
  deviation?: number
  /** The starting volatility $\sigma$ (default 0.06). */
  volatility?: number
  /** The system constant $\tau$ (default 0.5). */
  tau?: number
  /** Rating periods per round (default 1). */
  periodsPerRound?: number
  /** The lowest RD (default 0). */
  minDeviation?: number
  /** The largest volatility (default none). */
  maxVolatility?: number
  /** The lowest rating a player can fall to (default none). */
  floor?: number
}

/** TrueSkill on Elo's scale, with dynamics $\tau$ per round and a draw margin from the draw probability. */
export interface TrueSkillSpec {
  /** Selects TrueSkill. */
  kind: 'trueskill'
  /** The starting mean, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** $\sigma_0$ (default 350). */
  deviation?: number
  /** Performance noise $\beta$ (default `THURSTONE_BETA`). */
  beta?: number
  /** Skill drift per round: $\sigma^2 \gets \sigma^2 + \tau^2 t$ after $t$ rounds (default 10). */
  tau?: number
  /** The chance of a draw between equals, setting the draw margin (default 0.1). */
  drawProbability?: number
}

/**
 * A Kalman filter on the logistic model: each player's skill is a random walk with process sd `q` per round, and each
 * game is an extended-Kalman update with both players' variances (the vector form of Szczecinski and Tihon's
 * "simplified Kalman filter", 2023, https://doi.org/10.1515/jqas-2021-0061): with $z = (\mu_a - \mu_b)/s$,
 * $F = \sigma(z)$, $\omega = (v_a + v_b)/s^2$ and $s$ = `ELO_SCALE`,
 * $\mu_a \gets \mu_a + (v_a/s)(y - F) / (1 + F(1 - F)\omega)$ and
 * $v_a \gets v_a(1 - (v_a/s^2) F(1 - F) / (1 + F(1 - F)\omega))$ (and the mirror image for $b$).
 */
export interface KalmanSpec {
  /** Selects the Kalman filter. */
  kind: 'kalman'
  /** The starting mean, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** The starting standard deviation (default 350). */
  deviation?: number
  /** Process noise sd per round, in points (default 10). */
  q?: number
}

/** The settings of any online rating system, told apart by `kind`. */
export type RaterSpec = EloSpec | FideSpec | GlickoSpec | Glicko2Spec | TrueSkillSpec | KalmanSpec

/** The display name of each kind of rater. */
const RATER_NAMES: Record<RaterSpec['kind'], string> = {
  elo: 'Elo',
  fide: 'FIDE Elo',
  glicko: 'Glicko',
  glicko2: 'Glicko-2',
  trueskill: 'TrueSkill',
  kalman: 'Kalman (logistic)',
}

/**
 * FIDE's $dp$: the rating difference of a score fraction $p$ by the normal table,
 * $\sqrt{2} \cdot 200 \cdot \Phi^{-1}(p)$, within $\pm 800$.
 *
 * @param p The score fraction, clamped to $[0, 1]$.
 * @returns The rating difference, in points.
 *
 * @example From a lost match to a whitewash
 * print('dp at 0, 0.25, 0.5, 0.75, 1:', [0, 0.25, 0.5, 0.75, 1].map(fideDp))
 */
export function fideDp(p: number): number {
  return Math.max(-800, Math.min(800, Math.SQRT2 * 200 * normalQuantile(Math.min(1, Math.max(0, p)))))
}

/**
 * An online rater for `players` players. `fixed[p]` (not NaN) holds player p's known skill: the rater never updates it
 * and reports it with sd 0 (a field of known-strength opponents). Systems with dynamics widen a player's uncertainty
 * by the rounds since their last game (since round $-1$ before their first) before rating it.
 *
 * @param spec The system and its settings.
 * @param players The number of players.
 * @param fixed Known skills, one per player, with NaN for a player to be rated (default: everyone is rated).
 * @returns The rater, whose state changes with every `game`.
 *
 * @example One game between two new players, under four systems
 * for (const kind of ['elo', 'glicko', 'trueskill', 'kalman']) {
 *   const rater = createRater({ kind }, 2)
 *   rater.game({ a: 0, b: 1, score: 1 }, 0)
 *   print(rater.name, 'winner:', rater.estimate(0), 'loser:', rater.estimate(1))
 * }
 */
export function createRater(spec: RaterSpec, players: number, fixed?: ArrayLike<number>): OnlineRater {
  const isFixed = (p: number) => fixed !== undefined && !Number.isNaN(fixed[p])
  const name = RATER_NAMES[spec.kind]
  const last = new Int32Array(players).fill(-1)
  /** Rounds elapsed since p's last game (counting from round −1), 0 for a second game in a round; records the round. */
  const elapsed = (p: number, round: number) => {
    const t = round - last[p]
    last[p] = round
    return t
  }
  switch (spec.kind) {
    case 'elo': {
      const { k = 32, initial = 1500, floor = -Infinity } = spec
      const r = Float64Array.from({ length: players }, (_, p) => (isFixed(p) ? fixed![p] : startOf(initial, p)))
      return {
        name,
        predict: (a, b) => eloExpected(r[a], r[b]),
        game: ({ a, b, score }) => {
          const surprise = score - eloExpected(r[a], r[b])
          if (!isFixed(a)) r[a] = Math.max(floor, r[a] + k * surprise)
          if (!isFixed(b)) r[b] = Math.max(floor, r[b] - k * surprise)
        },
        estimate: (p) => ({ mean: r[p], sd: isFixed(p) ? 0 : NaN }),
      }
    }
    case 'fide': {
      const { entryGames = 5, placeholder = 1500, newcomerGames = 30 } = spec
      const r = Float64Array.from({ length: players }, (_, p) => (isFixed(p) ? fixed![p] : NaN))
      const games = new Int32Array(players)
      const opponentSum = new Float64Array(players)
      const scoreSum = new Float64Array(players)
      const elite = new Uint8Array(players)
      const seen = (p: number) => (Number.isNaN(r[p]) ? placeholder : r[p])
      const expected = (ra: number, rb: number) => eloExpected(ra, ra + Math.max(-400, Math.min(400, rb - ra)))
      const kOf = (p: number) => (games[p] - entryGames < newcomerGames ? 40 : elite[p] ? 10 : 20)
      const play = (p: number, opponent: number, score: number) => {
        if (isFixed(p)) return
        if (Number.isNaN(r[p])) {
          opponentSum[p] += opponent
          scoreSum[p] += score
          games[p]++
          if (games[p] >= entryGames) {
            const ra = (opponentSum[p] + 2 * 1800) / (games[p] + 2)
            r[p] = Math.min(2200, Math.round(ra + fideDp((scoreSum[p] + 1) / (games[p] + 2))))
          }
          return
        }
        const k = kOf(p)
        r[p] += k * (score - expected(r[p], opponent))
        games[p]++
        if (r[p] >= 2400) elite[p] = 1
      }
      return {
        name,
        predict: (a, b) => expected(seen(a), seen(b)),
        game: ({ a, b, score }) => {
          const ra = seen(a)
          const rb = seen(b)
          play(a, rb, score)
          play(b, ra, 1 - score)
        },
        estimate: (p) => ({ mean: r[p], sd: isFixed(p) ? 0 : NaN }),
      }
    }
    case 'glicko': {
      const { initial = 1500, deviation = 350, c = 15, minDeviation = 0, floor = -Infinity } = spec
      const s: GlickoRating[] = Array.from({ length: players }, (_, p) =>
        isFixed(p) ? { rating: fixed![p], deviation: 0 } : { rating: startOf(initial, p), deviation },
      )
      const grow = (p: number, round: number) => {
        const t = elapsed(p, round)
        if (t > 0 && !isFixed(p))
          s[p] = { ...s[p], deviation: Math.min(deviation, Math.sqrt(s[p].deviation ** 2 + c * c * t)) }
      }
      return {
        name,
        predict: (a, b) =>
          glickoExpected(s[a].rating, { rating: s[b].rating, deviation: Math.hypot(s[a].deviation, s[b].deviation) }),
        game: ({ a, b, score }, round) => {
          grow(a, round)
          grow(b, round)
          const ra = s[a]
          const rb = s[b]
          const clamp = (g: GlickoRating) => ({
            rating: Math.max(floor, g.rating),
            deviation: Math.max(minDeviation, g.deviation),
          })
          if (!isFixed(a)) s[a] = clamp(glickoUpdate(ra, [{ opponent: rb, score }]))
          if (!isFixed(b)) s[b] = clamp(glickoUpdate(rb, [{ opponent: ra, score: 1 - score }]))
        },
        estimate: (p) => ({ mean: s[p].rating, sd: s[p].deviation }),
      }
    }
    case 'glicko2': {
      const {
        initial = 1500,
        deviation = 350,
        volatility = 0.06,
        tau = 0.5,
        periodsPerRound = 1,
        minDeviation = 0,
        maxVolatility = Infinity,
        floor = -Infinity,
      } = spec
      const s: GlickoRating[] = Array.from({ length: players }, (_, p) =>
        isFixed(p)
          ? { rating: fixed![p], deviation: 0, volatility }
          : { rating: startOf(initial, p), deviation, volatility },
      )
      return {
        name,
        predict: (a, b) =>
          glickoExpected(s[a].rating, { rating: s[b].rating, deviation: Math.hypot(s[a].deviation, s[b].deviation) }),
        game: ({ a, b, score }, round) => {
          const ta = elapsed(a, round) * periodsPerRound
          const tb = elapsed(b, round) * periodsPerRound
          const ra = s[a]
          const rb = s[b]
          const clamp = (g: GlickoRating) => ({
            rating: Math.max(floor, g.rating),
            deviation: Math.min(deviation, Math.max(minDeviation, g.deviation)),
            volatility: Math.min(maxVolatility, g.volatility ?? volatility),
          })
          if (!isFixed(a)) s[a] = clamp(glicko2Update(ra, [{ opponent: rb, score }], { tau, elapsed: ta }))
          if (!isFixed(b)) s[b] = clamp(glicko2Update(rb, [{ opponent: ra, score: 1 - score }], { tau, elapsed: tb }))
        },
        estimate: (p) => ({ mean: s[p].rating, sd: s[p].deviation }),
      }
    }
    case 'trueskill': {
      const { initial = 1500, deviation = 350, beta = THURSTONE_BETA, tau = 10, drawProbability = 0.1 } = spec
      const eps = drawProbability > 0 ? drawMargin(drawProbability, beta) : 0
      const s: Rating[] = Array.from({ length: players }, (_, p) =>
        isFixed(p) ? { mean: fixed![p], sd: 0 } : { mean: startOf(initial, p), sd: deviation },
      )
      return {
        name,
        predict: (a, b) => {
          const c = Math.sqrt(2 * beta * beta + s[a].sd ** 2 + s[b].sd ** 2)
          const t = s[a].mean - s[b].mean
          return 0.5 * (normalCdf((t - eps) / c) + normalCdf((t + eps) / c))
        },
        game: ({ a, b, score }, round) => {
          for (const p of [a, b]) {
            const t = elapsed(p, round)
            if (!isFixed(p)) s[p] = { mean: s[p].mean, sd: Math.sqrt(s[p].sd ** 2 + tau * tau * t) }
          }
          // A draw needs a margin; with none it carries no information here.
          if (score === 0.5 && eps === 0) return
          const u = trueSkillUpdate(s[a], s[b], score === 1 ? 'win' : score === 0 ? 'loss' : 'draw', {
            beta,
            tau: 0,
            drawMargin: eps,
          })
          if (!isFixed(a)) s[a] = u.player1
          if (!isFixed(b)) s[b] = u.player2
        },
        estimate: (p) => ({ mean: s[p].mean, sd: s[p].sd }),
      }
    }
    case 'kalman': {
      const { initial = 1500, deviation = 350, q = 10 } = spec
      const m = Float64Array.from({ length: players }, (_, p) => (isFixed(p) ? fixed![p] : startOf(initial, p)))
      const v = Float64Array.from({ length: players }, (_, p) => (isFixed(p) ? 0 : deviation * deviation))
      const S = ELO_SCALE
      return {
        name,
        // The logistic averaged over the skill uncertainty, by the probit approximation 1/√(1 + πω/8).
        predict: (a, b) =>
          1 / (1 + Math.exp(-(m[a] - m[b]) / S / Math.sqrt(1 + (Math.PI * (v[a] + v[b])) / (8 * S * S)))),
        game: ({ a, b, score }, round) => {
          for (const p of [a, b]) {
            const t = elapsed(p, round)
            if (!isFixed(p)) v[p] += q * q * t
          }
          const F = 1 / (1 + Math.exp(-(m[a] - m[b]) / S))
          const h = F * (1 - F)
          const omega = (v[a] + v[b]) / (S * S)
          const gain = (score - F) / (1 + h * omega)
          const shrink = h / (1 + h * omega) / (S * S)
          const [va, vb] = [v[a], v[b]]
          if (!isFixed(a)) {
            m[a] += (va / S) * gain
            v[a] = va * (1 - va * shrink)
          }
          if (!isFixed(b)) {
            m[b] -= (vb / S) * gain
            v[b] = vb * (1 - vb * shrink)
          }
        },
        estimate: (p) => ({ mean: m[p], sd: Math.sqrt(v[p]) }),
      }
    }
  }
}

/** Estimates of every player after every round of a stream. */
export interface RatingTrace {
  /** The system's display name. */
  readonly name: string
  /** The number of players, the length of a row. */
  readonly players: number
  /** The number of rounds. */
  readonly rounds: number
  /**
   * Means on Elo's scale, row-major $(\text{rounds} + 1) \times \text{players}$: row 0 is the start, row $r + 1$ is
   * after round $r$.
   */
  readonly mean: Float64Array
  /** Their standard deviations, same layout (NaN for a system without one). */
  readonly sd: Float64Array
  /** Games each player has played by each row, same layout. */
  readonly played: Int32Array
  /** Player a's expected score before each game, in stream order. */
  readonly predicted: Float64Array
}

/**
 * The arrays of a `RatingTrace` for a stream: `played` filled in, `mean`, `sd` and `predicted` zeroed for the caller.
 *
 * @param stream The game stream.
 * @returns The row-major means, standard deviations and games played, and one prediction slot per game.
 */
function traceArrays(stream: GameStream) {
  const { players } = stream
  const R = stream.rounds.length
  const played = new Int32Array((R + 1) * players)
  const count = new Int32Array(players)
  stream.rounds.forEach((games, r) => {
    for (const { a, b } of games) {
      count[a]++
      count[b]++
    }
    played.set(count, (r + 1) * players)
  })
  return {
    mean: new Float64Array((R + 1) * players),
    sd: new Float64Array((R + 1) * players),
    played,
    predicted: new Float64Array(stream.rounds.reduce((n, g) => n + g.length, 0)),
  }
}

/**
 * Run a rating system over a stream: each game is predicted and then rated, in order, and every player's estimate is
 * recorded after each round.
 *
 * @param stream The games, round by round.
 * @param spec The system and its settings.
 * @param options Known skills of players who are not rated, `fixed`, as in `createRater`.
 * @returns The estimates after every round, the games played and the prediction of each game.
 *
 * @example Glicko and TrueSkill on four rounds
 * const rounds = [
 *   [{ a: 0, b: 1, score: 1 }],
 *   [{ a: 1, b: 2, score: 1 }],
 *   [{ a: 0, b: 2, score: 1 }],
 *   [{ a: 2, b: 1, score: 0.5 }],
 * ]
 * for (const kind of ['glicko', 'trueskill']) {
 *   const trace = rateStream({ players: 3, rounds }, { kind })
 *   print(trace.name, 'means:', trace.mean.slice(-3), 'sds:', trace.sd.slice(-3))
 * }
 */
export function rateStream(
  stream: GameStream,
  spec: RaterSpec,
  options: { fixed?: ArrayLike<number> } = {},
): RatingTrace {
  const { players } = stream
  const rater = createRater(spec, players, options.fixed)
  const out = traceArrays(stream)
  const write = (row: number) => {
    for (let p = 0; p < players; p++) {
      const e = rater.estimate(p)
      out.mean[row * players + p] = e.mean
      out.sd[row * players + p] = e.sd
    }
  }
  write(0)
  let g = 0
  stream.rounds.forEach((games, r) => {
    for (const result of games) {
      out.predicted[g++] = rater.predict(result.a, result.b)
      rater.game(result, r)
    }
    write(r + 1)
  })
  return { name: rater.name, players, rounds: stream.rounds.length, ...out }
}

// ── TrueSkill Through Time ───────────────────────────────────────────────────────────────────────────────────────────

/** Options of `trueSkillThroughTime`. */
export interface TrueSkillThroughTimeOptions {
  /** The prior mean at the start, one for everyone or one per player (default 1500). */
  initial?: Starts
  /** The prior standard deviation $\sigma_0$ at the start (default 350). */
  deviation?: number
  /** Performance noise $\beta$ (default `THURSTONE_BETA`). */
  beta?: number
  /** Skill drift $\tau$ per round (default 10); must be positive. */
  tau?: number
  /** The chance of a draw between equals, setting the draw margin (default 0.1; 0 ignores draws). */
  drawProbability?: number
  /** Known skills (NaN for a rated player), as in `createRater`. */
  fixed?: ArrayLike<number>
  /** Sweeps at most (default 100). */
  maxSweeps?: number
  /** The largest change of any mean in a sweep that ends the sweeps (default 1e-3 points). */
  tolerance?: number
  /** Weight of a game's old message in its update, in $[0, 1)$ (default 0). */
  damping?: number
}

/** A smoothed trace: every estimate conditions on every game, before and after it. */
export interface SmoothedTrace extends RatingTrace {
  /** The sweeps run. */
  readonly sweeps: number
  /** Whether the last sweep changed no mean by more than `tolerance`. */
  readonly converged: boolean
  /** The largest change of a mean in the last sweep. */
  readonly change: number
}

/**
 * TrueSkill Through Time (Dangauthier, Herbrich, Minka and Graepel, 2007) with a time step per round: each rated
 * player's skill is a chain with $s_p^{\text{start}} \sim \Gauss(\mu_0, \sigma_0^2)$ (row 0) and
 * $s_p^r \sim \Gauss(s_p^{r-1}, \tau^2)$ for every round $r$ (row $r + 1$), and each game is the TrueSkill factor on
 * the two players' skills of its round. Expectation propagation alternates two phases until the means stop moving:
 * (1) the chains, which are Gaussian and exact: forward and backward messages along each chain and the marginal of
 * every node, given the games' current messages; (2) the games, in stream order: each game's cavity is the node's
 * marginal without the game's message, the two-player TrueSkill update ($\tau = 0$) of the cavities gives new
 * marginals, and the new message is their ratio with the cavity. The result is the posterior of every skill at every
 * round given all games. `predicted` holds each game's expected score under these smoothed beliefs (hindsight, not a
 * forecast). A draw is skipped when the draw margin is 0, and a game whose cavity is not a proper Gaussian is skipped
 * for that sweep. Throws `DomainError` when `tau` is not positive.
 *
 * @param stream The games, round by round.
 * @param options The prior, $\beta$, $\tau$, the draw probability, known skills and the stopping rule.
 * @returns The smoothed estimates after every round, with the sweeps run and whether they converged.
 *
 * @example Smoothing a short stream
 * // The same four rounds as the online raters see them; the smoothed means barely move with the round.
 * const rounds = [
 *   [{ a: 0, b: 1, score: 1 }],
 *   [{ a: 1, b: 2, score: 1 }],
 *   [{ a: 0, b: 2, score: 1 }],
 *   [{ a: 2, b: 1, score: 0.5 }],
 * ]
 * const smooth = trueSkillThroughTime({ players: 3, rounds })
 * print('sweeps:', smooth.sweeps, 'converged:', smooth.converged)
 * print('start:', smooth.mean.slice(0, 3))
 * print('after the last round:', smooth.mean.slice(-3), 'sds:', smooth.sd.slice(-3))
 * print('online TrueSkill:', rateStream({ players: 3, rounds }, { kind: 'trueskill' }).mean.slice(-3))
 */
export function trueSkillThroughTime(stream: GameStream, options: TrueSkillThroughTimeOptions = {}): SmoothedTrace {
  const {
    initial = 1500,
    deviation = 350,
    beta = THURSTONE_BETA,
    tau = 10,
    drawProbability = 0.1,
    fixed,
    maxSweeps = 100,
    tolerance = 1e-3,
    damping = 0,
  } = options
  if (!(tau > 0)) throw new DomainError('trueSkillThroughTime', 'trueSkillThroughTime: tau must be positive')
  const { players } = stream
  const R = stream.rounds.length
  const isFixed = (p: number) => fixed !== undefined && !Number.isNaN(fixed[p])
  const eps = drawProbability > 0 ? drawMargin(drawProbability, beta) : 0
  const games = stream.rounds.flatMap((gs, r) => gs.map((g) => ({ ...g, r })))
  const G = games.length
  // Game messages (precision, precision × mean) into each side's node.
  const sitePi = new Float64Array(2 * G)
  const siteNu = new Float64Array(2 * G)
  // Per node [players × R]: summed game messages, forward and backward messages (precision form), marginals.
  const N = players * R
  const sPi = new Float64Array(N)
  const sNu = new Float64Array(N)
  const margPi = new Float64Array(N)
  const margNu = new Float64Array(N)
  const startPi = new Float64Array(players)
  const startNu = new Float64Array(players)
  const t2 = tau * tau
  const chains = () => {
    sPi.fill(0)
    sNu.fill(0)
    games.forEach((g, k) => {
      sPi[g.a * R + g.r] += sitePi[2 * k]
      sNu[g.a * R + g.r] += siteNu[2 * k]
      sPi[g.b * R + g.r] += sitePi[2 * k + 1]
      sNu[g.b * R + g.r] += siteNu[2 * k + 1]
    })
    const fMean = new Float64Array(R)
    const fVar = new Float64Array(R)
    const bMean = new Float64Array(R + 1)
    const bVar = new Float64Array(R + 1)
    for (let p = 0; p < players; p++) {
      if (isFixed(p)) continue
      const m0 = startOf(initial, p)
      const v0 = deviation * deviation
      // Forward: the message into node r from the start and the earlier nodes.
      let m = m0
      let v = v0 + t2
      for (let r = 0; r < R; r++) {
        fMean[r] = m
        fVar[r] = v
        const P = 1 / v + sPi[p * R + r]
        m = (m / v + sNu[p * R + r]) / P
        v = 1 / P + t2
      }
      // Backward: bMean/bVar[r] is the message into node r from the later nodes (r = R: none); index −1 is row 0.
      bVar[R] = Infinity
      bMean[R] = 0
      for (let r = R - 1; r >= 0; r--) {
        const Pb = Number.isFinite(bVar[r + 1]) ? 1 / bVar[r + 1] : 0
        const P = Pb + sPi[p * R + r]
        if (P > 0) {
          bMean[r] = (bMean[r + 1] * Pb + sNu[p * R + r]) / P
          bVar[r] = 1 / P + t2
        } else {
          bMean[r] = 0
          bVar[r] = Infinity
        }
      }
      for (let r = 0; r < R; r++) {
        const Pb = r + 1 <= R - 1 && Number.isFinite(bVar[r + 1]) ? 1 / bVar[r + 1] : 0
        margPi[p * R + r] = 1 / fVar[r] + sPi[p * R + r] + Pb
        margNu[p * R + r] = fMean[r] / fVar[r] + sNu[p * R + r] + (Pb > 0 ? bMean[r + 1] * Pb : 0)
      }
      // Row 0: the start's prior times the message from node 0 and beyond.
      const P0 = R > 0 && Number.isFinite(bVar[0]) ? 1 / bVar[0] : 0
      startPi[p] = 1 / v0 + P0
      startNu[p] = m0 / v0 + (P0 > 0 ? bMean[0] * P0 : 0)
    }
  }
  const belief = (p: number, r: number): Rating =>
    isFixed(p)
      ? { mean: fixed![p], sd: 0 }
      : { mean: margNu[p * R + r] / margPi[p * R + r], sd: 1 / Math.sqrt(margPi[p * R + r]) }
  let sweeps = 0
  let change = Infinity
  chains()
  while (sweeps < maxSweeps && change > tolerance) {
    const before = Float64Array.from({ length: N }, (_, i) => margNu[i] / margPi[i])
    games.forEach((g, k) => {
      if (g.score === 0.5 && eps === 0) return
      const sides = [g.a, g.b]
      const cav = sides.map((p, j) => {
        if (isFixed(p)) return { pi: Infinity, nu: 0, rating: { mean: fixed![p], sd: 0 } }
        const i = p * R + g.r
        const pi = margPi[i] - sitePi[2 * k + j]
        const nu = margNu[i] - siteNu[2 * k + j]
        return { pi, nu, rating: { mean: nu / pi, sd: 1 / Math.sqrt(pi) } }
      })
      if (cav.some((c) => !(c.pi > 0))) return
      const u = trueSkillUpdate(cav[0].rating, cav[1].rating, g.score === 1 ? 'win' : g.score === 0 ? 'loss' : 'draw', {
        beta,
        tau: 0,
        drawMargin: eps,
      })
      ;[u.player1, u.player2].forEach((post, j) => {
        const p = sides[j]
        if (isFixed(p)) return
        const i = p * R + g.r
        const newPi = (1 - damping) * (1 / post.sd ** 2 - cav[j].pi) + damping * sitePi[2 * k + j]
        const newNu = (1 - damping) * (post.mean / post.sd ** 2 - cav[j].nu) + damping * siteNu[2 * k + j]
        if (!Number.isFinite(newPi) || !Number.isFinite(newNu)) return
        margPi[i] += newPi - sitePi[2 * k + j]
        margNu[i] += newNu - siteNu[2 * k + j]
        sitePi[2 * k + j] = newPi
        siteNu[2 * k + j] = newNu
      })
    })
    chains()
    change = 0
    for (let i = 0; i < N; i++)
      if (margPi[i] > 0) change = Math.max(change, Math.abs(margNu[i] / margPi[i] - before[i]))
    sweeps++
  }
  const out = traceArrays(stream)
  for (let p = 0; p < players; p++) {
    out.mean[p] = isFixed(p) ? fixed![p] : startNu[p] / startPi[p]
    out.sd[p] = isFixed(p) ? 0 : 1 / Math.sqrt(startPi[p])
    for (let r = 0; r < R; r++) {
      const e = belief(p, r)
      out.mean[(r + 1) * players + p] = e.mean
      out.sd[(r + 1) * players + p] = e.sd
    }
  }
  games.forEach((g, k) => {
    const a = belief(g.a, g.r)
    const b = belief(g.b, g.r)
    const c = Math.sqrt(2 * beta * beta + a.sd ** 2 + b.sd ** 2)
    const t = a.mean - b.mean
    out.predicted[k] = 0.5 * (normalCdf((t - eps) / c) + normalCdf((t + eps) / c))
  })
  return {
    name: 'TrueSkill Through Time',
    players,
    rounds: R,
    ...out,
    sweeps,
    converged: change <= tolerance,
    change,
  }
}

// ── Settling and scales ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Games a player needed to settle: the games played by the first row from which the rating stays within
 * $\pm$`tolerance` (default 50) of its settled value (the mean over the last `tail`, default $\tfrac{1}{4}$, of the
 * rows) for `hold` rows (default 10; infinity: to the end). NaN while never rated or never settled.
 *
 * @param trace The estimates of a run, from `rateStream` or `trueSkillThroughTime`.
 * @param player The player's index.
 * @param options The band around the settled value, the fraction of rows that defines it and the rows to hold.
 * @param options.tolerance The half-width of the band, in rating points (default 50).
 * @param options.tail The fraction of the last rows whose mean is the settled value (default 0.25).
 * @param options.hold The rows the rating must stay inside the band (default 10; infinity: to the end).
 * @returns The games the player had played by the first settled row, or NaN.
 *
 * @example A 1700 player against a known 1500 field
 * // 80 rounds of one game each, won with Elo's probability (drawn from stream(0)); player 1 is fixed at 1500.
 * const u = toArray(uniform(stream(0), 0, 1, { shape: [80] }))
 * const rounds = u.map((ui) => [{ a: 0, b: 1, score: ui < eloExpected(1700, 1500) ? 1 : 0 }])
 * const trace = rateStream({ players: 2, rounds }, { kind: 'elo', k: 20 }, { fixed: [NaN, 1500] })
 * print('settled rating:', settledRatings(trace)[0])
 * print('games to settle within 50:', settlingGames(trace, 0))
 * print('games to settle within 25:', settlingGames(trace, 0, { tolerance: 25 }))
 */
export function settlingGames(
  trace: RatingTrace,
  player: number,
  options: { tolerance?: number; tail?: number; hold?: number } = {},
): number {
  const { tolerance = 50, tail = 0.25, hold = 10 } = options
  const rows = trace.rounds + 1
  const P = trace.players
  const settled = settledRatings(trace, tail)[player]
  if (!Number.isFinite(settled)) return NaN
  const inside = (r: number) => Math.abs(trace.mean[r * P + player] - settled) <= tolerance
  for (let r = 0; r < rows; r++) {
    let ok = true
    for (let k = r; k < Math.min(rows, r + hold); k++)
      if (!inside(k)) {
        ok = false
        break
      }
    if (ok) return trace.played[r * P + player]
  }
  return NaN
}

/**
 * The settled value of every player: the mean of the last `tail` (default $\tfrac{1}{4}$) of the rows, skipping NaN.
 *
 * @param trace The estimates of a run, from `rateStream` or `trueSkillThroughTime`.
 * @param tail The fraction of the rows, counted from the end, to average.
 * @returns One value per player (NaN for a player with no finite rating in those rows).
 *
 * @example The settled ratings of a short Elo run
 * // Player 0 wins, wins, loses and wins against player 1.
 * const rounds = [1, 1, 0, 1].map((score) => [{ a: 0, b: 1, score }])
 * const trace = rateStream({ players: 2, rounds }, { kind: 'elo' })
 * print('ratings by row:', trace.mean)
 * print('last half:', settledRatings(trace, 0.5))
 */
export function settledRatings(trace: RatingTrace, tail = 0.25): Float64Array {
  const rows = trace.rounds + 1
  const P = trace.players
  const from = Math.max(0, Math.floor(rows * (1 - tail)))
  return Float64Array.from({ length: P }, (_, p) => {
    let sum = 0
    let n = 0
    for (let r = from; r < rows; r++) {
      const v = trace.mean[r * P + p]
      if (Number.isFinite(v)) {
        sum += v
        n++
      }
    }
    return n > 0 ? sum / n : NaN
  })
}

/** How one rating scale maps onto another for the same players: $y \approx \text{intercept} + \text{slope} \cdot x$. */
export interface ScaleMap {
  /** The slope of the least-squares line (NaN when every $x$ is equal). */
  slope: number
  /** The intercept of the line. */
  intercept: number
  /** The mean of $y - x$: the offset between the scales. */
  offset: number
  /** The sd of $y - x$ across players. */
  spread: number
  /** The sd of the residuals of the line. */
  residualSd: number
  /** The number of players with both ratings finite. */
  n: number
}

/**
 * The least-squares line from ratings `x` to ratings `y` of the same players (pairs with a NaN skipped).
 *
 * @param x The ratings on the first scale, one per player.
 * @param y The ratings on the second scale, indexed like `x`.
 * @returns The line, the mean offset and the spreads.
 *
 * @example A stretched and shifted scale
 * // The fourth player has no rating on the first scale.
 * print(ratingScaleMap([1500, 1600, 1700, NaN, 1400], [1300, 1450, 1600, 1800, 1170]))
 */
export function ratingScaleMap(x: ArrayLike<number>, y: ArrayLike<number>): ScaleMap {
  const idx = Array.from({ length: Math.min(x.length, y.length) }, (_, i) => i).filter(
    (i) => Number.isFinite(x[i]) && Number.isFinite(y[i]),
  )
  const n = idx.length
  const mx = idx.reduce((s, i) => s + x[i], 0) / n
  const my = idx.reduce((s, i) => s + y[i], 0) / n
  let sxy = 0
  let sxx = 0
  let sd = 0
  for (const i of idx) {
    sxy += (x[i] - mx) * (y[i] - my)
    sxx += (x[i] - mx) ** 2
    sd += (y[i] - x[i] - (my - mx)) ** 2
  }
  const slope = sxx > 0 ? sxy / sxx : NaN
  const intercept = my - slope * mx
  let rss = 0
  for (const i of idx) rss += (y[i] - intercept - slope * x[i]) ** 2
  return {
    slope,
    intercept,
    offset: my - mx,
    spread: Math.sqrt(sd / Math.max(1, n - 1)),
    residualSd: Math.sqrt(rss / Math.max(1, n - 2)),
    n,
  }
}

// ── Real chess sites ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Lichess's Glicko-2 (sources: https://lichess.org/faq; lila's Glicko.scala,
 * https://github.com/lichess-org/lila/blob/master/modules/rating/src/main/Glicko.scala): start 1500 with RD 500
 * ("1500 ± 1000"), volatility 0.09 capped at 0.1, RD within $[45, 500]$, ratings floored at 400, $\tau = 0.75$, every
 * game rated on its own with 0.21436 rating periods per day elapsed. Provisional ("?") while RD $> 110$.
 *
 * @param daysPerRound The days one round of the stream stands for, which scales `periodsPerRound`.
 * @returns The settings, for `createRater` or `rateStream`.
 *
 * @example A new Lichess player wins a game
 * const rater = createRater(lichessSpec(), 2)
 * rater.game({ a: 0, b: 1, score: 1 }, 0)
 * print('winner:', rater.estimate(0), 'loser:', rater.estimate(1))
 * print('still provisional:', rater.estimate(0).sd > LICHESS_PROVISIONAL_RD)
 */
export function lichessSpec(daysPerRound = 1): Glicko2Spec {
  return {
    kind: 'glicko2',
    initial: 1500,
    deviation: 500,
    volatility: 0.09,
    maxVolatility: 0.1,
    tau: 0.75,
    periodsPerRound: 0.21436 * daysPerRound,
    minDeviation: 45,
    floor: 400,
  }
}

/** Lichess shows a rating as provisional while its RD exceeds 110 (https://lichess.org/faq). */
export const LICHESS_PROVISIONAL_RD = 110

/**
 * Chess.com's Glicko-1 (https://support.chess.com/en/articles/8566476-how-do-ratings-work-on-chess-com: Glicko, RD
 * grows with inactivity). Starting ratings are chosen by the player's self-declared level (`chessComStarts`). Not public, so
 * assumed: a starting RD of 350 and the cap (Glickman's default for an unrated player), the RD floor 30 (Glickman's
 * suggested threshold), $c = 18$ points per day (RD 50 to 350 in about a year of inactivity), the rating floor 100,
 * every game its own rating period.
 *
 * @param initial The starting ratings: one for everyone, or one per player (as `chessComStarts` gives them).
 * @param daysPerRound The days one round of the stream stands for: $c$ per round is $18\sqrt{\text{days}}$.
 * @returns The settings, for `createRater` or `rateStream`.
 *
 * @example Weekly rounds
 * print(chessComSpec(1200, 7))
 */
export function chessComSpec(initial: Starts, daysPerRound = 1): GlickoSpec {
  return { kind: 'glicko', initial, deviation: 350, c: 18 * Math.sqrt(daysPerRound), minDeviation: 30, floor: 100 }
}

/** Chess.com's sign-up levels and their starting ratings (New to chess … Advanced; reported by members). */
export const CHESS_COM_LEVELS = [400, 800, 1200, 1600] as const

/**
 * Starting ratings on chess.com's self-declared levels (`CHESS_COM_LEVELS`): each player picks the level nearest
 * $1200 + (\text{skill} - \text{pool mean}) + \text{error} \cdot z$ (an assumption: players judge themselves against
 * an average club player, with a standard deviation of `error` points).
 *
 * @param relativeSkill Each player's skill minus the pool's mean, in rating points.
 * @param noise One standard-normal draw $z$ per player.
 * @param error The standard deviation of a player's misjudgement, in rating points.
 * @returns The starting rating of each player, one of the levels.
 *
 * @example Five players judge themselves
 * const skill = [-700, -300, 0, 300, 600]
 * print('levels:', chessComStarts(skill, toArray(normals(stream(1), [5]))))
 * print('judged exactly:', chessComStarts(skill, [0, 0, 0, 0, 0]))
 */
export function chessComStarts(relativeSkill: ArrayLike<number>, noise: ArrayLike<number>, error = 200): Float64Array {
  return Float64Array.from(relativeSkill, (s, p) => {
    const guess = 1200 + s + error * noise[p]
    let best: number = CHESS_COM_LEVELS[0]
    for (const level of CHESS_COM_LEVELS) if (Math.abs(level - guess) < Math.abs(best - guess)) best = level
    return best
  })
}

/**
 * FIDE's Elo (`FideSpec`) with unrated opponents counted at 1500 (an assumption of the closed simulated pool).
 *
 * @returns The settings, for `createRater` or `rateStream`.
 *
 * @example A newcomer's first rating
 * // Player 0 meets known players rated 1600 and 1800 in turn, and scores 3.5 from the first five games.
 * const rater = createRater(fideSpec(), 3, [NaN, 1600, 1800])
 * const scores = [1, 0.5, 1, 0, 1, 1]
 * scores.forEach((score, i) => rater.game({ a: 0, b: 1 + (i % 2), score }, i))
 * print(fideSpec())
 * print('rating after six games:', rater.estimate(0).mean)
 */
export function fideSpec(): FideSpec {
  return { kind: 'fide', entryGames: 5, placeholder: 1500, newcomerGames: 30 }
}
