/**
 * Rating dynamics: every online rating system behind one interface, run on one game stream, each reporting a mean and
 * an uncertainty per player after every round; TrueSkill Through Time as the smoother over the whole stream; and the
 * settings of real chess sites.
 *
 * Everything is on Elo's scale: a difference of 400 points means odds of 10 : 1 under the logistic model. A round is the
 * unit of time (a day, a rating period): systems with dynamics widen a player's uncertainty by the rounds elapsed since
 * their last game, and every game is rated on its own, in order, as chess sites do.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import { eloExpected, glicko2Update, glickoExpected, glickoUpdate, type GlickoRating, type PairedResult } from './elo'
import { drawMargin, trueSkillUpdate, type Rating } from './examples'

/** Points per unit of log-odds: 400/ln 10 ≈ 173.7. */
export const ELO_SCALE = 400 / Math.LN10

/**
 * The performance noise β (Elo points) at which Thurstone's P(a beats b) = Φ((s_a − s_b)/(√2 β)) has the logistic's
 * slope at an even game: φ(0)/(√2 β) = ln 10/1600, so β = 1600 φ(0)/(√2 ln 10) ≈ 196.
 */
export const THURSTONE_BETA = 1600 / Math.sqrt(2 * Math.PI) / (Math.SQRT2 * Math.LN10)

/** A game stream: the games of each round, in the order played. Scores are player a's: 1, ½ or 0. */
export interface GameStream {
  readonly players: number
  readonly rounds: readonly (readonly PairedResult[])[]
}

/** A belief about one player's skill on Elo's scale: sd is NaN for a system without one, 0 for a fixed player. */
export interface SkillEstimate {
  mean: number
  sd: number
}

/** An online rating system over a fixed set of players. */
export interface OnlineRater {
  readonly name: string
  /** Player a's expected score against b before the game. */
  predict(a: number, b: number): number
  /** Rate one game played in `round` (rounds never decrease). */
  game(result: PairedResult, round: number): void
  estimate(player: number): SkillEstimate
}

/** Starting ratings: one for everyone, or one per player. */
type Starts = number | ArrayLike<number>
const startOf = (s: Starts, p: number) => (typeof s === 'number' ? s : s[p])

/** Elo with a fixed K. */
export interface EloSpec {
  kind: 'elo'
  k?: number
  initial?: Starts
  /** The lowest rating a player can fall to (default none). */
  floor?: number
}

/**
 * FIDE's Elo (FIDE Rating Regulations from 1 March 2024, https://handbook.fide.com/chapter/B022024). A newcomer is
 * unrated for `entryGames` games; the initial rating is Ra + dp, Ra the mean rating of the opponents and two
 * hypothetical opponents rated 1800 with draws against them, dp from the normal table (8.1.a) of the score fraction,
 * capped at 2200. K is 40 for the first 30 rated games, 20 below 2400, 10 once 2400 has been reached; a rating
 * difference beyond 400 counts as 400. Unrated opponents count at `placeholder` (an assumption of a closed simulated
 * pool, where nobody is rated at the start).
 */
export interface FideSpec {
  kind: 'fide'
  entryGames?: number
  placeholder?: number
  newcomerGames?: number
}

/** Glicko-1 with every game its own rating period and RD growing by c² per round without games. */
export interface GlickoSpec {
  kind: 'glicko'
  initial?: Starts
  /** RD of a new player, also the cap (default 350). */
  deviation?: number
  /** RD growth per round: RD ← √(RD² + c² t) (default 15). */
  c?: number
  /** The lowest RD (default 0; Glickman suggests about 30). */
  minDeviation?: number
  floor?: number
}

/** Glicko-2 with every game rated on its own and φ grown by the elapsed (possibly fractional) rating periods. */
export interface Glicko2Spec {
  kind: 'glicko2'
  initial?: Starts
  deviation?: number
  volatility?: number
  tau?: number
  /** Rating periods per round (default 1). */
  periodsPerRound?: number
  minDeviation?: number
  maxVolatility?: number
  floor?: number
}

/** TrueSkill on Elo's scale, with dynamics τ per round and a draw margin from the draw probability. */
export interface TrueSkillSpec {
  kind: 'trueskill'
  initial?: Starts
  /** σ₀ (default 350). */
  deviation?: number
  /** Performance noise β (default `THURSTONE_BETA`). */
  beta?: number
  /** Skill drift per round: σ² ← σ² + τ² t (default 10). */
  tau?: number
  /** The chance of a draw between equals, setting the draw margin (default 0.1). */
  drawProbability?: number
}

/**
 * A Kalman filter on the logistic model: each player's skill is a random walk with process sd `q` per round, and each
 * game is an extended-Kalman update with both players' variances (the vector form of Szczecinski and Tihon's
 * "simplified Kalman filter", 2023, https://doi.org/10.1515/jqas-2021-0061): with z = (μ_a − μ_b)/s, F = σ(z),
 * ω = (v_a + v_b)/s², μ_a += (v_a/s)(y − F)/(1 + F(1 − F)ω) and v_a ← v_a(1 − (v_a/s²)F(1 − F)/(1 + F(1 − F)ω)).
 */
export interface KalmanSpec {
  kind: 'kalman'
  initial?: Starts
  deviation?: number
  /** Process noise sd per round, in points (default 10). */
  q?: number
}

export type RaterSpec = EloSpec | FideSpec | GlickoSpec | Glicko2Spec | TrueSkillSpec | KalmanSpec

const RATER_NAMES: Record<RaterSpec['kind'], string> = {
  elo: 'Elo',
  fide: 'FIDE Elo',
  glicko: 'Glicko',
  glicko2: 'Glicko-2',
  trueskill: 'TrueSkill',
  kalman: 'Kalman (logistic)',
}

/** FIDE's dp: the rating difference of a score fraction p by the normal table, √2·200·Φ⁻¹(p), within ±800. */
export function fideDp(p: number): number {
  return Math.max(-800, Math.min(800, Math.SQRT2 * 200 * normalQuantile(Math.min(1, Math.max(0, p)))))
}

/**
 * An online rater for `players` players. `fixed[p]` (not NaN) holds player p's known skill: the rater never updates it
 * and reports it with sd 0 (a field of known-strength opponents).
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
  readonly name: string
  readonly players: number
  readonly rounds: number
  /** Means on Elo's scale, row-major [(rounds + 1) × players]: row 0 is the start, row r + 1 is after round r. */
  readonly mean: Float64Array
  /** Their standard deviations, same layout (NaN for a system without one). */
  readonly sd: Float64Array
  /** Games each player has played by each row, same layout. */
  readonly played: Int32Array
  /** Player a's expected score before each game, in stream order. */
  readonly predicted: Float64Array
}

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

/** Run a rating system over a stream (`fixed` as in `createRater`). */
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
  initial?: Starts
  deviation?: number
  beta?: number
  /** Skill drift per round (default 10). */
  tau?: number
  drawProbability?: number
  /** Known skills (NaN for a rated player), as in `createRater`. */
  fixed?: ArrayLike<number>
  /** Sweeps at most (default 100), and the largest change of any mean that ends them (default 1e-3 points). */
  maxSweeps?: number
  tolerance?: number
  /** Weight of a game's old message in its update, in [0, 1) (default 0). */
  damping?: number
}

/** A smoothed trace: every estimate conditions on every game, before and after it. */
export interface SmoothedTrace extends RatingTrace {
  readonly sweeps: number
  readonly converged: boolean
  /** The largest change of a mean in the last sweep. */
  readonly change: number
}

/**
 * TrueSkill Through Time (Dangauthier, Herbrich, Minka and Graepel, 2007) with a time step per round: each rated
 * player's skill is a chain s_p⁰ ~ N(μ₀, σ₀²) (row 0), s_pʳ ~ N(s_pʳ⁻¹, τ²) for every round r, and each game is the
 * TrueSkill factor on the two players' skills of its round. Expectation propagation alternates two phases until the
 * means stop moving: (1) the chains, which are Gaussian and exact: forward and backward messages along each chain and
 * the marginal of every node, given the games' current messages; (2) the games, in stream order: each game's cavity is
 * the node's marginal without the game's message, the two-player TrueSkill update (τ = 0) of the cavities gives new
 * marginals, and the new message is their ratio with the cavity. The result is the posterior of every skill at every
 * round given all games. `predicted` holds each game's expected score under these smoothed beliefs (hindsight, not a
 * forecast).
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
 * Games a player needed to settle: the games played by the first row from which the rating stays within ±`tolerance`
 * (default 50) of its settled value (the mean over the last `tail`, default ¼, of the rows) for `hold` rows (default
 * 10; Infinity: to the end). NaN while never rated or never settled.
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

/** The settled value of every player: the mean of the last `tail` (default ¼) of the rows, skipping NaN. */
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

/** How one rating scale maps onto another for the same players: y ≈ intercept + slope·x. */
export interface ScaleMap {
  slope: number
  intercept: number
  /** The mean of y − x: the offset between the scales. */
  offset: number
  /** The sd of y − x across players, and of the residuals of the line. */
  spread: number
  residualSd: number
  n: number
}

/** The least-squares line from ratings `x` to ratings `y` of the same players (pairs with a NaN skipped). */
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
 * ("1500 ± 1000"), volatility 0.09 capped at 0.1, RD within [45, 500], ratings floored at 400, τ = 0.75, every game rated
 * on its own with 0.21436 rating periods per day elapsed (`periodsPerRound` assumes a round is a day). Provisional ("?")
 * while RD > 110.
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
 * Chess.com's Glicko-1 (https://support.chess.com/en/articles/8566476-how-do-ratings-work-on-chess-com: Glicko, RD grows
 * with inactivity). Starting ratings are chosen by the player's self-declared level (`chessComStarts`). Not public, so
 * assumed: RD₀ 350 and the cap (Glickman's default for an unrated player), the RD floor 30 (Glickman's suggested
 * threshold), c = 18 points per day (RD 50 → 350 in about a year of inactivity), the rating floor 100, every game its
 * own rating period.
 */
export function chessComSpec(initial: Starts, daysPerRound = 1): GlickoSpec {
  return { kind: 'glicko', initial, deviation: 350, c: 18 * Math.sqrt(daysPerRound), minDeviation: 30, floor: 100 }
}

/** Chess.com's sign-up levels and their starting ratings (New to chess … Advanced; reported by members). */
export const CHESS_COM_LEVELS = [400, 800, 1200, 1600] as const

/**
 * Starting ratings on chess.com's self-declared levels: each player picks the level nearest 1200 + (skill − pool
 * mean) + noise (an assumption: players judge themselves against an average club player, with `noise` points of
 * error). `relativeSkill` is each player's skill minus the pool's mean; `noise` holds standard-normal draws.
 */
export function chessComStarts(relativeSkill: ArrayLike<number>, noise: ArrayLike<number>, error = 200): Float64Array {
  return Float64Array.from(relativeSkill, (s, p) => {
    const guess = 1200 + s + error * noise[p]
    let best: number = CHESS_COM_LEVELS[0]
    for (const level of CHESS_COM_LEVELS) if (Math.abs(level - guess) < Math.abs(best - guess)) best = level
    return best
  })
}

/** FIDE's Elo (`FideSpec`) with unrated opponents counted at 1500 (an assumption of the closed simulated pool). */
export function fideSpec(): FideSpec {
  return { kind: 'fide', entryGames: 5, placeholder: 1500, newcomerGames: 30 }
}
