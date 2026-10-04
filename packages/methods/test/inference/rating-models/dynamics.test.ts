import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { normalCdf } from 'aifn-compute/numerics/special'
import {
  THURSTONE_BETA,
  chessComSpec,
  createRater,
  eloRatings,
  fideDp,
  glickoRatings,
  lichessSpec,
  rateStream,
  ratingScaleMap,
  settlingGames,
  trueSkillEp,
  trueSkillThroughTime,
  type GameStream,
  type PairedResult,
} from 'aifn-methods/inference/rating-models'
import { focalPlayerStream, ratingPopulation, tournament, winProbability } from 'aifn-methods/data/synthetic'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

type Fixture = {
  constants: { mu: number; sigma: number; beta: number; tau: number; drawProbability: number }
  sigmaAtStart: number
  online: { games: [number, number, number][]; after: [number, number, number, number][] }
  smoothed: { games: [number, number, number, number][]; posteriors: [number, number, number, number][] }
}
const F = fixture<{ dynamics: Fixture }>('inference/rating-models').dynamics

/** One game per round. */
const oneARound = (results: readonly PairedResult[], players: number): GameStream => ({
  players,
  rounds: results.map((r) => [r]),
})

describe('online raters on one interface', () => {
  const t = tournament(stream(5), { players: 2, games: 60, spread: 0.5, draws: 0.3 })
  const s = oneARound(t.results, 2)
  it('Elo equals eloRatings', () => {
    const a = rateStream(s, { kind: 'elo', k: 24 })
    const b = eloRatings(t.results, { players: 2, k: 24 })
    expect(Array.from(a.mean)).toEqual(Array.from(b.history))
    expect(Array.from(a.predicted)).toEqual(Array.from(b.predicted))
  })
  it('Glicko and Glicko-2 equal glickoRatings with one game a period', () => {
    const g1 = rateStream(s, { kind: 'glicko', c: 0 })
    const r1 = glickoRatings(t.results, { players: 2, period: 1, c: 0 })
    g1.mean.forEach((v, i) => expect(v).toBeCloseTo(r1.history[i], 9))
    g1.sd.forEach((v, i) => expect(v).toBeCloseTo(r1.deviations![i], 9))
    const g2 = rateStream(s, { kind: 'glicko2', tau: 0.5 })
    const r2 = glickoRatings(t.results, { players: 2, period: 1, version: 'glicko2', tau: 0.5 })
    g2.mean.forEach((v, i) => expect(v).toBeCloseTo(r2.history[i], 6))
  })
  it('TrueSkill matches the trueskill package over a sequence with draws', () => {
    const { mu, sigma, beta, tau, drawProbability } = F.constants
    const games = F.online.games.map(([a, b, score]) => ({ a, b, score }))
    const tr = rateStream(oneARound(games, 2), {
      kind: 'trueskill',
      initial: mu,
      deviation: sigma,
      beta,
      tau,
      drawProbability,
    })
    F.online.after.forEach(([m0, s0, m1, s1], g) => {
      const row = (g + 1) * 2
      // The package's own erfc approximation limits the agreement to about 1e-6.
      expect(tr.mean[row]).toBeCloseTo(m0, 5)
      expect(tr.sd[row]).toBeCloseTo(s0, 5)
      expect(tr.mean[row + 1]).toBeCloseTo(m1, 5)
      expect(tr.sd[row + 1]).toBeCloseTo(s1, 5)
    })
  })
  it('the logistic Kalman filter is Glicko-1 against opponents of known skill', () => {
    const focal = focalPlayerStream(stream(2), { games: 80 })
    // Glicko's RD cap is its starting RD; a cap far away leaves both filters uncapped.
    const k = rateStream(focal, { kind: 'kalman', q: 20, deviation: 1e4 }, { fixed: focal.fixed })
    const g = rateStream(focal, { kind: 'glicko', c: 20, deviation: 1e4 }, { fixed: focal.fixed })
    for (let r = 1; r <= 80; r++) {
      expect(k.mean[r * focal.players]).toBeCloseTo(g.mean[r * focal.players], 4)
      expect(k.sd[r * focal.players]).toBeCloseTo(g.sd[r * focal.players], 4)
    }
  })
  it('fixed players never move and report sd 0', () => {
    const focal = focalPlayerStream(stream(3), { games: 20 })
    for (const kind of ['elo', 'glicko', 'glicko2', 'trueskill', 'kalman'] as const) {
      const tr = rateStream(focal, { kind }, { fixed: focal.fixed })
      expect(tr.mean[20 * focal.players + 5]).toBe(focal.fixed[5])
      expect(tr.sd[20 * focal.players + 5]).toBe(0)
    }
  })
})

describe('chess sites', () => {
  it('Lichess: RD from 500, never below 45; chess.com: RD never below 30', () => {
    const pop = ratingPopulation(stream(1), { players: 40, rounds: 80, gamesPerRound: 3 })
    const li = rateStream(pop, lichessSpec())
    expect(li.sd[0]).toBe(500)
    expect(Math.min(...li.sd)).toBeGreaterThanOrEqual(45 - 1e-9)
    expect(Math.min(...li.mean)).toBeGreaterThanOrEqual(400)
    const cc = rateStream(pop, chessComSpec(1200))
    expect(Math.min(...cc.sd)).toBeGreaterThanOrEqual(30 - 1e-9)
    // A closed pool keeps its level near its starting ratings: the scales differ by an offset.
    const m = ratingScaleMap(cc.mean.subarray(80 * 40), li.mean.subarray(80 * 40))
    expect(m.offset).toBeGreaterThan(200)
    expect(m.offset).toBeLessThan(400)
    expect(m.slope).toBeGreaterThan(0.85)
    expect(m.slope).toBeLessThan(1.25)
  })
  it("FIDE's dp follows the normal table 8.1.a", () => {
    expect(fideDp(0.5)).toBeCloseTo(0, 10)
    expect(Math.abs(fideDp(0.6) - 72)).toBeLessThan(2)
    expect(Math.abs(fideDp(0.7) - 149)).toBeLessThan(2)
    expect(Math.abs(fideDp(0.8) - 240)).toBeLessThan(3)
    expect(Math.abs(fideDp(0.9) - 366)).toBeLessThan(5)
    expect(fideDp(1)).toBe(800)
  })
  it('FIDE: unrated for 5 games, then Ra + dp with two draws against 1800', () => {
    const r = createRater({ kind: 'fide' }, 2)
    for (let g = 0; g < 4; g++) r.game({ a: 0, b: 1, score: 1 }, g)
    expect(r.estimate(0).mean).toBeNaN()
    r.game({ a: 0, b: 1, score: 1 }, 4)
    // Opponent at the 1500 placeholder five times, plus 1800 twice; 5 + 1 points of 7.
    const ra = (5 * 1500 + 2 * 1800) / 7
    expect(r.estimate(0).mean).toBe(Math.round(ra + fideDp(6 / 7)))
    expect(r.estimate(1).mean).toBe(Math.round(ra + fideDp(1 / 7)))
  })
  it('settling counts the games until the rating stays within the tolerance', () => {
    const tr = {
      name: 'x',
      players: 1,
      rounds: 7,
      mean: Float64Array.from([1500, 1300, 1150, 1080, 1010, 990, 1000, 1000]),
      sd: new Float64Array(8),
      played: Int32Array.from([0, 2, 4, 6, 8, 10, 12, 14]),
      predicted: new Float64Array(0),
    }
    // Settled: the mean of the last two rows, 1000; within ±50 from row 4 (1010) on, after 8 games.
    expect(settlingGames(tr, 0, { tolerance: 50, tail: 0.25, hold: Infinity })).toBe(8)
    // Within ±100 from row 3 (1080) on.
    expect(settlingGames(tr, 0, { tolerance: 100, tail: 0.25, hold: 1 })).toBe(6)
  })
})

describe('TrueSkill Through Time', () => {
  it('matches the trueskillthroughtime package', () => {
    const { mu, beta, tau, drawProbability } = F.constants
    const R = Math.max(...F.smoothed.games.map((g) => g[0])) + 1
    const rounds: PairedResult[][] = Array.from({ length: R }, () => [])
    for (const [r, a, b, score] of F.smoothed.games) rounds[r].push({ a, b, score })
    // The package starts a player's prior at its first game; ours at the row before, one τ² earlier.
    const s = trueSkillThroughTime(
      { players: 3, rounds },
      { initial: mu, deviation: F.sigmaAtStart, beta, tau, drawProbability, tolerance: 1e-12, maxSweeps: 2000 },
    )
    expect(s.converged).toBe(true)
    for (const [p, t, m, sd] of F.smoothed.posteriors) {
      expect(s.mean[(t + 1) * 3 + p]).toBeCloseTo(m, 5)
      expect(s.sd[(t + 1) * 3 + p]).toBeCloseTo(sd, 5)
    }
  })
  it('in one round it is EP over the match set', () => {
    const matches = [
      { winner: 0, loser: 1 },
      { winner: 1, loser: 2 },
      { winner: 0, loser: 2 },
      { winner: 2, loser: 1, draw: true },
    ]
    const tau = 1
    const s = trueSkillThroughTime(
      {
        players: 3,
        rounds: [matches.map((m) => ({ a: m.winner, b: m.loser, score: m.draw ? 0.5 : 1 }))],
      },
      { initial: 25, deviation: 25 / 3, beta: 25 / 6, tau, tolerance: 1e-12, maxSweeps: 2000 },
    )
    const prior = { mean: 25, sd: Math.sqrt((25 / 3) ** 2 + tau * tau) }
    const ep = run(trueSkillEp({ players: [prior, prior, prior], matches, beta: 25 / 6 }), undefined, 5000)
    for (let p = 0; p < 3; p++) {
      expect(s.mean[3 + p]).toBeCloseTo(ep.means.data[p], 6)
      expect(s.sd[3 + p]).toBeCloseTo(ep.sds.data[p], 6)
    }
  })
  it('sees a step on both sides: smaller error than the filter around the change', () => {
    const f = focalPlayerStream(stream(9), { games: 200, path: { kind: 'step', at: 100, size: 300 } })
    const online = rateStream(f, { kind: 'trueskill', tau: 20 }, { fixed: f.fixed })
    const smooth = trueSkillThroughTime(f, { tau: 20, fixed: f.fixed })
    const err = (tr: { mean: Float64Array }) => {
      let s = 0
      for (let g = 90; g < 130; g++) s += (tr.mean[(g + 1) * f.players] - f.truth[g]) ** 2
      return Math.sqrt(s / 40)
    }
    expect(err(smooth)).toBeLessThan(err(online))
  })
})

describe('simulated streams', () => {
  it('outcome probabilities are the Bradley–Terry and Thurstone models', () => {
    expect(winProbability(0)).toBe(0.5)
    expect(winProbability(400)).toBeCloseTo(10 / 11, 12)
    expect(winProbability(150, { outcome: 'thurstone', beta: 200 })).toBeCloseTo(
      normalCdf(150 / (Math.SQRT2 * 200)),
      12,
    )
    // Matched slopes at an even game.
    const h = 1e-3
    const slope = (o: 'bradley-terry' | 'thurstone') =>
      (winProbability(h, { outcome: o }) - winProbability(-h, { outcome: o })) / (2 * h)
    expect(slope('thurstone')).toBeCloseTo(slope('bradley-terry'), 8)
    expect(THURSTONE_BETA).toBeCloseTo(196, 0)
  })
  it('the scores follow the stated probabilities, draws keeping the expected score', () => {
    const f = focalPlayerStream(stream(4), { games: 20000, path: { kind: 'constant' }, opponents: 'field', draws: 0.4 })
    let score = 0
    let p = 0
    let draws = 0
    f.rounds.forEach(([g], i) => {
      score += g.score
      p += f.probability[i]
      if (g.score === 0.5) draws++
    })
    expect(Math.abs(score - p) / 20000).toBeLessThan(0.01)
    expect(draws / 20000).toBeGreaterThan(0.25)
  })
  it('skill and rating matchmaking pair players of similar skill', () => {
    const close = ratingPopulation(stream(6), { players: 100, rounds: 5, matchmaking: 'skill', window: 50 })
    const rated = ratingPopulation(stream(6), { players: 100, rounds: 60, matchmaking: 'rating', window: 50 })
    const random = ratingPopulation(stream(6), { players: 100, rounds: 5, matchmaking: 'random' })
    const gap = (pop: typeof close, from = 0) => {
      let s = 0
      let n = 0
      pop.rounds.forEach((games, r) =>
        (r < from ? [] : games).forEach(({ a, b }) => {
          s += Math.abs(pop.skills[r * 100 + a] - pop.skills[r * 100 + b])
          n++
        }),
      )
      return s / n
    }
    expect(gap(close)).toBeLessThan(gap(random) / 3)
    expect(gap(rated, 40)).toBeLessThan(gap(random) / 1.5)
    expect(close.rounds[0].length).toBe(100)
  })
})

describe('a break before the change', () => {
  it('uncertainty-tracking raters react faster after a break; Elo does not', () => {
    const path = { kind: 'step', at: 100, size: 300 } as const
    const lagOf = (pause: number, spec: Parameters<typeof rateStream>[1]) => {
      const f = focalPlayerStream(stream(11), {
        games: 200,
        path,
        pause: pause ? { at: 100, rounds: pause } : undefined,
      })
      const tr = rateStream(f, spec, { fixed: f.fixed })
      const est = Array.from({ length: 200 }, (_, g) => tr.mean[(f.gameRound[g] + 1) * f.players])
      return est.findIndex((v, g) => g >= 100 && v - 1500 >= 0.9 * 300) - 100
    }
    expect(lagOf(200, { kind: 'glicko', c: 15 })).toBeLessThan(lagOf(0, { kind: 'glicko', c: 15 }))
    expect(lagOf(200, { kind: 'trueskill', tau: 15 })).toBeLessThan(lagOf(0, { kind: 'trueskill', tau: 15 }))
    expect(lagOf(200, { kind: 'elo', k: 24 })).toBe(lagOf(0, { kind: 'elo', k: 24 }))
  })
})
