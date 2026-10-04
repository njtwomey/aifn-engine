import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  bradleyTerry,
  eloRatings,
  eloUpdate,
  fitIrt,
  glicko2Update,
  glickoRatings,
  glickoUpdate,
  irtProbability,
  itemInformation,
  plackettLuce,
  type PairedResult,
} from 'aifn-methods/inference/rating-models'
import { irtResponses, plackettLuceRankings, tournament } from 'aifn-methods/data/synthetic'
import { fixture } from '../../fixtures'

type Fixture = {
  pairwise: { players: number; pairs: [number, number][]; ml: number[]; map: number[] }
  rankings: { items: number; rankings: number[][]; ml: number[] }
}
const F = fixture<Fixture>('inference/rating-models')

const correlation = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const n = a.length
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += a[i] / n
    mb += b[i] / n
  }
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

describe('Elo', () => {
  it('moves both ratings by K times the surprise, conserving the total', () => {
    const u = eloUpdate(1600, 1400, 1, { k: 32 })
    expect(u.expected).toBeCloseTo(1 / (1 + 10 ** (-200 / 400)), 12)
    expect(u.a - 1600).toBeCloseTo(32 * (1 - u.expected), 12)
    expect(u.a + u.b).toBeCloseTo(3000, 12)
    // A draw between equals changes nothing.
    const d = eloUpdate(1500, 1500, 0.5)
    expect(d.a).toBe(1500)
  })
  it('ranks a simulated league in the order of the true skills', () => {
    const t = tournament(stream(3), { players: 8, games: 3000, spread: 1.2 })
    const run = eloRatings(t.results, { players: 8, k: 16 })
    expect(correlation(run.ratings, t.skills.subarray(0, 8))).toBeGreaterThan(0.9)
    expect(run.history.length).toBe(3001 * 8)
    expect(run.logLoss).toBeLessThan(Math.log(2))
  })
})

describe('Glicko', () => {
  // Glickman's worked examples: a 1500 (RD 200) player beats 1400 (30), loses to 1550 (100) and 1700 (300).
  const games = [
    { opponent: { rating: 1400, deviation: 30 }, score: 1 },
    { opponent: { rating: 1550, deviation: 100 }, score: 0 },
    { opponent: { rating: 1700, deviation: 300 }, score: 0 },
  ]
  it('reproduces the Glicko-1 example (Glickman, 1999)', () => {
    const r = glickoUpdate({ rating: 1500, deviation: 200 }, games)
    expect(r.rating).toBeCloseTo(1464.1, 0)
    expect(r.deviation).toBeCloseTo(151.4, 0)
  })
  it('reproduces the Glicko-2 example (Glickman, 2001)', () => {
    const r = glicko2Update({ rating: 1500, deviation: 200, volatility: 0.06 }, games, { tau: 0.5 })
    expect(r.rating).toBeCloseTo(1464.06, 1)
    expect(r.deviation).toBeCloseTo(151.52, 1)
    expect(r.volatility).toBeCloseTo(0.05999, 4)
  })
  it('grows the deviation of an idle player under Glicko-2 and shrinks active ones', () => {
    const idle = glicko2Update({ rating: 1500, deviation: 50, volatility: 0.06 }, [])
    expect(idle.deviation).toBeGreaterThan(50)
    const results: PairedResult[] = tournament(stream(1), { players: 6, games: 300 }).results
    for (const version of ['glicko', 'glicko2'] as const) {
      const run = glickoRatings(results, { players: 6, version, period: 10 })
      const last = run.deviations!.subarray(300 * 6)
      for (const d of last) expect(d).toBeLessThan(150)
    }
  })
})

describe('Bradley–Terry and Plackett–Luce by MM (choix)', () => {
  const results: PairedResult[] = F.pairwise.pairs.map(([w, l]) => ({ a: w, b: l, score: 1 }))
  it('matches choix.mm_pairwise, maximum likelihood and MAP', () => {
    const ml = bradleyTerry(results, { players: F.pairwise.players, tolerance: 1e-12, maxIterations: 100_000 })
    expect(ml.converged).toBe(true)
    ml.logStrength.forEach((v, i) => expect(v).toBeCloseTo(F.pairwise.ml[i], 7))
    const map = bradleyTerry(results, { players: F.pairwise.players, prior: 0.5, tolerance: 1e-12 })
    map.logStrength.forEach((v, i) => expect(v).toBeCloseTo(F.pairwise.map[i], 7))
  })
  it('never decreases the log-likelihood', () => {
    const fit = bradleyTerry(results, { players: F.pairwise.players })
    for (let i = 1; i < fit.logLikelihood.length; i++)
      expect(fit.logLikelihood[i]).toBeGreaterThanOrEqual(fit.logLikelihood[i - 1] - 1e-9)
  })
  it('matches choix.mm_rankings', () => {
    const fit = plackettLuce(F.rankings.rankings, { items: F.rankings.items, tolerance: 1e-12, maxIterations: 100_000 })
    fit.logStrength.forEach((v, i) => expect(v).toBeCloseTo(F.rankings.ml[i], 7))
  })
  it('recovers simulated Plackett–Luce strengths', () => {
    const { rankings, logStrength } = plackettLuceRankings(stream(4), { items: 6, rankings: 800, size: 4 })
    const fit = plackettLuce(rankings, { items: 6 })
    expect(correlation(fit.logStrength, logStrength)).toBeGreaterThan(0.95)
  })
})

describe('item response theory', () => {
  it('has information a²P(1 − P), largest at θ = b', () => {
    const p = irtProbability(0.3, 1.7, -0.2)
    expect(itemInformation(0.3, 1.7, -0.2)).toBeCloseTo(1.7 ** 2 * p * (1 - p), 12)
    expect(itemInformation(-0.2, 1.7, -0.2)).toBeGreaterThan(itemInformation(0.5, 1.7, -0.2))
    expect(irtProbability(-10, 1, 0, 0.25)).toBeCloseTo(0.25, 3)
  })
  it('recovers abilities and difficulties of simulated 2PL responses', () => {
    const d = irtResponses(stream(9), { persons: 300, items: 30 })
    const fit = fitIrt(d.responses, { model: '2pl' })
    expect(correlation(fit.difficulty, d.difficulty)).toBeGreaterThan(0.95)
    expect(correlation(fit.ability, d.ability)).toBeGreaterThan(0.85)
    expect(correlation(fit.discrimination, d.discrimination)).toBeGreaterThan(0.4)
    const rasch = fitIrt(d.responses, { model: '1pl' })
    expect(rasch.discrimination.every((a) => a === 1)).toBe(true)
    expect(fit.logLikelihood).toBeGreaterThan(rasch.logLikelihood)
  })
})
