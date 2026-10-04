/**
 * Simulated game streams with known skill trajectories, for watching rating systems track skill: a population of
 * players paired round by round (at random, or with opponents of similar rating as on chess sites), and one focal player whose skill
 * changes, facing opponents of known strength. Skills are on Elo's scale; outcomes come from the Bradley–Terry
 * (logistic) or Thurstone (probit) model of the true skills, with optional draws.
 */

import type { DatasetInfo, FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { normalCdf } from 'aifn-compute/numerics/special'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { THURSTONE_BETA, eloUpdate, type GameStream } from 'aifn-methods/inference/rating-models'
import type { PairedResult } from 'aifn-methods/inference/rating-models'

/** How a true skill moves over the rounds. */
export type SkillPath =
  | { kind: 'constant' }
  /** Jumps by `size` points at the start of round `at`. */
  | { kind: 'step'; at: number; size: number }
  /** Rises by `perRound` points a round from round `from` on. */
  | { kind: 'drift'; from: number; perRound: number }
  /** A Gaussian random walk with `sd` points a round. */
  | { kind: 'random-walk'; sd: number }

/** A skill's value during each of `rounds` rounds, starting at `start` (a random walk draws from `s`). */
export function skillPath(s: Stream, start: number, rounds: number, path: SkillPath): Float64Array {
  const out = new Float64Array(rounds)
  const steps = path.kind === 'random-walk' ? standardNormals(s, rounds) : null
  let v = start
  for (let r = 0; r < rounds; r++) {
    if (path.kind === 'step') v = start + (r >= path.at ? path.size : 0)
    else if (path.kind === 'drift') v = start + Math.max(0, r - path.from) * path.perRound
    else if (path.kind === 'random-walk' && r > 0) v += path.sd * steps![r]
    out[r] = v
  }
  return out
}

/** The outcome model of a game between true skills. */
export interface OutcomeOptions {
  /** `bradley-terry`: P = 1/(1 + 10^(−d/400)); `thurstone`: P = Φ(d/(√2 β)) (default `bradley-terry`). */
  outcome?: 'bradley-terry' | 'thurstone'
  /** Thurstone's performance noise (default `THURSTONE_BETA` ≈ 196, the logistic's slope at d = 0). */
  beta?: number
  /** The chance of a draw between equals, at most ½; draws thin out as the gap grows: P(draw) = draws · 4P(1 − P) (default 0). */
  draws?: number
}

/** P(a beats b), ignoring draws, for the skill difference d = s_a − s_b. */
export function winProbability(d: number, options: OutcomeOptions = {}): number {
  const { outcome = 'bradley-terry', beta = THURSTONE_BETA } = options
  return outcome === 'thurstone' ? normalCdf(d / (Math.SQRT2 * beta)) : 1 / (1 + Math.pow(10, -d / 400))
}

/**
 * One game for the skill difference d: P(draw) = draws · 4P(1 − P), carved equally from wins and losses, so a's expected
 * score stays P (draws must be at most ½). Uses one uniform `u`.
 */
function play(d: number, u: number, options: OutcomeOptions): { score: number; p: number } {
  const p = winProbability(d, options)
  const half = 2 * (options.draws ?? 0) * p * (1 - p)
  return { score: u < p - half ? 1 : u < p + half ? 0.5 : 0, p }
}

/** Options of `ratingPopulation`. */
export interface RatingPopulationOptions extends OutcomeOptions {
  players?: number
  rounds?: number
  /** Games each player plays per round (default 2). */
  gamesPerRound?: number
  /** Mean and sd of the starting true skills (default 0 and 350: skills relative to the pool). */
  mean?: number
  spread?: number
  /** Every player's skill path (default constant). */
  path?: SkillPath
  /**
   * `random`: uniform pairs. `rating` (default): players sorted by a matchmaking rating plus N(0, window²) noise and
   * paired with their neighbour, as a site pairs players of similar rating; the matchmaking rating is the simulator's
   * own Elo (K = 32, start 1500), so the games are the same whichever rating system is then run on them. `skill`: the
   * same on the true skills (games between near-equals, which say little about the scale).
   */
  matchmaking?: 'random' | 'rating' | 'skill'
  window?: number
}

/** A simulated stream with the truth. */
export interface RatingPopulation extends GameStream {
  /** True skills during each round, row-major [rounds × players]. */
  readonly skills: Float64Array
  /** The true expected score of player a in each game, in stream order. */
  readonly probability: Float64Array
}

/**
 * A population of players with known skill trajectories, paired every slot of every round by the matchmaking rule
 * (with an odd count, one player sits out a slot), outcomes from the true skills.
 */
export function ratingPopulation(s: Stream, options: RatingPopulationOptions = {}): RatingPopulation {
  const {
    players = 200,
    rounds = 60,
    gamesPerRound = 2,
    mean = 0,
    spread = 350,
    path = { kind: 'constant' },
    matchmaking = 'rating',
    window = 100,
  } = options
  if (!(players >= 2)) throw new DomainError('ratingPopulation', 'ratingPopulation: needs at least two players')
  const start = standardNormals(child(s, 'skills'), players)
  const skills = new Float64Array(rounds * players)
  for (let p = 0; p < players; p++) {
    const traj = skillPath(child(s, 'path', p), mean + spread * start[p], rounds, path)
    for (let r = 0; r < rounds; r++) skills[r * players + p] = traj[r]
  }
  const out: PairedResult[][] = []
  const probability: number[] = []
  const elo = new Float64Array(players).fill(1500)
  for (let r = 0; r < rounds; r++) {
    const games: PairedResult[] = []
    for (let slot = 0; slot < gamesPerRound; slot++) {
      const key = child(s, 'round', r, slot)
      const noise = standardNormals(child(key, 'pairing'), players)
      const order = Array.from({ length: players }, (_, p) => p)
      const sortKey =
        matchmaking === 'random'
          ? Float64Array.from(units(child(key, 'shuffle'), players))
          : Float64Array.from(
              order,
              (p) => (matchmaking === 'skill' ? skills[r * players + p] : elo[p]) + window * noise[p],
            )
      order.sort((i, j) => sortKey[i] - sortKey[j])
      const u = units(child(key, 'outcomes'), players)
      for (let k = 0; k + 1 < players; k += 2) {
        // Alternate colours by slot so neither side of a pairing is always player a.
        const [a, b] = slot % 2 === 0 ? [order[k], order[k + 1]] : [order[k + 1], order[k]]
        const g = play(skills[r * players + a] - skills[r * players + b], u[k], options)
        games.push({ a, b, score: g.score })
        probability.push(g.p)
        const u2 = eloUpdate(elo[a], elo[b], g.score)
        elo[a] = u2.a
        elo[b] = u2.b
      }
    }
    out.push(games)
  }
  return { players, rounds: out, skills, probability: Float64Array.from(probability) }
}

/** Options of `focalPlayerStream`. */
export interface FocalPlayerOptions extends OutcomeOptions {
  /** Games, one a round (default 300). */
  games?: number
  /** The focal player's skill before any change (default 1500). */
  start?: number
  /** The focal player's skill path (default a step of +200 at game 150). */
  path?: SkillPath
  /**
   * `close`: each opponent's known skill is the focal player's true skill plus N(0, window²) (matchmaking by skill);
   * `field`: drawn from N(start, window²) whatever the focal skill (default `close`).
   */
  opponents?: 'close' | 'field'
  window?: number
  /** A break of `rounds` rounds without games just before game `at` (default none). */
  pause?: { at: number; rounds: number }
  /** Opponents in the pool (default 80). */
  pool?: number
}

/** The focal player's stream: player 0 against opponents of known skill from a pool (players 1 …). */
export interface FocalPlayerStream extends GameStream {
  /** The true expected score of the focal player in each game. */
  readonly probability: Float64Array
  /** Known skills: NaN for the focal player, each opponent's skill otherwise (for `rateStream`'s `fixed`). */
  readonly fixed: Float64Array
  /** The focal player's true skill during each game. */
  readonly truth: Float64Array
  /** The round of each game (games and rounds differ by a break). */
  readonly gameRound: Int32Array
}

/**
 * One focal player (player 0) whose skill follows `path` (indexed by game), one game a round against an opponent of
 * known skill from a pool: with `close` opponents the pool spans the focal skill's range and each game takes the member
 * nearest the focal skill plus N(0, window²); with a `field` the pool is spread as N(start, window²) and each game takes
 * a member at random. A `pause` inserts rounds without games before game `pause.at`.
 */
export function focalPlayerStream(s: Stream, options: FocalPlayerOptions = {}): FocalPlayerStream {
  const {
    games = 300,
    start = 1500,
    path = { kind: 'step', at: 150, size: 200 },
    opponents = 'close',
    window = 150,
    pool = 80,
  } = options
  if (!(games >= 1 && pool >= 1))
    throw new DomainError('focalPlayerStream', 'focalPlayerStream: needs games and a pool')
  const truth = skillPath(child(s, 'path'), start, games, path)
  const players = pool + 1
  const fixed = new Float64Array(players)
  fixed[0] = NaN
  let lo = Infinity
  let hi = -Infinity
  for (const v of truth) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const spreadOf = standardNormals(child(s, 'pool'), pool)
  for (let k = 0; k < pool; k++)
    fixed[k + 1] =
      opponents === 'close'
        ? lo - 2 * window + ((hi - lo + 4 * window) * (k + 0.5)) / pool
        : start + window * spreadOf[k]
  const z = standardNormals(child(s, 'opponents'), games)
  const pick = units(child(s, 'pick'), games)
  const u = units(child(s, 'outcomes'), games)
  const rounds: PairedResult[][] = []
  const gameRound = new Int32Array(games)
  const probability = new Float64Array(games)
  for (let g = 0; g < games; g++) {
    if (options.pause && g === options.pause.at) for (let k = 0; k < options.pause.rounds; k++) rounds.push([])
    let b = 1 + Math.min(pool - 1, Math.floor(pick[g] * pool))
    if (opponents === 'close') {
      const target = truth[g] + window * z[g]
      for (let k = 1; k <= pool; k++) if (Math.abs(fixed[k] - target) < Math.abs(fixed[b] - target)) b = k
    }
    const res = play(truth[g] - fixed[b], u[g], options)
    probability[g] = res.p
    gameRound[g] = rounds.length
    rounds.push([{ a: 0, b, score: res.score }])
  }
  return { players, rounds, probability, fixed, truth, gameRound }
}

/** Flat knobs of a skill path (for the dataset registry). */
interface PathKnobs {
  path?: SkillPath['kind']
  at?: number
  size?: number
  perRound?: number
  sd?: number
}
const pathOfKnobs = (k: PathKnobs): SkillPath => {
  const { path = 'constant', at = 0, size = 0, perRound = 0, sd = 0 } = k
  return path === 'step'
    ? { kind: 'step', at, size }
    : path === 'drift'
      ? { kind: 'drift', from: at, perRound }
      : path === 'random-walk'
        ? { kind: 'random-walk', sd }
        : { kind: 'constant' }
}

/** `ratingPopulation` with flat knobs: every player's skill path given by `path`, `at`, `size`, `perRound`, `sd`. */
export function ratingMatches(
  s: Stream,
  knobs: Omit<RatingPopulationOptions, 'path'> & PathKnobs = {},
): RatingPopulation {
  return ratingPopulation(s, { ...knobs, path: pathOfKnobs(knobs) })
}

/** `focalPlayerStream` with flat knobs (the focal path by `path`, `at`, `size`, `perRound`, `sd`; default a step). */
export function focalPlayerMatches(
  s: Stream,
  knobs: Omit<FocalPlayerOptions, 'path' | 'pause'> & PathKnobs = {},
): FocalPlayerStream {
  const { path = 'step', at = 150, size = 200 } = knobs
  return focalPlayerStream(s, { ...knobs, path: pathOfKnobs({ ...knobs, path, at, size }) })
}

const PATH_KNOBS = {
  path: oneOf(['constant', 'step', 'drift', 'random-walk']),
  at: int(0, 100000, { default: 0 }),
  size: real(-1000, 1000, { default: 0 }),
  perRound: real(-50, 50, { default: 0 }),
  sd: real(0, 200, { default: 0 }),
}
const OUTCOME_KNOBS = {
  outcome: oneOf(['bradley-terry', 'thurstone']),
  draws: real(0, 0.5, { default: 0 }),
}
const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'ratingMatches',
    name: 'Rated population with known skills',
    summary:
      'Players with known skill paths paired round by round (by a matchmaking Elo, true skill, or at random), with Bradley–Terry or Thurstone outcomes and the true skills.',
    task: 'ranking',
    output: 'matches',
    knobs: space({
      players: int(2, 5000, { default: 200 }),
      rounds: int(1, 2000, { default: 60 }),
      gamesPerRound: int(1, 50, { default: 2 }),
      spread: real(0, 1000, { default: 350 }),
      matchmaking: oneOf(['rating', 'skill', 'random']),
      window: real(0, 1000, { default: 100 }),
      ...PATH_KNOBS,
      ...OUTCOME_KNOBS,
    }),
    truth: true,
    random: true,
    notes: ['skill-rating', 'elo-rating', 'glicko-and-glicko-2'],
    cite: ['bradley1952'],
  },
  ratingMatches,
)
dataset(
  {
    key: 'focalPlayerMatches',
    name: 'Focal player with a skill change',
    summary:
      'One player whose skill steps, drifts or wanders, against a pool of opponents of known skill; the true skill per game.',
    task: 'ranking',
    output: 'matches',
    knobs: space({
      games: int(1, 100000, { default: 300 }),
      start: real(-5000, 5000, { default: 1500 }),
      opponents: oneOf(['close', 'field']),
      window: real(0, 1000, { default: 150 }),
      pool: int(1, 1000, { default: 80 }),
      ...PATH_KNOBS,
      path: oneOf(['step', 'constant', 'drift', 'random-walk']),
      at: int(0, 100000, { default: 150 }),
      size: real(-1000, 1000, { default: 200 }),
      ...OUTCOME_KNOBS,
    }),
    truth: true,
    random: true,
    notes: ['skill-rating', 'trueskill', 'trueskill-through-time'],
  },
  focalPlayerMatches,
)

const fn = definer<FunctionInfo>('function', 'data/synthetic')
const NOTES = ['skill-rating', 'elo-rating', 'glicko-and-glicko-2', 'trueskill']

fn(
  {
    key: 'skillPath',
    name: 'Skill trajectory',
    summary: 'A true skill over rounds: constant, a step, a linear drift or a Gaussian random walk.',
    role: 'simulation',
    random: true,
    notes: NOTES,
  },
  skillPath,
)
fn(
  {
    key: 'winProbability',
    name: 'Win probability of a skill gap',
    summary: 'Bradley–Terry 1/(1 + 10^(−d/400)) or Thurstone Φ(d/(√2β)) for the skill difference d.',
    role: 'property',
    notes: ['paired-comparison-models', 'trueskill'],
    cite: ['bradley1952'],
  },
  winProbability,
)
fn(
  {
    key: 'ratingPopulation',
    name: 'Rated population with known skills',
    summary:
      'Players with known skill paths paired every round, at random or with close opponents, with outcomes from the true skills.',
    role: 'simulation',
    random: true,
    notes: NOTES,
    cite: ['bradley1952'],
  },
  ratingPopulation,
)
fn(
  {
    key: 'focalPlayerStream',
    name: 'Focal player with a skill change',
    summary: 'One player whose skill steps, drifts or wanders, each game against a new opponent of known skill.',
    role: 'simulation',
    random: true,
    notes: NOTES,
  },
  focalPlayerStream,
)
