/**
 * Simulated game streams with known skill trajectories, for watching rating systems track skill: a population of
 * players paired round by round (at random, or with opponents of similar rating as on chess sites), and one focal
 * player whose skill changes, facing opponents of known strength. Skills are on Elo's scale; outcomes come from the
 * Bradley–Terry (logistic) or Thurstone (probit) model of the true skills, with optional draws.
 *
 * Every stream is a `GameStream` (`players`, and the games of each round) that the rating systems of
 * `aifn-methods/inference/rating-models` read directly, with the truth beside it: the true skills and each game's
 * expected score.
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
  /** Stays at the start. */
  | { kind: 'constant' }
  /** Jumps by `size` points at the start of round `at`. */
  | { kind: 'step'; at: number; size: number }
  /** Rises by `perRound` points a round from round `from` on. */
  | { kind: 'drift'; from: number; perRound: number }
  /** A Gaussian random walk with `sd` points a round. */
  | { kind: 'random-walk'; sd: number }

/**
 * A skill's value during each of `rounds` rounds, starting at `start`: constant; a step of `size` from round `at` on;
 * a drift of `perRound` a round after round `from` (so round $\text{from} + 1$ is the first to move); or a Gaussian
 * random walk $v_r = v_{r-1} + \sigma z_r$ from $v_0$ = `start`.
 *
 * @param s The stream a random walk's steps are drawn from (one standard normal per round); unused by the other
 *   paths.
 * @param start The skill at round 0, in Elo points.
 * @param rounds The number of rounds.
 * @param path How the skill moves (`SkillPath`).
 * @returns The skill in each round.
 *
 * @example A step, a drift and a random walk
 * print('step:', skillPath(stream(1), 1500, 6, { kind: 'step', at: 3, size: 200 }))
 * print('drift:', skillPath(stream(1), 1500, 6, { kind: 'drift', from: 2, perRound: 10 }))
 * const w = skillPath(stream(1), 1500, 2000, { kind: 'random-walk', sd: 10 })
 * const steps = w.slice(1).map((v, r) => v - w[r])
 * print('random walk, first rounds:', w.slice(0, 4))
 * print('random walk, step sd (10):', Math.sqrt(steps.reduce((a, v) => a + v * v, 0) / steps.length))
 */
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
  /**
   * `bradley-terry`: $P = 1/(1 + 10^{-d/400})$; `thurstone`: $P = \Phi(d / (\sqrt{2} \beta))$, for the skill
   * difference $d$ (default `bradley-terry`).
   */
  outcome?: 'bradley-terry' | 'thurstone'
  /**
   * Thurstone's performance noise $\beta$ (default `THURSTONE_BETA`, about 196, which matches the logistic's slope at
   * $d = 0$).
   */
  beta?: number
  /**
   * The chance of a draw between equals, at most $\tfrac{1}{2}$; draws thin out as the gap grows:
   * $P(\text{draw}) = 4 \cdot \text{draws} \cdot P(1 - P)$ (default 0).
   */
  draws?: number
}

/**
 * $P(a \text{ beats } b)$, ignoring draws, for the skill difference $d = s_a - s_b$: the expected score of $a$.
 *
 * @param d The skill difference, in Elo points.
 * @param options The outcome model and Thurstone's $\beta$ (`draws` is not read).
 * @returns The probability, in $(0, 1)$.
 *
 * @example 400 points is ten to one under Bradley–Terry
 * print('even:', winProbability(0), ' +400:', winProbability(400), ' -400:', winProbability(-400))
 * print('Thurstone, +400:', winProbability(400, { outcome: 'thurstone' }))
 */
export function winProbability(d: number, options: OutcomeOptions = {}): number {
  const { outcome = 'bradley-terry', beta = THURSTONE_BETA } = options
  return outcome === 'thurstone' ? normalCdf(d / (Math.SQRT2 * beta)) : 1 / (1 + Math.pow(10, -d / 400))
}

/**
 * One game for the skill difference $d$: $P(\text{draw}) = 4 \cdot \text{draws} \cdot P(1 - P)$, carved equally from
 * wins and losses, so $a$'s expected score stays $P$ (draws must be at most $\tfrac{1}{2}$). Uses one uniform `u`.
 *
 * @param d The skill difference $s_a - s_b$.
 * @param u A uniform draw on $[0, 1)$ that decides the result.
 * @param options The outcome model and the draw rate.
 * @returns `score`, $a$'s score (1, $\tfrac{1}{2}$ or 0), and `p`, $a$'s expected score.
 */
function play(d: number, u: number, options: OutcomeOptions): { score: number; p: number } {
  const p = winProbability(d, options)
  const half = 2 * (options.draws ?? 0) * p * (1 - p)
  return { score: u < p - half ? 1 : u < p + half ? 0.5 : 0, p }
}

/** Options of `ratingPopulation`. */
export interface RatingPopulationOptions extends OutcomeOptions {
  /** Number of players, at least 2 (default 200). */
  players?: number
  /** Number of rounds (default 60). */
  rounds?: number
  /** Games each player plays per round (default 2). */
  gamesPerRound?: number
  /** Mean of the starting true skills (default 0: skills relative to the pool). */
  mean?: number
  /** Standard deviation of the starting true skills (default 350). */
  spread?: number
  /** Every player's skill path (default constant). */
  path?: SkillPath
  /**
   * `random`: uniform pairs. `rating` (default): players sorted by a matchmaking rating plus
   * $\Gauss(0, \text{window}^2)$ noise and paired with their neighbour, as a site pairs players of similar rating; the
   * matchmaking rating is the simulator's own Elo ($K = 32$, start 1500), so the games are the same whichever rating
   * system is then run on them. `skill`: the same on the true skills (games between near-equals, which say little
   * about the scale).
   */
  matchmaking?: 'random' | 'rating' | 'skill'
  /** Standard deviation of the noise added to the sort key of `rating` and `skill` matchmaking (default 100). */
  window?: number
}

/** A simulated stream with the truth. */
export interface RatingPopulation extends GameStream {
  /** True skills during each round, row-major, rounds by players: round $r$, player $p$ at `r * players + p`. */
  readonly skills: Float64Array
  /** The true expected score of player a in each game, in stream order. */
  readonly probability: Float64Array
}

/**
 * A population of players with known skill trajectories, paired every slot of every round by the matchmaking rule
 * (with an odd count, one player sits out a slot), outcomes from the true skills. Starting skills are
 * $\Gauss(\text{mean}, \text{spread}^2)$ from `child(s, 'skills')`, and each player's path draws from
 * `child(s, 'path', p)`. Colours alternate by slot, so neither side of a pairing is always player $a$. Throws
 * `DomainError` with fewer than two players.
 *
 * @param s The stream the skills, the pairings and the outcomes are drawn from.
 * @param options The size of the population, the skill paths, the matchmaking and the outcome model
 *   (`RatingPopulationOptions`).
 * @returns The games of every round, the true skills and every game's expected score.
 *
 * @example Every player plays twice a round, and the scores average to the expected ones
 * const pop = ratingPopulation(stream(1), { players: 200, rounds: 5 })
 * const games = pop.rounds.flat()
 * const start = pop.skills.slice(0, pop.players)
 * print('rounds:', pop.rounds.length, ' games per round:', pop.rounds[0].length, ' first game:', games[0])
 * print('sd of the starting skills (350):', Math.sqrt(start.reduce((a, v) => a + v * v, 0) / start.length))
 * const mean = (v) => v.reduce((a, u) => a + u, 0) / v.length
 * print('mean score:', mean(games.map((g) => g.score)), ' mean expected:', mean(pop.probability))
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
   * `close`: each opponent's known skill is near the focal player's true skill plus $\Gauss(0, \text{window}^2)$
   * (matchmaking by skill); `field`: drawn from $\Gauss(\text{start}, \text{window}^2)$ whatever the focal skill
   * (default `close`).
   */
  opponents?: 'close' | 'field'
  /** The spread of the opponents' skills around their target, in Elo points (default 150). */
  window?: number
  /** A break of `rounds` rounds without games just before game `at` (default none). */
  pause?: { at: number; rounds: number }
  /** Opponents in the pool (default 80). */
  pool?: number
}

/** The focal player's stream: player 0 against opponents of known skill from a pool (players $1, \dots$). */
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
 * known skill from a pool: with `close` opponents the pool is evenly spaced over the focal skill's range widened by
 * $2 \cdot \text{window}$ on each side, and each game takes the member nearest the focal skill plus
 * $\Gauss(0, \text{window}^2)$; with a `field` the pool is spread as $\Gauss(\text{start}, \text{window}^2)$ and each
 * game takes a member at random. A `pause` inserts rounds without games before game `pause.at`. Throws `DomainError`
 * unless there is at least one game and one opponent.
 *
 * @param s The stream the path, the pool, the opponents and the outcomes are drawn from.
 * @param options The games, the focal player's start and path, the opponents, the pause and the outcome model
 *   (`FocalPlayerOptions`).
 * @returns The games, one a round (none in a pause), with the opponents' known skills in `fixed`, the focal player's
 *   true skill per game in `truth`, each game's round and its expected score.
 *
 * @example The default step of 200 at game 150, against close opponents
 * const f = focalPlayerStream(stream(1))
 * print('players:', f.players, ' rounds:', f.rounds.length, ' first game:', f.rounds[0][0])
 * print('true skill at games 149 and 150:', f.truth[149], f.truth[150])
 * const opp = f.rounds.map((r) => f.fixed[r[0].b])
 * const avg = (v) => v.reduce((a, u) => a + u, 0) / v.length
 * print('mean opponent before and after the step:', avg(opp.slice(0, 150)), avg(opp.slice(150)))
 *
 * @example A pause adds empty rounds
 * const f = focalPlayerStream(stream(1), { games: 10, pause: { at: 5, rounds: 3 } })
 * print('rounds:', f.rounds.length, ' round of each game:', f.gameRound)
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
  /** The path's kind (default `constant`). */
  path?: SkillPath['kind']
  /** The round of a step, or the round a drift starts from (default 0). */
  at?: number
  /** The size of a step (default 0). */
  size?: number
  /** The drift per round (default 0). */
  perRound?: number
  /** The random walk's step standard deviation (default 0). */
  sd?: number
}
/**
 * The skill path the flat knobs describe; the knobs the kind does not use are ignored.
 *
 * @param k The flat knobs.
 * @returns The path.
 */
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

/**
 * `ratingPopulation` with flat knobs: every player's skill path given by `path`, `at`, `size`, `perRound`, `sd`, as
 * the dataset registry passes them.
 *
 * @param s The stream the population is drawn from.
 * @param knobs The options of `ratingPopulation`, with the path as flat knobs (default constant).
 * @returns The population, as `ratingPopulation` returns it.
 *
 * @example Every player drifts up by 5 points a round
 * const pop = ratingMatches(stream(1), { players: 10, rounds: 4, path: 'drift', perRound: 5 })
 * print('player 0 by round:', [0, 1, 2, 3].map((r) => pop.skills[r * pop.players]))
 */
export function ratingMatches(
  s: Stream,
  knobs: Omit<RatingPopulationOptions, 'path'> & PathKnobs = {},
): RatingPopulation {
  return ratingPopulation(s, { ...knobs, path: pathOfKnobs(knobs) })
}

/**
 * `focalPlayerStream` with flat knobs, as the dataset registry passes them: the focal path by `path`, `at`, `size`,
 * `perRound`, `sd` (default a step of 200 at game 150), and no pause.
 *
 * @param s The stream the games are drawn from.
 * @param knobs The options of `focalPlayerStream` without `pause`, with the path as flat knobs.
 * @returns The stream, as `focalPlayerStream` returns it.
 *
 * @example A step of 100 at game 20
 * const f = focalPlayerMatches(stream(1), { games: 40, at: 20, size: 100 })
 * print('true skill at games 19 and 20:', f.truth[19], f.truth[20])
 */
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
