/**
 * Simulated comparisons for skill rating: a round of random pairings between players with known (optionally drifting)
 * skills, rankings drawn from a Plackett–Luce model, and item responses drawn from a 2PL item-response model. Skills
 * are on the log-odds scale: $\pr(a \text{ beats } b) = \sigma(s_a - s_b)$, $\sigma$ the logistic function, which is
 * Elo's scale divided by $400/\ln 10 \approx 173.7$.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { PairedResult } from 'aifn-methods/inference/rating-models'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `tournament`. */
export interface TournamentOptions {
  /** Players (default 12). */
  players?: number
  /** Games, each between two players drawn uniformly (default 600). */
  games?: number
  /** Standard deviation of the initial skills, in log-odds (default 1). */
  spread?: number
  /** Standard deviation of each skill's random-walk step per game (default 0: fixed skills). */
  drift?: number
  /** The chance of a draw between equal players, in $[0, 1]$; draws thin out as the skill gap grows (default 0). */
  draws?: number
}

/** A simulated tournament: results in order and the true skills before each game. */
export interface Tournament {
  /** The games in order: the two players' indices and the first player's score (1 win, 0.5 draw, 0 loss). */
  readonly results: PairedResult[]
  /**
   * True skills, row-major, $(\text{games} + 1) \times \text{players}$; row $g$ holds the skills before game $g$ (row 0
   * the start, the last row the skills after the final game).
   */
  readonly skills: Float64Array
  /** The number of players. */
  readonly players: number
}

/**
 * Random pairings with outcomes from the Bradley–Terry (logistic) model,
 * $p = \pr(a \text{ beats } b) = \sigma(s_a - s_b)$. Each game is between two distinct players drawn uniformly. With
 * `draws` $= \delta > 0$, a game is drawn with probability $4\delta p(1 - p)$, so draws are commonest between equals.
 * With `drift` $> 0$ every skill takes a Gaussian random-walk step after each game. Throws `DomainError` with fewer
 * than two players.
 *
 * @param s The stream the initial skills, the drift steps, the pairings and the outcomes are drawn from (children
 *   `'skills'`, `'drift'`, `'pairs'` and `'outcomes'`).
 * @param options The number of players and games, the spread and drift of the skills, and the draw rate.
 * @returns The results in order, the true skills before each game, and the number of players.
 *
 * @example Results agree with the skills that produced them
 * const t = tournament(stream(1), { players: 6, games: 2000 })
 * print('games:', t.results.length, ' skills:', t.skills.length, '=', 2001, 'x', t.players)
 * print('first results [a, b, score]:', t.results.slice(0, 3).map((r) => [r.a, r.b, r.score]))
 * // Fixed skills: row 0 holds them for every game.
 * const gap = t.results.map(({ a, b }) => t.skills[a] - t.skills[b])
 * const won = t.results.filter((r, g) => (r.score === 1) === gap[g] > 0).length
 * print('share won by the stronger player:', won / 2000)
 * print('mean predicted chance of that:', gap.reduce((x, d) => x + 1 / (1 + Math.exp(-Math.abs(d))), 0) / 2000)
 */
export function tournament(s: Stream, options: TournamentOptions = {}): Tournament {
  const { players = 12, games = 600, spread = 1, drift = 0, draws = 0 } = options
  if (!(players >= 2)) throw new DomainError('tournament', 'tournament: needs at least two players')
  const z = standardNormals(child(s, 'skills'), players)
  const current = Float64Array.from(z, (v) => spread * v)
  const steps = drift > 0 ? standardNormals(child(s, 'drift'), games * players) : null
  const pick = units(child(s, 'pairs'), 2 * games)
  const u = units(child(s, 'outcomes'), 2 * games)
  const skills = new Float64Array((games + 1) * players)
  const results: PairedResult[] = []
  skills.set(current, 0)
  for (let g = 0; g < games; g++) {
    const a = Math.floor(pick[2 * g] * players)
    let b = Math.floor(pick[2 * g + 1] * (players - 1))
    if (b >= a) b++
    const p = 1 / (1 + Math.exp(current[b] - current[a]))
    const draw = u[2 * g + 1] < draws * 4 * p * (1 - p)
    results.push({ a, b, score: draw ? 0.5 : u[2 * g] < p ? 1 : 0 })
    if (steps) for (let k = 0; k < players; k++) current[k] += drift * steps[g * players + k]
    skills.set(current, (g + 1) * players)
  }
  return { results, skills, players }
}

/**
 * Rankings drawn from a Plackett–Luce model: each a full ranking of `size` items chosen uniformly from all items, built
 * by picking the next item from those left with probability proportional to its strength $e^{\lambda_i}$. The log
 * strengths $\lambda_i$ are drawn as $\Gauss(0, v^2)$, $v$ the `spread`. `size` is not checked against `items` and must
 * not exceed it.
 *
 * @param s The stream the strengths (child `'strengths'`) and each ranking (child `'ranking'`, $r$) are drawn from.
 * @param options `items` (default 8), the number of items; `rankings` (default 200), the number of rankings; `size`
 *   (default 4), the items in each ranking; `spread` (default 1), the standard deviation of the log strengths.
 * @returns `rankings`, each a list of item indices from first to last, and `logStrength`, the true $\lambda_i$ of every
 *   item.
 *
 * @example The strongest item comes first as often as the model says
 * const { rankings, logStrength } = plackettLuceRankings(stream(1), { items: 3, size: 3, rankings: 2000 })
 * print('log strengths:', logStrength)
 * print('first rankings:', rankings.slice(0, 3))
 * const w = Array.from(logStrength, Math.exp)
 * const top = w.indexOf(Math.max(...w))
 * print('P(strongest first):', w[top] / w.reduce((a, v) => a + v, 0))
 * print('share first:', rankings.filter((r) => r[0] === top).length / 2000)
 */
export function plackettLuceRankings(
  s: Stream,
  options: { items?: number; rankings?: number; size?: number; spread?: number } = {},
): { rankings: number[][]; logStrength: Float64Array } {
  const { items = 8, rankings: count = 200, size = 4, spread = 1 } = options
  const logStrength = Float64Array.from(standardNormals(child(s, 'strengths'), items), (v) => spread * v)
  const rankings: number[][] = []
  for (let r = 0; r < count; r++) {
    const u = units(child(s, 'ranking', r), items + size)
    // A uniform subset of `size` items by a partial Fisher–Yates shuffle.
    const pool = Array.from({ length: items }, (_, i) => i)
    for (let k = 0; k < size; k++) {
      const j = k + Math.floor(u[k] * (items - k))
      ;[pool[k], pool[j]] = [pool[j], pool[k]]
    }
    const left = pool.slice(0, size)
    const ranking: number[] = []
    for (let k = 0; k < size; k++) {
      let total = 0
      for (const i of left) total += Math.exp(logStrength[i])
      let v = u[items + k] * total
      let at = left.length - 1
      for (let m = 0; m < left.length; m++) {
        v -= Math.exp(logStrength[left[m]])
        if (v <= 0) {
          at = m
          break
        }
      }
      ranking.push(left[at])
      left.splice(at, 1)
    }
    rankings.push(ranking)
  }
  return { rankings, logStrength }
}

/**
 * Item responses from a 2PL model: person $p$ answers item $i$ correctly with probability
 * $\sigma(a_i(\theta_p - b_i))$, with abilities $\theta_p \sim \Gauss(0, 1)$, difficulties $b_i \sim \Gauss(0, 1)$ and
 * discriminations $\log a_i \sim \Gauss(0, 0.3^2)$ ($a_i = 1$ with `rasch`).
 *
 * @param s The stream the abilities, difficulties, discriminations and responses are drawn from (children `'ability'`,
 *   `'difficulty'`, `'discrimination'` and `'responses'`).
 * @param options `persons` (default 200) and `items` (default 20), the size of the table; `rasch` (default false), to
 *   fix every discrimination at 1 (the Rasch model).
 * @returns `responses`, a persons $\times$ items tensor of 1 (correct) and 0; and the true `ability` $\theta$ (one per
 *   person), `difficulty` $b$ and `discrimination` $a$ (one per item).
 *
 * @example Harder items are answered correctly less often
 * const r = irtResponses(stream(1), { persons: 500, items: 5, rasch: true })
 * const y = toArray(r.responses)
 * print('responses:', r.responses.shape, ' first rows:', y.slice(0, 3))
 * print('difficulty:', r.difficulty)
 * print('share correct:', [0, 1, 2, 3, 4].map((i) => y.reduce((a, row) => a + row[i], 0) / 500))
 */
export function irtResponses(
  s: Stream,
  options: { persons?: number; items?: number; rasch?: boolean } = {},
): { responses: Tensor; ability: Float64Array; difficulty: Float64Array; discrimination: Float64Array } {
  const { persons = 200, items = 20, rasch = false } = options
  const ability = standardNormals(child(s, 'ability'), persons)
  const difficulty = standardNormals(child(s, 'difficulty'), items)
  const discrimination = rasch
    ? new Float64Array(items).fill(1)
    : Float64Array.from(standardNormals(child(s, 'discrimination'), items), (v) => Math.exp(0.3 * v))
  const u = units(child(s, 'responses'), persons * items)
  const y = new Float64Array(persons * items)
  for (let p = 0; p < persons; p++)
    for (let i = 0; i < items; i++) {
      const q = 1 / (1 + Math.exp(-discrimination[i] * (ability[p] - difficulty[i])))
      y[p * items + i] = u[p * items + i] < q ? 1 : 0
    }
  return { responses: fromData(y, [persons, items]), ability, difficulty, discrimination }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'tournament',
    name: 'Simulated tournament',
    summary: 'Random pairings between players with known, optionally drifting, skills and Bradley–Terry outcomes.',
    role: 'simulation',
    random: true,
    notes: ['elo-rating', 'glicko-and-glicko-2', 'paired-comparison-models'],
    cite: ['bradley1952'],
  },
  tournament,
)
fn(
  {
    key: 'plackettLuceRankings',
    name: 'Plackett–Luce rankings',
    summary: 'Rankings of random subsets of items drawn as successive choices with probability ∝ strength.',
    role: 'simulation',
    random: true,
    notes: ['plackett-luce-model'],
    cite: ['plackett1975', 'luce1959'],
  },
  plackettLuceRankings,
)
fn(
  {
    key: 'irtResponses',
    name: 'Item responses',
    summary: 'Correct/incorrect answers of persons to items under a 2PL (or Rasch) item-response model.',
    role: 'simulation',
    random: true,
    notes: ['item-response-theory'],
    cite: ['rasch1960', 'lord1968'],
  },
  irtResponses,
)
