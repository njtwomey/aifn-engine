/**
 * Simulated comparisons for skill rating: a round of random pairings between players with known (optionally drifting)
 * skills, rankings drawn from a Plackett–Luce model, and item responses drawn from a 2PL item-response model. Skills
 * are on the log-odds scale: P(a beats b) = σ(s_a − s_b), which is Elo's scale divided by 400/ln 10 ≈ 173.7.
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
  /** The chance of a draw between equal players; draws thin out as the skill gap grows (default 0). */
  draws?: number
}

/** A simulated tournament: results in order and the true skills before each game. */
export interface Tournament {
  readonly results: PairedResult[]
  /** True skills, row-major [(games + 1) × players]; row g holds the skills before game g (row 0 the start). */
  readonly skills: Float64Array
  readonly players: number
}

/**
 * Random pairings with outcomes from the Bradley–Terry (logistic) model, P(a beats b) = σ(s_a − s_b). With `draws` > 0,
 * a game is drawn with probability draws · 4 p(1 − p), so draws are commonest between equals.
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

/** Rankings drawn from a Plackett–Luce model: each a full ranking of `size` items chosen uniformly from all items. */
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

/** Item responses from a 2PL model with θ ~ N(0, 1), b ~ N(0, 1) and log a ~ N(0, 0.3²) (a = 1 with `rasch`). */
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
