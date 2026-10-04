/**
 * Agents for two-action control problems such as the cart-pole (push left = 0, push right = 1):
 *
 * - **`lqrBangBangAgent`**: linearise the environment's differentiable dynamics about the origin (x = 0, u = 0) by
 *   autodiff (`lineariseDynamics`), solve the discrete-time LQR (`dlqr`, Q = I, R = r) and push in the direction of
 *   the LQR force u = −K x. Bang-bang control with the LQR's switching surface: no learning, a model-based baseline.
 * - **`crossEntropyAgent`**: the cross-entropy method over linear threshold policies a = [w · (o, 1) > 0]
 *   (Szita and Lőrincz, 2006, "Learning Tetris using the noisy cross-entropy method", Neural Computation 18(12)). Each
 *   generation draws a population of weight vectors from N(μ, diag σ²), plays one episode with each, keeps the elite
 *   fraction with the highest returns, and refits μ and σ to them, adding a decaying noise to σ so that the search does
 *   not collapse early. It is an episode-level learner on the `Agent` protocol: `act` plays the current member,
 *   `learn` sums its return and refits after the last member. The state is plain data (it checkpoints), and the
 *   population of each generation is drawn from a seed fixed in `init`, so `learn` stays pure.
 */

import { dlqr } from 'aifn-compute/dynamics/control'
import type { Agent, AgentInfo, EnvironmentShape } from 'aifn-compute/foundation/contracts'
import { child, integers, normal, stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { domainDimension, int, real, space } from 'aifn-compute/foundation/space'
import { eye, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { lineariseDynamics } from './swing-up'

// ── LQR bang-bang ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `lqrBangBangAgent`. */
export interface LqrBangBangOptions {
  /** The LQR's force weight R (default 1); the state weight is Q = I. */
  r?: number
}

/** The bang-bang agent's state: the LQR gain K on the observed state. */
export interface LqrBangBangState {
  K: readonly number[]
}

/**
 * Pushes right (action 1) when the LQR force −K o about the origin is positive, else left (action 0). `init` needs a
 * `dynamics` model whose state is the observation and whose action is a force.
 */
export function lqrBangBangAgent({ r = 1 }: LqrBangBangOptions = {}): Agent<LqrBangBangState, Float64Array, number> {
  return {
    name: 'LQR bang-bang',
    init: (env: EnvironmentShape) => {
      const model = env.model
      if (model?.kind !== 'dynamics' || model.actionSize !== 1)
        throw new TypeError(`lqrBangBangAgent: ${env.name} has no dynamics model with a scalar force`)
      const n = model.stateSize
      const lin = lineariseDynamics(model, new Float64Array(n), [0])
      return { K: toRows(dlqr(lin, eye(n), [[r]]).K)[0] }
    },
    act({ K }, obs) {
      let u = 0
      for (let i = 0; i < K.length; i++) u -= K[i] * obs[i]
      return { action: u > 0 ? 1 : 0, scores: Float64Array.of(u) }
    },
    greedy: ({ K }, obs) => (K.reduce((u, k, i) => u - k * obs[i], 0) > 0 ? 1 : 0),
    learn: (g) => g,
  }
}

// ── Linear threshold policies ────────────────────────────────────────────────────────────────────────────────────────

/** w · (o, 1): the last weight is the bias. */
function score(w: ArrayLike<number>, obs: ArrayLike<number>): number {
  let s = w[obs.length]
  for (let i = 0; i < obs.length; i++) s += w[i] * obs[i]
  return s
}

/** A fixed linear threshold policy: action 1 when w · (o, 1) > 0, else 0. It learns nothing. */
export function linearPolicyAgent(weights: readonly number[]): Agent<null, Float64Array, number> {
  return {
    name: 'linear policy',
    init: () => null,
    act: (_, obs) => {
      const s = score(weights, obs)
      return { action: s > 0 ? 1 : 0, scores: Float64Array.of(s) }
    },
    greedy: (_, obs) => (score(weights, obs) > 0 ? 1 : 0),
    learn: (g) => g,
  }
}

// ── The cross-entropy method ─────────────────────────────────────────────────────────────────────────────────────────

/** Options of `crossEntropyAgent`. */
export interface CrossEntropyOptions {
  /** Policies per generation (default 20). */
  population?: number
  /** The fraction kept as the elite (default 0.2). */
  eliteFraction?: number
  /** The initial standard deviation of every weight (default 0.3). */
  initialStd?: number
  /** Noise added to the refitted σ: noise / (1 + generation) (default 0.5). */
  noise?: number
}

/** Summary of one finished generation. */
export interface GenerationSummary {
  generation: number
  meanReturn: number
  minReturn: number
  maxReturn: number
  /** The mean return of the elite. */
  eliteReturn: number
}

/** The cross-entropy agent's state: the search distribution, the population being played, and the last generation. */
export interface CrossEntropyState {
  seed: number
  generation: number
  mean: number[]
  std: number[]
  /** This generation's weight vectors, each of length dim(o) + 1. */
  population: number[][]
  /** The member playing the current episode. */
  member: number
  /** The returns of the members played so far this generation. */
  returns: number[]
  /** The current episode's return so far. */
  episodeReturn: number
  /** The last finished generation, or null before the first. */
  last: GenerationSummary | null
}

/** Generation g's population, drawn from the seed: pure. */
function draw(seed: number, generation: number, mean: readonly number[], std: readonly number[], size: number) {
  const z = toFlat(normal(child(stream(seed), 'generation', generation), 0, 1, { shape: [size * mean.length] }))
  return Array.from({ length: size }, (_, k) => mean.map((m, i) => m + std[i] * z[k * mean.length + i]))
}

/** The cross-entropy method over linear threshold policies (module docs), for a box observation and two actions. */
export function crossEntropyAgent({
  population = 20,
  eliteFraction = 0.2,
  initialStd = 0.3,
  noise = 0.5,
}: CrossEntropyOptions = {}): Agent<CrossEntropyState, Float64Array, number> {
  const elite = Math.max(1, Math.round(population * eliteFraction))
  return {
    name: 'cross-entropy method',
    init: (env, s) => {
      if (env.action.kind !== 'discrete' || env.action.n !== 2)
        throw new TypeError(`crossEntropyAgent: ${env.name} needs two actions`)
      const dim = domainDimension(env.observation) + 1
      const seed = integers(s, 2 ** 31)
      const mean = new Array<number>(dim).fill(0)
      const std = new Array<number>(dim).fill(initialStd)
      return {
        seed,
        generation: 0,
        mean,
        std,
        population: draw(seed, 0, mean, std, population),
        member: 0,
        returns: [],
        episodeReturn: 0,
        last: null,
      }
    },
    act(g, obs) {
      const s = score(g.population[g.member], obs)
      return { action: s > 0 ? 1 : 0, scores: Float64Array.of(s) }
    },
    // The learnt policy is the search distribution's mean.
    greedy: (g, obs) => (score(g.mean, obs) > 0 ? 1 : 0),
    scalars: (g) => ({
      'generation mean return': g.last?.meanReturn ?? NaN,
      'elite mean return': g.last?.eliteReturn ?? NaN,
      generation: g.generation,
    }),
    learn(g, t) {
      const episodeReturn = g.episodeReturn + t.reward
      if (!(t.terminated || t.truncated)) return { ...g, episodeReturn }
      const returns = [...g.returns, episodeReturn]
      if (returns.length < population) return { ...g, returns, member: g.member + 1, episodeReturn: 0 }
      // The generation is complete: refit to the elite.
      const order = returns.map((_, k) => k).sort((a, b) => returns[b] - returns[a])
      const top = order.slice(0, elite).map((k) => g.population[k])
      const extra = noise / (1 + g.generation)
      const mean = g.mean.map((_, i) => top.reduce((s, w) => s + w[i], 0) / elite)
      const std = mean.map((m, i) => Math.sqrt(top.reduce((s, w) => s + (w[i] - m) ** 2, 0) / elite) + extra)
      const generation = g.generation + 1
      const sum = returns.reduce((a, b) => a + b, 0)
      return {
        ...g,
        generation,
        mean,
        std,
        population: draw(g.seed, generation, mean, std, population),
        member: 0,
        returns: [],
        episodeReturn: 0,
        last: {
          generation: g.generation,
          meanReturn: sum / population,
          minReturn: Math.min(...returns),
          maxReturn: Math.max(...returns),
          eliteReturn: order.slice(0, elite).reduce((s, k) => s + returns[k], 0) / elite,
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const agent = definer<AgentInfo>('agent', 'gym/agents/control')

agent(
  {
    key: 'lqrBangBangAgent',
    name: 'LQR bang-bang',
    summary:
      'Pushes in the direction of the LQR force about the origin, from autodiff Jacobians of the model: a two-action baseline.',
    params: space({ r: real(1e-3, 100, { default: 1, scale: 'log', label: 'force weight R' }) }),
    requires: { observation: 'box', action: 'discrete', model: 'dynamics', families: ['control'] },
    notes: ['linear-quadratic-regulator'],
  },
  lqrBangBangAgent,
)

agent(
  {
    key: 'crossEntropyAgent',
    name: 'Cross-entropy method',
    summary:
      'Searches linear threshold policies by the cross-entropy method: sample a population, keep the elite, refit, repeat.',
    params: space({
      population: int(4, 200, { default: 20, label: 'population' }),
      eliteFraction: real(0.05, 0.5, { default: 0.2, label: 'elite fraction' }),
      initialStd: real(0.05, 5, { default: 0.3, label: 'initial σ' }),
      noise: real(0, 2, { default: 0.5, label: 'extra noise' }),
    }),
    requires: { observation: 'box', action: 'discrete', families: ['control'] },
    notes: ['reinforcement-learning'],
    random: true,
  },
  crossEntropyAgent,
)
