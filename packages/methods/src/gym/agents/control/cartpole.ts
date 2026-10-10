/**
 * Agents for two-action control problems such as the cart-pole (push left = 0, push right = 1):
 *
 * - **`lqrBangBangAgent`**: linearise the environment's differentiable dynamics about the origin
 *   ($\xvec = \zeros$, $u = 0$) by autodiff (`lineariseDynamics`), solve the discrete-time LQR (`dlqr`,
 *   $\Qmat = \Imat$, $\Rmat = r$) and push in the direction of the LQR force $u = -\Kmat\xvec$. Bang-bang control with
 *   the LQR's switching surface: no learning, a model-based baseline.
 * - **`crossEntropyAgent`**: the cross-entropy method over linear threshold policies
 *   $a = [\wvec \cdot (\ovec, 1) > 0]$ (Szita and Lőrincz, 2006, "Learning Tetris using the noisy cross-entropy
 *   method", Neural Computation 18(12)). Each generation draws a population of weight vectors from
 *   $\Gauss(\muvec, \diag \sigmavec^2)$, plays one episode with each, keeps the elite fraction with the highest
 *   returns, and refits $\muvec$ and $\sigmavec$ to them, adding a decaying noise to $\sigmavec$ so that the search
 *   does not collapse early. It is an episode-level learner on the `Agent` protocol: `act` plays the current member,
 *   `learn` sums its return and refits after the last member. The state is plain data (it checkpoints), and the
 *   population of each generation is drawn from a seed fixed in `init`, so `learn` stays pure.
 * - **`linearPolicyAgent`**: one fixed linear threshold policy, to play a given weight vector.
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
  /** The LQR's force weight $\Rmat$ (default 1); the state weight is $\Qmat = \Imat$. */
  r?: number
}

/** The bang-bang agent's state: the LQR gain $\Kmat$ on the observed state. */
export interface LqrBangBangState {
  /** The gain $\Kmat$, one entry per state variable: the LQR force is $u = -\Kmat\ovec$. */
  K: readonly number[]
}

/**
 * Pushes right (action 1) when the LQR force $-\Kmat\ovec$ about the origin is positive, else left (action 0); the
 * force is the decision's `scores`. `init` needs a `dynamics` model whose state is the observation and whose action is
 * a scalar force, or it throws `TypeError`. It learns nothing.
 *
 * @param options The LQR's weights.
 * @param options.r The force weight $\Rmat$ against the state weight $\Qmat = \Imat$.
 * @returns The agent, named `'LQR bang-bang'`.
 *
 * @example Bang-bang control of a cart on a frictionless track
 * // State (position, velocity), force u, dt = 0.1: a double integrator. Action 1 pushes with +1, action 0 with -1.
 * const A = tensor([[1, 0.1], [0, 1]])
 * const B = tensor([0, 0.1])
 * const model = { kind: 'dynamics', stateSize: 2, actionSize: 1, transition: (x, u) => add(matmul(A, x), mul(B, u)) }
 * const agent = lqrBangBangAgent()
 * const g = agent.init({ name: 'cart', model }, stream(0))
 * print('K =', g.K)
 * let p = 1
 * let v = 0
 * for (let t = 0; t < 80; t++) {
 *   const u = agent.act(g, Float64Array.of(p, v), stream(t)).action === 1 ? 1 : -1
 *   p += 0.1 * v
 *   v += 0.1 * u
 * }
 * print('position and velocity after 8 s:', [p, v])
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

/**
 * $\wvec \cdot (\ovec, 1)$: the last weight is the bias.
 *
 * @param w The weights: one per observation entry, then the bias.
 * @param obs The observation $\ovec$.
 * @returns The score; the policy pushes right when it is positive.
 */
function score(w: ArrayLike<number>, obs: ArrayLike<number>): number {
  let s = w[obs.length]
  for (let i = 0; i < obs.length; i++) s += w[i] * obs[i]
  return s
}

/**
 * A fixed linear threshold policy: action 1 when $\wvec \cdot (\ovec, 1) > 0$, else 0; the score is the decision's
 * `scores`. It learns nothing, and its state is null.
 *
 * @param weights The weights $\wvec$: one per observation entry, then the bias.
 * @returns The agent, named `'linear policy'`.
 *
 * @example A threshold on the first observation
 * const agent = linearPolicyAgent([1, 0, -0.5])
 * const g = agent.init({}, stream(0))
 * print('at (1, 3):', agent.act(g, Float64Array.of(1, 3), stream(1)))
 * print('at (0, 3):', agent.act(g, Float64Array.of(0, 3), stream(2)).action)
 */
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
  /** Noise added to the refitted $\sigmavec$: `noise` $/ (1 + \text{generation})$ (default 0.5). */
  noise?: number
}

/** Summary of one finished generation. */
export interface GenerationSummary {
  /** The generation's number, from 0. */
  generation: number
  /** The mean return of its members. */
  meanReturn: number
  /** The lowest return of a member. */
  minReturn: number
  /** The highest return of a member. */
  maxReturn: number
  /** The mean return of the elite. */
  eliteReturn: number
}

/** The cross-entropy agent's state: the search distribution, the population being played, and the last generation. */
export interface CrossEntropyState {
  /** The seed every generation's population is drawn from, fixed in `init`. */
  seed: number
  /** The current generation's number, from 0. */
  generation: number
  /** The search distribution's mean $\muvec$: also the learnt policy's weights. */
  mean: number[]
  /** The search distribution's standard deviations $\sigmavec$. */
  std: number[]
  /** This generation's weight vectors, each of length $\dim(\ovec) + 1$. */
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

/**
 * Generation $g$'s population, drawn from the seed: pure. Member $k$'s weight $i$ is
 * $\mu_i + \sigma_i z_{ki}$ with $z$ standard normal draws.
 *
 * @param seed The seed fixed in `init`.
 * @param generation The generation $g$, which names the child stream drawn from.
 * @param mean The mean $\muvec$ of every weight.
 * @param std The standard deviation $\sigmavec$ of every weight.
 * @param size The number of members.
 * @returns `size` weight vectors.
 */
function draw(seed: number, generation: number, mean: readonly number[], std: readonly number[], size: number) {
  const z = toFlat(normal(child(stream(seed), 'generation', generation), 0, 1, { shape: [size * mean.length] }))
  return Array.from({ length: size }, (_, k) => mean.map((m, i) => m + std[i] * z[k * mean.length + i]))
}

/**
 * The cross-entropy method over linear threshold policies (module docs), for a box observation and two actions
 * (`init` throws `TypeError` otherwise). The weights start at $\muvec = \zeros$ and `initialStd`; `greedy` plays the
 * mean $\muvec$, and `scalars` report the last generation's mean and elite returns and the generation.
 *
 * @param options The population and the refit.
 * @param options.population The members per generation: the episodes a generation takes.
 * @param options.eliteFraction The share of the population refitted to; at least one member.
 * @param options.initialStd The starting standard deviation of every weight.
 * @param options.noise The extra spread $\nu$ added to every refitted standard deviation as $\nu / (1 + g)$ after
 *   generation $g$.
 * @returns The agent, named `'cross-entropy method'`.
 *
 * @example Five generations on a ten-step task
 * // Each step the observation is -1 or 1, and the action matching its sign (0 or 1) pays 1; episodes last 10 steps.
 * const agent = crossEntropyAgent({ population: 10 })
 * const env = { name: 'sign', observation: { kind: 'box', shape: [1], low: [-1], high: [1] } }
 * let g = agent.init({ ...env, action: { kind: 'discrete', n: 2 } }, stream(0))
 * const s = stream(1)
 * for (let episode = 0; episode < 50; episode++)
 *   for (let t = 0; t < 10; t++) {
 *     const o = Float64Array.of(uniform(s) < 0.5 ? -1 : 1)
 *     const action = agent.act(g, o, s).action
 *     const reward = action === (o[0] > 0 ? 1 : 0) ? 1 : 0
 *     g = agent.learn(g, { observation: o, action, reward, next: o, terminated: t === 9, truncated: false })
 *   }
 * print('generations:', g.generation)
 * print('last generation:', g.last)
 * print('mean weights (o, bias):', g.mean)
 */
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
