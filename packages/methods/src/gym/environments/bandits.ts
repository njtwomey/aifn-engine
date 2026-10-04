/**
 * Bandit environments (docs/aifn-gym.md §5): Bernoulli and Gaussian multi-armed bandits and a linear contextual bandit,
 * each an `Environment` with horizon 1 whose action is an arm. Only the pulled arm's reward is revealed, but every
 * round's `step` draws every arm's reward from its stream, in arm order, so the pulled arm's reward does not depend on
 * which arm an agent pulls: two agents run on the same stream see the same reward for the same arm in the same round
 * (common random numbers), and differences between their curves come from the agents. The oracle knows the arms'
 * expected rewards, from which the rollout scores pseudo-regret.
 */

import type { Environment, EnvironmentInfo } from 'aifn-compute/foundation/contracts'
import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { boxDomain, discreteDomain, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const read = (v: Tensor | readonly number[]) => Float64Array.from(isTensor(v) ? toFlat(v) : v)

/** A context-free bandit: one state (0), one observation (0), an arm per action. */
export type BanditEnvironment = Environment<number, number, number>

/** A context-free bandit over arms whose round draws every arm's reward with `draw(stream)` (in arm order). */
function armBandit(name: string, mu: Float64Array, draw: (s: Stream) => Float64Array) {
  let best = -Infinity
  for (const m of mu) best = Math.max(best, m)
  const env: BanditEnvironment = {
    name,
    observation: discreteDomain(1),
    action: discreteDomain(
      mu.length,
      Array.from(mu, (_, a) => `arm ${a + 1}`),
    ),
    gamma: 1,
    horizon: 1,
    reset: () => ({ state: 0, observation: 0 }),
    step: (_, a, s) => ({ state: 0, observation: 0, reward: draw(s)[a], terminated: true, truncated: false }),
    oracle: { expectedReward: (_, a) => mu[a], bestExpectedReward: () => best },
  }
  return env
}

/** Options for `bernoulliBandit`. */
export interface BernoulliBanditOptions {
  /** P(reward = 1) per arm. Default [0.3, 0.5, 0.7]. */
  means?: Tensor | readonly number[]
}

/** A Bernoulli bandit: arm a pays 1 with probability μ_a and 0 otherwise. */
export function bernoulliBandit({ means = [0.3, 0.5, 0.7] }: BernoulliBanditOptions = {}): BanditEnvironment {
  const mu = read(means)
  if (mu.some((p) => !(p >= 0 && p <= 1)))
    throw new DomainError('bernoulliBandit', 'bernoulliBandit: means must be probabilities')
  return armBandit('Bernoulli bandit', mu, (s) => mu.map((p) => (uniform(s) < p ? 1 : 0)))
}

/** Options for `gaussianBandit`. */
export interface GaussianBanditOptions {
  /** The arms' means. Default [0, 0.5, 1]. */
  means?: Tensor | readonly number[]
  /** The rewards' standard deviation, one for all arms or one per arm. Default 1. */
  sd?: number | readonly number[]
}

/** A Gaussian bandit: arm a pays N(μ_a, σ_a²). */
export function gaussianBandit({ means = [0, 0.5, 1], sd = 1 }: GaussianBanditOptions = {}): BanditEnvironment {
  const mu = read(means)
  const sds = typeof sd === 'number' ? new Float64Array(mu.length).fill(sd) : Float64Array.from(sd)
  return armBandit('Gaussian bandit', mu, (s) => mu.map((m, a) => m + sds[a] * normal(s)))
}

/** Options for `linearBandit`. */
export interface LinearBanditOptions {
  /** The true parameter θ*. Default [1, 0.5]. */
  theta?: readonly number[]
  /** Arms per round. Default 5. */
  arms?: number
  /**
   * `fixed`: the same unit vectors every round, evenly spread in angle (2-D) or the coordinate axes and their negatives;
   * `random`: fresh arms each round, with random directions and lengths in [0.5, 1] (the contextual setting).
   */
  mode?: 'fixed' | 'random'
  /** Standard deviation of the Gaussian reward noise. Default 0.3. */
  noise?: number
}

/**
 * A linear bandit: arm x pays xᵀθ* + ε with ε ~ N(0, noise²) (Abbasi-Yadkori, Pál and Szepesvári, 2011, NeurIPS). The
 * round's arms are its state and observation, a flat arms × d array (row a is arm a's features) in the box [−1, 1];
 * `reset` draws them (random mode) and `step` draws every arm's noise.
 */
export function linearBandit({
  theta: th = [1, 0.5],
  arms: k = 5,
  mode = 'fixed',
  noise = 0.3,
}: LinearBanditOptions = {}): Environment<Float64Array, Float64Array, number> {
  const theta = Float64Array.from(th)
  const d = theta.length
  const fixed = new Float64Array(k * d)
  for (let i = 0; i < k; i++) {
    if (d === 2) {
      const angle = (2 * Math.PI * i) / k + 0.3
      fixed[i * d] = Math.cos(angle)
      fixed[i * d + 1] = Math.sin(angle)
    } else fixed[i * d + (i % d)] = i < d ? 1 : -1
  }
  const means = (x: Float64Array) =>
    Float64Array.from({ length: k }, (_, a) => {
      let m = 0
      for (let j = 0; j < d; j++) m += x[a * d + j] * theta[j]
      return m
    })
  return {
    name: 'linear bandit',
    observation: boxDomain(-1, 1, { shape: [k, d] }),
    action: discreteDomain(k),
    gamma: 1,
    horizon: 1,
    reset(s) {
      if (mode === 'fixed') return { state: fixed, observation: fixed }
      const x = new Float64Array(k * d)
      for (let a = 0; a < k; a++) {
        const v = Float64Array.from({ length: d }, () => normal(s))
        const norm = Math.hypot(...v)
        const length = 0.5 + 0.5 * uniform(s)
        for (let j = 0; j < d; j++) x[a * d + j] = (length * v[j]) / norm
      }
      return { state: x, observation: x }
    },
    step(x, a, s) {
      const rewards = means(x).map((m) => m + noise * normal(s))
      return { state: x, observation: x, reward: rewards[a], terminated: true, truncated: false }
    },
    oracle: {
      expectedReward: (x, a) => means(x)[a],
      bestExpectedReward: (x) => Math.max(...means(x)),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const environment = definer<EnvironmentInfo>('environment', 'gym/environments')
const arms = { observation: 'discrete', action: 'discrete', capabilities: ['oracle'] } as const

environment(
  {
    key: 'bernoulliBandit',
    name: 'Bernoulli bandit',
    summary: 'Arms paying 1 with their mean probability, else 0 (means default 0.3, 0.5, 0.7).',
    family: 'bandit',
    params: space({}),
    ...arms,
    notes: ['multi-armed-bandit', 'thompson-sampling'],
  },
  bernoulliBandit,
)

environment(
  {
    key: 'gaussianBandit',
    name: 'Gaussian bandit',
    summary: 'Arms paying Gaussian rewards around their means (default 0, 0.5, 1; sd 1).',
    family: 'bandit',
    params: space({ sd: real(0.01, 5, { default: 1 }) }),
    ...arms,
    notes: ['multi-armed-bandit', 'upper-confidence-bound-algorithm'],
  },
  gaussianBandit,
)

environment(
  {
    key: 'linearBandit',
    name: 'Linear bandit',
    summary: 'Arms are feature vectors whose mean reward is linear in an unknown θ* (default θ* = (1, 0.5)).',
    family: 'contextual-bandit',
    params: space({
      arms: int(2, 50, { default: 5 }),
      mode: oneOf(['fixed', 'random']),
      noise: real(0, 3, { default: 0.3 }),
    }),
    observation: 'box',
    action: 'discrete',
    capabilities: ['oracle'],
    notes: ['contextual-bandit', 'linucb', 'linear-thompson-sampling'],
  },
  linearBandit,
)
