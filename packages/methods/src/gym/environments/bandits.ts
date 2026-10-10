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

/**
 * A tensor or array of numbers as a fresh `Float64Array`, in row-major order.
 *
 * @param v The values: a tensor (flattened) or a plain array.
 * @returns A copy of the values.
 */
const read = (v: Tensor | readonly number[]) => Float64Array.from(isTensor(v) ? toFlat(v) : v)

/** A context-free bandit: one state (0), one observation (0), an arm per action, horizon 1. */
export type BanditEnvironment = Environment<number, number, number>

/**
 * A context-free bandit over arms whose round draws every arm's reward with `draw(stream)` (in arm order) and pays
 * the pulled arm's. The oracle reports each arm's mean and the best mean, for pseudo-regret.
 *
 * @param name The environment's readable name.
 * @param mu The arms' expected rewards $\mu_a$, one per arm; their count is the number of actions.
 * @param draw Draws one reward per arm from the stream, in arm order (a vector as long as `mu`).
 * @returns The bandit, with arms named `arm 1`, `arm 2`, ...
 */
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
  /** $\Pr(r = 1)$ per arm, each in $[0, 1]$. Default `[0.3, 0.5, 0.7]`. */
  means?: Tensor | readonly number[]
}

/**
 * A Bernoulli bandit: arm $a$ pays 1 with probability $\mu_a$ and 0 otherwise, $r \sim \Bern(\mu_a)$. Throws
 * `DomainError` when a mean is not a probability.
 *
 * @param options The arms.
 * @param options.means The success probability $\mu_a$ of each arm, as an array or a tensor; their count is the
 *   number of arms.
 * @returns The bandit: action $a$ pulls arm $a + 1$, and the oracle knows the means.
 *
 * @example Pull each arm of a two-armed bandit a few times
 * const env = bernoulliBandit({ means: [0.2, 0.9] })
 * const s = stream(0)
 * const { state } = env.reset(s)
 * print('arm 1 pays', [1, 2, 3, 4, 5].map(() => env.step(state, 0, s).reward))
 * print('arm 2 pays', [1, 2, 3, 4, 5].map(() => env.step(state, 1, s).reward))
 * print('best expected reward', env.oracle.bestExpectedReward(state))
 */
export function bernoulliBandit({ means = [0.3, 0.5, 0.7] }: BernoulliBanditOptions = {}): BanditEnvironment {
  const mu = read(means)
  if (mu.some((p) => !(p >= 0 && p <= 1)))
    throw new DomainError('bernoulliBandit', 'bernoulliBandit: means must be probabilities')
  return armBandit('Bernoulli bandit', mu, (s) => mu.map((p) => (uniform(s) < p ? 1 : 0)))
}

/** Options for `gaussianBandit`. */
export interface GaussianBanditOptions {
  /** The arms' means $\mu_a$. Default `[0, 0.5, 1]`. */
  means?: Tensor | readonly number[]
  /** The rewards' standard deviation $\sigma_a$, one for all arms or one per arm. Default 1. */
  sd?: number | readonly number[]
}

/**
 * A Gaussian bandit: arm $a$ pays $r \sim \Gauss(\mu_a, \sigma_a^2)$.
 *
 * @param options The arms.
 * @param options.means The mean $\mu_a$ of each arm, as an array or a tensor; their count is the number of arms.
 * @param options.sd The standard deviation $\sigma_a$: one number for every arm, or one per arm.
 * @returns The bandit: action $a$ pulls arm $a + 1$, and the oracle knows the means.
 *
 * @example Noisy rewards around each arm's mean
 * const env = gaussianBandit({ means: [0, 1], sd: 0.1 })
 * const s = stream(0)
 * const { state } = env.reset(s)
 * print('arm 1 pays', env.step(state, 0, s).reward)
 * print('arm 2 pays', env.step(state, 1, s).reward)
 * print('expected', env.oracle.expectedReward(state, 0), env.oracle.expectedReward(state, 1))
 */
export function gaussianBandit({ means = [0, 0.5, 1], sd = 1 }: GaussianBanditOptions = {}): BanditEnvironment {
  const mu = read(means)
  const sds = typeof sd === 'number' ? new Float64Array(mu.length).fill(sd) : Float64Array.from(sd)
  return armBandit('Gaussian bandit', mu, (s) => mu.map((m, a) => m + sds[a] * normal(s)))
}

/** Options for `linearBandit`. */
export interface LinearBanditOptions {
  /** The true parameter $\thetavec^*$, of length $d$ (the arms' dimension). Default `[1, 0.5]`. */
  theta?: readonly number[]
  /** Arms per round, $k$. Default 5. */
  arms?: number
  /**
   * `fixed` (default): the same unit vectors every round, evenly spread in angle when $d = 2$, else the coordinate axes
   * and then their negatives; `random`: fresh arms each round, with uniformly random directions and lengths uniform on
   * $[0.5, 1]$ (the contextual setting).
   */
  mode?: 'fixed' | 'random'
  /** Standard deviation $\sigma$ of the Gaussian reward noise. Default 0.3. */
  noise?: number
}

/**
 * A linear bandit: arm $\xvec$ pays $\xvec^\top\thetavec^* + \varepsilon$ with $\varepsilon \sim \Gauss(0, \sigma^2)$
 * (Abbasi-Yadkori, Pál and Szepesvári, 2011, NeurIPS). The round's arms are its state and observation, a flat
 * $k \times d$ array (row $a$ is arm $a$'s features) in the box $[-1, 1]$; `reset` draws them (random mode) and `step`
 * draws every arm's noise. The oracle knows each arm's mean $\xvec_a^\top\thetavec^*$.
 *
 * @param options The parameter, the arms and the noise.
 * @param options.theta The true parameter $\thetavec^*$; its length is the arms' dimension $d$.
 * @param options.arms The number of arms per round, $k$.
 * @param options.mode `fixed` for the same arms every round, `random` for fresh arms drawn at each reset.
 * @param options.noise The standard deviation $\sigma$ of the reward noise.
 * @returns The bandit, whose state and observation are the round's $k \times d$ arm features, row-major.
 *
 * @example The fixed arms and their mean rewards
 * const env = linearBandit({ theta: [1, 0], arms: 4, noise: 0 })
 * const { state } = env.reset(stream(0))
 * print('arms (rows)', tensor(Array.from(state), [4, 2]))
 * print('means', [0, 1, 2, 3].map((a) => env.oracle.expectedReward(state, a)))
 * print('arm 1 pays', env.step(state, 0, stream(1)).reward)
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

/** Registers an environment factory of `gym/environments`. */
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
