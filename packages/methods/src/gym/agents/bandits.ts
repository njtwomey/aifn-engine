/**
 * Bandit policies as `Agent`s (docs/aifn-gym.md §5), with the KL-UCB index and the Lai–Robbins lower bound on regret.
 *
 * `init` builds the policy's statistics from the action domain (one arm per action, $K$ arms; a domain that is not
 * discrete throws `DomainError`), `act` picks an arm and reports the index, sample or estimate of every arm it
 * maximised (`scores`) and, for randomised policies, its probabilities, and `learn` records the pulled arm's reward.
 * States are plain data, and `learn` returns a new one. The round number $t$ is one more than the pulls so far. Ties
 * between the best arms are broken uniformly at random in `act`, and towards the first arm in `greedy`. The linear
 * policies read the round's context, a flat $K \times d$ observation (row $a$ holds arm $a$'s features), and take $d$
 * from the observation domain's shape.
 */

import type { Agent, AgentInfo, Decision, EnvironmentShape, Transition } from 'aifn-compute/foundation/contracts'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { beta as betaDraw } from 'aifn-compute/probability/samplers'
import { integers, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Pull counts and reward sums per arm: the state of every context-free policy. */
export interface ArmStatistics {
  /** The number of pulls of each arm, by arm index. */
  counts: Float64Array
  /** The sum of the rewards each arm has returned, by arm index. */
  sums: Float64Array
}

/**
 * The index of the largest score, ties broken uniformly at random (a draw is made only when there is a tie).
 *
 * @param scores One score per arm.
 * @param s The stream the tie-break draws from.
 * @returns The chosen arm.
 */
function argmaxRandom(scores: Float64Array, s: Stream): number {
  let best = -Infinity
  let ties: number[] = []
  scores.forEach((v, i) => {
    if (v > best) {
      best = v
      ties = [i]
    } else if (v === best) ties.push(i)
  })
  return ties.length === 1 ? ties[0] : ties[integers(s, ties.length)]
}

/**
 * The number of arms: the size of the environment's discrete action domain. Throws `DomainError` for any other domain.
 *
 * @param env The environment the policy is initialised for.
 * @returns The number of actions $K$.
 */
function armCount(env: EnvironmentShape): number {
  if (env.action.kind !== 'discrete')
    throw new DomainError('armCount', 'bandit policies need a discrete action domain (the arms)')
  return env.action.n
}

/**
 * Empty statistics: no pulls and no reward for every arm.
 *
 * @param arms The number of arms $K$.
 * @returns Zeroed `counts` and `sums` of $K$ entries each.
 */
const stats = (arms: number): ArmStatistics => ({ counts: new Float64Array(arms), sums: new Float64Array(arms) })

/**
 * The statistics after one more pull: the arm's count rises by 1 and its sum by the reward.
 *
 * @param st The statistics before the pull, with any further fields of the policy's state; not modified.
 * @param arm The arm pulled.
 * @param reward The reward it returned.
 * @returns A copy of `st` with new `counts` and `sums`; other fields are carried over as they are.
 */
function record<T extends ArmStatistics>(st: T, arm: number, reward: number): T {
  const counts = st.counts.slice()
  const sums = st.sums.slice()
  counts[arm] += 1
  sums[arm] += reward
  return { ...st, counts, sums }
}

/**
 * The first index of the largest score: a deterministic greedy choice.
 *
 * @param scores One score per arm.
 * @returns The lowest index holding the largest score.
 */
const firstMax = (scores: ArrayLike<number>) => {
  let arg = 0
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[arg]) arg = i
  return arg
}

/**
 * The share of pulls that went to the most-pulled arm: how settled a policy is, a training-curve scalar.
 *
 * @param st The policy's statistics.
 * @returns `top arm share`, the largest count over the total (0 before any pull).
 */
const armScalars = (st: ArmStatistics) => {
  const n = st.counts.reduce((a, b) => a + b, 0)
  return { 'top arm share': n > 0 ? Math.max(...st.counts) / n : 0 }
}

/**
 * The round $t = 1, 2, \dots$ about to be played: one more than the pulls so far.
 *
 * @param st The policy's statistics.
 * @returns The round number $t$.
 */
const round = (st: ArmStatistics) => st.counts.reduce((a, b) => a + b, 0) + 1
/**
 * The empirical mean reward $\hat{\mu}_a$ of every arm, 0 for an arm never pulled.
 *
 * @param st The policy's statistics.
 * @returns One mean per arm.
 */
const means = (st: ArmStatistics) => st.counts.map((c, i) => (c > 0 ? st.sums[i] / c : 0))
/**
 * The first arm never pulled.
 *
 * @param st The policy's statistics.
 * @returns Its index, or $-1$ once every arm has been pulled.
 */
const firstUnpulled = (st: ArmStatistics) => st.counts.findIndex((c) => c === 0)

/**
 * A context-free policy on arm statistics: `choose(st, t, s)` picks in round $t$; `learn` records the reward, and
 * `greedy` is the first arm of largest empirical mean.
 *
 * @param name The agent's readable name.
 * @param choose The decision of a round from the statistics, the round number $t$ and the act's stream.
 * @returns The agent, whose state is the `ArmStatistics`.
 */
function armAgent(
  name: string,
  choose: (st: ArmStatistics, t: number, s: Stream) => Decision<number>,
): Agent<ArmStatistics, unknown, number> {
  return {
    name,
    init: (env) => stats(armCount(env)),
    act: (st, _, s) => choose(st, round(st), s),
    greedy: (st) => firstMax(means(st)),
    scalars: armScalars,
    learn: (st, tr) => record(st, tr.action, tr.reward),
  }
}

/**
 * An index policy: pulls every arm once, in order, then the arm with the largest index (ties at random). While arms
 * remain unpulled, the scores are $\infty$ for those and $-\infty$ for the rest.
 *
 * @param name The agent's readable name.
 * @param index The index of every arm from the statistics and the round number $t$; called only once every arm has a
 *   pull, so it may divide by the counts.
 * @returns The agent, whose state is the `ArmStatistics`.
 */
function indexAgent(name: string, index: (st: ArmStatistics, t: number) => Float64Array) {
  return armAgent(name, (st, t, s) => {
    const first = firstUnpulled(st)
    const scores = first >= 0 ? st.counts.map((c) => (c === 0 ? Infinity : -Infinity)) : index(st, t)
    return { action: first >= 0 ? first : argmaxRandom(scores, s), scores }
  })
}

/**
 * Round robin over the arms (an A/B/n test): in round $t$ it pulls arm $(t - 1) \bmod K$, whatever the rewards. Its
 * `scores` are the empirical means, and `greedy` the arm with the best of them. Regret grows linearly: the baseline of
 * the bandit policies.
 *
 * @returns The agent, named `'uniform'`.
 *
 * @example Every arm gets the same share
 * const means = [0.2, 0.5, 0.8]
 * const agent = uniformPolicy()
 * const s = stream(1)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 300; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 300)
 * print('greedy arm:', agent.greedy(g))
 */
export function uniformPolicy(): Agent<ArmStatistics, unknown, number> {
  return armAgent('uniform', (st, t) => ({ action: (t - 1) % st.counts.length, scores: means(st) }))
}

/**
 * Explore then commit: $m$ pulls of every arm in turn (rounds $t \le mK$), then the arm of best empirical mean. The
 * means are recomputed every round, so after exploring it plays greedily: it stays with its choice unless that arm's
 * mean falls below another's.
 *
 * @param options The length of the exploration phase.
 * @param options.m The pulls of every arm before committing.
 * @returns The agent, named after $m$.
 *
 * @example Explore 10 pulls an arm, then commit
 * const means = [0.2, 0.5, 0.8]
 * const agent = exploreThenCommit({ m: 10 })
 * const s = stream(2)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 300; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 300)
 */
export function exploreThenCommit({ m = 50 }: { m?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return armAgent(`explore-then-commit (m = ${m})`, (st, t, s) => {
    const k = st.counts.length
    const scores = means(st)
    return { action: t <= m * k ? (t - 1) % k : argmaxRandom(scores, s), scores }
  })
}

/**
 * $\varepsilon$-greedy: every arm once, in order, then with probability $\varepsilon$ a uniformly random arm (which may
 * be the best) and otherwise the best empirical mean. With `decay: c` the exploration rate is $\min(1, cK/t)$ (Auer,
 * Cesa-Bianchi and Fischer, 2002, Machine Learning 47, §3). Its decision reports the probabilities
 * $\varepsilon / K$ for every arm plus $1 - \varepsilon$ for the greedy one.
 *
 * @param options The exploration rate, fixed or decaying.
 * @param options.epsilon The fixed exploration rate $\varepsilon$, used when `decay` is not given.
 * @param options.decay The constant $c$ of the decaying rate $\min(1, cK/t)$; when given, `epsilon` is ignored.
 * @returns The agent, named after its rate.
 *
 * @example Action counts of an epsilon-greedy agent
 * const means = [0.2, 0.5, 0.8]
 * const agent = epsilonGreedy({ epsilon: 0.1 })
 * const s = stream(3)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 * print('probabilities now:', agent.act(g, 0, s).probabilities)
 */
export function epsilonGreedy({ epsilon = 0.1, decay }: { epsilon?: number; decay?: number } = {}): Agent<
  ArmStatistics,
  unknown,
  number
> {
  return armAgent(
    decay === undefined ? `ε-greedy (ε = ${epsilon})` : `decaying ε-greedy (c = ${decay})`,
    (st, t, s) => {
      const k = st.counts.length
      const first = firstUnpulled(st)
      const scores = means(st)
      if (first >= 0) return { action: first, scores }
      const eps = decay === undefined ? epsilon : Math.min(1, (decay * k) / t)
      const greedy = argmaxRandom(scores, s)
      const probabilities = new Float64Array(k).fill(eps / k)
      probabilities[greedy] += 1 - eps
      return { action: uniform(s) < eps ? integers(s, k) : greedy, scores, probabilities }
    },
  )
}

/**
 * UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): every arm once, then the arm of largest index
 * $\hat{\mu}_a + \sqrt{c \ln t / n_a}$, for rewards in $[0, 1]$; $\hat{\mu}_a$ is arm $a$'s empirical mean and $n_a$
 * its pulls. The indices are the decision's `scores`.
 *
 * @param options The width of the confidence bonus.
 * @param options.c The constant $c$ of the bonus; 2 is Auer et al.'s UCB1.
 * @returns The agent, named `'UCB1'`.
 *
 * @example UCB1 finds the best of three Bernoulli arms
 * const means = [0.2, 0.5, 0.8]
 * const agent = ucb1()
 * const s = stream(4)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 * print('indices now:', agent.act(g, 0, s).scores)
 */
export function ucb1({ c = 2 }: { c?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return indexAgent('UCB1', (st, t) => st.counts.map((n, i) => st.sums[i] / n + Math.sqrt((c * Math.log(t)) / n)))
}

/**
 * $\KL(\Bern(p) \,\Vert\, \Bern(q)) = p \log(p/q) + (1 - p) \log((1 - p)/(1 - q))$ in nats, with $0 \log 0 = 0$;
 * infinite when $q$ is 0 or 1 and $p$ is not.
 *
 * @param p The success probability of the first distribution, in $[0, 1]$.
 * @param q The success probability of the second, in $[0, 1]$.
 * @returns The divergence, $\ge 0$ (0 when $p = q$).
 *
 * @example The divergence grows as q moves from p
 * print('KL(0.5, 0.5) =', klBernoulli(0.5, 0.5))
 * print('KL(0.5, 0.6) =', klBernoulli(0.5, 0.6))
 * print('KL(0.5, 0.9) =', klBernoulli(0.5, 0.9))
 * print('KL(0.5, 1) =', klBernoulli(0.5, 1))
 */
export function klBernoulli(p: number, q: number): number {
  const term = (x: number, y: number) => (x === 0 ? 0 : y === 0 ? Infinity : x * Math.log(x / y))
  return term(p, q) + term(1 - p, 1 - q)
}

/**
 * The KL-UCB index: the largest $q \in [p, 1]$ with $n \KL(p, q) \le \text{level}$ (Garivier and Cappé, 2011, COLT),
 * where $\KL$ is `klBernoulli`, by bisection to $10^{-10}$ (at most 60 halvings).
 *
 * @param p The arm's empirical mean, in $[0, 1]$.
 * @param n The arm's number of pulls.
 * @param level The exploration level, $\ln t$ for KL-UCB.
 * @returns The index $q$: 1 when even $q = 1$ is within the level, else the bisection's lower end.
 *
 * @example The index shrinks towards the mean as pulls accumulate
 * const level = Math.log(1000)
 * print('n = 10:', klUcbIndex(0.5, 10, level))
 * print('n = 100:', klUcbIndex(0.5, 100, level))
 * print('n = 1000:', klUcbIndex(0.5, 1000, level))
 */
export function klUcbIndex(p: number, n: number, level: number): number {
  let lo = p
  let hi = 1
  if (n * klBernoulli(p, hi) <= level) return 1
  for (let it = 0; it < 60 && hi - lo > 1e-10; it++) {
    const mid = (lo + hi) / 2
    if (n * klBernoulli(p, mid) <= level) lo = mid
    else hi = mid
  }
  return lo
}

/**
 * KL-UCB for rewards in $[0, 1]$ (Garivier and Cappé, 2011): every arm once, then the arm of largest index
 * `klUcbIndex` $(\hat{\mu}_a, n_a, \ln t + c \ln \ln t)$, with $\hat{\mu}_a$ arm $a$'s empirical mean and $n_a$ its
 * pulls. Asymptotically optimal for Bernoulli arms.
 *
 * @param options The exploration level.
 * @param options.c The constant $c$ of the $\ln \ln t$ term (applied from $t = 2$); 0 drops the term.
 * @returns The agent, named `'KL-UCB'`.
 *
 * @example KL-UCB on three Bernoulli arms
 * const means = [0.2, 0.5, 0.8]
 * const agent = klUcb()
 * const s = stream(5)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 */
export function klUcb({ c = 0 }: { c?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return indexAgent('KL-UCB', (st, t) => {
    const level = Math.log(t) + (c > 0 && t > 1 ? c * Math.log(Math.log(t)) : 0)
    return st.counts.map((n, i) => klUcbIndex(st.sums[i] / n, n, level))
  })
}

/**
 * Thompson sampling for Bernoulli rewards (Thompson, 1933, Biometrika 25; Agrawal and Goyal, 2012, COLT): draw
 * $\theta_a \sim \Beta(\alpha + s_a, \beta + n_a - s_a)$ for every arm, with $s_a$ its reward sum and $n_a$ its pulls,
 * and pull the largest draw. The draws are the decision's `scores`. Rewards in $(0, 1)$ count as fractional successes;
 * a reward above 1 can make the second parameter non-positive.
 *
 * @param options The $\Beta(\alpha, \beta)$ prior of every arm's success probability.
 * @param options.alpha The prior's $\alpha$: pseudo-successes.
 * @param options.beta The prior's $\beta$: pseudo-failures.
 * @returns The agent, named `'Thompson sampling'`.
 *
 * @example Thompson sampling on three Bernoulli arms
 * const means = [0.2, 0.5, 0.8]
 * const agent = thompsonBernoulli()
 * const s = stream(6)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 */
export function thompsonBernoulli({ alpha = 1, beta = 1 }: { alpha?: number; beta?: number } = {}): Agent<
  ArmStatistics,
  unknown,
  number
> {
  return armAgent('Thompson sampling', (st, _t, s) => {
    const scores = st.counts.map((n, i) => betaDraw(s, alpha + st.sums[i], beta + n - st.sums[i]))
    return { action: argmaxRandom(scores, s), scores }
  })
}

/**
 * Thompson sampling for Gaussian rewards with known noise standard deviation $\sigma$ and a $\Gauss(\mu_0, \tau_0^2)$
 * prior on each mean: the posterior of arm $a$ is $\Gauss(m_a, v_a)$ with $1/v_a = 1/\tau_0^2 + n_a/\sigma^2$ and
 * $m_a = v_a (\mu_0/\tau_0^2 + s_a/\sigma^2)$ ($n_a$ its pulls, $s_a$ its reward sum). It draws a mean from every
 * posterior (the decision's `scores`) and pulls the largest.
 *
 * @param options The prior and the noise.
 * @param options.priorMean The prior mean $\mu_0$ of every arm.
 * @param options.priorSd The prior standard deviation $\tau_0$.
 * @param options.noiseSd The rewards' noise standard deviation $\sigma$, taken as known.
 * @returns The agent, named `'Gaussian Thompson sampling'`.
 *
 * @example Gaussian Thompson sampling on arms with noisy rewards
 * const means = [0, 0.5, 1]
 * const agent = thompsonGaussian({ noiseSd: 1 })
 * const s = stream(7)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = means[a] + normal(s)
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 */
export function thompsonGaussian({ priorMean = 0, priorSd = 1, noiseSd = 1 } = {}): Agent<
  ArmStatistics,
  unknown,
  number
> {
  return armAgent('Gaussian Thompson sampling', (st, _t, s) => {
    const scores = st.counts.map((n, i) => {
      const precision = 1 / priorSd ** 2 + n / noiseSd ** 2
      const mean = (priorMean / priorSd ** 2 + st.sums[i] / noiseSd ** 2) / precision
      return mean + normal(s) / Math.sqrt(precision)
    })
    return { action: argmaxRandom(scores, s), scores }
  })
}

/** The state of EXP3: log-weights per arm, with counts and sums for display. */
export interface Exp3State extends ArmStatistics {
  /** The log-weight $\log w_a$ of each arm, 0 at the start. */
  logWeights: Float64Array
}

/**
 * EXP3 for adversarial rewards in $[0, 1]$ (Auer, Cesa-Bianchi, Freund and Schapire, 2002, SIAM J. Comput. 32(1)):
 * arm $a$ is drawn with probability $p_a = (1 - \gamma) w_a / \sum_b w_b + \gamma/K$, and the pulled arm's weight grows
 * by $\exp(\gamma \hat{r} / K)$ with the importance-weighted reward $\hat{r} = r / p_a$ ($p$ from the weights that
 * chose the arm, which `learn` recomputes). The probabilities are the decision's `scores` and `probabilities`, and
 * `greedy` is the arm of largest weight.
 *
 * @param options The exploration mix.
 * @param options.gamma The share $\gamma$ of uniform exploration, in $(0, 1]$; it also scales the weight update.
 * @returns The agent, named after $\gamma$.
 *
 * @example EXP3 shifts its weight to the best arm
 * const means = [0.2, 0.5, 0.8]
 * const agent = exp3({ gamma: 0.1 })
 * const s = stream(8)
 * let g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * let total = 0
 * for (let t = 0; t < 500; t++) {
 *   const a = agent.act(g, 0, s).action
 *   const r = uniform(s) < means[a] ? 1 : 0
 *   total += r
 *   g = agent.learn(g, { observation: 0, action: a, reward: r, next: 0, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('average reward:', total / 500)
 * print('probabilities now:', agent.act(g, 0, s).probabilities)
 */
export function exp3({ gamma = 0.1 }: { gamma?: number } = {}): Agent<Exp3State, unknown, number> {
  const probs = (lw: Float64Array) => {
    const k = lw.length
    const mx = Math.max(...lw)
    const w = lw.map((v) => Math.exp(v - mx))
    const total = w.reduce((a, b) => a + b, 0)
    return w.map((v) => (1 - gamma) * (v / total) + gamma / k)
  }
  return {
    name: `EXP3 (γ = ${gamma})`,
    init: (env) => {
      const k = armCount(env)
      return { ...stats(k), logWeights: new Float64Array(k) }
    },
    act(st, _, s) {
      const p = probs(st.logWeights)
      let u = uniform(s)
      let arm = p.length - 1
      for (let a = 0; a < p.length; a++) {
        if (u < p[a]) {
          arm = a
          break
        }
        u -= p[a]
      }
      return { action: arm, scores: p, probabilities: p }
    },
    greedy: (st) => firstMax(st.logWeights),
    scalars: armScalars,
    learn(st, tr) {
      const k = st.logWeights.length
      const p = probs(st.logWeights)[tr.action]
      const logWeights = st.logWeights.slice()
      logWeights[tr.action] += (gamma * (tr.reward / p)) / k
      return { ...record(st, tr.action, tr.reward), logWeights }
    },
  }
}

// ── Linear policies ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The ridge statistics of a linear policy: $\Vmat = \lambda\Imat + \sum \xvec\xvec^\top$ (kept as its inverse) and
 * $\bvec = \sum r\xvec$, over the features $\xvec$ of the arms pulled and their rewards $r$.
 */
export interface RidgeState extends ArmStatistics {
  /** $\Vmat^{-1}$, $d \times d$ row-major, kept by Sherman–Morrison updates. */
  vInverse: Float64Array
  /** $\bvec = \sum r\xvec$, $d$ values. */
  b: Float64Array
  /** The ridge estimate $\hat{\thetavec} = \Vmat^{-1}\bvec$, $d$ values. */
  theta: Float64Array
  /** The feature dimension $d$. */
  dim: number
}

/**
 * The ridge statistics before any pull: $\Vmat^{-1} = \Imat/\lambda$, $\bvec = \hat{\thetavec} = \zeros$. Throws
 * `DomainError` unless the observation domain is a $K \times d$ box and the actions are discrete.
 *
 * @param env The environment: its observation shape gives $d$, its action domain $K$.
 * @param lambda The ridge penalty $\lambda > 0$.
 * @returns The initial `RidgeState`.
 */
function ridgeInit(env: EnvironmentShape, lambda: number): RidgeState {
  const o = env.observation
  if (o.kind !== 'box' || o.shape.length !== 2)
    throw new DomainError(
      'ridgeInit',
      'linear bandit policies need an arms × features box observation (a linear bandit)',
    )
  const d = o.shape[1]
  const vInverse = new Float64Array(d * d)
  for (let i = 0; i < d; i++) vInverse[i * d + i] = 1 / lambda
  return { ...stats(armCount(env)), vInverse, b: new Float64Array(d), theta: new Float64Array(d), dim: d }
}

/**
 * Arm $a$'s features: row $a$ of the flat $K \times d$ context, as a view (not a copy).
 *
 * @param ctx The round's context, $Kd$ values row-major.
 * @param a The arm.
 * @param d The feature dimension.
 * @returns Entries $ad$ to $ad + d - 1$ of `ctx`.
 */
const featuresOf = (ctx: Float64Array, a: number, d: number) => ctx.subarray(a * d, (a + 1) * d)

/**
 * The ridge statistics after a pull: $\Vmat^{-1}$ by a Sherman–Morrison rank-one update with the pulled arm's
 * features $\xvec$, $\bvec$ plus $r\xvec$, and $\hat{\thetavec} = \Vmat^{-1}\bvec$ recomputed. The `learn` of the
 * linear policies.
 *
 * @param st The statistics before the pull; not modified.
 * @param tr The transition: `observation` is the context the arm was pulled in, `action` the arm, `reward` $r$.
 * @returns The new statistics, with the pull counted.
 */
function ridgeUpdate(st: RidgeState, tr: Transition<Float64Array, number>): RidgeState {
  const d = st.dim
  const x = featuresOf(tr.observation, tr.action, d)
  const r = tr.reward
  const A = st.vInverse
  // Sherman–Morrison: (V + x xᵀ)⁻¹ = V⁻¹ − V⁻¹x xᵀV⁻¹ / (1 + xᵀV⁻¹x).
  const Ax = new Float64Array(d)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ax[i] += A[i * d + j] * x[j]
  const denom = 1 + x.reduce((s, v, i) => s + v * Ax[i], 0)
  const next = A.slice()
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) next[i * d + j] -= (Ax[i] * Ax[j]) / denom
  const b = st.b.map((v, i) => v + r * x[i])
  const theta = new Float64Array(d)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) theta[i] += next[i * d + j] * b[j]
  return { ...record(st, tr.action, r), vInverse: next, b, theta, dim: d }
}

/**
 * The confidence width $\lVert \xvec \rVert_{\Amat} = \sqrt{\xvec^\top\Amat\xvec}$ (negative rounding clamped to 0).
 *
 * @param A The matrix $\Amat$ ($\Vmat^{-1}$), $d \times d$ row-major.
 * @param x The features $\xvec$, $d$ values.
 * @param d The feature dimension.
 * @returns The width.
 */
function width(A: Float64Array, x: Float64Array, d: number): number {
  let s = 0
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) s += x[i] * A[i * d + j] * x[j]
  return Math.sqrt(Math.max(0, s))
}

/**
 * The score of every arm under parameter $\wvec$: $\xvec_a^\top\wvec$, plus a per-arm bonus when given.
 *
 * @param st The policy's state, for $K$ and $d$.
 * @param ctx The round's context, $K \times d$ row-major.
 * @param w The parameter $\wvec$, $d$ values.
 * @param bonus A bonus of an arm's features, added to its score (the confidence width of LinUCB).
 * @returns $K$ scores.
 */
function linearScores(st: RidgeState, ctx: Float64Array, w: Float64Array, bonus?: (x: Float64Array) => number) {
  const k = st.counts.length
  const d = st.dim
  return Float64Array.from({ length: k }, (_, a) => {
    const x = featuresOf(ctx, a, d)
    return x.reduce((acc, v, i) => acc + v * w[i], 0) + (bonus ? bonus(x) : 0)
  })
}

/**
 * LinUCB (Li, Chu, Langford and Schapire, 2010, WWW; the disjoint-free form of Abbasi-Yadkori et al., 2011): pull the
 * arm maximising $\xvec^\top\hat{\thetavec} + \alpha \lVert \xvec \rVert_{\Vmat^{-1}}$, with the ridge estimate
 * $\hat{\thetavec} = \Vmat^{-1}\bvec$ shared by all arms. $\alpha = 0$ is the greedy ridge policy. Needs a
 * $K \times d$ box observation (the round's context), or `init` throws `DomainError`.
 *
 * @param options The exploration bonus and the ridge penalty.
 * @param options.alpha The bonus weight $\alpha$.
 * @param options.lambda The ridge penalty $\lambda$: $\Vmat$ starts at $\lambda\Imat$.
 * @returns The agent, named after $\alpha$.
 *
 * @example LinUCB learns the parameter of a two-arm linear bandit
 * const theta = [0.2, 0.7]
 * const ctx = new Float64Array([1, 0, 0, 1])
 * const agent = linUcb({ alpha: 1 })
 * const s = stream(9)
 * const observation = { kind: 'box', shape: [2, 2], low: [0, 0, 0, 0], high: [1, 1, 1, 1] }
 * let g = agent.init({ observation, action: { kind: 'discrete', n: 2 } }, s)
 * for (let t = 0; t < 200; t++) {
 *   const a = agent.act(g, ctx, s).action
 *   const r = theta[a] + 0.1 * normal(s)
 *   g = agent.learn(g, { observation: ctx, action: a, reward: r, next: ctx, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('estimate of theta:', g.theta)
 */
export function linUcb({ alpha = 1, lambda = 1 }: { alpha?: number; lambda?: number } = {}): Agent<
  RidgeState,
  Float64Array,
  number
> {
  return {
    name: alpha === 0 ? 'greedy ridge' : `LinUCB (α = ${alpha})`,
    init: (env) => ridgeInit(env, lambda),
    act(st, ctx, s) {
      const scores = linearScores(st, ctx, st.theta, (x) => alpha * width(st.vInverse, x, st.dim))
      return { action: argmaxRandom(scores, s), scores }
    },
    greedy: (st, ctx) => firstMax(linearScores(st, ctx, st.theta)),
    scalars: armScalars,
    learn: ridgeUpdate,
  }
}

/**
 * Linear Thompson sampling (Agrawal and Goyal, 2013, ICML): draw
 * $\tilde{\thetavec} \sim \Gauss(\hat{\thetavec}, v^2 \Vmat^{-1})$ (through the Cholesky factor of $v^2\Vmat^{-1}$) and
 * pull the arm maximising $\xvec^\top\tilde{\thetavec}$. Needs a $K \times d$ box observation, or `init` throws
 * `DomainError`.
 *
 * @param options The posterior scale and the ridge penalty.
 * @param options.v The scale $v$ of the posterior draw; larger explores more.
 * @param options.lambda The ridge penalty $\lambda$: $\Vmat$ starts at $\lambda\Imat$.
 * @returns The agent, named after $v$.
 *
 * @example Linear Thompson sampling on a two-arm linear bandit
 * const theta = [0.2, 0.7]
 * const ctx = new Float64Array([1, 0, 0, 1])
 * const agent = linearThompson({ v: 0.5 })
 * const s = stream(10)
 * const observation = { kind: 'box', shape: [2, 2], low: [0, 0, 0, 0], high: [1, 1, 1, 1] }
 * let g = agent.init({ observation, action: { kind: 'discrete', n: 2 } }, s)
 * for (let t = 0; t < 200; t++) {
 *   const a = agent.act(g, ctx, s).action
 *   const r = theta[a] + 0.1 * normal(s)
 *   g = agent.learn(g, { observation: ctx, action: a, reward: r, next: ctx, terminated: true, truncated: false })
 * }
 * print('pulls per arm:', g.counts)
 * print('estimate of theta:', g.theta)
 */
export function linearThompson({ v = 0.5, lambda = 1 }: { v?: number; lambda?: number } = {}): Agent<
  RidgeState,
  Float64Array,
  number
> {
  return {
    name: `linear Thompson sampling (v = ${v})`,
    init: (env) => ridgeInit(env, lambda),
    act(st, ctx, s) {
      const d = st.dim
      const L = toFlat(
        cholesky(
          fromData(
            st.vInverse.map((x) => x * v * v),
            [d, d],
          ),
        ).L,
      )
      const z = Float64Array.from({ length: d }, () => normal(s))
      const draw = st.theta.map((m, i) => {
        let acc = m
        for (let j = 0; j <= i; j++) acc += L[i * d + j] * z[j]
        return acc
      })
      const scores = linearScores(st, ctx, draw)
      return { action: argmaxRandom(scores, s), scores }
    },
    greedy: (st, ctx) => firstMax(linearScores(st, ctx, st.theta)),
    scalars: armScalars,
    learn: ridgeUpdate,
  }
}

// ── Bounds ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Lai–Robbins lower bound for Bernoulli arms (Lai and Robbins, 1985, Adv. Appl. Math. 6): every consistent policy
 * has $\expect[R_T] \ge (1 + o(1)) \ln T \sum_{a : \mu_a < \mu^*} (\mu^* - \mu_a) / \KL(\mu_a, \mu^*)$, with $R_T$ the
 * regret after $T$ rounds and $\KL$ the Bernoulli divergence (`klBernoulli`).
 *
 * @param means The arms' success probabilities $\mu_a$; $\mu^*$ is the largest.
 * @param t The rounds $T$ to evaluate the bound at: a tensor or an array of numbers.
 * @returns `constant`, the sum (the coefficient of $\ln T$), and `bound`, the constant times $\ln T$ at each `t`.
 *
 * @example The regret floor of three Bernoulli arms
 * const { constant, bound } = laiRobbinsBound([0.2, 0.5, 0.8], [10, 100, 1000])
 * print('constant:', constant)
 * print('bound at T = 10, 100, 1000:', bound)
 */
export function laiRobbinsBound(
  means: readonly number[],
  t: Tensor | readonly number[],
): { constant: number; bound: Tensor } {
  const best = Math.max(...means)
  const constant = means.reduce((acc, m) => (m < best ? acc + (best - m) / klBernoulli(m, best) : acc), 0)
  const ts = 'shape' in t ? Array.from(t.data) : t
  return { constant, bound: fromData(Float64Array.from(ts, (v) => constant * Math.log(v))) }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const agent = definer<AgentInfo>('agent', 'gym/agents')
const armRequires = { action: 'discrete', families: ['bandit', 'contextual-bandit'] } as const
const linearRequires = { observation: 'box', action: 'discrete', families: ['contextual-bandit'] } as const

agent(
  {
    key: 'uniformPolicy',
    name: 'Uniform (round robin)',
    summary: 'Pulls the arms in turn, an A/B/n test: linear regret, the baseline of the bandit policies.',
    params: space({}),
    requires: armRequires,
    notes: ['multi-armed-bandit'],
  },
  uniformPolicy,
)
agent(
  {
    key: 'exploreThenCommit',
    name: 'Explore then commit',
    summary: 'Pulls every arm m times in turn, then the best empirical mean for ever.',
    params: space({ m: int(1, 500, { default: 50 }) }),
    requires: armRequires,
    notes: ['multi-armed-bandit'],
    random: true,
  },
  exploreThenCommit,
)
agent(
  {
    key: 'epsilonGreedy',
    name: 'ε-greedy',
    summary: 'The best empirical mean, or with probability ε a uniformly random arm (optionally decaying as cK/t).',
    params: space({ epsilon: real(0, 1, { default: 0.1, label: 'ε' }) }),
    requires: armRequires,
    notes: ['multi-armed-bandit'],
    random: true,
  },
  epsilonGreedy,
)
agent(
  {
    key: 'ucb1',
    name: 'UCB1',
    summary: 'Pulls the arm with the largest upper confidence bound, the mean plus √(c ln t / n).',
    params: space({ c: real(0, 10, { default: 2 }) }),
    requires: armRequires,
    notes: ['upper-confidence-bound-algorithm', 'multi-armed-bandit'],
    random: true,
  },
  ucb1,
)
agent(
  {
    key: 'klUcb',
    name: 'KL-UCB',
    summary: 'Pulls the arm with the largest KL upper confidence bound; asymptotically optimal for Bernoulli arms.',
    params: space({ c: real(0, 3, { default: 0 }) }),
    requires: armRequires,
    notes: ['upper-confidence-bound-algorithm'],
    random: true,
  },
  klUcb,
)
agent(
  {
    key: 'thompsonBernoulli',
    name: 'Thompson sampling (Bernoulli)',
    summary: 'Draws each arm’s success probability from its Beta posterior and pulls the largest draw.',
    params: space({
      alpha: real(0.1, 10, { default: 1, label: 'α' }),
      beta: real(0.1, 10, { default: 1, label: 'β' }),
    }),
    requires: armRequires,
    notes: ['thompson-sampling'],
    random: true,
  },
  thompsonBernoulli,
)
agent(
  {
    key: 'thompsonGaussian',
    name: 'Thompson sampling (Gaussian)',
    summary: 'Draws each arm’s mean from its Gaussian posterior (known noise) and pulls the largest draw.',
    params: space({ noiseSd: real(0.01, 10, { default: 1 }) }),
    requires: armRequires,
    notes: ['thompson-sampling'],
    random: true,
  },
  thompsonGaussian,
)
agent(
  {
    key: 'exp3',
    name: 'EXP3',
    summary:
      'Exponential weights on importance-weighted rewards, mixed with uniform exploration: for adversarial arms.',
    params: space({ gamma: real(0.001, 1, { default: 0.1, label: 'γ' }) }),
    requires: armRequires,
    notes: ['multi-armed-bandit'],
    random: true,
  },
  exp3,
)
agent(
  {
    key: 'linUcb',
    name: 'LinUCB',
    summary: 'Ridge regression on the arms’ features plus an exploration bonus α‖x‖ in the inverse design matrix.',
    params: space({
      alpha: real(0, 5, { default: 1, label: 'α' }),
      lambda: real(0.01, 10, { default: 1, label: 'λ' }),
    }),
    requires: linearRequires,
    notes: ['linucb', 'contextual-bandit'],
    random: true,
  },
  linUcb,
)
agent(
  {
    key: 'linearThompson',
    name: 'Linear Thompson sampling',
    summary: 'Draws θ from the ridge posterior N(θ̂, v²V⁻¹) and pulls the arm whose features score highest under it.',
    params: space({ v: real(0.01, 5, { default: 0.5 }), lambda: real(0.01, 10, { default: 1, label: 'λ' }) }),
    requires: linearRequires,
    notes: ['linear-thompson-sampling', 'contextual-bandit'],
    random: true,
  },
  linearThompson,
)
