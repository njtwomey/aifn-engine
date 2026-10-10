/**
 * Seeded decision data: logged contextual-bandit feedback (with the true policy value beside it), logged slates under
 * an additive reward, and loss sequences for prediction with expert advice (stochastic, switching, and the
 * oscillating sequence that defeats follow-the-leader). The estimators these feed are in
 * `aifn-compute/learning/off-policy` and `aifn-compute/optim/online`.
 *
 * Every quantity is drawn from its own child stream of `s`, so a problem or a log is fixed by its key and knobs, and
 * the truth an estimator aims at (`trueValue`, the best expert) is returned beside the data.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import {
  bernoulli,
  categorical,
  child,
  normal,
  permutation,
  uniform,
  type Stream,
} from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'
import { checkCount, labels, matrix, vector, type DatasetMeta } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

// ── Contextual bandits ───────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `banditProblem` and `loggedBandit`. */
export interface BanditOptions {
  /** Rounds $n$ (default 1000). */
  n?: number
  /** Actions $K$ (default 5). */
  actions?: number
  /** Context features $d$ (default 3). */
  features?: number
  /**
   * How sharply the logging policy prefers its favourite actions:
   * $\pi_0 = \operatorname{softmax}(\beta_0 s_0(\xvec, \cdot))$, with $s_0(\xvec, a) = \xvec^\top \phivec_a$ a linear
   * score unrelated to the reward. 0 is uniform logging; large values make propensities extreme (default 1).
   */
  loggingSharpness?: number
  /**
   * The share $\varepsilon$ of uniform exploration mixed into the logging policy,
   * $\pi_0 \leftarrow (1 - \varepsilon)\pi_0 + \varepsilon/K$ (default 0.05).
   */
  exploration?: number
  /**
   * How sharply the target policy follows the true mean reward: $\pi = \operatorname{softmax}(\beta \mu(\xvec, \cdot))$
   * (default 8).
   */
  targetSharpness?: number
  /**
   * The standard deviation of the error $e$ added to the reward model's logits,
   * $\hat{q} = \sigma(\operatorname{logit} \mu + e + \text{bias})$ (default 0.5).
   */
  modelError?: number
  /** A constant shift of the reward model's logits, a systematic bias (default 0). */
  modelBias?: number
}

/** A contextual bandit with known truth: contexts, mean rewards, the logging and target policies and a reward model. */
export interface BanditProblem {
  /** Contexts $\xvec_i$, $n \times d$. */
  contexts: Tensor
  /** The mean reward $\mu(\xvec_i, a) \in (0, 1)$ of every action, $n \times K$ (rewards are Bernoulli). */
  meanRewards: Tensor
  /** The logging policy $\pi_0(a \mid \xvec_i)$, $n \times K$. */
  logging: Tensor
  /** The target policy $\pi(a \mid \xvec_i)$, $n \times K$. */
  target: Tensor
  /**
   * An imperfect reward model $\hat{q}(\xvec_i, a)$, $n \times K$, for the direct method and doubly robust
   * estimators.
   */
  rewardModel: Tensor
  /**
   * The target's value on these contexts, $\frac{1}{n} \sum_i \sum_a \pi(a \mid \xvec_i) \mu(\xvec_i, a)$: what every
   * estimator aims at.
   */
  trueValue: number
  /** The logging policy's own value on these contexts, the same average under $\pi_0$. */
  loggingValue: number
}

/** Logged feedback: the logging policy's actions, their propensities and the Bernoulli rewards. */
export interface BanditLogDraw {
  /** The logged action $a_i$ of every round (int32, length $n$). */
  actions: Tensor
  /** The logging policy's probability $\pi_0(a_i \mid \xvec_i)$ of the logged action (length $n$). */
  propensities: Tensor
  /** The observed reward $r_i$, 0 or 1 (float64, length $n$). */
  rewards: Tensor
}

/**
 * The softmax of a row of scores at inverse temperature $\beta$, $e^{\beta z_k} / \sum_j e^{\beta z_j}$, computed
 * after subtracting the largest $\beta z_k$ so that it cannot overflow.
 *
 * @param z The scores, one per action; not modified.
 * @param beta The inverse temperature $\beta$: 0 gives the uniform distribution.
 * @returns The probabilities, one per action.
 */
const softmaxRow = (z: ArrayLike<number>, beta: number): F64 => {
  let hi = -Infinity
  for (let k = 0; k < z.length; k++) hi = Math.max(hi, beta * z[k])
  const e = Float64Array.from(z, (v) => Math.exp(beta * v - hi))
  const total = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / total)
}

/**
 * A contextual bandit with known truth. Contexts are standard normal;
 * $\mu(\xvec, a) = \sigma(\xvec^\top \thetavec_a + b_a)$ with $\thetavec_a \sim \Gauss(\zeros, \Imat)$ and
 * $b_a \sim \Gauss(0, 0.5^2)$; the logging policy is a softmax of a second, unrelated linear score (plus $\varepsilon$
 * exploration), so its preferences disagree with the reward; the target is a softmax of $\mu$; the reward model
 * perturbs $\mu$'s logits by Gaussian noise and a bias. Every quantity is drawn from child streams of `s`, so a problem
 * is fixed by its key and knobs. Throws `DomainError` when `n` is not a non-negative integer.
 *
 * @param s The stream the contexts, the weights of the reward and the logging score, and the model's errors are drawn
 *   from.
 * @param options The size of the problem and the shape of the two policies and the reward model (`BanditOptions`).
 * @returns The contexts, the mean rewards, both policies and the reward model as $n \times d$ and $n \times K$
 *   matrices, with the target's and the logging policy's true values.
 *
 * @example The target policy is worth more than the logging one
 * const p = banditProblem(stream(1), { n: 200, actions: 3 })
 * print('contexts:', p.contexts.shape, ' mean rewards:', p.meanRewards.shape)
 * print('first row: mu', toArray(p.meanRewards)[0], ' logging', toArray(p.logging)[0], ' target', toArray(p.target)[0])
 * print('true value:', p.trueValue, ' logging value:', p.loggingValue)
 */
export function banditProblem(s: Stream, options: BanditOptions = {}): BanditProblem {
  const {
    n = 1000,
    actions: K = 5,
    features: d = 3,
    loggingSharpness = 1,
    exploration = 0.05,
    targetSharpness = 8,
    modelError = 0.5,
    modelBias = 0,
  } = options
  checkCount(n, 'banditProblem')
  const draw = (name: string, count: number, sd = 1) => {
    const c = child(s, name)
    return Float64Array.from({ length: count }, () => sd * normal(c))
  }
  const x = draw('contexts', n * d)
  const theta = draw('theta', d * K)
  const bias = draw('bias', K, 0.5)
  const phi = draw('logging', d * K)
  const err = draw('model', n * K, modelError)
  const mu = new Float64Array(n * K)
  const q = new Float64Array(n * K)
  const p0 = new Float64Array(n * K)
  const pi = new Float64Array(n * K)
  let trueValue = 0
  let loggingValue = 0
  for (let i = 0; i < n; i++) {
    const logits = new Float64Array(K)
    const score = new Float64Array(K)
    for (let a = 0; a < K; a++) {
      let z = bias[a]
      let u = 0
      for (let j = 0; j < d; j++) {
        z += x[i * d + j] * theta[j * K + a]
        u += x[i * d + j] * phi[j * K + a]
      }
      logits[a] = z
      score[a] = u
      mu[i * K + a] = sigmoid(z) as number
      q[i * K + a] = sigmoid(z + err[i * K + a] + modelBias) as number
    }
    const lp = softmaxRow(score, loggingSharpness)
    const tp = softmaxRow(
      Float64Array.from({ length: K }, (_, a) => mu[i * K + a]),
      targetSharpness,
    )
    for (let a = 0; a < K; a++) {
      p0[i * K + a] = (1 - exploration) * lp[a] + exploration / K
      pi[i * K + a] = tp[a]
      trueValue += tp[a] * mu[i * K + a]
      loggingValue += p0[i * K + a] * mu[i * K + a]
    }
  }
  return {
    contexts: matrix(x, n, d),
    meanRewards: matrix(mu, n, K),
    logging: matrix(p0, n, K),
    target: matrix(pi, n, K),
    rewardModel: matrix(q, n, K),
    trueValue: trueValue / n,
    loggingValue: loggingValue / n,
  }
}

/**
 * Draw one log from a problem: $a_i \sim \pi_0(\cdot \mid \xvec_i)$, $r_i \sim \Bern(\mu(\xvec_i, a_i))$, with
 * the propensity $\pi_0(a_i \mid \xvec_i)$. The actions come from `child(s, 'actions')`, the rewards from
 * `child(s, 'rewards')`.
 *
 * @param s The stream the log is drawn from.
 * @param problem The problem, as `banditProblem` returns it; only its `logging` and `meanRewards` are read.
 * @returns The logged actions, their propensities and the rewards, one per round.
 *
 * @example An importance-weighted estimate of the target's value
 * const p = banditProblem(stream(1), { n: 2000, actions: 3 })
 * const log = logBandit(stream(2), p)
 * const [a, w, r] = [toArray(log.actions), toArray(log.propensities), toArray(log.rewards)]
 * print('first actions:', a.slice(0, 5), ' propensities:', w.slice(0, 5))
 * const pi = toArray(p.target)
 * print('IPS estimate:', r.reduce((acc, v, i) => acc + (v * pi[i][a[i]]) / w[i], 0) / r.length)
 * print('true value:', p.trueValue)
 */
export function logBandit(s: Stream, problem: BanditProblem): BanditLogDraw {
  const [n, K] = problem.logging.shape
  const p0 = dense.data(problem.logging)
  const mu = dense.data(problem.meanRewards)
  const ca = child(s, 'actions')
  const cr = child(s, 'rewards')
  const a = new Int32Array(n)
  const prop = new Float64Array(n)
  const r = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    a[i] = categorical(ca, p0.subarray(i * K, (i + 1) * K))
    prop[i] = p0[i * K + a[i]]
    r[i] = bernoulli(cr, mu[i * K + a[i]])
  }
  return { actions: labels(a), propensities: vector(prop), rewards: vector(r) }
}

/** A logged contextual bandit: the problem's truth and one log drawn from it. */
export interface LoggedBandit extends BanditProblem, BanditLogDraw {
  /** The dataset's name, description, task (`'decision'`), feature names and stream key. */
  meta: DatasetMeta
}

/**
 * A contextual bandit with known truth and one log drawn from its logging policy (`banditProblem` on
 * `child(s, 'problem')`, then `logBandit` on `child(s, 'log')`). Throws `DomainError` when `n` is not a non-negative
 * integer.
 *
 * @param s The stream the problem and the log are drawn from.
 * @param options The size of the problem and the shape of the two policies and the reward model (`BanditOptions`).
 * @returns The problem's truth and the log in one record, with dataset metadata.
 *
 * @example The logged rewards average to the logging policy's value
 * const b = loggedBandit(stream(1), { n: 2000, actions: 4 })
 * const r = toArray(b.rewards)
 * print('contexts:', b.contexts.shape, ' actions:', b.actions.shape)
 * print('mean logged reward:', r.reduce((a, v) => a + v, 0) / r.length, ' logging value:', b.loggingValue)
 */
export function loggedBandit(s: Stream, options: BanditOptions = {}): LoggedBandit {
  const problem = banditProblem(child(s, 'problem'), options)
  const log = logBandit(child(s, 'log'), problem)
  const [n, K] = problem.logging.shape
  return {
    ...problem,
    ...log,
    meta: {
      name: 'logged bandit',
      description: `${n} rounds of a ${K}-action contextual bandit with Bernoulli rewards, logged by a softmax policy that disagrees with the reward; the target follows the reward.`,
      task: 'decision',
      featureNames: Array.from({ length: problem.contexts.shape[1] }, (_, j) => `x${j}`),
      key: s.key,
    },
  }
}

// ── Slates ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `slateBandit`. */
export interface SlateBanditOptions {
  /** Rounds $n$ (default 2000). */
  n?: number
  /** Items $m$ (default 6). */
  items?: number
  /** Slots $l$, at most $m$ (default 3). */
  slots?: number
  /** Reward noise standard deviation (default 0.1). */
  noise?: number
}

/** Logged slates with an additive reward and a uniform logging policy. */
export interface SlateBandit {
  /** The logged slates, $n \times l$ (item indices as float64): distinct items, uniformly ordered. */
  slates: Tensor
  /** The rewards $r_i = \sum_j \phi(j, s_{ij}) + \varepsilon_i$, length $n$. */
  rewards: Tensor
  /**
   * The target slate, the same each round, $n \times l$: the best items in quality order, which is the best assignment
   * of items to slots under $\phi$.
   */
  target: Tensor
  /** $\phi(j, a)$, the additive reward of item $a$ in slot $j$, $l \times m$. */
  slotValues: Tensor
  /** The target's true value $\sum_j \phi(j, t_j)$. */
  trueValue: number
  /** The number of items $m$. */
  items: number
  /** The dataset's name, description, task (`'decision'`), slot names and stream key. */
  meta: DatasetMeta
}

/**
 * Slates of $l$ distinct items from $m$, logged uniformly at random (the first $l$ of a random permutation); the reward
 * is additive over (slot, item) pairs, $\phi(j, a) = u_a / (j + 1)$ for slot $j = 0, \dots, l - 1$, with item
 * qualities $u_a \sim \Unif(0, 1)$ (earlier slots matter more), plus Gaussian noise. The target shows the best items
 * in quality order. The pseudo-inverse estimator is unbiased here; slate IPS is too, but sees the target slate in only
 * one round in $m! / (m - l)!$. Throws `DomainError` when $l > m$ or `n` is not a non-negative integer.
 *
 * @param s The stream the qualities (`child(s, 'quality')`), the slates and the noise are drawn from.
 * @param options The rounds, items and slots, and the noise's standard deviation (`SlateBanditOptions`).
 * @returns The logged slates and rewards, the target slate, the slot values $\phi$ and the target's true value.
 *
 * @example The mean logged reward is the uniform policy's value
 * const b = slateBandit(stream(1), { n: 2000, items: 5, slots: 2 })
 * const phi = toArray(b.slotValues)
 * const r = toArray(b.rewards)
 * print('slates:', b.slates.shape, ' first slates:', toArray(b.slates).slice(0, 3), ' target:', toArray(b.target)[0])
 * print('mean reward:', r.reduce((a, v) => a + v, 0) / r.length)
 * print('uniform value:', phi.reduce((a, row) => a + row.reduce((c, v) => c + v, 0) / row.length, 0))
 * print('target value:', b.trueValue)
 */
export function slateBandit(s: Stream, options: SlateBanditOptions = {}): SlateBandit {
  const { n = 2000, items: m = 6, slots: l = 3, noise = 0.1 } = options
  checkCount(n, 'slateBandit')
  if (l > m) throw new DomainError('slateBandit', `slateBandit: ${l} slots need at least ${l} items, got ${m}`)
  const cu = child(s, 'quality')
  const u = Float64Array.from({ length: m }, () => uniform(cu))
  const phi = new Float64Array(l * m)
  for (let j = 0; j < l; j++) for (let a = 0; a < m; a++) phi[j * m + a] = u[a] / (j + 1)
  const order = Array.from({ length: m }, (_, a) => a).sort((a, b) => u[b] - u[a])
  const best = order.slice(0, l)
  const trueValue = best.reduce((acc, a, j) => acc + phi[j * m + a], 0)
  const cs = child(s, 'slates')
  const ce = child(s, 'noise')
  const slates = new Float64Array(n * l)
  const target = new Float64Array(n * l)
  const r = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const p = Array.from(toFlat(permutation(cs, m)))
    let v = 0
    for (let j = 0; j < l; j++) {
      slates[i * l + j] = p[j]
      target[i * l + j] = best[j]
      v += phi[j * m + p[j]]
    }
    r[i] = v + noise * normal(ce)
  }
  return {
    slates: matrix(slates, n, l),
    rewards: vector(r),
    target: matrix(target, n, l),
    slotValues: matrix(phi, l, m),
    trueValue,
    items: m,
    meta: {
      name: 'logged slates',
      description: `${n} slates of ${l} items from ${m}, logged uniformly at random, with a reward additive over (slot, item) pairs.`,
      task: 'decision',
      featureNames: Array.from({ length: l }, (_, j) => `slot ${j + 1}`),
      key: s.key,
    },
  }
}

// ── Expert advice ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `expertGame`. */
export interface ExpertGameOptions {
  /** Rounds $T$ (default 1000). */
  rounds?: number
  /** Experts $N$ (default 10). */
  experts?: number
  /**
   * `stochastic`: Bernoulli losses of mean $0.5 - \text{gap}$ for one expert drawn at random and 0.5 for the rest;
   * `switching`: the same, with the best expert changing every $\lceil T/(\text{switches} + 1) \rceil$ rounds, in a
   * random order of the experts; `follow-the-leader trap`: two experts with losses $(\tfrac{1}{2}, 0)$, then $(0, 1)$,
   * $(1, 0), \dots$ alternating, so the leader always loses next (the even-numbered experts copy expert 0 and the odd
   * ones expert 1); `random`: uniform losses in $[0, 1]$. Default `stochastic`.
   */
  kind?: 'stochastic' | 'switching' | 'follow-the-leader trap' | 'random'
  /** The gap between the best expert's mean loss and the others' (`stochastic` and `switching`, default 0.1). */
  gap?: number
  /** Switches of the best expert (`switching`, default 3). */
  switches?: number
}

/** A loss sequence for prediction with expert advice. */
export interface ExpertGame {
  /** Losses $\ell_{t,i} \in [0, 1]$, $T \times N$. */
  losses: Tensor
  /** The best expert of each round's regime, length $T$ (int32; the stochastic and switching kinds, $-1$ otherwise). */
  best: Tensor
  /** The dataset's name, description, task (`'decision'`), expert names and stream key. */
  meta: DatasetMeta
}

/**
 * A sequence of expert losses in $[0, 1]$ of a named kind (see `ExpertGameOptions.kind`). The losses come from
 * `child(s, 'losses')` and the order of the best experts from `child(s, 'order')`; the follow-the-leader trap draws
 * nothing. Throws `DomainError` when the number of rounds is not a non-negative integer.
 *
 * @param s The stream the losses and the best experts are drawn from.
 * @param options The rounds, the experts, the kind and its gap and switches (`ExpertGameOptions`).
 * @returns The losses ($T \times N$) and the best expert of every round.
 *
 * @example The trap: each expert loses right after it leads
 * const g = expertGame(stream(1), { rounds: 6, experts: 2, kind: 'follow-the-leader trap' })
 * print('losses:', toArray(g.losses))
 *
 * @example The stochastic game: one expert loses gap less on average
 * const g = expertGame(stream(1), { rounds: 2000, experts: 4, gap: 0.1 })
 * const L = toArray(g.losses)
 * print('best expert:', toArray(g.best)[0])
 * print('mean losses:', [0, 1, 2, 3].map((i) => L.reduce((a, row) => a + row[i], 0) / L.length))
 */
export function expertGame(s: Stream, options: ExpertGameOptions = {}): ExpertGame {
  const { rounds: T = 1000, experts: N = 10, kind = 'stochastic', gap = 0.1, switches = 3 } = options
  checkCount(T, 'expertGame')
  const L = new Float64Array(T * N)
  const best = new Int32Array(T).fill(-1)
  const c = child(s, 'losses')
  if (kind === 'follow-the-leader trap') {
    for (let t = 0; t < T; t++)
      for (let i = 0; i < N; i++) {
        const e = i % 2
        L[t * N + i] = t === 0 ? (e === 0 ? 0.5 : 0) : (t % 2 === 1) === (e === 0) ? 0 : 1
      }
  } else if (kind === 'random') {
    for (let k = 0; k < T * N; k++) L[k] = uniform(c)
  } else {
    const order = Array.from(toFlat(permutation(child(s, 'order'), N)))
    const span = kind === 'switching' ? Math.ceil(T / (switches + 1)) : T
    for (let t = 0; t < T; t++) {
      const leader = order[Math.floor(t / span) % N]
      best[t] = leader
      for (let i = 0; i < N; i++) {
        const mean = i === leader ? 0.5 - gap : 0.5
        L[t * N + i] = bernoulli(c, mean)
      }
    }
  }
  return {
    losses: matrix(L, T, N),
    best: fromData(best, [T]),
    meta: {
      name: `expert game (${kind})`,
      description: `${T} rounds of losses in [0, 1] for ${N} experts: ${kind}.`,
      task: 'decision',
      featureNames: Array.from({ length: N }, (_, i) => `expert ${i}`),
      key: s.key,
    },
  }
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'loggedBandit',
    name: 'Logged contextual bandit',
    summary: 'Logged actions, propensities and Bernoulli rewards, with the true value of a target policy.',
    task: 'decision',
    output: 'log',
    knobs: space({
      n: int(1, 100000, { default: 1000 }),
      actions: int(2, 50, { default: 5 }),
      features: int(1, 20, { default: 3 }),
      loggingSharpness: real(0, 20, { default: 1 }),
      exploration: real(0, 1, { default: 0.05 }),
      targetSharpness: real(0, 50, { default: 8 }),
      modelError: real(0, 5, { default: 0.5 }),
      modelBias: real(-5, 5, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['off-policy-evaluation', 'logging-policies-and-propensities'],
    cite: ['dudik2011'],
  },
  loggedBandit,
)

dataset(
  {
    key: 'slateBandit',
    name: 'Logged slates',
    summary: 'Uniformly logged slates with a reward additive over (slot, item) pairs.',
    task: 'decision',
    output: 'log',
    knobs: space({
      n: int(1, 100000, { default: 2000 }),
      items: int(2, 20, { default: 6 }),
      slots: int(1, 10, { default: 3 }),
      noise: real(0, 2, { default: 0.1 }),
    }),
    truth: false,
    random: true,
    notes: ['slate-off-policy-evaluation'],
    cite: ['swaminathan2017slate'],
  },
  slateBandit,
)

dataset(
  {
    key: 'expertGame',
    name: 'Expert advice game',
    summary: 'Loss sequences for N experts: stochastic, switching, random, or the follow-the-leader trap.',
    task: 'decision',
    output: 'game',
    knobs: space({
      rounds: int(1, 100000, { default: 1000 }),
      experts: int(2, 1000, { default: 10 }),
      kind: oneOf(['stochastic', 'switching', 'follow-the-leader trap', 'random'], { default: 'stochastic' }),
      gap: real(0, 0.5, { default: 0.1 }),
      switches: int(0, 100, { default: 3 }),
    }),
    truth: false,
    random: true,
    notes: ['hedge-and-exponential-weights', 'tracking-the-best-expert', 'follow-the-regularised-leader'],
    cite: ['cesabianchi2006'],
  },
  expertGame,
)
