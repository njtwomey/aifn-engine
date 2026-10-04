/**
 * Seeded decision data: logged contextual-bandit feedback (with the true policy value beside it), logged slates under
 * an additive reward, and loss sequences for prediction with expert advice (stochastic, switching, and the
 * oscillating sequence that defeats follow-the-leader). The estimators these feed are in `aifn-compute/learning/off-policy`
 * and `aifn-compute/optim/online`.
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
  /** Rounds n (default 1000). */
  n?: number
  /** Actions K (default 5). */
  actions?: number
  /** Context features d (default 3). */
  features?: number
  /**
   * How sharply the logging policy prefers its favourite actions: π₀ = softmax(β₀ s₀(x, ·)), with s₀ a linear score
   * unrelated to the reward. 0 is uniform logging; large values make propensities extreme (default 1).
   */
  loggingSharpness?: number
  /** The share ε of uniform exploration mixed into the logging policy, π₀ ← (1 − ε)π₀ + ε/K (default 0.05). */
  exploration?: number
  /** How sharply the target policy follows the true mean reward: π = softmax(β μ(x, ·)) (default 8). */
  targetSharpness?: number
  /** The standard deviation of the error added to the reward model's logits, q̂ = σ(logit μ + e) (default 0.5). */
  modelError?: number
  /** A constant shift of the reward model's logits, a systematic bias (default 0). */
  modelBias?: number
}

/** A contextual bandit with known truth: contexts, mean rewards, the logging and target policies and a reward model. */
export interface BanditProblem {
  /** Contexts xᵢ [n, d]. */
  contexts: Tensor
  /** The mean reward μ(xᵢ, a) ∈ (0, 1) of every action [n, K] (rewards are Bernoulli). */
  meanRewards: Tensor
  /** The logging policy π₀(a | xᵢ) [n, K]. */
  logging: Tensor
  /** The target policy π(a | xᵢ) [n, K]. */
  target: Tensor
  /** An imperfect reward model q̂(xᵢ, a) [n, K], for the direct method and doubly robust estimators. */
  rewardModel: Tensor
  /** The target's value on these contexts, (1/n) Σᵢ Σₐ π(a | xᵢ) μ(xᵢ, a): what every estimator aims at. */
  trueValue: number
  /** The logging policy's own value on these contexts. */
  loggingValue: number
}

/** Logged feedback: the logging policy's actions, their propensities and the Bernoulli rewards. */
export interface BanditLogDraw {
  actions: Tensor
  propensities: Tensor
  rewards: Tensor
}

const softmaxRow = (z: ArrayLike<number>, beta: number): F64 => {
  let hi = -Infinity
  for (let k = 0; k < z.length; k++) hi = Math.max(hi, beta * z[k])
  const e = Float64Array.from(z, (v) => Math.exp(beta * v - hi))
  const total = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / total)
}

/**
 * A contextual bandit with known truth. Contexts are standard normal; μ(x, a) = σ(xᵀθₐ + bₐ) with θ, b normal; the
 * logging policy is a softmax of a second, unrelated linear score (plus ε exploration), so its preferences disagree
 * with the reward; the target is a softmax of μ; the reward model perturbs μ's logits by noise and a bias. Every
 * quantity is drawn from child streams of `s`, so a problem is fixed by its key and knobs.
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

/** Draw one log from a problem: aᵢ ~ π₀(· | xᵢ), rᵢ ~ Bernoulli(μ(xᵢ, aᵢ)), with the propensity π₀(aᵢ | xᵢ). */
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
  meta: DatasetMeta
}

/** A contextual bandit with known truth and one log drawn from its logging policy (`banditProblem`, `logBandit`). */
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
  /** Rounds n (default 2000). */
  n?: number
  /** Items m (default 6). */
  items?: number
  /** Slots l (default 3). */
  slots?: number
  /** Reward noise standard deviation (default 0.1). */
  noise?: number
}

/** Logged slates with an additive reward and a uniform logging policy. */
export interface SlateBandit {
  /** The logged slates [n, l]: distinct items, uniformly ordered. */
  slates: Tensor
  /** rᵢ = Σⱼ φ(j, sᵢⱼ) + noise [n]. */
  rewards: Tensor
  /** The target slate, the same each round [n, l]: the best assignment of items to slots by φ, greedily. */
  target: Tensor
  /** φ(j, a), the additive reward of item a in slot j [l, m]. */
  slotValues: Tensor
  /** The target's true value Σⱼ φ(j, tⱼ). */
  trueValue: number
  items: number
  meta: DatasetMeta
}

/**
 * Slates of l distinct items from m, logged uniformly at random; the reward is additive over (slot, item) pairs,
 * φ(j, a) = u_a / (j + 1) with item qualities u ~ U(0, 1) (higher slots matter more), plus Gaussian noise. The target
 * shows the best items in quality order. The pseudo-inverse estimator is unbiased here; slate IPS is too, but almost
 * never sees the target slate.
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
  /** Rounds T (default 1000). */
  rounds?: number
  /** Experts N (default 10). */
  experts?: number
  /**
   * `stochastic`: losses Bernoulli with means spread by `gap`, expert 0 best; `switching`: the best expert changes
   * every T/(switches + 1) rounds; `follow-the-leader trap`: two experts with losses (½, 0), then (0, 1), (1, 0), …
   * alternating, so the leader always loses next (the rest copy expert 0 or 1); `random`: uniform losses in [0, 1].
   */
  kind?: 'stochastic' | 'switching' | 'follow-the-leader trap' | 'random'
  /** The gap between the best expert's mean loss and the others' (default 0.1). */
  gap?: number
  /** Switches of the best expert (`switching`, default 3). */
  switches?: number
}

/** A loss sequence for prediction with expert advice. */
export interface ExpertGame {
  /** Losses ℓ_{t,i} ∈ [0, 1] [T, N]. */
  losses: Tensor
  /** The best expert of each round's regime [T] (the stochastic and switching kinds; −1 otherwise). */
  best: Tensor
  meta: DatasetMeta
}

/** A sequence of expert losses in [0, 1] of a named kind (see `ExpertGameOptions.kind`). */
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
