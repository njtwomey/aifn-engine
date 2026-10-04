/**
 * Bandit policies as `Agent`s (docs/aifn-gym.md §5): `init` builds the policy's statistics from the action domain (one
 * arm per action), `act` picks an arm and reports the index, sample or estimate of every arm it maximised (`scores`)
 * and, for randomised policies, its probabilities, and `learn` records the pulled arm's reward. States are plain data.
 * The round number t is one more than the pulls so far. The linear policies read the round's context, a flat
 * arms × d observation, and take d from the observation domain's shape. Also: the KL-UCB index and the Lai–Robbins
 * lower bound.
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
  counts: Float64Array
  sums: Float64Array
}

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

function armCount(env: EnvironmentShape): number {
  if (env.action.kind !== 'discrete')
    throw new DomainError('armCount', 'bandit policies need a discrete action domain (the arms)')
  return env.action.n
}

const stats = (arms: number): ArmStatistics => ({ counts: new Float64Array(arms), sums: new Float64Array(arms) })

function record<T extends ArmStatistics>(st: T, arm: number, reward: number): T {
  const counts = st.counts.slice()
  const sums = st.sums.slice()
  counts[arm] += 1
  sums[arm] += reward
  return { ...st, counts, sums }
}

/** The first index of the largest score: a deterministic greedy choice. */
const firstMax = (scores: ArrayLike<number>) => {
  let arg = 0
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[arg]) arg = i
  return arg
}

/** The share of pulls that went to the most-pulled arm: how settled a policy is, a training-curve scalar. */
const armScalars = (st: ArmStatistics) => {
  const n = st.counts.reduce((a, b) => a + b, 0)
  return { 'top arm share': n > 0 ? Math.max(...st.counts) / n : 0 }
}

/** Round t = 1, 2, …: one more than the pulls so far. */
const round = (st: ArmStatistics) => st.counts.reduce((a, b) => a + b, 0) + 1
const means = (st: ArmStatistics) => st.counts.map((c, i) => (c > 0 ? st.sums[i] / c : 0))
const firstUnpulled = (st: ArmStatistics) => st.counts.findIndex((c) => c === 0)

/** A context-free policy on arm statistics: `choose(st, t, s)` picks in round t; `learn` records the reward. */
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

/** An index policy: pulls every arm once, then the arm with the largest index. */
function indexAgent(name: string, index: (st: ArmStatistics, t: number) => Float64Array) {
  return armAgent(name, (st, t, s) => {
    const first = firstUnpulled(st)
    const scores = first >= 0 ? st.counts.map((c) => (c === 0 ? Infinity : -Infinity)) : index(st, t)
    return { action: first >= 0 ? first : argmaxRandom(scores, s), scores }
  })
}

/** Round robin over the arms (an A/B/n test): arm (t − 1) mod K. */
export function uniformPolicy(): Agent<ArmStatistics, unknown, number> {
  return armAgent('uniform', (st, t) => ({ action: (t - 1) % st.counts.length, scores: means(st) }))
}

/** Explore then commit: m pulls of every arm in turn, then the best empirical mean for ever. */
export function exploreThenCommit({ m = 50 }: { m?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return armAgent(`explore-then-commit (m = ${m})`, (st, t, s) => {
    const k = st.counts.length
    const scores = means(st)
    return { action: t <= m * k ? (t - 1) % k : argmaxRandom(scores, s), scores }
  })
}

/**
 * ε-greedy: with probability ε a uniformly random arm, otherwise the best empirical mean. With `decay: c` the
 * exploration rate is min(1, cK/t) (Auer, Cesa-Bianchi and Fischer, 2002, Machine Learning 47, §3).
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
 * UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): the index μ̂_a + √(c ln t / n_a), c = 2 by default, for rewards in
 * [0, 1].
 */
export function ucb1({ c = 2 }: { c?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return indexAgent('UCB1', (st, t) => st.counts.map((n, i) => st.sums[i] / n + Math.sqrt((c * Math.log(t)) / n)))
}

/** KL(Bernoulli(p) ‖ Bernoulli(q)) in nats, with 0 log 0 = 0; infinite when q is 0 or 1 and p is not. */
export function klBernoulli(p: number, q: number): number {
  const term = (x: number, y: number) => (x === 0 ? 0 : y === 0 ? Infinity : x * Math.log(x / y))
  return term(p, q) + term(1 - p, 1 - q)
}

/**
 * The KL-UCB index: the largest q ∈ [p, 1] with n · kl(p, q) ≤ level (Garivier and Cappé, 2011, COLT), by bisection to
 * 1e-10.
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

/** KL-UCB for rewards in [0, 1]: the index `klUcbIndex(μ̂_a, n_a, ln t + c ln ln t)`, c = 0 by default. */
export function klUcb({ c = 0 }: { c?: number } = {}): Agent<ArmStatistics, unknown, number> {
  return indexAgent('KL-UCB', (st, t) => {
    const level = Math.log(t) + (c > 0 && t > 1 ? c * Math.log(Math.log(t)) : 0)
    return st.counts.map((n, i) => klUcbIndex(st.sums[i] / n, n, level))
  })
}

/**
 * Thompson sampling for Bernoulli rewards (Thompson, 1933, Biometrika 25; Agrawal and Goyal, 2012, COLT): draw
 * θ_a ~ Beta(α + s_a, β + n_a − s_a) and pull the largest. Rewards outside {0, 1} are used as fractional successes.
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
 * Thompson sampling for Gaussian rewards with known noise sd σ and a N(μ₀, τ₀²) prior on each mean: the posterior of
 * arm a is N(m_a, v_a) with 1/v_a = 1/τ₀² + n_a/σ², m_a = v_a (μ₀/τ₀² + s_a/σ²).
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
  logWeights: Float64Array
}

/**
 * EXP3 for adversarial rewards in [0, 1] (Auer, Cesa-Bianchi, Freund and Schapire, 2002, SIAM J. Comput. 32(1)):
 * p_a = (1 − γ) w_a / Σ w + γ/K, and the pulled arm's weight grows by exp(γ r̂ / K) with the importance-weighted reward
 * r̂ = r / p_a (p from the weights that chose the arm, which `learn` recomputes).
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

/** The ridge statistics of a linear policy: V = λI + Σ x xᵀ (and its inverse) and b = Σ r x. */
export interface RidgeState extends ArmStatistics {
  /** V⁻¹, d × d row-major, kept by Sherman–Morrison updates. */
  vInverse: Float64Array
  b: Float64Array
  /** θ̂ = V⁻¹ b. */
  theta: Float64Array
  dim: number
}

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

/** Arm a's features: row a of the flat arms × d context. */
const featuresOf = (ctx: Float64Array, a: number, d: number) => ctx.subarray(a * d, (a + 1) * d)

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

function width(A: Float64Array, x: Float64Array, d: number): number {
  let s = 0
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) s += x[i] * A[i * d + j] * x[j]
  return Math.sqrt(Math.max(0, s))
}

/** The score of every arm under parameter w: xₐᵀw (+ a per-arm bonus). */
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
 * arm maximising xᵀθ̂ + α ‖x‖_{V⁻¹}, with the ridge estimate θ̂ = V⁻¹ b. α = 0 is the greedy ridge policy.
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
 * Linear Thompson sampling (Agrawal and Goyal, 2013, ICML): draw θ̃ ~ N(θ̂, v² V⁻¹) and pull the arm maximising xᵀθ̃.
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
 * has E[R_T] ≥ (1 + o(1)) ln T · Σ_{a: μ_a < μ*} (μ* − μ_a) / kl(μ_a, μ*). Returns the constant and the bound at each t.
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
