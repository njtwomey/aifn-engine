/**
 * Group-relative policy optimisation (Shao et al., 2024) of a softmax policy over a few actions per prompt, with the
 * variants of its later relatives, and the coverage problem on which reinforcement learning with verifiable rewards
 * sharpens a policy rather than teaching it new strategies (Yue et al., 2025).
 *
 * A problem gives each prompt $x$ the logits of its $K$ actions as a function of shared parameters $\thetavec$, and a
 * reward for each prompt and action. Each step samples a group of $G$ actions per prompt from the current policy
 * $\pi_{\mathrm{old}}$, scores them by `groupAdvantages`, and takes one or more optimiser steps on the clipped surrogate
 *
 * $$\mathcal{L}(\thetavec) = -\frac{1}{N} \sum_i \min\bigl(\rho_i A_i,\ \operatorname{clip}(\rho_i, 1 - \epsilon_{\mathrm{low}}, 1 + \epsilon_{\mathrm{high}}) A_i\bigr) + \beta\, \frac{1}{N} \sum_i k_3(i),$$
 *
 * where $\rho_i = \pi_{\thetavec}(a_i \mid x_i) / \pi_{\mathrm{old}}(a_i \mid x_i)$ and $k_3$ is Schulman's estimator of
 * $\mathrm{KL}(\pi_{\thetavec} \,\|\, \pi_{\mathrm{ref}})$ at the sample, $\pi_{\mathrm{ref}}$ the policy at the start.
 * Each response is one action, so the per-token aggregations coincide. The gradient comes from `valueAndGrad` and the
 * step from `aifn-compute/optim/first-order`'s `adamRule` or `sgdRule`.
 *
 * Samples of the same action for the same prompt share $\rho$, so the surrogate is evaluated per prompt on $K$-vectors:
 * $\sum_{i: a_i = a} \min(\rho_a A_i, c_a A_i) = \min(\rho_a, c_a) P_a + \max(\rho_a, c_a) N_a$, with $c_a$ the clipped
 * ratio and $P_a$, $N_a$ the sums of the positive and the negative advantages of the samples of $a$.
 */

import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { categorical, child, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  clip as clipValue,
  dot,
  exp,
  fromData,
  maximum,
  minimum,
  mul,
  sub,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { logSoftmax } from 'aifn-compute/numerics/special'
import { adamRule, applyUpdates, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { groupAdvantages, type Baseline, type Scale } from './advantages'

/**
 * A policy-optimisation problem with several prompts: the logits of each prompt's actions as a function of shared
 * parameters, and a reward for each prompt and action.
 */
export interface GroupPolicyProblem {
  /** The number of prompts $X$. */
  readonly prompts: Size
  /** The number of actions $K$ (candidate responses or strategies) per prompt. */
  readonly actions: Size
  /**
   * The logits of the $K$ actions for prompt `x`, written with `aifn-compute/foundation/tensor` operations so that the
   * gradient flows through them.
   */
  logits(theta: Value, x: Index): Value
  /** The reward of action `a` for prompt `x`. */
  reward(x: Index, a: Index): number
}

/** A PPO clipping range: one $\epsilon$ for both sides, or DAPO's decoupled `low` and `high`. */
export type ClipRange = number | { low: number; high: number }

/** Options of `grpoTrace`. */
export interface GrpoOptions {
  /** The number of actions $G$ sampled per prompt per step, at least 2. */
  groupSize: Size
  /** The optimiser's step size. */
  learningRate: number
  /** The seed of the samples. */
  seed: number
  /** The optimiser (default `'adam'`). */
  optimiser?: 'adam' | 'sgd'
  /** The prompts sampled per step (default all of them, in order); fewer are drawn at random each step. */
  promptsPerStep?: Size
  /** The weight $\beta \ge 0$ of the KL penalty to the starting policy (default 0, as in DAPO and Dr. GRPO). */
  beta?: number
  /** The clipping range (default 0.2; DAPO's clip-higher is `{ low: 0.2, high: 0.28 }`). */
  clip?: ClipRange
  /** The advantage's baseline and scale (default GRPO's group mean and group standard deviation). */
  advantage?: { baseline?: Baseline; scale?: Scale }
  /** Drop groups whose rewards are all equal, as DAPO's dynamic sampling does (default false). */
  dynamicSampling?: boolean
  /** Optimiser steps on each sampled batch, $\mu$ (default 1; the clipping only acts when it is above 1). */
  updatesPerBatch?: Size
}

/** One step of `grpoTrace`. */
export interface GrpoSnapshot {
  /** Steps taken: 0 is the start. */
  step: Size
  /** The parameters $\thetavec$. */
  theta: Float64Array
  /** The policy $\pi_{\thetavec}(\cdot \mid x)$ of every prompt. */
  probs: Float64Array[]
  /** The expected reward under the policy, averaged over prompts (exact, not sampled). */
  meanReward: number
  /** The rewards of the samples of the step's batch: their mean (NaN at step 0). */
  sampledReward: number
  /**
   * The standard deviation of each group's rewards ($G - 1$ divisor), averaged over the step's groups, as TRL's
   * `reward_std` (NaN at step 0).
   */
  sampledRewardStd: number
  /** The share of the step's groups whose rewards were all equal, so they had no gradient (0 at step 0). */
  fracZeroStd: number
  /** The policy's entropy, averaged over prompts. */
  entropy: number
  /** $\mathrm{KL}(\pi_{\thetavec} \,\|\, \pi_{\mathrm{ref}})$ to the starting policy, averaged over prompts (exact). */
  kl: number
  /** The share of the step's sample updates whose ratio was clipped, so they had no gradient (0 at step 0). */
  clipFraction: number
  /** The surrogate loss of the step's batch at its first update (NaN at step 0). */
  loss: number
}

/**
 * A softmax's probabilities from its logits.
 *
 * @param logits The logits.
 * @returns The probabilities, summing to 1.
 */
export function softmaxOf(logits: ArrayLike<number>): Float64Array {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i])
  const e = Float64Array.from(logits, (v) => Math.exp(v - max))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/**
 * The policy's probabilities for every prompt.
 *
 * @param problem The problem.
 * @param theta The parameters.
 * @returns $\pi_{\thetavec}(\cdot \mid x)$ for each prompt $x$.
 */
export function policyOf(problem: GroupPolicyProblem, theta: Tensor): Float64Array[] {
  return Array.from({ length: problem.prompts }, (_, x) => softmaxOf(toFlat(problem.logits(theta, x) as Tensor)))
}

/**
 * The entropy and the KL divergence to a reference, of one prompt's policy.
 *
 * @param p The policy's probabilities.
 * @param q The reference's probabilities.
 * @returns $H(p)$ and $\mathrm{KL}(p \,\|\, q)$, in nats.
 */
export function entropyAndKl(p: ArrayLike<number>, q: ArrayLike<number>): { entropy: number; kl: number } {
  let entropy = 0
  let kl = 0
  for (let i = 0; i < p.length; i++)
    if (p[i] > 0) {
      entropy -= p[i] * Math.log(p[i])
      kl += p[i] * Math.log(p[i] / q[i])
    }
  return { entropy, kl }
}

/**
 * The optimiser rule.
 *
 * @param optimiser Its name.
 * @param learningRate Its step size.
 * @returns The update rule.
 */
export function ruleOf(optimiser: 'adam' | 'sgd', learningRate: number): UpdateRule {
  return optimiser === 'adam' ? adamRule({ stepSize: learningRate }) : sgdRule({ stepSize: learningRate })
}

/** One prompt's sampled group, summarised per action for the surrogate. */
type Group = { x: Index; positive: Tensor; negative: Tensor; counts: Tensor; logpOld: Float64Array }

/**
 * The clipped surrogate of a batch of groups, with the KL penalty, as a function of the parameters.
 *
 * @param problem The problem.
 * @param groups The batch, summarised per action.
 * @param logpRef The starting policy's log-probabilities, per prompt.
 * @param low The lower clipping range $\epsilon_{\mathrm{low}}$.
 * @param high The upper clipping range $\epsilon_{\mathrm{high}}$.
 * @param beta The KL weight.
 * @returns The loss $\mathcal{L}(\thetavec)$.
 */
function surrogate(
  problem: GroupPolicyProblem,
  groups: readonly Group[],
  logpRef: readonly Float64Array[],
  low: number,
  high: number,
  beta: number,
): (theta: Value) => Value {
  const N = groups.reduce((n, g) => n + toFlat(g.counts).reduce((a, b) => a + b, 0), 0)
  return (theta) => {
    let total: Value = 0
    for (const g of groups) {
      const logp = logSoftmax(problem.logits(theta, g.x))
      const ratio = exp(sub(logp, fromData(g.logpOld, [problem.actions])))
      const clipped = clipValue(ratio, 1 - low, 1 + high)
      const gain = add(dot(minimum(ratio, clipped), g.positive), dot(maximum(ratio, clipped), g.negative))
      total = sub(total, gain)
      if (beta > 0) {
        // k3 = π_ref/π − log(π_ref/π) − 1 at each sample, weighted by how often each action was drawn.
        const logRatio = sub(fromData(logpRef[g.x], [problem.actions]), logp)
        const k3 = sub(sub(exp(logRatio), logRatio), 1)
        total = add(total, mul(beta, dot(k3, g.counts)))
      }
    }
    return mul(1 / N, total)
  }
}

/**
 * The parameters, the policy and its statistics as a snapshot.
 *
 * @param problem The problem.
 * @param step The step.
 * @param theta The parameters.
 * @param ref The starting policy, per prompt.
 * @param batch The step's batch statistics.
 * @returns The snapshot.
 */
function snapshotOf(
  problem: GroupPolicyProblem,
  step: Size,
  theta: Tensor,
  ref: readonly Float64Array[],
  batch: Pick<GrpoSnapshot, 'sampledReward' | 'sampledRewardStd' | 'fracZeroStd' | 'clipFraction' | 'loss'>,
): GrpoSnapshot {
  const probs = policyOf(problem, theta)
  let meanReward = 0
  let entropy = 0
  let kl = 0
  probs.forEach((p, x) => {
    for (let a = 0; a < problem.actions; a++) meanReward += (p[a] * problem.reward(x, a)) / problem.prompts
    const s = entropyAndKl(p, ref[x])
    entropy += s.entropy / problem.prompts
    kl += s.kl / problem.prompts
  })
  return { step, theta: Float64Array.from(toFlat(theta)), probs, meanReward, entropy, kl, ...batch }
}

/**
 * Group-relative policy optimisation as a trace: one snapshot per step, from the start (step 0), each holding the
 * parameters, every prompt's policy, its expected reward, entropy and KL to the start, and the step's batch
 * statistics in TRL's terms (the share of groups with equal rewards, the clipped share). Deterministic from the seed:
 * step $t$ draws from `child(stream(seed), 'step', t)`.
 *
 * The options cover GRPO's relatives: Dr. GRPO is `advantage: { scale: 'none' }` with `beta: 0`; DAPO is
 * `clip: { low: 0.2, high: 0.28 }` with `dynamicSampling: true` and `beta: 0`; RLOO's baseline is
 * `advantage: { baseline: 'leave-one-out', scale: 'none' }`.
 *
 * @param problem The prompts, the policy's logits and the rewards.
 * @param theta0 The starting parameters, which also define the reference policy of the KL penalty.
 * @param options The group size, the step size, the seed and the variant's options.
 * @param steps The number of steps.
 * @returns A generator of `steps + 1` snapshots.
 *
 * @example Sharpening on the coverage problem
 * const problem = coverageProblem({ prompts: 40, strategies: 8, density: 0.25, seed: 1 })
 * const trace = [...grpoTrace(problem, new Float64Array(8), { groupSize: 8, learningRate: 0.1, seed: 1 }, 60)]
 * for (const s of [trace[0], trace[60]]) print(`step ${s.step}: expected reward`, s.meanReward, ' entropy', s.entropy)
 */
export function* grpoTrace(
  problem: GroupPolicyProblem,
  theta0: ArrayLike<number>,
  options: GrpoOptions,
  steps: Size,
): Generator<GrpoSnapshot> {
  const {
    groupSize: G,
    learningRate,
    seed,
    optimiser = 'adam',
    beta = 0,
    clip = 0.2,
    advantage = {},
    dynamicSampling = false,
    updatesPerBatch = 1,
  } = options
  const X = problem.prompts
  const K = problem.actions
  const promptsPerStep = options.promptsPerStep ?? X
  const where = 'grpoTrace'
  if (!(Number.isInteger(G) && G >= 2)) throw new DomainError(where, `${where}: groupSize must be at least 2, got ${G}`)
  if (!(Number.isInteger(promptsPerStep) && promptsPerStep >= 1 && promptsPerStep <= X))
    throw new DomainError(where, `${where}: promptsPerStep must be an integer from 1 to ${X}`)
  if (!(Number.isInteger(updatesPerBatch) && updatesPerBatch >= 1))
    throw new DomainError(where, `${where}: updatesPerBatch must be a positive integer`)
  if (!(Number.isInteger(steps) && steps >= 0)) throw new DomainError(where, `${where}: steps must be non-negative`)
  const { low, high } = typeof clip === 'number' ? { low: clip, high: clip } : clip
  const root = stream(seed)
  let theta = fromData(Float64Array.from(theta0), [theta0.length])
  const ref = policyOf(problem, theta)
  const logpRef = ref.map((p) => p.map(Math.log))
  const rule = ruleOf(optimiser, learningRate)
  let state = rule.init(theta)
  yield snapshotOf(problem, 0, theta, ref, {
    sampledReward: NaN,
    sampledRewardStd: NaN,
    fracZeroStd: 0,
    clipFraction: 0,
    loss: NaN,
  })
  for (let t = 1; t <= steps; t++) {
    const s = child(root, 'step', t)
    const old = policyOf(problem, theta)
    // The step's prompts: all of them, or a random subset drawn without replacement.
    let chosen = Array.from({ length: X }, (_, x) => x)
    if (promptsPerStep < X) {
      const order = chosen.map((x) => ({ x, u: uniform(child(s, 'prompt', x)) })).sort((a, b) => a.u - b.u)
      chosen = order.slice(0, promptsPerStep).map((o) => o.x)
    }
    const groups: Group[] = []
    let rewardSum = 0
    let rewardCount = 0
    let zeroStd = 0
    let stdSum = 0
    for (const x of chosen) {
      const gs: Stream = child(s, 'group', x)
      const actions = Array.from({ length: G }, () => categorical(gs, old[x]))
      const rewards = actions.map((a) => problem.reward(x, a))
      rewards.forEach((r) => (rewardSum += r))
      rewardCount += G
      const mean = rewards.reduce((a, b) => a + b, 0) / G
      stdSum += Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / (G - 1))
      const equal = rewards.every((r) => r === rewards[0])
      if (equal) zeroStd++
      if (equal && dynamicSampling) continue
      const adv = groupAdvantages(rewards, advantage)
      const positive = new Float64Array(K)
      const negative = new Float64Array(K)
      const counts = new Float64Array(K)
      actions.forEach((a, i) => {
        counts[a]++
        if (adv[i] >= 0) positive[a] += adv[i]
        else negative[a] += adv[i]
      })
      groups.push({
        x,
        positive: fromData(positive, [K]),
        negative: fromData(negative, [K]),
        counts: fromData(counts, [K]),
        logpOld: old[x].map(Math.log),
      })
    }
    let loss = NaN
    let clipped = 0
    let samples = 0
    if (groups.length) {
      const objective = valueAndGrad(surrogate(problem, groups, logpRef, low, high, beta))
      for (let u = 0; u < updatesPerBatch; u++) {
        // The clipped share at the parameters of this update: samples whose ratio is past the side their advantage
        // pushes towards.
        const now = policyOf(problem, theta)
        for (const g of groups) {
          const [counts, positive, negative] = [toFlat(g.counts), toFlat(g.positive), toFlat(g.negative)]
          for (let a = 0; a < K; a++) {
            if (!counts[a]) continue
            const ratio = now[g.x][a] / Math.exp(g.logpOld[a])
            samples += counts[a]
            if (positive[a] > 0 && ratio > 1 + high) clipped += counts[a]
            else if (negative[a] < 0 && ratio < 1 - low) clipped += counts[a]
          }
        }
        const { value, grad } = objective(theta)
        if (u === 0) loss = Number(value)
        const step = rule.update(grad as Tensor, state, theta)
        state = step.state
        theta = applyUpdates(theta, step.updates) as Tensor
      }
    }
    yield snapshotOf(problem, t, theta, ref, {
      sampledReward: rewardSum / rewardCount,
      sampledRewardStd: stdSum / chosen.length,
      fracZeroStd: zeroStd / chosen.length,
      clipFraction: samples ? clipped / samples : 0,
      loss,
    })
  }
}

/** Options of `coverageProblem`. */
export interface CoverageOptions {
  /** The number of prompts $X$. */
  prompts: Size
  /** The number of strategies $S$, the actions shared by every prompt. */
  strategies: Size
  /** The probability that a strategy solves a prompt, from 0 to 1. */
  density: number
  /** The seed of the solves matrix. */
  seed: number
}

/**
 * The coverage problem of reinforcement learning with verifiable rewards: $X$ prompts share $S$ strategies, strategy
 * $s$ solves prompt $x$ when $C_{xs} = 1$ (each entry drawn independently with probability `density`, from
 * `child(stream(seed), 'solves')`), and the reward is $C_{xs}$. The policy is one softmax over strategies,
 * $\pi_{\thetavec}(s) \propto e^{\theta_s}$, shared by every prompt, so training raises the strategies that solve most
 * prompts; a prompt that only a rare strategy solves then becomes harder to solve at a large sampling budget. The
 * chance that a prompt is solved by one sample is $p_x = \sum_s \pi_s C_{xs}$, and its pass@$k$ is
 * $1 - (1 - p_x)^k$ (`passAtKExact`).
 *
 * @param options The number of prompts and strategies, the density and the seed.
 * @returns The problem, with the solves matrix $\Cmat$ ($X \times S$, row-major) as `solves`.
 *
 * @example How often each strategy solves a prompt
 * const problem = coverageProblem({ prompts: 40, strategies: 8, density: 0.25, seed: 1 })
 * const solved = Array.from({ length: 8 }, (_, s) => Array.from({ length: 40 }, (_, x) => problem.reward(x, s)).reduce((a, b) => a + b, 0))
 * print('prompts solved by each strategy:', solved)
 */
export function coverageProblem(options: CoverageOptions): GroupPolicyProblem & { solves: Uint8Array } {
  const { prompts: X, strategies: S, density, seed } = options
  const where = 'coverageProblem'
  if (!(Number.isInteger(X) && X >= 1)) throw new DomainError(where, `${where}: prompts must be a positive integer`)
  if (!(Number.isInteger(S) && S >= 2)) throw new DomainError(where, `${where}: strategies must be at least 2`)
  if (!(density >= 0 && density <= 1)) throw new DomainError(where, `${where}: density must be in [0, 1]`)
  const u = toFlat(uniform(child(stream(seed), 'solves'), 0, 1, { shape: [X * S] }) as Tensor)
  const solves = Uint8Array.from(u, (v) => (v < density ? 1 : 0))
  return {
    prompts: X,
    actions: S,
    solves,
    logits: (theta) => theta,
    reward: (x, a) => solves[x * S + a],
  }
}

/**
 * The probability that one sample solves each prompt: $p_x = \sum_s \pi(s \mid x) R(x, s)$ for a 0/1 reward.
 *
 * @param problem The problem.
 * @param probs The policy of every prompt, as a snapshot holds it.
 * @returns $p_x$ for each prompt.
 *
 * @example Before and after training, with pass@k averaged over prompts
 * const problem = coverageProblem({ prompts: 40, strategies: 8, density: 0.25, seed: 1 })
 * const trace = [...grpoTrace(problem, new Float64Array(8), { groupSize: 8, learningRate: 0.1, seed: 1 }, 80)]
 * const passAt = (probs, k) => solveProbabilities(problem, probs).reduce((a, p) => a + passAtKExact(p, k), 0) / 40
 * for (const k of [1, 16, 256]) print(`pass@${k}: start`, passAt(trace[0].probs, k), ' trained', passAt(trace[80].probs, k))
 */
export function solveProbabilities(problem: GroupPolicyProblem, probs: readonly ArrayLike<number>[]): Float64Array {
  return Float64Array.from({ length: problem.prompts }, (_, x) => {
    let p = 0
    for (let a = 0; a < problem.actions; a++) p += probs[x][a] * problem.reward(x, a)
    return p
  })
}
