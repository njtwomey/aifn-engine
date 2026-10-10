/**
 * Post-training on a toy policy: one prompt, $K$ candidate responses with fixed features $\phivec_k \in \reals^m$, and
 * the softmax policy $\pi_{\thetavec}(k) \propto \exp(\thetavec^\top \phivec_k)$, trained by supervised fine-tuning,
 * direct preference optimisation (or IPO, or SimPO) or group-relative policy optimisation, step by step.
 *
 * The policy is small enough to see whole: every snapshot holds all $K$ probabilities, the loss, and the statistics
 * TRL logs for each method (`logps/chosen`, `rewards/margins`, `rewards/accuracies` for DPO; `reward`,
 * `reward_std`, `frac_reward_zero_std` for GRPO), with the exact KL divergence to the starting policy and the
 * entropy. The starting policy is the reference of DPO and of GRPO's KL penalty. The losses are those of
 * `aifn-compute/learning/losses` (`dpo`, `ipo`, `simpo`), the gradients come from `valueAndGrad`, and the steps from
 * `adamRule` or `sgdRule`; GRPO runs `grpoTrace` on the one-prompt problem.
 */

import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { add, fromData, get, matmul, mul, neg, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { dpo, ipo, simpo } from 'aifn-compute/learning/losses'
import { logSoftmax } from 'aifn-compute/numerics/special'
import { applyUpdates } from 'aifn-compute/optim/first-order'
import type { Baseline, Scale } from './advantages'
import { entropyAndKl, grpoTrace, ruleOf, softmaxOf, type ClipRange, type GroupPolicyProblem } from './grpo'

/** A toy policy: the features of the $K$ responses ($K \times m$) and the starting parameters ($m$). */
export type ToyPolicy = {
  /** The features $\Phimat$, one row per candidate response. */
  features: Tensor | readonly (readonly number[])[]
  /** The starting parameters $\thetavec_0$, which also define the reference policy. */
  theta0: Tensor | readonly number[]
}

/** The method and its data. */
export type PostTrainingMethod =
  | {
      /** Supervised fine-tuning: maximise the log-likelihood of the target responses. */
      method: 'sft'
      /** The indices of the target responses (with repeats for weight). */
      targets: ArrayLike<number>
    }
  | {
      /** Preference optimisation on pairs of responses. */
      method: 'dpo'
      /** The preference pairs, by response index. */
      pairs: readonly { chosen: Index; rejected: Index }[]
      /** DPO's $\beta$; for `'ipo'` it is $\tau$, for `'simpo'` the reward scale $\beta$. */
      beta: number
      /** The loss (default `'dpo'`). */
      loss?: 'dpo' | 'ipo' | 'simpo'
      /** SimPO's margin $\gamma$ (default 0). */
      gamma?: number
      /** The weight of a negative log-likelihood term on the chosen responses, as in RPO (default 0). */
      nllWeight?: number
    }
  | {
      /** Group-relative policy optimisation with a fixed reward per response. */
      method: 'grpo'
      /** The reward of each response. */
      rewards: ArrayLike<number>
      /** The group size $G$, at least 2. */
      groupSize: Size
      /** The KL weight $\beta$ (default 0). */
      beta?: number
      /** The clipping range (default 0.2). */
      clip?: ClipRange
      /** The advantage's baseline and scale (default GRPO's). */
      advantage?: { baseline?: Baseline; scale?: Scale }
      /** Drop groups whose rewards are all equal, as DAPO does (default false). */
      dynamicSampling?: boolean
      /** Optimiser steps per sampled group (default 1). */
      updatesPerBatch?: Size
    }

/** Options of `postTrainingTrace`: the method, and the run's step size, length, seed and optimiser. */
export type PostTrainingOptions = PostTrainingMethod & {
  /** The optimiser's step size. */
  learningRate: number
  /** The number of steps, at most 5000. */
  steps: Size
  /** The seed of GRPO's samples (unused by SFT and DPO, which are deterministic). */
  seed: number
  /** The optimiser (default `'adam'`). */
  optimiser?: 'adam' | 'sgd'
}

/** One step of `postTrainingTrace`. */
export type PostTrainingSnapshot = {
  /** Steps taken: 0 is the start. */
  step: Size
  /** The policy's probabilities of the $K$ responses. */
  probs: Float64Array
  /** SFT and DPO: the loss at the snapshot's parameters. GRPO: the surrogate loss of the step's group (NaN at 0). */
  loss: number
  /** DPO: the mean log-probability of the chosen responses (TRL's `logps/chosen`). */
  logpChosen?: number
  /** DPO: the mean log-probability of the rejected responses (`logps/rejected`). */
  logpRejected?: number
  /**
   * DPO: the mean implicit reward margin $\beta(\rho_w - \rho_l)$ over the pairs (`rewards/margins`), with $\rho$ the
   * log-ratio $\log \pi_{\thetavec} - \log \pi_{\thetavec_0}$ of the chosen ($w$) and rejected ($l$) response.
   */
  margin?: number
  /** DPO: the share of pairs whose implicit reward favours the chosen response (`rewards/accuracies`). */
  accuracy?: number
  /** GRPO: the mean reward of the step's group (`reward`; NaN at step 0). */
  rewardMean?: number
  /** GRPO: the standard deviation of the step's group rewards (`reward_std`; NaN at step 0). */
  rewardStd?: number
  /** GRPO: 1 when the step's group had equal rewards and so no gradient, else 0 (`frac_reward_zero_std`). */
  fracZeroStd?: number
  /** $\mathrm{KL}(\pi_{\thetavec} \,\|\, \pi_{\thetavec_0})$, exact. */
  kl: number
  /** The policy's entropy $H(\pi_{\thetavec})$, in nats. */
  entropy: number
}

/**
 * Read a toy policy's features and parameters, checking their shapes.
 *
 * @param policy The policy.
 * @returns The features as a $K \times m$ tensor, the parameters as an $m$ tensor, and $K$.
 */
function readPolicy(policy: ToyPolicy): { features: Tensor; theta0: Tensor; K: Size } {
  const rows = Array.isArray(policy.features) ? (policy.features as readonly (readonly number[])[]) : null
  const features = rows
    ? fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])
    : (policy.features as Tensor)
  const theta0 = Array.isArray(policy.theta0)
    ? fromData(Float64Array.from(policy.theta0 as readonly number[]), [(policy.theta0 as readonly number[]).length])
    : (policy.theta0 as Tensor)
  if (features.shape.length !== 2 || theta0.shape.length !== 1 || features.shape[1] !== theta0.shape[0])
    throw new ShapeError(
      'postTrainingTrace',
      `postTrainingTrace: features must be K × m and theta0 of length m, got [${features.shape}] and [${theta0.shape}]`,
    )
  return { features, theta0, K: features.shape[0] }
}

/**
 * Check that response indices are responses of the policy.
 *
 * @param indices The indices.
 * @param K The number of responses.
 * @param what What they are, for the error message.
 */
function checkIndices(indices: ArrayLike<number>, K: Size, what: string): void {
  for (let i = 0; i < indices.length; i++)
    if (!(Number.isInteger(indices[i]) && indices[i] >= 0 && indices[i] < K))
      throw new DomainError(
        'postTrainingTrace',
        `postTrainingTrace: ${what} ${indices[i]} is not a response 0 … ${K - 1}`,
      )
}

/**
 * Train a toy softmax policy by SFT, DPO (or IPO, or SimPO) or GRPO and record every step: the probabilities of all
 * responses, the loss, the method's TRL statistics, and the KL divergence to the start and the entropy. SFT and DPO
 * take full-batch gradient steps and are deterministic; GRPO samples one group per step from `seed` (`grpoTrace`).
 *
 * @param policy The features of the responses and the starting parameters.
 * @param options The method and its data, the step size, the number of steps, the seed and the optimiser.
 * @returns `steps + 1` snapshots, from the start.
 *
 * @example DPO raises the chosen response and lowers the rejected one
 * const policy = { features: [[1, 0], [0, 1], [1, 1], [-1, 0]], theta0: [0, 0] }
 * const pairs = [{ chosen: 2, rejected: 3 }]
 * const trace = postTrainingTrace(policy, { method: 'dpo', pairs, beta: 0.5, learningRate: 0.05, steps: 100, seed: 1 })
 * for (const s of [trace[0], trace[100]]) print(`step ${s.step}: probs`, s.probs, ' margin', s.margin, ' KL', s.kl)
 *
 * @example SFT on one target, and the entropy it removes
 * const policy = { features: [[1, 0], [0, 1], [1, 1], [-1, 0]], theta0: [0, 0] }
 * const trace = postTrainingTrace(policy, { method: 'sft', targets: [1], learningRate: 0.1, steps: 50, seed: 1 })
 * print('entropy: start', trace[0].entropy, ' after 50 steps', trace[50].entropy)
 *
 * @example GRPO moves mass to the rewarded responses
 * const policy = { features: [[1, 0], [0, 1], [1, 1], [-1, 0]], theta0: [0, 0] }
 * const options = { method: 'grpo', rewards: [0, 1, 1, 0], groupSize: 8, learningRate: 0.05, steps: 100, seed: 1 }
 * const trace = postTrainingTrace(policy, options)
 * print('start', trace[0].probs, ' after 100 steps', trace[100].probs)
 */
export function postTrainingTrace(policy: ToyPolicy, options: PostTrainingOptions): PostTrainingSnapshot[] {
  const { features, theta0, K } = readPolicy(policy)
  const { learningRate, steps, seed, optimiser = 'adam' } = options
  if (!(Number.isInteger(steps) && steps >= 0 && steps <= 5000))
    throw new DomainError(
      'postTrainingTrace',
      `postTrainingTrace: steps must be an integer from 0 to 5000, got ${steps}`,
    )
  if (!(learningRate > 0))
    throw new DomainError('postTrainingTrace', 'postTrainingTrace: learningRate must be positive')
  const logitsOf = (theta: Value): Value => matmul(features, theta)
  const ref = softmaxOf(toFlat(logitsOf(theta0) as Tensor))
  const stats = (probs: Float64Array) => entropyAndKl(probs, ref)

  if (options.method === 'grpo') {
    if (options.rewards.length !== K)
      throw new ShapeError(
        'postTrainingTrace',
        `postTrainingTrace: ${options.rewards.length} rewards for ${K} responses`,
      )
    const problem: GroupPolicyProblem = {
      prompts: 1,
      actions: K,
      logits: (theta) => logitsOf(theta),
      reward: (_x, a) => options.rewards[a],
    }
    const out: PostTrainingSnapshot[] = []
    for (const s of grpoTrace(problem, toFlat(theta0), { ...options, optimiser, learningRate, seed }, steps)) {
      const probs = s.probs[0]
      out.push({
        step: s.step,
        probs,
        loss: s.loss,
        rewardMean: s.sampledReward,
        rewardStd: s.sampledRewardStd,
        fracZeroStd: s.fracZeroStd,
        ...stats(probs),
      })
    }
    return out
  }

  // SFT and DPO: a deterministic loss of the parameters, minimised by full-batch steps.
  let loss: (theta: Value) => Value
  let extras: (logp: Float64Array) => Partial<PostTrainingSnapshot> = () => ({})
  if (options.method === 'sft') {
    const targets = Array.from(options.targets)
    if (!targets.length) throw new DomainError('postTrainingTrace', 'postTrainingTrace: SFT needs at least one target')
    checkIndices(targets, K, 'target')
    loss = (theta) => {
      const logp = logSoftmax(logitsOf(theta))
      let total: Value = 0
      for (const k of targets) total = add(total, get(logp, k))
      return mul(-1 / targets.length, total)
    }
  } else {
    const { pairs, beta, loss: kind = 'dpo', gamma = 0, nllWeight = 0 } = options
    if (!pairs.length) throw new DomainError('postTrainingTrace', 'postTrainingTrace: DPO needs at least one pair')
    checkIndices(
      pairs.flatMap((p) => [p.chosen, p.rejected]),
      K,
      'response',
    )
    const logRef = ref.map(Math.log)
    const w = pairs.map((p) => p.chosen)
    const l = pairs.map((p) => p.rejected)
    loss = (theta) => {
      const logp = logSoftmax(logitsOf(theta))
      const gather = (idx: number[]) => idx.map((k) => get(logp, k))
      const pw = gather(w)
      const pl = gather(l)
      let total: Value = 0
      pairs.forEach((_, i) => {
        const term =
          kind === 'dpo'
            ? dpo(pw[i], pl[i], logRef[w[i]], logRef[l[i]], { beta })
            : kind === 'ipo'
              ? ipo(pw[i], pl[i], logRef[w[i]], logRef[l[i]], { tau: beta })
              : simpo(pw[i], pl[i], { beta, gamma })
        total = add(total, nllWeight ? add(term, mul(nllWeight, neg(pw[i]))) : term)
      })
      return mul(1 / pairs.length, total)
    }
    extras = (logp) => {
      const rewards = pairs.map((p) => ({
        w: beta * (logp[p.chosen] - logRef[p.chosen]),
        l: beta * (logp[p.rejected] - logRef[p.rejected]),
      }))
      const mean = (f: (i: number) => number) => pairs.reduce((a, _, i) => a + f(i), 0) / pairs.length
      return {
        logpChosen: mean((i) => logp[w[i]]),
        logpRejected: mean((i) => logp[l[i]]),
        margin: mean((i) => rewards[i].w - rewards[i].l),
        accuracy: mean((i) => (rewards[i].w > rewards[i].l ? 1 : 0)),
      }
    }
  }
  const rule = ruleOf(optimiser, learningRate)
  const objective = valueAndGrad(loss)
  let theta = theta0
  let state = rule.init(theta)
  const out: PostTrainingSnapshot[] = []
  for (let t = 0; ; t++) {
    const { value, grad } = objective(theta)
    const logp = toFlat(logSoftmax(logitsOf(theta)) as Tensor)
    const probs = Float64Array.from(logp, Math.exp)
    out.push({ step: t, probs, loss: Number(value), ...extras(Float64Array.from(logp)), ...stats(probs) })
    if (t === steps) return out
    const step = rule.update(grad as Tensor, state, theta)
    state = step.state
    theta = applyUpdates(theta, step.updates) as Tensor
  }
}
