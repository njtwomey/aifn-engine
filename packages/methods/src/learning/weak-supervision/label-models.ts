/**
 * Label models: estimate each example's class from several noisy votes (labelling functions or crowd workers) without
 * any true labels. Majority vote counts votes; the Dawid–Skene model (Dawid and Skene, 1979) learns each voter's full
 * confusion matrix by expectation–maximisation; the data-programming label model (Ratner et al., 2016) learns each
 * labelling function's accuracy and coverage by maximising the marginal likelihood of the votes. A vote is a class in
 * 0 … K − 1, or −1 for an abstention (the voter said nothing).
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, stream as makeStream } from 'aifn-compute/foundation/random'
import {
  add,
  exp,
  fromData,
  logsumexp,
  mean,
  mul,
  neg,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { methodTraining } from 'aifn-compute/nn/training'
import { logSigmoid, logSoftmax, softmax } from 'aifn-compute/numerics/special'

/** Votes: an n × m matrix of classes, −1 where voter j abstained on example i. */
export type Votes = { readonly n: Size; readonly m: Size; readonly votes: Int32Array }

/** Votes from rows (example by voter) or an int32 tensor [n, m]. */
export function votesOf(v: Tensor | readonly (readonly number[])[]): Votes {
  if (Array.isArray(v)) {
    const rows = v as number[][]
    return { n: rows.length, m: rows[0]?.length ?? 0, votes: Int32Array.from(rows.flat()) }
  }
  const t = v as Tensor
  return { n: t.shape[0], m: t.shape[1], votes: Int32Array.from(toFlat(t)) }
}

/**
 * Majority vote: each example's class probabilities are the shares of its non-abstaining votes, ties split evenly; an
 * example with no votes gets the uniform distribution. Returns [n, K].
 */
export function majorityVote(votes: Votes, classes: Size): Tensor {
  const { n, m } = votes
  const out = new Float64Array(n * classes)
  for (let i = 0; i < n; i++) {
    const count = new Float64Array(classes)
    for (let j = 0; j < m; j++) {
      const v = votes.votes[i * m + j]
      if (v >= 0) count[v]++
    }
    const top = Math.max(...count)
    const winners = top === 0 ? classes : count.filter((c) => c === top).length
    for (let k = 0; k < classes; k++) out[i * classes + k] = top === 0 || count[k] === top ? 1 / winners : 0
  }
  return fromData(out, [n, classes])
}

// ── Dawid–Skene ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `dawidSkeneSteps`. */
export type DawidSkeneOptions = {
  /** Pseudo-counts added to every confusion cell and class count in the M-step (default 0.01). */
  smoothing?: number
}

/** The state of `dawidSkeneSteps`. */
export interface DawidSkeneState extends Status {
  t: Size
  /** Class priors π [K]. */
  priors: Tensor
  /** Confusion matrices θ [m, K, K]: θ_j[k][l] = P(voter j says l | class k). */
  confusions: Tensor
  /** Posterior class probabilities T [n, K]. */
  posteriors: Tensor
  /** The log-likelihood of the votes under the current parameters. */
  logLikelihood: number
}

/**
 * The Dawid–Skene model fitted by expectation–maximisation, as a step-through algorithm. Each class k has a prior π_k
 * and each voter j a confusion matrix θ_j; given the class, votes are independent. The E-step sets
 * T_ik ∝ π_k Π_j θ_j[k][v_ij] over the votes cast; the M-step sets π_k ∝ Σ_i T_ik and θ_j[k][l] ∝ Σ_i T_ik 1[v_ij = l].
 * The first M-step starts from the majority-vote posteriors. With `smoothing` the M-step is a MAP step under a
 * Dirichlet prior, so the quantity that never decreases is the log-likelihood plus that log prior. Deterministic.
 */
export function dawidSkeneSteps(
  votes: Votes,
  classes: Size,
  options: DawidSkeneOptions = {},
): Algorithm<void, DawidSkeneState> {
  const { smoothing = 0.01 } = options
  const { n, m } = votes
  const K = classes
  const mStep = (T: Float64Array) => {
    const priors = new Float64Array(K).fill(smoothing)
    const conf = new Float64Array(m * K * K).fill(smoothing)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        const t = T[i * K + k]
        priors[k] += t
        for (let j = 0; j < m; j++) {
          const v = votes.votes[i * m + j]
          if (v >= 0) conf[(j * K + k) * K + v] += t
        }
      }
    const total = priors.reduce((a, b) => a + b, 0)
    priors.forEach((p, k) => (priors[k] = p / total))
    for (let j = 0; j < m; j++)
      for (let k = 0; k < K; k++) {
        let row = 0
        for (let l = 0; l < K; l++) row += conf[(j * K + k) * K + l]
        for (let l = 0; l < K; l++) conf[(j * K + k) * K + l] /= row
      }
    return { priors, conf }
  }
  const eStep = (priors: Float64Array, conf: Float64Array) => {
    // log π_k + Σ_j log θ_j[k][v_ij] for every example and class; the posteriors are its row softmax and the
    // log-likelihood the sum of its row log-sum-exps.
    const lp = new Float64Array(n * K)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        let s = Math.log(priors[k])
        for (let j = 0; j < m; j++) {
          const v = votes.votes[i * m + j]
          if (v >= 0) s += Math.log(conf[(j * K + k) * K + v])
        }
        lp[i * K + k] = s
      }
    const L = fromData(lp, [n, K])
    const T = Float64Array.from(toFlat(softmax(L)))
    const ll = toFlat(logsumexp(L, -1) as Tensor).reduce((a, b) => a + b, 0)
    return { T, ll }
  }
  const state = (t: Size, priors: Float64Array, conf: Float64Array, T: Float64Array, ll: number): DawidSkeneState => ({
    t,
    priors: fromData(priors, [K]),
    confusions: fromData(conf, [m, K, K]),
    posteriors: fromData(T, [n, K]),
    logLikelihood: ll,
  })
  return {
    name: 'dawid-skene-em',
    init: () => {
      const { priors, conf } = mStep(Float64Array.from(toFlat(majorityVote(votes, K))))
      const { T, ll } = eStep(priors, conf)
      return state(0, priors, conf, T, ll)
    },
    step: (s) => {
      const { priors, conf } = mStep(Float64Array.from(s.posteriors.data))
      const { T, ll } = eStep(priors, conf)
      const next = state(s.t + 1, priors, conf, T, ll)
      return { ...next, converged: Math.abs(ll - s.logLikelihood) < 1e-9 * (1 + Math.abs(ll)) }
    },
  }
}

/** Run Dawid–Skene EM to convergence (or `maxSteps`, default 200) and return the final state. */
export function dawidSkene(
  votes: Votes,
  classes: Size,
  options: DawidSkeneOptions & { maxSteps?: Size } = {},
): DawidSkeneState {
  const alg = dawidSkeneSteps(votes, classes, options)
  const s0 = makeStream('dawid-skene')
  let s = alg.init(undefined, s0)
  for (let t = 0; t < (options.maxSteps ?? 200) && !s.converged; t++) s = alg.step(s, { t, stream: child(s0, t) })
  return s
}

// ── The data-programming label model ─────────────────────────────────────────────────────────────────────────────────

/** Parameters of the label model: log class balance, and each function's accuracy and coverage on the logit scale. */
export type LabelModelParams = { classBalance: Tensor; accuracy: Tensor; coverage: Tensor }

/** A fitted label model: the estimates and the posteriors they give. */
export type LabelModel = {
  /** Class priors [K]. */
  classBalance: Float64Array
  /** P(λ_j = y | λ_j votes) for each function. */
  accuracy: Float64Array
  /** P(λ_j votes) for each function. */
  coverage: Float64Array
  /** Posterior class probabilities [n, K]. */
  posteriors: Tensor
  /** The final negative log-likelihood per example. */
  loss: number
  steps: Size
}

/**
 * The log-likelihood of each example's votes under the label model, log Σ_y π_y Π_j p(λ_ij | y), with
 * p(λ = −1 | y) = 1 − β_j, p(λ = y | y) = β_j α_j and p(λ = l | y) = β_j (1 − α_j)/(K − 1) for each wrong class l, as a
 * differentiable [n, K] matrix of per-class terms (before the log-sum-exp over y).
 */
function jointTerms(p: LabelModelParams, votes: Votes, K: Size): Value {
  const { n, m } = votes
  const logAcc = logSigmoid(p.accuracy) // [m]
  const logErr = sub(logSigmoid(neg(p.accuracy)), Math.log(Math.max(1, K - 1)))
  const logCov = logSigmoid(p.coverage)
  const logAbs = logSigmoid(neg(p.coverage))
  // Indicator constants: for each example and class, how each function's vote relates to y.
  const agree = new Float64Array(n * K * m)
  const disagree = new Float64Array(n * K * m)
  const abstain = new Float64Array(n * K * m)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      const v = votes.votes[i * m + j]
      for (let k = 0; k < K; k++) {
        const o = (i * K + k) * m + j
        if (v < 0) abstain[o] = 1
        else if (v === k) agree[o] = 1
        else disagree[o] = 1
      }
    }
  const shape = [n, K, m]
  const perVote = add(
    add(mul(fromData(agree, shape), add(logCov, logAcc)), mul(fromData(disagree, shape), add(logCov, logErr))),
    mul(fromData(abstain, shape), logAbs),
  )
  return add(sum(perVote, -1), logSoftmax(p.classBalance))
}

/** Options of `labelModel`. */
export type LabelModelOptions = {
  /** L-BFGS steps at most (default 200). */
  maxSteps?: Size
  /** Fix the class balance at uniform instead of learning it (default false). */
  fixedBalance?: boolean
}

/**
 * The data-programming label model (Ratner, De Sa, Wu, Selsam and Ré, 2016; the generative model behind Snorkel):
 * labelling functions are conditionally independent given the class; function j votes with probability β_j and, when
 * it votes, is right with probability α_j and otherwise names one of the K − 1 wrong classes uniformly. The class
 * balance, α and β are fitted by maximising the marginal likelihood Σ_i log Σ_y π_y Π_j p(λ_ij | y), with gradients from
 * `aifn-compute/foundation/autodiff` and full-batch L-BFGS from `aifn-compute/nn/training`; the posteriors weight each function by its
 * learned accuracy. Accuracies start at 0.7 so the solution with the classes' names swapped is not chosen.
 */
export function labelModel(votes: Votes, classes: Size, options: LabelModelOptions = {}): LabelModel {
  const K = classes
  const { m } = votes
  const init: LabelModelParams = {
    classBalance: fromData(new Float64Array(K), [K]),
    accuracy: fromData(new Float64Array(m).fill(Math.log(0.7 / 0.3)), [m]),
    coverage: fromData(new Float64Array(m), [m]),
  }
  const loss = (p: LabelModelParams) => {
    const q = options.fixedBalance ? { ...p, classBalance: init.classBalance } : p
    return neg(mean(logsumexp(jointTerms(q, votes, K), -1)))
  }
  const alg = methodTraining<LabelModelParams, Record<string, Tensor>>((p) => loss(p), {}, { method: 'lbfgs' })
  const s0 = makeStream('label-model')
  let s = alg.init({ params: init }, s0)
  const maxSteps = options.maxSteps ?? 200
  let t = 0
  for (; t < maxSteps && !s.stopped; t++) s = alg.step(s, { t, stream: child(s0, t) })
  const p = s.params
  const terms = jointTerms(options.fixedBalance ? { ...p, classBalance: init.classBalance } : p, votes, K) as Tensor
  const posteriors = exp(sub(terms, logsumexp(terms, -1, true))) as Tensor
  const sig = (v: number) => 1 / (1 + Math.exp(-v))
  const bal = toFlat(logSoftmax(options.fixedBalance ? init.classBalance : p.classBalance)).map(Math.exp)
  return {
    classBalance: Float64Array.from(bal),
    accuracy: Float64Array.from(toFlat(p.accuracy), sig),
    coverage: Float64Array.from(toFlat(p.coverage), sig),
    posteriors,
    loss: s.loss,
    steps: t,
  }
}
