/**
 * Label models: estimate each example's class from several noisy votes (labelling functions or crowd workers) without
 * any true labels. Majority vote counts votes; the Dawid–Skene model (Dawid and Skene, 1979) learns each voter's full
 * confusion matrix by expectation–maximisation; the data-programming label model (Ratner et al., 2016) learns each
 * labelling function's accuracy and coverage by maximising the marginal likelihood of the votes. A vote is a class in
 * $0, \dots, K - 1$, or $-1$ for an abstention (the voter said nothing). Every model returns posterior class
 * probabilities as an $n \times K$ tensor, one row per example.
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

/**
 * Votes: an $n \times m$ matrix of classes, $-1$ where voter $j$ abstained on example $i$. `n` is the number of
 * examples, `m` the number of voters, and `votes` the matrix, row-major: voter `j`'s vote on example `i` is
 * `votes[i * m + j]`.
 */
export type Votes = { readonly n: Size; readonly m: Size; readonly votes: Int32Array }

/**
 * Votes from rows (example by voter) or a tensor of shape $[n, m]$. The values are copied into an `Int32Array`, so a
 * fractional vote is truncated; nothing is checked.
 *
 * @param v The votes: one row per example with one entry per voter, each a class in $0, \dots, K - 1$ or $-1$ for an
 *   abstention. With rows, `m` is the length of the first row (0 when there are none).
 * @returns The votes as `Votes`.
 *
 * @example Three examples, two voters, one abstention
 * const v = votesOf([[0, 1], [1, 1], [-1, 0]])
 * print('n =', v.n, ' m =', v.m)
 * print('votes =', v.votes)
 */
export function votesOf(v: Tensor | readonly (readonly number[])[]): Votes {
  if (Array.isArray(v)) {
    const rows = v as number[][]
    return { n: rows.length, m: rows[0]?.length ?? 0, votes: Int32Array.from(rows.flat()) }
  }
  const t = v as Tensor
  return { n: t.shape[0], m: t.shape[1], votes: Int32Array.from(toFlat(t)) }
}

/**
 * Majority vote: each example's whole probability goes to the class with the most non-abstaining votes, split evenly
 * between the classes tied for most; an example with no votes gets the uniform distribution.
 *
 * @param votes The votes, $n$ examples by $m$ voters. A vote outside $0, \dots, K - 1$ other than $-1$ is not counted.
 * @param classes The number of classes $K$.
 * @returns The posteriors, $n \times K$: 1 on the winning class, $1/t$ on each of $t$ tied classes, 0 elsewhere.
 *
 * @example A clear winner, a tie and an example nobody voted on
 * const v = votesOf([[0, 0, 1], [0, 1, -1], [-1, -1, -1]])
 * print('posteriors =', majorityVote(v, 2))
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

/** Options of `dawidSkeneSteps` and `dawidSkene`. */
export type DawidSkeneOptions = {
  /** Pseudo-counts added to every confusion cell and class count in the M-step (default 0.01). */
  smoothing?: number
}

/** The state of `dawidSkeneSteps`. */
export interface DawidSkeneState extends Status {
  /** EM steps taken (0 after the start from majority vote). */
  t: Size
  /** Class priors $\pivec$, $K$ values. */
  priors: Tensor
  /**
   * Confusion matrices $\thetavec$, $m \times K \times K$: $\theta_{jkl} = p(\text{voter } j \text{ says } l \mid
   * \text{class } k)$, each row summing to 1.
   */
  confusions: Tensor
  /** Posterior class probabilities $\Tmat$, $n \times K$. */
  posteriors: Tensor
  /** The log-likelihood of the votes under the current parameters. */
  logLikelihood: number
}

/**
 * The Dawid–Skene model (Dawid and Skene, 1979) fitted by expectation–maximisation, as a step-through algorithm. Each
 * class $k$ has a prior $\pi_k$ and each voter $j$ a confusion matrix $\thetavec_j$; given the class, votes are
 * independent. The E-step sets $T_{ik} \propto \pi_k \prod_j \theta_{jk v_{ij}}$ over the votes cast; the M-step sets
 * $\pi_k \propto s + \sum_i T_{ik}$ and $\theta_{jkl} \propto s + \sum_i T_{ik} \indicator[v_{ij} = l]$, with $s$ the
 * `smoothing`. The start is an M-step from the majority-vote posteriors and an E-step; each step is one M-step and one
 * E-step, and the state is `converged` when the log-likelihood changes by less than $10^{-9}(1 + \lvert \ell \rvert)$.
 * With `smoothing` the M-step is a MAP step under a Dirichlet prior, so the quantity that never decreases is the
 * log-likelihood plus that log prior. Deterministic: the stream is not used.
 *
 * @param votes The votes, $n$ examples by $m$ voters, $-1$ for an abstention.
 * @param classes The number of classes $K$.
 * @param options The `smoothing` pseudo-count.
 * @returns The algorithm, to run with `run` or `trace` (its input is unused).
 *
 * @example Three annotators, one of them careless, step by step
 * const s = stream(1)
 * const right = [0.9, 0.8, 0.55]
 * const truth = Array.from({ length: 60 }, () => (uniform(s) < 0.5 ? 1 : 0))
 * const rows = truth.map((y) => right.map((a) => (uniform(s) < a ? y : 1 - y)))
 * const alg = dawidSkeneSteps(votesOf(rows), 2)
 * const ll = [0, 1, 2, 5, 50].map((t) => run(alg, undefined, t).logLikelihood)
 * print('log-likelihood after 0, 1, 2, 5 and 50 steps:', ll)
 * const end = run(alg, undefined, 50)
 * print('P(class 1) of the first five examples:', end.posteriors.data.filter((_, i) => i % 2).slice(0, 5))
 * print('their true classes:', truth.slice(0, 5))
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

/**
 * Run Dawid–Skene EM (`dawidSkeneSteps`) to convergence, or for `maxSteps` steps, and return the final state.
 *
 * @param votes The votes, $n$ examples by $m$ voters, $-1$ for an abstention.
 * @param classes The number of classes $K$.
 * @param options The `smoothing` pseudo-count of `DawidSkeneOptions`, and `maxSteps`, the most EM steps taken (default
 *   200).
 * @returns The final state: priors, confusion matrices, posteriors and log-likelihood.
 *
 * @example Recover five annotators' accuracies without any true label
 * const s = stream(2)
 * const right = [0.95, 0.85, 0.75, 0.65, 0.55]
 * const truth = Array.from({ length: 100 }, () => (uniform(s) < 0.4 ? 1 : 0))
 * const rows = truth.map((y) => right.map((a) => (uniform(s) < a ? y : 1 - y)))
 * const fit = dawidSkene(votesOf(rows), 2)
 * const c = fit.confusions.data
 * print('true accuracies:', right)
 * print('P(says 0 | class 0) per annotator:', [0, 1, 2, 3, 4].map((j) => c[4 * j]))
 * print('P(says 1 | class 1) per annotator:', [0, 1, 2, 3, 4].map((j) => c[4 * j + 3]))
 * print('priors (truth has 40% of class 1):', fit.priors, ' EM steps:', fit.t)
 */
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

/**
 * Parameters of the label model: `classBalance`, $K$ unnormalised log class priors (the priors are its softmax), and
 * `accuracy` and `coverage`, $m$ values each, every labelling function's $\alpha_j$ and $\beta_j$ on the logit scale.
 */
export type LabelModelParams = { classBalance: Tensor; accuracy: Tensor; coverage: Tensor }

/** A fitted label model: the estimates and the posteriors they give. */
export type LabelModel = {
  /** Class priors $\pivec$, $K$ values (uniform with `fixedBalance`). */
  classBalance: Float64Array
  /** $\alpha_j = p(\lambda_j = y \mid \lambda_j \ne -1)$ for each function $j$. */
  accuracy: Float64Array
  /** $\beta_j = p(\lambda_j \ne -1)$ for each function $j$. */
  coverage: Float64Array
  /** Posterior class probabilities, $n \times K$. */
  posteriors: Tensor
  /** The final negative log marginal likelihood, averaged over the examples. */
  loss: number
  /** L-BFGS steps taken. */
  steps: Size
}

/**
 * The log-likelihood of each example's votes under the label model, $\log \sum_y \pi_y \prod_j p(\lambda_{ij} \mid y)$,
 * with $p(\lambda = -1 \mid y) = 1 - \beta_j$, $p(\lambda = y \mid y) = \beta_j \alpha_j$ and
 * $p(\lambda = l \mid y) = \beta_j (1 - \alpha_j)/(K - 1)$ for each wrong class $l$, as a differentiable $n \times K$
 * matrix of the per-class terms $\log \pi_y + \sum_j \log p(\lambda_{ij} \mid y)$ (before the log-sum-exp over $y$).
 * With $K = 1$ the divisor $K - 1$ is taken as 1.
 *
 * @param p The parameters: logits of the class balance, the accuracies $\alpha_j$ and the coverages $\beta_j$.
 * @param votes The votes, $n$ examples by $m$ voters, $-1$ for an abstention.
 * @param K The number of classes.
 * @returns The $n \times K$ matrix of per-class log terms.
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
  /** The most L-BFGS steps taken (default 200). */
  maxSteps?: Size
  /** Fix the class balance at uniform instead of learning it (default false). */
  fixedBalance?: boolean
}

/**
 * The data-programming label model (Ratner, De Sa, Wu, Selsam and Ré, 2016; the generative model behind Snorkel):
 * labelling functions are conditionally independent given the class; function $j$ votes with probability $\beta_j$
 * and, when it votes, is right with probability $\alpha_j$ and otherwise names one of the $K - 1$ wrong classes
 * uniformly. The class balance $\pivec$, $\alphavec$ and $\betavec$ are fitted by maximising the marginal likelihood
 * $\sum_i \log \sum_y \pi_y \prod_j p(\lambda_{ij} \mid y)$, with gradients from `aifn-compute/foundation/autodiff`
 * and full-batch L-BFGS from `aifn-compute/nn/training`, until it stops or after `maxSteps` steps; the posteriors
 * weight each function by its learned accuracy. Accuracies start at 0.7 so the solution with the classes' names
 * swapped is not chosen; coverages start at 0.5 and the class balance at uniform. Deterministic.
 *
 * @param votes The votes, $n$ examples by $m$ labelling functions, $-1$ for an abstention.
 * @param classes The number of classes $K$.
 * @param options The most L-BFGS steps, and whether the class balance is fixed at uniform.
 * @returns The estimated class balance, accuracies and coverages, the posteriors, the final loss and the steps taken.
 *
 * @example Three noisy labelling functions with known accuracies and coverages
 * const s = stream(3)
 * const acc = [0.9, 0.75, 0.6]
 * const cov = [0.5, 0.7, 0.9]
 * const truth = Array.from({ length: 200 }, () => (uniform(s) < 0.5 ? 1 : 0))
 * const rows = truth.map((y) => acc.map((a, j) => (uniform(s) < cov[j] ? (uniform(s) < a ? y : 1 - y) : -1)))
 * const fit = labelModel(votesOf(rows), 2)
 * print('true accuracy:', acc, ' estimated:', fit.accuracy)
 * print('true coverage:', cov, ' estimated:', fit.coverage)
 * print('class balance:', fit.classBalance, ' steps:', fit.steps)
 *
 * @example Two functions disagree: the posterior sides with the more accurate one
 * const s = stream(4)
 * const acc = [0.95, 0.6, 0.6]
 * const truth = Array.from({ length: 200 }, () => (uniform(s) < 0.5 ? 1 : 0))
 * const rows = truth.map((y) => acc.map((a) => (uniform(s) < a ? y : 1 - y)))
 * const fit = labelModel(votesOf([...rows, [0, 1, 1]]), 2)
 * print('accuracies:', fit.accuracy)
 * print('posterior of the votes [0, 1, 1]:', fit.posteriors.data.slice(-2))
 * print('majority vote of the same:', majorityVote(votesOf([[0, 1, 1]]), 2))
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
