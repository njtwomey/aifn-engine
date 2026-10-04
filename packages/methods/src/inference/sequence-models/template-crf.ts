/**
 * The linear-chain CRF with CRF++-style feature templates (Lafferty, McCallum & Pereira 2001; Kudo 2005, CRF++;
 * Okazaki 2007, CRFsuite), in the notation of Twomey, Diethe & Flach (2016):
 *
 *   P(y | x) = (1/Z) Π_n ψ_n(y_n) Π_{n ≥ 1} Ψ_{n−1}(y_{n−1}, y_n),
 *   ψ_n(k) = exp Σ_{u fires at n} λ_{u,k},   Ψ_{n−1}(i, k) = exp Σ_{b fires at n} λ_{b,i,k}.
 *
 * Every unigram string u of the feature index (`aifn-compute/text/features`) is K feature functions
 * f_{u,k}(y_n, x, n) = [u fires at n][y_n = k], and every bigram string b is K² functions
 * f_{b,i,k}(y_{n−1}, y_n, x, n) = [b fires at n][y_{n−1} = i][y_n = k]. The weights λ are one vector: the unigram
 * block (string u, label k at u·K + k), then the bigram block (string b, labels i, k at U·K + b·K² + i·K + k).
 *
 * Inference runs the chain engines of `aifn-compute/inference/exact` on the log-potentials log ψ_n (N × K) and log Ψ_n
 * ((N − 1) × K × K): forward messages α_n = Ψ_{n−1}ᵀ γ_{n−1} with γ_n = α_n ⊙ ψ_n, backward messages
 * β_n = Ψ_n δ_{n+1} with δ_{n+1} = β_{n+1} ⊙ ψ_{n+1}, marginals ∝ α_n ⊙ ψ_n ⊙ β_n, and Viterbi.
 *
 * Training minimises the regularised negative conditional log-likelihood
 *
 *   L(λ) = −Σ_m log P(y_m | x_m) + c₁‖λ‖₁ + c₂‖λ‖²
 *
 * (CRFsuite's c1 and c2; CRF++'s `-c C` is c₂ = 1/(2C) with L2, c₁ = 1/C with L1). Its gradient is expected minus
 * observed feature counts (Sutton & McCallum 2012, eq. 5.6), read from the node and pairwise marginals. L-BFGS (c₁ = 0)
 * and OWL-QN (c₁ > 0) come from `aifn-compute/optim/second-order`; SGD and Adam take minibatches of sequences.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  chainForwardBackward,
  chainViterbi,
  posteriorDecode,
  type ChainMarginals,
  type PosteriorDecoding,
  type ViterbiResult,
} from 'aifn-compute/inference/exact'
import { adamRule, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { owlqn, type OwlqnState } from 'aifn-compute/optim/second-order'
import {
  encodeTemplateRows,
  featureIndex,
  type EncodedSequence,
  type FeatureIndex,
  type FeatureTemplates,
  type TemplateMacro,
  type TokenRows,
} from 'aifn-compute/text/features'

/** A training sequence: token rows (one per position, string columns) and a label per position. */
export interface LabelledSequence {
  readonly rows: TokenRows
  readonly labels: readonly string[]
}

/** A linear-chain CRF over the strings of a feature index. */
export interface TemplateCrf {
  readonly kind: 'template-crf'
  readonly index: FeatureIndex
  /** The label set, in weight order. */
  readonly labels: readonly string[]
  /** λ: unigram block (U × K), then bigram block (B × K × K). */
  readonly weights: Float64Array
}

/** The number of weights of a CRF over `index` with `K` labels: U·K + B·K². */
export const crfWeightCount = (index: FeatureIndex, K: Size): Size =>
  index.unigram.length * K + index.bigram.length * K * K

/** A CRF over an index and labels, with zero weights unless given. */
export function templateCrf(index: FeatureIndex, labels: readonly string[], weights?: ArrayLike<number>): TemplateCrf {
  const D = crfWeightCount(index, labels.length)
  if (labels.length < 2) throw new DomainError('templateCrf', 'templateCrf: needs at least two labels')
  if (weights && weights.length !== D)
    throw new DomainError('templateCrf', `templateCrf: expected ${D} weights, got ${weights.length}`)
  return { kind: 'template-crf', index, labels, weights: weights ? Float64Array.from(weights) : new Float64Array(D) }
}

/** The labels of the data in order of first appearance. */
export function labelSet(data: readonly LabelledSequence[]): string[] {
  const seen = new Set<string>()
  for (const s of data) for (const y of s.labels) seen.add(y)
  return [...seen]
}

/**
 * The log-potentials of an encoded sequence: log ψ_n(k) (N × K) and log Ψ_{n}(i, k) between positions n and n + 1
 * ((N − 1) × K × K), each the sum of the weights of the strings that fire.
 */
export function crfLogPotentials(
  weights: ArrayLike<number>,
  e: EncodedSequence,
  U: Size,
  K: Size,
): { logUnary: Matrix; logPairwise: Tensor } {
  const N = e.length
  const unary = new Float64Array(N * K)
  for (let n = 0; n < N; n++)
    for (let j = e.unigramStart[n]; j < e.unigramStart[n + 1]; j++) {
      const at = e.unigramIds[j] * K
      for (let k = 0; k < K; k++) unary[n * K + k] += weights[at + k]
    }
  const KK = K * K
  const pair = new Float64Array(Math.max(N - 1, 0) * KK)
  const off = U * K
  for (let n = 1; n < N; n++)
    for (let j = e.bigramStart[n]; j < e.bigramStart[n + 1]; j++) {
      const at = off + e.bigramIds[j] * KK
      for (let q = 0; q < KK; q++) pair[(n - 1) * KK + q] += weights[at + q]
    }
  return { logUnary: fromData(unary, [N, K]), logPairwise: fromData(pair, [Math.max(N - 1, 0), K, K]) }
}

const potentialsOf = (crf: TemplateCrf, rows: TokenRows) =>
  crfLogPotentials(crf.weights, encodeTemplateRows(crf.index, rows), crf.index.unigram.length, crf.labels.length)

/** Marginals of a sequence: log α, log β, P(y_n | x), pairwise marginals and log Z (log-space forward–backward). */
export function templateCrfMarginals(crf: TemplateCrf, rows: TokenRows): ChainMarginals {
  const p = potentialsOf(crf, rows)
  return chainForwardBackward(p.logUnary, p.logPairwise)
}

/** The Viterbi labelling of a sequence (`path` as label ids, `labels` as strings). */
export function templateCrfViterbi(crf: TemplateCrf, rows: TokenRows): ViterbiResult & { labels: string[] } {
  const p = potentialsOf(crf, rows)
  const v = chainViterbi(p.logUnary, p.logPairwise)
  return { ...v, labels: Array.from(toFlat(v.path), (k) => crf.labels[k]) }
}

/**
 * Posterior (max-marginal) decoding of a sequence, ŷ_n = argmax_k P(y_n = k | x) (`posteriorDecode` on the
 * forward–backward marginals), with the labels as strings and the path's unnormalised log score. It maximises the
 * expected number of correct labels; Viterbi maximises the probability of the whole labelling.
 */
export function templateCrfPosterior(
  crf: TemplateCrf,
  rows: TokenRows,
): PosteriorDecoding & { labels: string[]; logScore: number } {
  const p = potentialsOf(crf, rows)
  const d = posteriorDecode(chainForwardBackward(p.logUnary, p.logPairwise).marginals)
  const path = Array.from(toFlat(d.path))
  return { ...d, labels: path.map((k) => crf.labels[k]), logScore: pathScore(p, path, crf.labels.length) }
}

/** The unnormalised log score Σ_n log ψ_n(y_n) + Σ_n log Ψ_n−1(y_n−1, y_n) of a labelling (label ids). */
export function templateCrfScore(crf: TemplateCrf, rows: TokenRows, path: ArrayLike<number>): number {
  return pathScore(potentialsOf(crf, rows), path, crf.labels.length)
}

function pathScore(p: { logUnary: Matrix; logPairwise: Tensor }, path: ArrayLike<number>, K: Size): number {
  let s = 0
  for (let n = 0; n < path.length; n++) {
    s += p.logUnary.data[n * K + path[n]]
    if (n > 0) s += p.logPairwise.data[(n - 1) * K * K + path[n - 1] * K + path[n]]
  }
  return s
}

/** The log-potentials of a sequence under a CRF (log ψ and log Ψ). */
export function templateCrfPotentials(crf: TemplateCrf, rows: TokenRows): { logUnary: Matrix; logPairwise: Tensor } {
  return potentialsOf(crf, rows)
}

/** A feature string that fires at a position, with the template it came from, its id and its weights. */
export interface FiringFeature {
  readonly kind: 'unigram' | 'bigram'
  /** The expanded string, e.g. `U01:o`. */
  readonly string: string
  /** The template line it expands. */
  readonly template: string
  /** The template's macros: the cells %x[r,c] it reads. */
  readonly macros: readonly TemplateMacro[]
  /** Its id in the index, or −1 when the string was not seen in training (it then fires nothing). */
  readonly id: number
  /** λ_{u,k} per label k (unigram, length K), or λ_{b,i,k} row-major (bigram, K × K). Zeros when unknown. */
  readonly weights: readonly number[]
}

/** Every template's string at position n (bigram templates only for n ≥ 1), with its weights. */
export function firingFeatures(crf: TemplateCrf, rows: TokenRows, n: Size): FiringFeature[] {
  const K = crf.labels.length
  const U = crf.index.unigram.length
  const out: FiringFeature[] = []
  for (const t of crf.index.templates.templates) {
    if (t.kind === 'bigram' && n === 0) continue
    let s = t.pieces[0]
    t.macros.forEach((m, k) => {
      const at = n + m.row
      s +=
        (at < 0 ? `_B${at}` : at >= rows.length ? `_B+${at - rows.length + 1}` : rows[at][m.column]) + t.pieces[k + 1]
    })
    const id = (t.kind === 'unigram' ? crf.index.unigramIds : crf.index.bigramIds).get(s) ?? -1
    const width = t.kind === 'unigram' ? K : K * K
    const base = t.kind === 'unigram' ? id * K : U * K + id * K * K
    out.push({
      kind: t.kind,
      string: s,
      template: t.text,
      macros: t.macros,
      id,
      weights: Array.from({ length: width }, (_, q) => (id < 0 ? 0 : crf.weights[base + q])),
    })
  }
  return out
}

/** A unigram string's weights per label, for ranking. */
export interface WeightedString {
  readonly string: string
  readonly id: number
  /** λ_{u,k} per label k. */
  readonly weights: readonly number[]
  /** How often it fired in training. */
  readonly count: number
}

/**
 * The unigram strings with the largest |λ_{u,k}| for label k (`label` given), or with the largest max_k |λ_{u,k}|
 * (no label), at most `count`; zero weights are left out. With `relative`, strings are ranked by how much they favour
 * label k over the others, λ_{u,k} − mean_j λ_{u,j}, largest first (with two labels, half the log-odds they add).
 */
export function topFeatures(
  crf: TemplateCrf,
  count: Size,
  label?: Size,
  options: { relative?: boolean } = {},
): WeightedString[] {
  const K = crf.labels.length
  const scoreOf = (u: number) => {
    if (label !== undefined && options.relative) {
      let mean = 0
      for (let k = 0; k < K; k++) mean += crf.weights[u * K + k] / K
      return crf.weights[u * K + label] - mean
    }
    if (label !== undefined) return Math.abs(crf.weights[u * K + label])
    let m = 0
    for (let k = 0; k < K; k++) m = Math.max(m, Math.abs(crf.weights[u * K + k]))
    return m
  }
  return crf.index.unigram
    .map((_, u) => u)
    .filter((u) => scoreOf(u) > 0)
    .sort((a, b) => scoreOf(b) - scoreOf(a))
    .slice(0, count)
    .map((u) => ({
      string: crf.index.unigram[u],
      id: u,
      weights: Array.from({ length: K }, (_, k) => crf.weights[u * K + k]),
      count: crf.index.unigramCounts[u],
    }))
}

/** The K × K weights of the plain transition string `B` (or of another bigram string), or null if it is not indexed. */
export function transitionWeights(crf: TemplateCrf, string = 'B'): number[][] | null {
  const id = crf.index.bigramIds.get(string)
  if (id === undefined) return null
  const K = crf.labels.length
  const base = crf.index.unigram.length * K + id * K * K
  return Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, k) => crf.weights[base + i * K + k]))
}

/** The number of non-zero weights. */
export function activeWeights(weights: ArrayLike<number>): Size {
  let a = 0
  for (let i = 0; i < weights.length; i++) if (weights[i] !== 0) a++
  return a
}

// ── Likelihood and gradient ──────────────────────────────────────────────────────────────────────────────────────────

/** A sequence encoded once for training: feature ids and label ids. */
export interface EncodedLabelled {
  readonly sequence: EncodedSequence
  readonly labels: Int32Array
}

/** Encode training data against an index and a label set. Unknown labels throw. */
export function encodeLabelled(
  index: FeatureIndex,
  labels: readonly string[],
  data: readonly LabelledSequence[],
): EncodedLabelled[] {
  const ids = new Map(labels.map((y, k) => [y, k]))
  return data.map((s, m) => {
    if (s.labels.length !== s.rows.length)
      throw new DomainError(
        'encodeLabelled',
        `encodeLabelled: sequence ${m} has ${s.rows.length} rows but ${s.labels.length} labels`,
      )
    return {
      sequence: encodeTemplateRows(index, s.rows),
      labels: Int32Array.from(s.labels, (y) => {
        const k = ids.get(y)
        if (k === undefined) throw new DomainError('encodeLabelled', `encodeLabelled: unknown label '${y}'`)
        return k
      }),
    }
  })
}

/**
 * Σ −log P(y | x) over `data` and its gradient (added into `grad`): for each string u firing at n,
 * ∂/∂λ_{u,k} += P(y_n = k | x) − [y_n = k]; for each bigram string b firing at n ≥ 1,
 * ∂/∂λ_{b,i,k} += P(y_{n−1} = i, y_n = k | x) − [y_{n−1} = i, y_n = k].
 */
export function crfNegLogLikelihood(
  weights: ArrayLike<number>,
  data: readonly EncodedLabelled[],
  U: Size,
  K: Size,
  grad?: Float64Array,
): number {
  const KK = K * K
  const off = U * K
  let nll = 0
  for (const { sequence: e, labels: y } of data) {
    const N = e.length
    if (N === 0) continue
    const p = crfLogPotentials(weights, e, U, K)
    const fb = chainForwardBackward(p.logUnary, p.logPairwise)
    let score = 0
    for (let n = 0; n < N; n++) {
      score += p.logUnary.data[n * K + y[n]]
      if (n > 0) score += p.logPairwise.data[(n - 1) * KK + y[n - 1] * K + y[n]]
    }
    nll += fb.logZ - score
    if (!grad) continue
    const P = fb.marginals.data
    const Q = fb.pairwise.data
    for (let n = 0; n < N; n++)
      for (let j = e.unigramStart[n]; j < e.unigramStart[n + 1]; j++) {
        const at = e.unigramIds[j] * K
        for (let k = 0; k < K; k++) grad[at + k] += P[n * K + k]
        grad[at + y[n]] -= 1
      }
    for (let n = 1; n < N; n++)
      for (let j = e.bigramStart[n]; j < e.bigramStart[n + 1]; j++) {
        const at = off + e.bigramIds[j] * KK
        for (let q = 0; q < KK; q++) grad[at + q] += Q[(n - 1) * KK + q]
        grad[at + y[n - 1] * K + y[n]] -= 1
      }
  }
  return nll
}

/**
 * The smooth training objective −Σ log P(y | x) + c₂‖λ‖² as an `ObjectiveFn` of the weight vector (the L1 term is
 * OWL-QN's `l1`).
 */
export function crfObjective(data: readonly EncodedLabelled[], U: Size, K: Size, c2: number) {
  return (w: Vector) => {
    const x = toFlat(w)
    const g = new Float64Array(x.length)
    let value = crfNegLogLikelihood(x, data, U, K, g)
    if (c2 > 0)
      for (let i = 0; i < x.length; i++) {
        value += c2 * x[i] * x[i]
        g[i] += 2 * c2 * x[i]
      }
    return { value, grad: fromData(g, [x.length]) }
  }
}

// ── Training ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The optimisers of `crfTraining`. */
export type CrfOptimizer = 'lbfgs' | 'owlqn' | 'sgd' | 'adam'

/** Options of `crfTraining`. */
export interface CrfTrainingOptions {
  /** L-BFGS (c₂ only), OWL-QN (c₁ and c₂: elastic net), SGD or Adam (c₂ only). Default 'lbfgs'. */
  optimizer?: CrfOptimizer
  /** L1 strength c₁ (OWL-QN only). Default 0. */
  c1?: number
  /** L2 strength c₂. Default 0.01. */
  c2?: number
  /** Curvature pairs kept by L-BFGS / OWL-QN (default 10). */
  memory?: Size
  /** Stop when the (pseudo-)gradient norm per sequence falls below this (default 1e-5). */
  tolerance?: number
  /** SGD / Adam: step size (defaults 0.1 and 0.05) and sequences per minibatch (default 16; 0 for all, full batch). */
  stepSize?: number
  batchSize?: Size
}

/** The state of `crfTraining`: one quasi-Newton iteration, or one epoch of SGD / Adam, per step. */
export interface CrfTrainingState extends Status {
  /** λ. */
  readonly weights: Vector
  /** The full objective L(λ) = NLL + c₁‖λ‖₁ + c₂‖λ‖². */
  readonly objective: number
  /** −Σ log P(y | x) over the training data. */
  readonly nll: number
  /** The norm of the (pseudo-)gradient of L. */
  readonly gradNorm: number
  /** Non-zero weights. */
  readonly active: Size
  /** Objective and gradient evaluations (full passes) so far, counting a minibatch pass as its share. */
  readonly evaluations: number
  /** The quasi-Newton state (L-BFGS / OWL-QN), or null. */
  readonly inner: OwlqnState | null
  /** The first-order rule's state (SGD / Adam), or null. */
  readonly rule: unknown
  readonly stalled: boolean
}

const l1Norm = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += Math.abs(x[i])
  return a
}

/**
 * Train a template CRF's weights on encoded data (see the module comment), as a step-through algorithm: each step is
 * one L-BFGS or OWL-QN iteration (the line search included), or one epoch of minibatch SGD / Adam over a shuffled
 * order drawn from `ctx.stream`. The state carries the objective, the NLL, the gradient norm and the number of non-zero
 * weights. `init` takes the starting weights (zeros when undefined).
 */
export function crfTraining(
  problem: CrfProblem,
  options: CrfTrainingOptions = {},
): Algorithm<ArrayLike<number> | undefined, CrfTrainingState> {
  const { optimizer = 'lbfgs', c2 = 0.01, memory = 10, tolerance = 1e-5 } = options
  const { encoded: data, U, K, dimension } = problem
  const c1 = optimizer === 'owlqn' ? (options.c1 ?? 0) : 0
  const batchSize = options.batchSize === 0 ? Math.max(1, data.length) : (options.batchSize ?? 16)
  const M = Math.max(1, data.length)
  const f = crfObjective(data, U, K, c2)
  const quasiNewton = optimizer === 'lbfgs' || optimizer === 'owlqn'
  // The tolerance is per sequence, so it does not tighten as the data grow.
  const qn = owlqn(f, { l1: c1, memory, tolerance: tolerance * M })
  const rule: UpdateRule<unknown> | null = quasiNewton
    ? null
    : ((optimizer === 'sgd'
        ? sgdRule({ stepSize: options.stepSize ?? 0.1 })
        : adamRule({ stepSize: options.stepSize ?? 0.05 })) as UpdateRule<unknown>)
  const full = (w: Float64Array) => {
    const out = f(fromData(w, [w.length]))
    return { value: out.value, nll: out.value - c2 * w.reduce((a, v) => a + v * v, 0), grad: toFlat(out.grad) }
  }
  const stateOf = (w: Float64Array, extra: Partial<CrfTrainingState> & { t: number; evaluations: number }) => {
    const ev = full(w)
    return {
      t: extra.t,
      weights: fromData(w, [w.length]),
      objective: ev.value + c1 * l1Norm(w),
      nll: ev.nll,
      gradNorm: Math.hypot(...ev.grad),
      active: activeWeights(w),
      evaluations: extra.evaluations + 1,
      inner: null,
      rule: extra.rule ?? null,
      stalled: false,
      converged: false,
      diverged: !Number.isFinite(ev.value),
    } satisfies CrfTrainingState
  }
  return {
    name: `crf-${optimizer}`,
    init: (w0, s) => {
      if (w0 && w0.length !== dimension)
        throw new DomainError('crfTraining', `crfTraining: expected ${dimension} starting weights, got ${w0.length}`)
      const w = w0 ? Float64Array.from(w0) : new Float64Array(dimension)
      if (quasiNewton) {
        const inner = qn.init({ x0: w }, s)
        return {
          t: 0,
          weights: inner.x,
          objective: inner.value,
          nll: inner.smoothValue - c2 * l2(toFlat(inner.x)),
          gradNorm: inner.gradNorm,
          active: inner.nonzero,
          evaluations: 1,
          inner,
          rule: null,
          stalled: false,
          converged: inner.converged,
          diverged: inner.diverged,
        }
      }
      return stateOf(w, { t: 0, evaluations: 0, rule: rule!.init(fromData(w, [w.length])) })
    },
    step: (st, ctx) => {
      if (quasiNewton) {
        const inner = qn.step(st.inner!, ctx)
        return {
          t: st.t + 1,
          weights: inner.x,
          objective: inner.value,
          nll: inner.smoothValue - c2 * l2(toFlat(inner.x)),
          gradNorm: inner.gradNorm,
          active: inner.nonzero,
          evaluations: inner.evaluations,
          inner,
          rule: null,
          stalled: inner.stalled,
          converged: inner.converged,
          diverged: inner.diverged,
        }
      }
      // One epoch: minibatches of a shuffled order; each minibatch's gradient is of L/M restricted to it.
      let w = Float64Array.from(toFlat(st.weights))
      let rs = st.rule
      const order = Array.from(toFlat(permutation(child(ctx.stream, 'order'), data.length)))
      for (let b = 0; b < order.length; b += batchSize) {
        const batch = order.slice(b, b + batchSize).map((i) => data[i])
        const g = new Float64Array(w.length)
        crfNegLogLikelihood(w, batch, U, K, g)
        const scale = 1 / batch.length
        for (let i = 0; i < w.length; i++) g[i] = g[i] * scale + (2 * c2 * w[i]) / M
        const out = rule!.update(fromData(g, [w.length]), rs, fromData(w, [w.length]))
        rs = out.state
        const u = toFlat(out.updates as Tensor)
        w = w.map((v, i) => v + u[i])
      }
      return stateOf(w, { t: st.t + 1, evaluations: st.evaluations, rule: rs })
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

const l2 = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += x[i] * x[i]
  return a
}

/** Options of `fitTemplateCrf`. */
export interface FitTemplateCrfOptions extends CrfTrainingOptions {
  /** CRF++'s `-f`: keep strings seen at least this often (default 1). */
  minFrequency?: Size
  /** The label set (default: the data's labels in order of first appearance). */
  labels?: readonly string[]
  /** The most steps (default 200). */
  maxSteps?: Size
}

/** A training problem: the index of the data, the labels, the encoded data and the sizes (U strings, K labels). */
export interface CrfProblem {
  readonly index: FeatureIndex
  readonly labels: readonly string[]
  readonly encoded: readonly EncodedLabelled[]
  readonly U: Size
  readonly K: Size
  /** The number of weights, U·K + B·K². */
  readonly dimension: Size
}

/** Index the data's strings (with `minFrequency`) and encode it: what `crfTraining` takes. */
export function crfProblem(
  templates: FeatureTemplates,
  data: readonly LabelledSequence[],
  options: { minFrequency?: Size; labels?: readonly string[] } = {},
): CrfProblem {
  const index = featureIndex(
    templates,
    data.map((s) => s.rows),
    { minFrequency: options.minFrequency ?? 1 },
  )
  const labels = options.labels ?? labelSet(data)
  const encoded = encodeLabelled(index, labels, data)
  const U = index.unigram.length
  const K = labels.length
  return { index, labels, encoded, U, K, dimension: crfWeightCount(index, K) }
}

/** CRF++'s `-c C` as CRFsuite strengths: c₂ = 1/(2C) for L2, c₁ = 1/C for L1. */
export function crfppRegularisation(C: number, kind: 'L1' | 'L2' = 'L2'): { c1: number; c2: number } {
  if (!(C > 0)) throw new DomainError('crfppRegularisation', 'crfppRegularisation: C must be > 0')
  return kind === 'L2' ? { c1: 0, c2: 1 / (2 * C) } : { c1: 1 / C, c2: 0 }
}

/** A snapshot of `crfTrainingRun`. */
export interface CrfSnapshot {
  readonly step: Size
  readonly maxSteps: Size
  readonly crf: TemplateCrf
  /** Per step so far: objective, NLL, gradient norm and non-zero weights. */
  readonly history: {
    readonly objective: readonly number[]
    readonly nll: readonly number[]
    readonly gradNorm: readonly number[]
    readonly active: readonly number[]
  }
  readonly converged: boolean
  readonly done: boolean
  /** Milliseconds since the run started. */
  readonly ms: number
}

/**
 * Train a template CRF from token rows and labels, yielding a snapshot after every step (step 0 first): the model so
 * far and the objective, NLL, gradient norm and active-weight history. A generator, so the lab's worker streams it.
 */
export function* crfTrainingRun(
  templateSource: FeatureTemplates,
  data: readonly LabelledSequence[],
  options: FitTemplateCrfOptions & { seed?: string | number } = {},
): Generator<CrfSnapshot> {
  const { maxSteps = 200, seed = 'crf' } = options
  const t0 = Date.now()
  const p = crfProblem(templateSource, data, options)
  const alg = crfTraining(p, options)
  const s0 = stream(seed)
  let s = alg.init(new Float64Array(p.dimension), child(s0, 'init'))
  const history = { objective: [s.objective], nll: [s.nll], gradNorm: [s.gradNorm], active: [s.active] }
  const snap = (done: boolean): CrfSnapshot => ({
    step: s.t,
    maxSteps,
    crf: { kind: 'template-crf', index: p.index, labels: p.labels, weights: Float64Array.from(toFlat(s.weights)) },
    history: {
      objective: [...history.objective],
      nll: [...history.nll],
      gradNorm: [...history.gradNorm],
      active: [...history.active],
    },
    converged: s.converged === true,
    done,
    ms: Date.now() - t0,
  })
  const finished = () => s.t >= maxSteps || (alg.done?.(s) ?? false)
  yield snap(finished())
  while (!finished()) {
    s = alg.step(s, { t: s.t, stream: child(s0, 'step', s.t) })
    history.objective.push(s.objective)
    history.nll.push(s.nll)
    history.gradNorm.push(s.gradNorm)
    history.active.push(s.active)
    yield snap(finished())
  }
}
