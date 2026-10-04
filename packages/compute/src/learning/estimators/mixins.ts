/**
 * Capability mixins (plan §5.1): build `decide`, `expect` and `sample` from what a model already has, and complete a
 * partial forward pass with a readout. Each returns a new plain object; the model passed in is not changed.
 */

import type {
  Decides,
  Distribution,
  Expects,
  Fitted,
  Predicts,
  Samples,
  Scores,
} from 'aifn-compute/foundation/contracts'
import type { Stream } from 'aifn-compute/foundation/random'
import { argmax, dense, fromData, reshape, type Tensor } from 'aifn-compute/foundation/tensor'
import type { InputOf } from './capabilities'
import { asTensor, classProbabilities, expectation, isClassDistribution } from './distribution'
import { sizeOf } from './util'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * How `withDecision` turns a model's outputs into a decision per input.
 *
 * - `'argmax'`: the class with the highest score (from `score`, [N, K]) or the highest predictive probability.
 * - `'mode'`: the predictive's mode (`mode` field or method), or the most probable class.
 * - `{ threshold: t }`: class 1 when P(y = 1 | x) ≥ t (from a Bernoulli or two-class predictive), or when the score
 *   ([N] or [N, 1]) is at least t when the model has no predictive.
 * - `{ costs: C }`: the Bayes decision under a K×K cost matrix, where C[i][j] is the cost of deciding j when the truth
 *   is i: argminⱼ Σᵢ P(i | x) C[i][j] (Duda, Hart and Stork, 2001, "Pattern Classification", §2.2).
 */
export type DecisionRule =
  'argmax' | 'mode' | { threshold: number } | { costs: Tensor | readonly (readonly number[])[] }

type Decidable = Scores<never> | Predicts<never, Distribution>

/** The argmax class of each row of scores or probabilities [..., K], flattened to [N]. */
function argmaxRows(p: Tensor): Tensor {
  const k = p.shape[p.shape.length - 1]
  return argmax(reshape(p, [sizeOf(p.shape) / k, k]), 1)
}

function costMatrix(costs: Tensor | readonly (readonly number[])[]): { c: Float64Array; k: number } {
  if (Array.isArray(costs)) {
    const k = costs.length
    const c = new Float64Array(k * k)
    ;(costs as readonly (readonly number[])[]).forEach((row, i) => {
      if (row.length !== k) throw new ShapeError('withDecision', 'withDecision: the cost matrix must be square')
      row.forEach((v, j) => (c[i * k + j] = v))
    })
    return { c, k }
  }
  const t = costs as Tensor
  if (t.shape.length !== 2 || t.shape[0] !== t.shape[1])
    throw new ShapeError('withDecision', 'withDecision: the cost matrix must be K×K')
  return { c: dense.data(t), k: t.shape[0] }
}

/** The decision function for `rule` on `model`. */
function decider(model: Decidable, rule: DecisionRule): (x: unknown) => Tensor {
  const predictive = (model as Partial<Predicts<unknown, Distribution>>).predictive?.bind(model)
  const score = (model as Partial<Scores<unknown>>).score?.bind(model)
  if (rule === 'argmax') {
    if (score) {
      return (x) => {
        const s = score(x)
        if (s.shape.length < 2) {
          throw new ShapeError(
            'score',
            "withDecision('argmax'): a score of shape [N] has one column; use { threshold: 0 }",
          )
        }
        return argmaxRows(s)
      }
    }
    return (x) => argmaxRows(classProbabilities(predictive!(x)))
  }
  if (rule === 'mode') {
    if (!predictive) throw new DomainError('score', "withDecision('mode'): the model has no predictive")
    return (x) => {
      const d = predictive(x)
      return isClassDistribution(d) ? argmaxRows(classProbabilities(d)) : asTensor(d.mode())
    }
  }
  if ('threshold' in rule) {
    const t = rule.threshold
    return (x) => {
      let p: Float64Array
      if (predictive) {
        const d = predictive(x)
        if (!isClassDistribution(d))
          throw new DomainError('withDecision', 'withDecision(threshold): the predictive has no class probabilities')
        const c = classProbabilities(d)
        const [n, k] = c.shape
        if (k !== 2) throw new DomainError('withDecision', 'withDecision(threshold): needs two classes')
        const probs = dense.data(c)
        p = Float64Array.from({ length: n }, (_, i) => probs[2 * i + 1])
      } else {
        const s = score!(x)
        if (sizeOf(s.shape) !== s.shape[0])
          throw new ShapeError('withDecision', 'withDecision(threshold): the score must be [N] or [N, 1]')
        p = dense.data(s)
      }
      return fromData(
        Int32Array.from(p, (v) => (v >= t ? 1 : 0)),
        [p.length],
      )
    }
  }
  const { c, k } = costMatrix(rule.costs)
  if (!predictive) throw new DomainError('withDecision', 'withDecision(costs): the model has no predictive')
  return (x) => {
    const pk = classProbabilities(predictive(x))
    const [n, classes] = pk.shape
    const probs = dense.data(pk)
    if (classes !== k)
      throw new ShapeError('withDecision', `withDecision(costs): ${classes} classes but a ${k}×${k} cost matrix`)
    const out = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      let best = 0
      let bestCost = Infinity
      for (let j = 0; j < k; j++) {
        let risk = 0
        for (let truth = 0; truth < k; truth++) risk += probs[i * k + truth] * c[truth * k + j]
        if (risk < bestCost) [best, bestCost] = [j, risk]
      }
      out[i] = best
    }
    return fromData(out, [n])
  }
}

/**
 * A copy of `model` that also decides, by `rule` (see `DecisionRule`) from its `score` or `predictive`. Decisions are
 * int32 class indices [N].
 *
 * @example const classifier = withDecision(model, { costs: [[0, 1], [5, 0]] }) // a false negative costs 5
 */
export function withDecision<M extends Decidable>(model: M, rule: DecisionRule): M & Decides<InputOf<M>, Tensor> {
  return { ...model, decide: decider(model, rule) } as M & Decides<InputOf<M>, Tensor>
}

/**
 * A copy of `model` that also gives expectations E[f(y) | x] from its predictive (the mean without `f`): a finite sum
 * for class distributions, Gauss–Hermite quadrature for other univariate laws (see `expectation`). For an ordinal
 * model with an ordered-categorical predictive, `expect(x)` is E[y].
 */
export function withExpectation<M extends Predicts<never, Distribution>>(model: M): M & Expects<InputOf<M>> {
  const predictive = (model as unknown as Predicts<unknown, Distribution>).predictive.bind(model)
  return { ...model, expect: (x: unknown, f?: (y: number) => number) => expectation(predictive(x), f) } as M &
    Expects<InputOf<M>>
}

/** A copy of `model` that also samples: `sample(s, x, n)` draws from `predictive(x)`. */
export function withSampling<M extends Predicts<never, Distribution>>(model: M): M & Samples<InputOf<M>, Tensor> {
  const predictive = (model as unknown as Predicts<unknown, Distribution>).predictive.bind(model)
  return {
    ...model,
    sample: (s: Stream, x: unknown, n?: number) =>
      asTensor(predictive(x).sample(s, { shape: n === undefined ? [] : [n] })),
  } as M & Samples<InputOf<M>, Tensor>
}

/** Completers a readout may define: each maps the head (and the input) to one output. */
export type Completers<H, X> = {
  predictive?: (head: H, x: X) => Distribution
  decide?: (head: H, x: X) => unknown
  score?: (head: H, x: X) => Tensor
  transform?: (head: H, x: X) => unknown
}

/** The capabilities a set of completers gives: one method per key, returning what that completer returns. */
export type ReadoutOf<X, C> = {
  [K in keyof C]: C[K] extends (head: never, x: never) => infer R ? (x: X) => R : never
}

/**
 * Complete a model's partial forward pass `forward(x) → head` into new capabilities. `complete` is either a record of
 * completers (`predictive`, `decide`, `score`, `transform`), each `(head, x) => output`, or a single function
 * returning a distribution (short for `{ predictive }`). The result is a new fitted model whose capabilities follow
 * the completers' return types: `readout(m, (h) => bernoulliPredictive(sigmoid(h)))` predicts a Bernoulli, and has no
 * `decide` unless one is given (or added with `withDecision`). Capabilities of `model` with the same names are
 * replaced.
 *
 * @example
 * const ordinal = readout(latent, { predictive: (eta) => orderedCategorical(eta, cutpoints) })
 */
export function readout<M extends Fitted<never, unknown>, D extends Distribution>(
  model: M,
  complete: (head: HeadOfModel<M>, x: InputOf<M>) => D,
): Omit<M, 'predictive'> & Predicts<InputOf<M>, D>
export function readout<M extends Fitted<never, unknown>, C extends Completers<HeadOfModel<M>, InputOf<M>>>(
  model: M,
  complete: C,
): Omit<M, keyof C> & ReadoutOf<InputOf<M>, C>
export function readout(
  model: Fitted<unknown, unknown>,
  complete: ((head: unknown, x: unknown) => Distribution) | Completers<unknown, unknown>,
): unknown {
  const completers: Completers<unknown, unknown> = typeof complete === 'function' ? { predictive: complete } : complete
  const forward = model.forward.bind(model)
  const out: Record<string, unknown> = { ...model }
  for (const [key, f] of Object.entries(completers)) {
    if (typeof f !== 'function') continue
    out[key] = (x: unknown) => (f as (head: unknown, x: unknown) => unknown)(forward(x), x)
  }
  return out
}

/** The head type of a model with `forward`. */
type HeadOfModel<M> = M extends { forward(x: never): infer H } ? H : never
