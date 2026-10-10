/**
 * Capability mixins (plan §5.1): build `decide`, `expect` and `sample` from what a model already has, and complete a
 * partial forward pass with a readout. Each returns a new plain object; the model passed in is not changed.
 *
 * The copy is shallow (the model's fields are spread into a new object), and the new methods call the model's own,
 * bound to it, so a mixin can wrap any fitted model, including one made by another mixin.
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
 * - `'argmax'`: the class with the highest score (from `score`, $N \times K$; preferred when the model has one) or the
 *   highest predictive probability.
 * - `'mode'`: the predictive's mode (its `mode()` method), or the most probable class of a class distribution.
 * - `{ threshold: t }`: class 1 when $\Pr(y = 1 \mid x) \ge t$ (from a Bernoulli or two-class predictive), or when
 *   the score ($N$ values, or $N \times 1$) is at least $t$ when the model has no predictive.
 * - `{ costs: C }`: the Bayes decision under a $K \times K$ cost matrix $\Cmat$, where $C_{ij}$ is the cost of
 *   deciding $j$ when the truth is $i$: $\argmin_j \sum_i \Pr(i \mid x) C_{ij}$ (Duda, Hart and Stork, 2001, "Pattern
 *   Classification", §2.2).
 */
export type DecisionRule =
  'argmax' | 'mode' | { threshold: number } | { costs: Tensor | readonly (readonly number[])[] }

type Decidable = Scores<never> | Predicts<never, Distribution>

/**
 * The argmax class of each row of scores or probabilities, flattened to one class per row.
 *
 * @param p Scores or probabilities whose last axis is over the $K$ classes.
 * @returns The index of the largest entry of each row, $N$ values for $N$ rows.
 */
function argmaxRows(p: Tensor): Tensor {
  const k = p.shape[p.shape.length - 1]
  return argmax(reshape(p, [sizeOf(p.shape) / k, k]), 1)
}

/**
 * A cost matrix as a dense row-major array. Throws `ShapeError` when it is not square.
 *
 * @param costs The $K \times K$ costs, as nested arrays (row $i$ the truth) or a matrix tensor.
 * @returns The $K^2$ costs, row-major, and $K$.
 */
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

/**
 * The decision function for `rule` on `model` (see `DecisionRule`). Throws `DomainError` at once when the rule needs a
 * predictive the model lacks (`'mode'`, costs); other mismatches (a one-column score under `'argmax'`, a non-class
 * predictive under a threshold, a cost matrix of the wrong size) throw when the decision is made.
 *
 * @param model The model, with `score` or `predictive` (or both).
 * @param rule How to decide.
 * @returns A function from inputs to int32 decisions, one per input (the mode's values under `'mode'` for a
 *   non-class predictive).
 */
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
 * int32 class indices, one per input (under `'mode'`, a non-class predictive's modes, as they are). An existing
 * `decide` is replaced.
 *
 * @param model The fitted model, with `score` or `predictive`; not modified.
 * @param rule How to decide: `'argmax'`, `'mode'`, `{ threshold }` or `{ costs }`.
 * @returns A new model with the same fields and methods and a `decide`.
 *
 * @example A false negative that costs five times a false positive lowers the threshold to 1/6
 * // The inputs are already the probabilities of class 1.
 * const model = { predictive: (x) => bernoulliPredictive(x) }
 * const x = tensor([0.1, 0.2, 0.6])
 * print('threshold 0.5:', withDecision(model, { threshold: 0.5 }).decide(x))
 * print('costs:', withDecision(model, { costs: [[0, 1], [5, 0]] }).decide(x))
 *
 * @example The argmax of scores
 * const scorer = { score: (x) => x }
 * print(withDecision(scorer, 'argmax').decide(tensor([[0.1, 2, -1], [3, 0, 0]])))
 */
export function withDecision<M extends Decidable>(model: M, rule: DecisionRule): M & Decides<InputOf<M>, Tensor> {
  return { ...model, decide: decider(model, rule) } as M & Decides<InputOf<M>, Tensor>
}

/**
 * A copy of `model` that also gives expectations $\expect[f(y) \mid x]$ from its predictive (the mean without `f`): a
 * finite sum for class distributions, Gauss–Hermite quadrature for other univariate laws (see `expectation`). For an
 * ordinal model with an ordered-categorical predictive, `expect(x)` is $\expect[y]$.
 *
 * @param model The fitted model, with `predictive`; not modified.
 * @returns A new model with the same fields and methods and `expect(x, f)`.
 *
 * @example The mean and second moment of a Gaussian predictive
 * const model = withExpectation({ predictive: (x) => gaussianPredictive(x, full(x.shape, 2)) })
 * const x = tensor([0, 1])
 * print('E[y | x]:', model.expect(x))
 * print('E[y^2 | x] = mean^2 + 4:', model.expect(x, (y) => y * y))
 */
export function withExpectation<M extends Predicts<never, Distribution>>(model: M): M & Expects<InputOf<M>> {
  const predictive = (model as unknown as Predicts<unknown, Distribution>).predictive.bind(model)
  return { ...model, expect: (x: unknown, f?: (y: number) => number) => expectation(predictive(x), f) } as M &
    Expects<InputOf<M>>
}

/**
 * A copy of `model` that also samples: `sample(s, x, n)` draws from `predictive(x)` with the stream `s`, $n$ draws
 * per input on a new leading axis when $n$ is given, else one.
 *
 * @param model The fitted model, with `predictive`; not modified.
 * @returns A new model with the same fields and methods and `sample(s, x, n)`.
 *
 * @example Seeded draws from a Gaussian predictive
 * const model = withSampling({ predictive: (x) => gaussianPredictive(x, full(x.shape, 0.1)) })
 * const draws = model.sample(stream(0), tensor([0, 10]), 3)
 * print('shape:', draws.shape)
 * print('draws:', draws)
 */
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
  /** The predictive distribution from the head. */
  predictive?: (head: H, x: X) => Distribution
  /** The decisions from the head. */
  decide?: (head: H, x: X) => unknown
  /** The scores from the head. */
  score?: (head: H, x: X) => Tensor
  /** The transformed inputs from the head. */
  transform?: (head: H, x: X) => unknown
}

/** The capabilities a set of completers gives: one method per key, returning what that completer returns. */
export type ReadoutOf<X, C> = {
  [K in keyof C]: C[K] extends (head: never, x: never) => infer R ? (x: X) => R : never
}

/**
 * Complete a model's partial forward pass, `forward(x)` returning a head, into new capabilities. `complete` is either
 * a record of completers (`predictive`, `decide`, `score`, `transform`), each `(head, x) => output`, or a single
 * function returning a distribution (short for `{ predictive }`). The result is a new fitted model whose capabilities
 * follow the completers' return types: `readout(m, (h) => bernoulliPredictive(sigmoid(h)))` predicts a Bernoulli, and
 * has no `decide` unless one is given (or added with `withDecision`). Capabilities of `model` with the same names are
 * replaced; each call runs `forward` afresh.
 *
 * @param model The fitted model with `forward`; not modified.
 * @param complete A function from the head (and the input) to a distribution, or a record of completers.
 * @returns A new model with the fields and methods of `model` and one method per completer.
 *
 * @example A logistic readout of a linear head
 * const latent = { forward: (x) => sub(mul(x, 2), 1) }
 * const model = readout(latent, (h) => bernoulliPredictive(map(h, (v) => 1 / (1 + Math.exp(-v)))))
 * print('heads:', model.forward(tensor([0, 0.5, 2])))
 * print('P(y = 1 | x):', model.predictive(tensor([0, 0.5, 2])).mean())
 *
 * @example Several capabilities from one head
 * const latent = { forward: (x) => sub(mul(x, 2), 1) }
 * const model = readout(latent, { score: (h) => h, decide: (h) => map(h, (v) => (v >= 0 ? 1 : 0)) })
 * print('capabilities:', capabilities(model))
 * print('scores:', model.score(tensor([0, 0.5, 2])))
 * print('decisions:', model.decide(tensor([0, 0.5, 2])))
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
