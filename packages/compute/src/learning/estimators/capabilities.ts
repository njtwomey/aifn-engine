/**
 * Typed capabilities of fitted models (plan §5.1), after scikit-learn's mixins (Pedregosa et al., 2011; Buitinck et
 * al., 2013, "API design for machine learning software: experiences from the scikit-learn project"), but checked by
 * the compiler: a figure can only ask a model for what it declares. The capability interfaces and their names
 * (`Capability`) are defined once, in `aifn-compute/foundation/contracts`; this file adds the guards that test for
 * them.
 *
 * A model has a capability when it has a method of that name (`training`, a field, is the exception), so the guards
 * work on any object at run time and narrow its type for the compiler. The type helpers read a model's input, head,
 * predictive and decision types from its methods.
 */

import type {
  Capability,
  Decides,
  Distribution,
  Expects,
  Fitted,
  Predicts,
  Samples,
  Scores,
  Trained,
  Transforms,
} from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'

// The capability interfaces are defined once, in `aifn-compute/foundation/contracts`.
export type {
  Capability,
  Decides,
  Expects,
  Fitted,
  Model,
  ModelInfo,
  Predicts,
  Samples,
  Scores,
  Task,
  Trained,
  Transforms,
} from 'aifn-compute/foundation/contracts'

/** The input type of a model, read from whichever capability it has; `unknown` when it has none. */
export type InputOf<M> = M extends { forward(x: infer X): unknown }
  ? X
  : M extends { decide(x: infer X): unknown }
    ? X
    : M extends { predictive(x: infer X): unknown }
      ? X
      : M extends { score(x: infer X): unknown }
        ? X
        : M extends { transform(x: infer X): unknown }
          ? X
          : M extends { expect(x: infer X, ...rest: never[]): unknown }
            ? X
            : unknown

/** The head type of a model's forward pass. */
export type HeadOf<M> = M extends { forward(x: never): infer H } ? H : never

/** The distribution type a model predicts. */
export type PredictiveOf<M> = M extends { predictive(x: never): infer D } ? D : never

/** The decision type of a model. */
export type DecisionOf<M> = M extends { decide(x: never): infer Y } ? Y : never

/**
 * True when `m` is an object with a method named `key`.
 *
 * @param m Anything.
 * @param key The capability's name, which is also its method's name.
 * @returns Whether `m[key]` is a function.
 */
const has = (m: unknown, key: Capability) =>
  typeof m === 'object' && m !== null && typeof (m as Record<string, unknown>)[key] === 'function'

/**
 * True when the model has a partial forward pass: a `forward(x)` method returning a head that a readout completes.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example A model with a forward pass
 * const latent = { forward: (x) => mul(x, 2) }
 * print('latent:', hasForward(latent))
 * print('empty:', hasForward({}))
 */
export function hasForward<M>(m: M): m is M & Fitted<InputOf<M>, unknown> {
  return has(m, 'forward')
}

/**
 * True when the model decides: a `decide(x)` method returning one decision (a class or a value) per input.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example Only a method counts
 * print('method:', hasDecide({ decide: (x) => x }))
 * print('field:', hasDecide({ decide: 1 }))
 */
export function hasDecide<M>(m: M): m is M & Decides<InputOf<M>, Tensor> {
  return has(m, 'decide')
}

/**
 * True when the model returns a predictive distribution: a `predictive(x)` method.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example A model that predicts a Gaussian
 * const model = { predictive: (x) => gaussianPredictive(x, ones(x.shape)) }
 * print('has predictive:', hasPredictive(model))
 * print('predictive mean:', model.predictive(tensor([1, 2])).mean())
 */
export function hasPredictive<M>(m: M): m is M & Predicts<InputOf<M>, Distribution> {
  return has(m, 'predictive')
}

/**
 * True when the model gives expectations: an `expect(x, f)` method.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example withExpectation adds the capability
 * const model = { predictive: (x) => gaussianPredictive(x, ones(x.shape)) }
 * print('before:', hasExpect(model))
 * print('after:', hasExpect(withExpectation(model)))
 */
export function hasExpect<M>(m: M): m is M & Expects<InputOf<M>> {
  return has(m, 'expect')
}

/**
 * True when the model scores: a `score(x)` method returning per-class or per-item scores.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example A model that scores
 * print('scores:', hasScore({ score: (x) => x }))
 * print('decides only:', hasScore({ decide: (x) => x }))
 */
export function hasScore<M>(m: M): m is M & Scores<InputOf<M>> {
  return has(m, 'score')
}

/**
 * True when the model transforms its inputs: a `transform(x)` method, as a fitted scaler or encoder has.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example A fitted scaler transforms
 * const scaler = { transform: (x) => mul(x, 0.5) }
 * print('transforms:', hasTransform(scaler))
 */
export function hasTransform<M>(m: M): m is M & Transforms<InputOf<M>, unknown> {
  return has(m, 'transform')
}

/**
 * True when the model samples: a `sample(s, x, n)` method drawing from a stream.
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m` has the capability; the type of `m` is narrowed when it does.
 *
 * @example withSampling adds the capability
 * const model = { predictive: (x) => gaussianPredictive(x, ones(x.shape)) }
 * print('before:', hasSample(model))
 * print('after:', hasSample(withSampling(model)))
 */
export function hasSample<M>(m: M): m is M & Samples<InputOf<M>, Tensor> {
  return has(m, 'sample')
}

/**
 * True when the model kept a training trace: a `training` field holding an object (unlike the other capabilities, a
 * field, not a method).
 *
 * @param m The model, or anything else (which gives false).
 * @returns Whether `m.training` is an object; the type of `m` is narrowed when it is.
 *
 * @example A trace is a field
 * print('with a trace:', hasTraining({ training: { series: {} } }))
 * print('without:', hasTraining({ decide: (x) => x }))
 */
export function hasTraining<M>(m: M): m is M & Trained {
  return typeof m === 'object' && m !== null && typeof (m as Partial<Trained>).training === 'object'
}

/** Every capability, in the order `capabilities` reports them (checked against the contract's `Capability`). */
const ALL = [
  'forward',
  'decide',
  'predictive',
  'expect',
  'score',
  'transform',
  'sample',
] as const satisfies readonly Capability[]

/** Compile-time check that `ALL` lists every capability of the contract. */
const complete: [Exclude<Capability, (typeof ALL)[number]>] extends [never] ? true : never = true
void complete

/**
 * The capabilities a model has, as method names in a fixed order: `forward`, `decide`, `predictive`, `expect`,
 * `score`, `transform`, `sample`. A training trace is not listed (see `hasTraining`).
 *
 * @param m The model, or anything else (which has none).
 * @returns The names of the capabilities it has.
 *
 * @example A Gaussian model with decisions, expectations and draws
 * const base = { predictive: (x) => gaussianPredictive(x, ones(x.shape)) }
 * print('base:', capabilities(base))
 * print('completed:', capabilities(withSampling(withExpectation(withDecision(base, 'mode')))))
 */
export function capabilities(m: unknown): Capability[] {
  return ALL.filter((c) => has(m, c))
}
