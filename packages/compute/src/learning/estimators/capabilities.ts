/**
 * Typed capabilities of fitted models (plan §5.1), after scikit-learn's mixins (Pedregosa et al., 2011; Buitinck et
 * al., 2013, "API design for machine learning software: experiences from the scikit-learn project"), but checked by
 * the compiler: a figure can only ask a model for what it declares. The capability interfaces and their names
 * (`Capability`) are defined once, in `aifn-compute/foundation/contracts`; this file adds the guards that test for them.
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

const has = (m: unknown, key: Capability) =>
  typeof m === 'object' && m !== null && typeof (m as Record<string, unknown>)[key] === 'function'

/** True when the model has a partial forward pass. */
export function hasForward<M>(m: M): m is M & Fitted<InputOf<M>, unknown> {
  return has(m, 'forward')
}

/** True when the model decides. */
export function hasDecide<M>(m: M): m is M & Decides<InputOf<M>, Tensor> {
  return has(m, 'decide')
}

/** True when the model returns a predictive distribution. */
export function hasPredictive<M>(m: M): m is M & Predicts<InputOf<M>, Distribution> {
  return has(m, 'predictive')
}

/** True when the model gives expectations. */
export function hasExpect<M>(m: M): m is M & Expects<InputOf<M>> {
  return has(m, 'expect')
}

/** True when the model scores. */
export function hasScore<M>(m: M): m is M & Scores<InputOf<M>> {
  return has(m, 'score')
}

/** True when the model transforms its inputs. */
export function hasTransform<M>(m: M): m is M & Transforms<InputOf<M>, unknown> {
  return has(m, 'transform')
}

/** True when the model samples. */
export function hasSample<M>(m: M): m is M & Samples<InputOf<M>, Tensor> {
  return has(m, 'sample')
}

/** True when the model kept a training trace. */
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

/** The capabilities a model has, in a fixed order. */
export function capabilities(m: unknown): Capability[] {
  return ALL.filter((c) => has(m, c))
}
