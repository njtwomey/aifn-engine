/**
 * Pipelines (plan §5.4): transforms chained into a final estimator, with exactly the capabilities of the final step,
 * after scikit-learn's `Pipeline` (Buitinck et al., 2013).
 */

import type { Capability, Dataset, FitOptions } from 'aifn-compute/learning/estimators'
import type {
  Decides,
  Expects,
  Fitted,
  Predicts,
  Samples,
  Scores,
  Trained,
  Transforms,
} from 'aifn-compute/learning/estimators'
import type { Distribution } from 'aifn-compute/learning/estimators'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Anything with a `fit`: an estimator or a transformer. */
// oxlint-disable-next-line no-explicit-any -- estimator data types vary per step; the step list is typed below
export type AnyEstimator = { readonly name: string; fit(data: any, options?: FitOptions): object }

/** A step whose fitted model transforms (every step but the last). */
// oxlint-disable-next-line no-explicit-any -- as above
export type TransformStep = { readonly name: string; fit(data: any, options?: FitOptions): Transforms<any, any> }

/** The fitted model of a step. */
export type FittedOf<E> = E extends { fit(data: never, options?: never): infer M } ? M : never

/** The input features of a step (the `x` of the data it fits on). */
export type InputOfStep<E> = E extends { fit(data: infer D, options?: never): unknown }
  ? D extends { x: infer X }
    ? X
    : never
  : never

/** The last element of a tuple. */
export type Last<T extends readonly unknown[]> = T extends readonly [...unknown[], infer L] ? L : never

/**
 * The capabilities of a fitted model `M`, re-expressed on inputs `X`: each capability `M` has, and no other. This is
 * how a pipeline, or any wrapper that preprocesses its inputs, has exactly the capabilities of its final model.
 */
export type Lift<M, X> = (M extends Fitted<never, infer H> ? Fitted<X, H> : unknown) &
  (M extends Decides<never, infer Y> ? Decides<X, Y> : unknown) &
  (M extends Predicts<never, infer D extends Distribution> ? Predicts<X, D> : unknown) &
  (M extends Expects<never> ? Expects<X> : unknown) &
  (M extends Scores<never> ? Scores<X> : unknown) &
  (M extends Transforms<never, infer Z> ? Transforms<X, Z> : unknown) &
  (M extends Samples<never, infer Y> ? Samples<X, Y> : unknown) &
  (M extends Trained<infer S> ? Trained<S> : unknown)

/** The fitted steps of a pipeline, in order. */
export type FittedSteps<S extends readonly unknown[]> = { readonly [K in keyof S]: FittedOf<S[K]> }

/** A fitted pipeline: the fitted steps, and the final step's capabilities on the pipeline's inputs. */
export type PipelineModel<S extends readonly AnyEstimator[]> = Lift<FittedOf<Last<S>>, InputOfStep<S[0]>> & {
  readonly kind: 'model'
  /** Which composition made the model. */
  readonly composition: 'pipeline'
  /** Each step's fitted model, in order: the scalers' means and scales, the model's weights. */
  readonly steps: FittedSteps<S>
  /** The steps' names, in order. */
  readonly names: readonly string[]
  /** Run every step but the last: the features the final model sees. */
  features(x: InputOfStep<S[0]>): unknown
  /** The output of every step on x: `[x, step₀(x), step₁(step₀(x)), …]` up to the final model's input. */
  stages(x: InputOfStep<S[0]>): unknown[]
}

/** The capabilities a pipeline lifts from its final step (`sample` takes a stream first and is lifted separately). */
const LIFTED = [
  'forward',
  'decide',
  'predictive',
  'expect',
  'score',
  'transform',
] as const satisfies readonly Capability[]

/**
 * A pipeline of transforms ending in an estimator (or another transform). Fitting fits each step on the previous
 * steps' output of the training inputs (targets and groups pass through unchanged); step k gets the stream
 * `child(stream, 'step', k)`. The fitted pipeline has exactly the capabilities of its final step, on the pipeline's
 * inputs, and exposes every step's fitted state in `steps`.
 *
 * @example
 * const model = pipeline(standardScaler(), logisticRegression({ l2: 0.1 })).fit({ x, y })
 * model.steps[0].mean // the scaler's fitted means
 * model.predictive(xNew) // Bernoulli, after scaling xNew
 */
export function pipeline<const S extends readonly [...TransformStep[], AnyEstimator]>(
  ...steps: S
): {
  readonly name: string
  readonly steps: S
  fit(data: Dataset<InputOfStep<S[0]>, unknown>, options?: FitOptions): PipelineModel<S>
} {
  if (steps.length === 0) throw new DomainError('pipeline', 'pipeline: needs at least one step')
  return {
    name: `pipeline(${steps.map((s) => s.name).join(', ')})`,
    steps,
    fit(data, options = {}) {
      const fitted: object[] = []
      let x: unknown = data.x
      steps.forEach((step, k) => {
        const stream: Stream | undefined = options.stream && child(options.stream, 'step', k)
        const model = step.fit({ ...data, x }, { ...options, stream })
        fitted.push(model)
        if (k < steps.length - 1) x = (model as Transforms<unknown, unknown>).transform(x)
      })
      const head = fitted.slice(0, -1) as Transforms<unknown, unknown>[]
      const final = fitted[fitted.length - 1] as Record<string, unknown>
      const features = (input: unknown) => head.reduce((z, m) => m.transform(z), input)
      const out: Record<string, unknown> = {
        kind: 'model',
        composition: 'pipeline',
        steps: fitted,
        names: steps.map((s) => s.name),
        features,
        stages: (input: unknown) => {
          const all = [input]
          for (const m of head) all.push(m.transform(all[all.length - 1]))
          return all
        },
      }
      for (const cap of LIFTED) {
        const f = final[cap]
        if (typeof f === 'function')
          out[cap] = (input: unknown, ...rest: unknown[]) => f.call(final, features(input), ...rest)
      }
      if (typeof final.sample === 'function') {
        const sample = final.sample as (s: Stream, x: unknown, n?: number) => unknown
        out.sample = (s: Stream, input: unknown, n?: number) => sample.call(final, s, features(input), n)
      }
      if (final.training !== undefined) out.training = final.training
      return out as PipelineModel<S>
    },
  }
}
