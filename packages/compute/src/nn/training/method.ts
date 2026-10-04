/**
 * Training by a chosen method with one state shape, so a run can offer either: first-order updates through
 * `trainingLoop` (any pytree update rule, minibatch or full batch), or full-batch L-BFGS through `fullBatchTraining`
 * (Liu & Nocedal, 1989). Every state carries `params`, the training `loss` of those parameters, the evaluations of the
 * loss so far and the stop flags; once L-BFGS stops (converged or stalled), further steps return the state unchanged.
 */

import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adamRule, type AdamRuleOptions, type UpdateRule } from 'aifn-compute/optim/first-order'
import type { Context } from 'aifn-compute/nn/layers'
import { fullBatchTraining } from './fullBatch'
import { trainingLoop, type Batch } from './train'

/**
 * How to train: a first-order update rule through `trainingLoop` (`first-order` with any rule, or `adam` as plain data,
 * which a worker can receive), or full-batch L-BFGS.
 */
export type TrainingMethod =
  | (AdamRuleOptions & {
      method: 'adam'
      /** Examples per step (default the whole set). */
      batchSize?: Size
      /** Clip the gradient's global norm. */
      clipNorm?: Scalar
    })
  | {
      method: 'first-order'
      /** The update rule (default `trainingLoop`'s, Adam with step 0.01). */
      optimizer?: UpdateRule<unknown>
      /** Examples per step (default the whole set). */
      batchSize?: Size
      /** Clip the gradient's global norm. */
      clipNorm?: Scalar
    }
  | {
      method: 'lbfgs'
      /** Curvature pairs kept, m (default 10). */
      memory?: Size
      /** Stop when ‖∇‖₂ ≤ tolerance (default 1e-6). */
      tolerance?: Scalar
    }

/** The state of `methodTraining`. */
export type MethodTrainingState<P extends Params> = {
  readonly t: Size
  readonly params: P
  /** The loss of `params`: on this step's minibatch for first-order training, on the whole set for L-BFGS. */
  readonly loss: Scalar
  /** Loss-and-gradient evaluations so far (L-BFGS's line search may take several per step). */
  readonly evaluations: Size
  readonly converged: boolean
  readonly diverged: boolean
  /** The method stopped: further steps change nothing. */
  readonly stopped: boolean
  /** The underlying algorithm's state. */
  readonly inner: unknown
}

/**
 * Train `params` on `loss` over `data` by `method`. The loss takes the parameters, a batch of `data` (the whole set for
 * L-BFGS) and, for first-order training, the layers' context (dropout streams, buffers); L-BFGS passes none, so a model
 * with dropout or batch norm should train first-order.
 */
export function methodTraining<P extends Params, B extends Batch>(
  loss: (params: P, batch: B, ctx?: Context) => Value,
  data: B,
  method: TrainingMethod,
): Algorithm<{ params: P }, MethodTrainingState<P>> {
  if (method.method === 'lbfgs') {
    const alg = fullBatchTraining<P, B>({
      loss: (p, d) => loss(p, d),
      data,
      method: 'lbfgs',
      options: { memory: method.memory, tolerance: method.tolerance },
    })
    type S = ReturnType<typeof alg.init>
    const wrap = (s: S): MethodTrainingState<P> => ({
      t: s.t,
      params: s.params,
      loss: s.value,
      evaluations: s.evaluations,
      converged: s.converged,
      diverged: s.diverged,
      stopped: alg.done?.(s) ?? false,
      inner: s,
    })
    return {
      name: 'method-training-lbfgs',
      init: (input, s) => wrap(alg.init(input, s)),
      step: (state, ctx) => (state.stopped ? { ...state, t: state.t + 1 } : wrap(alg.step(state.inner as S, ctx))),
      done: (state) => state.stopped,
    }
  }
  let optimizer = method.method === 'first-order' ? method.optimizer : undefined
  if (method.method === 'adam') {
    const { stepSize, epsilon, beta1, beta2, weightDecay, decoupled } = method
    optimizer = adamRule({ stepSize, epsilon, beta1, beta2, weightDecay, decoupled })
  }
  const alg = trainingLoop<P, B>({
    loss,
    data,
    optimizer: optimizer as UpdateRule<unknown> | undefined,
    batchSize: method.batchSize,
    clipNorm: method.clipNorm,
  })
  type S = ReturnType<typeof alg.init>
  const wrap = (s: S): MethodTrainingState<P> => ({
    t: s.t,
    params: s.params,
    loss: s.loss,
    evaluations: s.t + 1,
    converged: false,
    diverged: s.diverged,
    stopped: s.diverged,
    inner: s,
  })
  return {
    name: 'method-training-first-order',
    init: (input, s) => wrap(alg.init(input, s)),
    step: (state, ctx) => wrap(alg.step(state.inner as S, ctx)),
    done: (state) => state.diverged,
  }
}
