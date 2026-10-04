/**
 * Training an energy-based model p_θ(x) ∝ exp(−E_θ(x)) by persistent contrastive divergence as a traceable algorithm:
 * each step draws negatives by short-run Langevin from a replay buffer of persistent chains
 * (`aifn-compute/inference/stochastic`'s `persistentLangevin`; Tieleman, 2008; Du & Mordatch, 2019) and takes one optimiser
 * step on `contrastiveDivergenceLoss` (mean E on data − mean E on negatives), plus an optional supervised term on the
 * same minibatch. With the term the softmax cross-entropy of a classifier whose energy is −logsumexp of its logits,
 * this is JEM (Grathwohl et al., 2019, Algorithm 1).
 */

import type { Scalar, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { grad as gradOf, valueAndGrad, type ValueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  mean,
  mul,
  neg,
  sum,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  persistentLangevin,
  chainBuffer,
  type PersistentLangevinOptions,
  type ChainBuffer,
} from 'aifn-compute/inference/stochastic'
import { contrastiveDivergenceLoss } from 'aifn-compute/learning/losses'
import { adamRule, applyUpdates, globalNorm, type UpdateRule } from 'aifn-compute/optim/first-order'
import type { Batch } from './train'

/** Options of `contrastiveDivergence`. */
export type ContrastiveDivergenceTrainingOptions<P extends Params, B extends Batch & { x: Tensor }> = {
  /** The energy E_θ of a batch of points [n, d], one value per row ([n]). */
  energy: (params: P, x: Value) => Value
  /** The training set; `x` [N, d] holds the points, other fields (labels) go to `supervised`. */
  data: B
  /** Points per step. Default 64. */
  batchSize?: Size
  /** The update rule. Default `adamRule({ stepSize: 1e-3 })`. */
  optimizer?: UpdateRule<unknown>
  /** The sampler of negatives: Langevin steps, step size, noise, restart probability, fresh points and bound. */
  sampler: PersistentLangevinOptions
  /** Persistent chains kept in the replay buffer. Default 1000. */
  bufferSize?: Size
  /** Weight of the generative term (the contrastive-divergence surrogate). Default 1; 0 trains `supervised` alone. */
  generativeWeight?: Scalar
  /** The energy-magnitude penalty α of `contrastiveDivergenceLoss`. Default 0. */
  regularisation?: Scalar
  /** A supervised loss on the minibatch, added to the generative term (JEM's cross-entropy). */
  supervised?: (params: P, batch: B) => Value
  /** Flag divergence when the loss is not finite or exceeds this in size. Default 1e8. */
  divergeAbove?: Scalar
}

/** The state of `contrastiveDivergence` after t updates. */
export interface ContrastiveDivergenceState<P extends Params> extends Status {
  readonly t: Size
  readonly params: P
  readonly optimizer: unknown
  /** The persistent chains. */
  readonly buffer: ChainBuffer
  /** The negatives of the last step ([n, d]; empty before the first). */
  readonly negatives: Tensor
  /** The total loss, its generative (CD) part and its supervised part, before the last update (NaN at t = 0). */
  readonly loss: Scalar
  readonly generativeLoss: Scalar
  readonly supervisedLoss: Scalar
  /** Mean energy of the minibatch's data and of the negatives (NaN at t = 0). */
  readonly dataEnergy: Scalar
  readonly sampleEnergy: Scalar
  readonly gradNorm: Scalar
  readonly diverged: boolean
}

const scalarOf = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : toFlat(raw)[0]
}

/**
 * Persistent contrastive-divergence training. `init` takes `{ params }`; the buffer starts from `sampler.fresh` on the
 * init stream. Step t draws a minibatch (without replacement, from `child(s, 'batch')`), negatives by
 * `persistentLangevin` under the current parameters (`child(s, 'negatives')`), and applies one update of the gradient
 * of w_gen·(mean E(x) − mean E(x′)) + supervised. Set `generativeWeight` to 0 to train the supervised term alone with
 * the same minibatches (the baseline JEM is compared against).
 */
export function contrastiveDivergence<P extends Params, B extends Batch & { x: Tensor }>(
  options: ContrastiveDivergenceTrainingOptions<P, B>,
): Algorithm<{ params: P }, ContrastiveDivergenceState<P>> {
  const { energy, data, sampler, supervised, regularisation = 0, divergeAbove = 1e8 } = options
  const weight = options.generativeWeight ?? 1
  const rule = options.optimizer ?? (adamRule({ stepSize: 1e-3 }) as UpdateRule<unknown>)
  const N = data.x.shape[0]
  const d = data.x.shape[1]
  const n = Math.min(options.batchSize ?? 64, N)
  const bufferSize = options.bufferSize ?? 1000
  // The loss's parts at the last evaluation, read off the traced values (valueAndGrad returns only the total).
  let seen = { generative: NaN, supervised: NaN, dataEnergy: NaN, sampleEnergy: NaN }
  const total: (params: P, batch: B, negatives: Tensor) => ValueAndGrad<Value, unknown> = valueAndGrad(
    (params: P, batch: B, negatives: Tensor): Value => {
      let loss: Value = 0
      seen = { generative: NaN, supervised: NaN, dataEnergy: NaN, sampleEnergy: NaN }
      if (weight !== 0) {
        const positive = energy(params, batch.x)
        const negative = energy(params, negatives)
        const generative = contrastiveDivergenceLoss(positive, negative, { regularisation })
        seen.generative = scalarOf(generative)
        seen.dataEnergy = scalarOf(mean(positive))
        seen.sampleEnergy = scalarOf(mean(negative))
        loss = weight === 1 ? generative : mul(weight, generative)
      }
      if (supervised) {
        const term = supervised(params, batch)
        seen.supervised = scalarOf(term)
        loss = add(loss, term)
      }
      return loss
    },
    {},
  )
  return {
    name: `contrastive-divergence-${rule.name}`,
    init: ({ params }, s) => ({
      t: 0,
      params,
      optimizer: rule.init(params),
      buffer:
        weight === 0 ? { samples: fromData(new Float64Array(0), [0, d]) } : chainBuffer(s, bufferSize, sampler.fresh),
      negatives: fromData(new Float64Array(0), [0, d]),
      loss: NaN,
      generativeLoss: NaN,
      supervisedLoss: NaN,
      dataEnergy: NaN,
      sampleEnergy: NaN,
      gradNorm: NaN,
      diverged: false,
    }),
    step: (state, ctx: StepContext) => {
      // A minibatch without replacement: a partial Fisher–Yates shuffle.
      const pick = child(ctx.stream, 'batch')
      const order = Int32Array.from({ length: N }, (_, i) => i)
      for (let i = 0; i < n; i++) {
        const j = i + integers(pick, N - i)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const ids = Array.from(order.subarray(0, n))
      const batch = Object.fromEntries(
        Object.entries(data).map(([key, v]) => [key, unwrap(take(v, ids)) as Tensor]),
      ) as B
      let buffer = state.buffer
      let negatives = state.negatives
      if (weight !== 0) {
        const score = (x: Tensor) => neg(gradOf((y: Value) => sum(energy(state.params, y)))(x) as Tensor) as Tensor
        const draw = persistentLangevin(score, buffer, child(ctx.stream, 'negatives'), n, sampler)
        buffer = draw.buffer
        negatives = draw.x
      }
      const { value, grad } = total(state.params, batch, negatives)
      const loss = scalarOf(value as Value)
      const { updates, state: next } = rule.update(grad as Params, state.optimizer, state.params)
      return {
        t: state.t + 1,
        params: applyUpdates(state.params, updates),
        optimizer: next,
        buffer,
        negatives,
        loss,
        generativeLoss: seen.generative,
        supervisedLoss: seen.supervised,
        dataEnergy: seen.dataEnergy,
        sampleEnergy: seen.sampleEnergy,
        gradNorm: globalNorm(grad as Params),
        diverged: !Number.isFinite(loss) || Math.abs(loss) > divergeAbove,
      }
    },
  }
}
