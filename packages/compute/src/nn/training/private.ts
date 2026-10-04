/**
 * Differentially private training, DP-SGD (Abadi et al., 2016): each step draws a Poisson sample of the examples (each
 * kept with probability q), computes every sampled example's gradient (`vmap(grad(loss))`), clips each to L2 norm C,
 * sums them, adds N(0, σ²C²) noise and divides by the expected batch size qn (`aifn-compute/probability/privacy`'s
 * `clipAndNoise`), then applies any update rule of `aifn-compute/optim/first-order`. The privacy spent after t steps is
 * reported as ε at the given δ by Rényi-DP accounting of the subsampled Gaussian mechanism (`dpSgdEpsilon`), as
 * Opacus's RDP accountant; the guarantee covers the released parameters of every step.
 */

import type { Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { treeMap, type Params } from 'aifn-compute/foundation/pytree'
import { child, uniform } from 'aifn-compute/foundation/random'
import { take, toFlat, unwrap, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adamRule, applyUpdates, type UpdateRule } from 'aifn-compute/optim/first-order'
import { clipAndNoise, rdpSubsampledGaussian, rdpToEpsilon } from 'aifn-compute/probability/privacy'
import type { Batch } from './train'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `privateTraining`. */
export type PrivateTrainingOptions<P extends Params, B extends Batch> = {
  /** The loss of the parameters on one example (each field of `data` without its leading axis). */
  loss: (params: P, example: B) => Value
  /** The training set: named tensors whose first axis indexes the examples. */
  data: B
  /** Expected examples per step, qn (default 64, capped at n); sets the sampling rate q. */
  batchSize?: Size
  /** Per-example clipping norm C (default 1). */
  clipNorm?: Scalar
  /** Noise multiplier σ: the noise's standard deviation over C (default 1; 0 trains without noise). */
  noiseMultiplier?: Scalar
  /** The update rule (default `adamRule({ stepSize: 0.01 })`). */
  optimizer?: UpdateRule<unknown>
  /** The δ at which ε is reported (default 1e-5). */
  delta?: Scalar
}

/** The state of `privateTraining` after t updates. */
export interface PrivateTrainingState<P extends Params> extends Status {
  readonly t: Size
  readonly params: P
  readonly optimizer: unknown
  /** The mean loss over this step's sample (NaN when the sample was empty). */
  readonly loss: Scalar
  /** Examples in this step's Poisson sample. */
  readonly batchSize: Size
  /** The mean per-example gradient norm before clipping. */
  readonly gradNorm: Scalar
  /** Share of the sample whose gradients were clipped. */
  readonly clippedShare: Scalar
  /** ε at `delta` spent by the t updates so far (∞ without noise). */
  readonly epsilon: Scalar
  readonly diverged: boolean
}

function examplesOf(data: Batch): Size {
  const sizes = Object.values(data).map((t) => t.shape[0])
  if (sizes.length === 0 || sizes.some((s) => s !== sizes[0]))
    throw new ShapeError('privateTraining', 'privateTraining: data fields differ in length')
  return sizes[0]
}

/** DP-SGD as a traceable algorithm; `init` takes `{ params }`. The sample of step t is drawn from its stream. */
export function privateTraining<P extends Params, B extends Batch>(
  options: PrivateTrainingOptions<P, B>,
): Algorithm<{ params: P }, PrivateTrainingState<P>> {
  const { loss, data, clipNorm = 1, noiseMultiplier = 1, delta = 1e-5 } = options
  const rule = options.optimizer ?? (adamRule({ stepSize: 0.01 }) as UpdateRule<unknown>)
  const n = examplesOf(data)
  const expected = Math.min(options.batchSize ?? 64, n)
  const q = expected / n
  const perExampleGrads = vmap(
    grad((p: P, example: B) => loss(p, example)),
    { inAxes: [null, 0] },
  )
  // RDP adds over steps: one step's RDP, scaled by t, gives the ε after t steps (`dpSgdEpsilon`).
  const perStep = noiseMultiplier > 0 ? rdpSubsampledGaussian(q, noiseMultiplier, 1) : null
  const perExampleLoss = vmap((p: P, example: B) => loss(p, example), { inAxes: [null, 0] })
  return {
    name: 'dp-sgd',
    init: ({ params }) => ({
      t: 0,
      params,
      optimizer: rule.init(params),
      loss: NaN,
      batchSize: 0,
      gradNorm: NaN,
      clippedShare: 0,
      epsilon: 0,
      diverged: false,
    }),
    step: (state, ctx) => {
      const u = toFlat(uniform(child(ctx.stream, 'sample'), 0, 1, { shape: [n] }))
      const ids: number[] = []
      for (let i = 0; i < n; i++) if (u[i] < q) ids.push(i)
      let perExample: unknown
      let lossValue = NaN
      if (ids.length > 0) {
        const batch = Object.fromEntries(Object.entries(data).map(([k, v]) => [k, unwrap(take(v, ids)) as Tensor])) as B
        perExample = perExampleGrads(state.params, batch)
        const losses = toFlat(perExampleLoss(state.params, batch) as Tensor)
        lossValue = losses.reduce((a, b) => a + b, 0) / losses.length
      } else perExample = treeMap(state.params, (leaf) => zeros([1, ...(typeof leaf === 'number' ? [] : leaf.shape)]))
      const noisy = clipAndNoise(perExample as P, clipNorm, noiseMultiplier, child(ctx.stream, 'noise'), expected)
      const { updates, state: next } = rule.update(noisy.gradient as Params, state.optimizer, state.params)
      const params = applyUpdates(state.params, updates)
      const t = state.t + 1
      const norms = noisy.norms
      return {
        t,
        params,
        optimizer: next,
        loss: lossValue,
        batchSize: ids.length,
        gradNorm: ids.length ? norms.reduce((a, b) => a + b, 0) / norms.length : NaN,
        clippedShare: ids.length ? noisy.clippedShare : 0,
        epsilon: perStep
          ? rdpToEpsilon(
              perStep.map((r) => r * t),
              delta,
            ).epsilon
          : Infinity,
        diverged: ids.length > 0 && !Number.isFinite(lossValue),
      }
    },
  }
}
