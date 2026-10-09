/**
 * Differentially private training, DP-SGD (Abadi et al., 2016), as a traceable algorithm.
 *
 * Each step draws a Poisson sample of the examples (each kept with probability $q$), computes every sampled example's
 * gradient (`vmap(grad(loss))`), clips each to Euclidean norm $C$, sums them, adds $\Gauss(0, \sigma^2 C^2)$ noise and
 * divides by the expected batch size $qn$ (`aifn-compute/probability/privacy`'s `clipAndNoise`), then applies any
 * update rule of `aifn-compute/optim/first-order`. The privacy spent after $t$ steps is reported as $\varepsilon$ at
 * the given $\delta$ by Rényi-DP accounting of the subsampled Gaussian mechanism (as `dpSgdEpsilon` computes it), as
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
  /** Expected examples per step, $qn$ (default 64, capped at $n$); sets the sampling rate $q$. */
  batchSize?: Size
  /** Per-example clipping norm $C$ (default 1). */
  clipNorm?: Scalar
  /** Noise multiplier $\sigma$: the noise's standard deviation over $C$ (default 1; 0 trains without noise). */
  noiseMultiplier?: Scalar
  /** The update rule (default `adamRule({ stepSize: 0.01 })`). */
  optimizer?: UpdateRule<unknown>
  /** The $\delta$ at which $\varepsilon$ is reported (default 1e-5). */
  delta?: Scalar
}

/** The state of `privateTraining` after $t$ updates. */
export interface PrivateTrainingState<P extends Params> extends Status {
  /** Updates applied so far. */
  readonly t: Size
  /** The parameters after $t$ updates. */
  readonly params: P
  /** The update rule's state. */
  readonly optimizer: unknown
  /**
   * The mean loss over this step's sample, at the parameters before the update (NaN at $t = 0$ and when the sample
   * was empty).
   */
  readonly loss: Scalar
  /** Examples in this step's Poisson sample. */
  readonly batchSize: Size
  /** The mean per-example gradient norm before clipping (NaN at $t = 0$ and when the sample was empty). */
  readonly gradNorm: Scalar
  /** Share of the sample whose gradients were clipped. */
  readonly clippedShare: Scalar
  /** $\varepsilon$ at `delta` spent by the $t$ updates so far (0 at $t = 0$; $\infty$ without noise). */
  readonly epsilon: Scalar
  /** Whether the loss of a non-empty sample is not finite; a run stops here. */
  readonly diverged: boolean
}

/**
 * The data's number of examples. Throws `ShapeError` when the fields differ in length or there are none.
 *
 * @param data The training set, whose fields' first axes index the examples.
 * @returns The common length of the first axes.
 */
function examplesOf(data: Batch): Size {
  const sizes = Object.values(data).map((t) => t.shape[0])
  if (sizes.length === 0 || sizes.some((s) => s !== sizes[0]))
    throw new ShapeError('privateTraining', 'privateTraining: data fields differ in length')
  return sizes[0]
}

/**
 * DP-SGD as a traceable algorithm; `init` takes `{ params }`. The sample of step $t$ is drawn from `child(s, 'sample')`
 * of its stream `s`, and the noise from `child(s, 'noise')`. An empty sample still takes a step, on noise alone.
 *
 * @param options The per-example loss, the data, the expected batch size, the clipping norm, the noise multiplier,
 *   the update rule and the $\delta$ of the reported $\varepsilon$.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example Fit $y = 2x + 1$ privately: the loss falls while $\varepsilon$ grows with every step
 * const x = linspace(-1, 1, 20)
 * const data = { x, y: add(mul(2, x), 1) }
 * const loss = (p, e) => square(sub(add(mul(p.w, e.x), p.b), e.y))
 * const alg = privateTraining({ loss, data, batchSize: 10 })
 * const tr = trace(alg, { params: { w: 0, b: 0 } }, 300, {
 *   every: 100,
 *   record: { loss: (s) => s.loss, epsilon: (s) => s.epsilon },
 * })
 * print('steps:', tr.index)
 * print('loss:', tr.series.loss)
 * print('epsilon:', tr.series.epsilon)
 */
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
