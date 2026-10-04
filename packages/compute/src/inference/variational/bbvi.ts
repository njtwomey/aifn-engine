/**
 * Black-box variational inference (Ranganath, Gerrish & Blei, 2014; Kucukelbir et al., 2017): stochastic gradient
 * ascent on the ELBO over a Gaussian family, with `aifn-compute/optim`'s Adam as the optimiser.
 */

import type { LogDensity, Status } from 'aifn-compute/foundation/contracts'
import { adam, type FirstOrderState, type StepSize } from 'aifn-compute/optim/first-order'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { elboGradient, type Baseline, type GradientEstimator } from './elbo'
import { fullRankGaussian, meanFieldGaussian, type GaussianFamily, type VectorLike } from './family'

/** The state of `bbvi`. */
export interface BbviState extends Status {
  t: number
  /** The variational parameters λ. */
  lambda: Vector
  /** The mean, scale factor and covariance of q_λ. */
  mean: Vector
  scale: Matrix
  covariance: Matrix
  /** The ELBO estimated from this step's gradient draws (noisy). */
  elbo: number
  /** The ELBO gradient estimate at λ and its norm. */
  grad: Vector
  gradNorm: number
  /** The optimiser's state (Adam minimising −ELBO): moments in `slots`, the last update in `update`. */
  optimiser: FirstOrderState
  /** The divergence minimised: always the reverse KL(q ‖ p). */
  objective: 'KL(q‖p)'
  diverged: boolean
}

/** Options for `bbvi`. */
export type BbviOptions = {
  /** `mean-field`, `full-rank`, or a family object. Default mean field. */
  family?: 'mean-field' | 'full-rank' | GaussianFamily
  estimator?: GradientEstimator
  /** Draws per gradient estimate. Default 1 (reparameterisation) or 10 (score). */
  samples?: number
  baseline?: Baseline
  /** Adam's step size, or a schedule t ↦ η_t. Default 0.05. */
  stepSize?: StepSize
}

/** The start: λ₀, or a mean and standard deviation (default zeros and 1). */
export type BbviStart = { lambda0?: VectorLike; mean0?: VectorLike; sd0?: number }

/**
 * Black-box VI: maximise the ELBO over q_λ by Adam on stochastic gradients from `elboGradient`. Step t estimates the
 * gradient with draws from its step stream (`ctx.stream`; the initial evaluation from the init stream), so the run is
 * a pure function of its root key: each step builds Adam on that step's objective and advances the embedded Adam
 * state by one step.
 */
export function bbvi(target: LogDensity, options: BbviOptions = {}): Algorithm<BbviStart, BbviState> {
  const name = 'bbvi'
  const d = target.dim
  const fam = options.family ?? 'mean-field'
  const family = typeof fam === 'string' ? (fam === 'full-rank' ? fullRankGaussian(d) : meanFieldGaussian(d)) : fam
  const stepSize = options.stepSize ?? 0.05
  const objectiveAt = (s: Stream) => (lambda: Vector) => {
    const g = elboGradient(s, target, family, lambda, options)
    return { value: -g.elbo, grad: toFlat(g.grad).map((v) => -v) }
  }
  const optimiserAt = (s: Stream) => adam(objectiveAt(s), { stepSize, tolerance: -1, divergeAbove: Infinity })
  const wrap = (opt: FirstOrderState, t: number): BbviState => ({
    t,
    lambda: opt.x,
    mean: family.mean(opt.x),
    scale: family.scale(opt.x),
    covariance: family.covariance(opt.x),
    elbo: -opt.value,
    grad: fromData(Float64Array.from(toFlat(opt.grad), (v) => -v)),
    gradNorm: opt.gradNorm,
    optimiser: opt,
    objective: 'KL(q‖p)',
    diverged: opt.diverged,
  })
  return {
    name,
    init: ({ lambda0, mean0, sd0 = 1 }, s) => {
      const x0 = lambda0 ?? family.parameters(mean0 ?? new Float64Array(d), sd0)
      return wrap(optimiserAt(s).init({ x0 }, child(s, 'optimiser')), 0)
    },
    step: (s, ctx) =>
      wrap(optimiserAt(ctx.stream).step(s.optimiser, { t: ctx.t, stream: child(ctx.stream, 'optimiser') }), s.t + 1),
  }
}
