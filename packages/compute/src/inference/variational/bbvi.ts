/**
 * Black-box variational inference (Ranganath, Gerrish & Blei, 2014; Kucukelbir et al., 2017): stochastic gradient
 * ascent on the ELBO over a Gaussian family, with `aifn-compute/optim`'s Adam as the optimiser. It needs only the
 * target's log density (and its gradient, or autodiff, for the reparameterisation estimator), not a conjugate model.
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
  /** The number of steps taken (0 after init). */
  t: number
  /** The variational parameters $\lambdavec$. */
  lambda: Vector
  /** The mean $\muvec$ of $q_{\lambdavec}$. */
  mean: Vector
  /** The scale factor $\Lmat$ of $q_{\lambdavec}$ ($d \times d$). */
  scale: Matrix
  /** The covariance $\Lmat\Lmat^\top$ of $q_{\lambdavec}$ ($d \times d$). */
  covariance: Matrix
  /** The ELBO estimated from this step's gradient draws (noisy). */
  elbo: number
  /** The ELBO gradient estimate at $\lambdavec$. */
  grad: Vector
  /** The norm of the gradient estimate. */
  gradNorm: number
  /**
   * The optimiser's state (Adam minimising $-\operatorname{ELBO}$): moments in `slots`, the last update in `update`.
   */
  optimiser: FirstOrderState
  /** The divergence minimised: always the reverse $\KL(q \,\|\, p)$. */
  objective: 'KL(q‖p)'
  /** Whether Adam reported divergence (a non-finite objective or iterate). */
  diverged: boolean
}

/** Options for `bbvi`. */
export type BbviOptions = {
  /** `mean-field`, `full-rank`, or a family object. Default mean field. */
  family?: 'mean-field' | 'full-rank' | GaussianFamily
  /** The ELBO gradient estimator, as `elboGradient` takes it. Default `'reparameterisation'`. */
  estimator?: GradientEstimator
  /** Draws per gradient estimate. Default 1 (reparameterisation) or 10 (score). */
  samples?: number
  /** The score estimator's baseline, as `elboGradient` takes it. */
  baseline?: Baseline
  /** Adam's step size, or a schedule $t \mapsto \eta_t$. Default 0.05. */
  stepSize?: StepSize
}

/**
 * The start: `lambda0`, the initial $\lambdavec_0$ itself; or else `mean0` and `sd0`, a mean (default zeros) and a
 * standard deviation for every coordinate (default 1) that the family turns into $\lambdavec_0$.
 */
export type BbviStart = { lambda0?: VectorLike; mean0?: VectorLike; sd0?: number }

/**
 * Black-box VI: maximise the ELBO over $q_{\lambdavec}$ by Adam on stochastic gradients from `elboGradient`. Step $t$
 * estimates the gradient with draws from its step stream (`ctx.stream`; the initial evaluation from the init stream),
 * so the run is a pure function of its root key: each step builds Adam on that step's objective and advances the
 * embedded Adam state by one step. Adam never stops on its own (no tolerance), so the run lasts as many steps as it is
 * given; divergence is reported in `diverged`.
 *
 * @param target The unnormalised log density $\log \tilde p$ to approximate; its `dim` sets the family's dimension.
 * @param options The family, the gradient estimator and its draws and baseline, and Adam's step size.
 * @returns The algorithm; its start is a `BbviStart` (`{}` for the defaults) and it needs a random stream.
 *
 * @example Recover a conjugate Gaussian posterior
 * // log p(m) = 3m - 2m^2 + c: the posterior of a mean m with prior N(0, 1) after three unit-noise observations
 * // summing to 3, which is N(0.75, 0.5^2).
 * const target = { kind: 'log-density', dim: 1, logDensity: (m) => sum(sub(mul(3, m), mul(2, mul(m, m)))) }
 * const s = run(bbvi(target, { samples: 10 }), {}, 300, { stream: stream(0) })
 * print('mean =', s.mean, 'exact 0.75')
 * print('sd =', Math.sqrt(s.covariance.data[0]), 'exact 0.5')
 * // The state's `elbo` is a noisy one-step estimate; `elbo` with many draws is tighter.
 * print('ELBO =', elbo(stream(1), target, meanFieldGaussian(1), s.lambda).value)
 * print('log Z =', 1.125 + Math.log(Math.PI / 2) / 2)
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
