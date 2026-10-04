/**
 * Regression losses of predictions ŷ and targets y, elementwise over broadcast shapes: squared, absolute, Huber,
 * log-cosh, pinball (quantile) and asymmetric squared (expectile) losses of the residual, and the negative
 * log-likelihoods of Poisson and Gaussian predictive distributions. Each names the statistic of y | x that its
 * population minimiser estimates.
 */

import { Normal } from 'aifn-compute/probability/distributions'
import { logFactorial, softplus } from 'aifn-compute/numerics/special'
import {
  abs,
  add,
  exp,
  less,
  log,
  maximum,
  mul,
  neg,
  square,
  sub,
  unwrap,
  where,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { constant, defineLoss, reduce, type ReductionOptions, type Target } from './core'
import { DomainError } from 'aifn-compute/foundation/errors'

const regressionInfo = (key: string, name: string, target: string, note = 'regression-losses') =>
  ({ key, name, family: 'regression', inputs: 'values', notes: [note], target }) as const

/** The squared error (ŷ − y)², averaged by default (`torch.nn.MSELoss`). */
export const meanSquaredErrorLoss = defineLoss(
  {
    ...regressionInfo('meanSquaredErrorLoss', 'Squared error', 'the conditional mean'),
    pairedMetric: 'meanSquaredError',
  },
  (predictions: Value, targets: Target, { reduction }: ReductionOptions = {}): Value =>
    reduce(square(sub(predictions, constant(targets))), reduction),
)

/** The absolute error |ŷ − y|, averaged by default (`torch.nn.L1Loss`). Its gradient is sign(ŷ − y), 0 at a tie. */
export const meanAbsoluteErrorLoss = defineLoss(
  {
    ...regressionInfo('meanAbsoluteErrorLoss', 'Absolute error', 'the conditional median'),
    pairedMetric: 'meanAbsoluteError',
  },
  (predictions: Value, targets: Target, { reduction }: ReductionOptions = {}): Value =>
    reduce(abs(sub(predictions, constant(targets))), reduction),
)

/** Options of `huber`. */
export type HuberOptions = ReductionOptions & {
  /** The residual size δ > 0 where the loss turns from quadratic to linear. Default 1. */
  delta?: number
}

/**
 * The Huber loss (Huber, 1964) of the residual r = ŷ − y: r²/2 for |r| ≤ δ and δ(|r| − δ/2) beyond, so large residuals
 * pull with a bounded force δ. Matches `torch.nn.HuberLoss`.
 */
export const huber = defineLoss(
  regressionInfo('huber', 'Huber loss', 'a location between the conditional mean and median'),
  (predictions: Value, targets: Target, { reduction, delta = 1 }: HuberOptions = {}): Value => {
    const r = sub(predictions, constant(targets))
    const a = abs(r)
    const quadratic = mul(0.5, square(r))
    const linear = mul(delta, sub(a, delta / 2))
    return reduce(where(less(unwrap(a), delta), quadratic, linear), reduction)
  },
)

/**
 * The log-cosh loss log cosh(ŷ − y): r²/2 near zero and |r| − log 2 in the tails; smooth everywhere, with gradient
 * tanh(r). Computed as |r| + softplus(−2|r|) − log 2, which does not overflow for large residuals.
 */
export const logCosh = defineLoss(
  regressionInfo('logCosh', 'Log-cosh loss', 'a location between the conditional mean and median'),
  (predictions: Value, targets: Target, { reduction }: ReductionOptions = {}): Value => {
    const a = abs(sub(predictions, constant(targets)))
    return reduce(sub(add(a, softplus(mul(-2, a))), Math.LN2), reduction)
  },
)

/** Options of `pinball`. */
export type PinballOptions = ReductionOptions & {
  /** The quantile level τ ∈ (0, 1). Default 0.5 (half the absolute error). */
  quantile?: number
}

/**
 * The pinball (quantile) loss ρ_τ(y − ŷ) = max(τ(y − ŷ), (τ − 1)(y − ŷ)) (Koenker & Bassett, 1978): under-predictions
 * cost τ per unit and over-predictions 1 − τ, so its minimiser is the conditional τ-quantile.
 */
export const pinball = defineLoss(
  regressionInfo('pinball', 'Pinball (quantile) loss', 'the conditional τ-quantile'),
  (predictions: Value, targets: Target, { reduction, quantile = 0.5 }: PinballOptions = {}): Value => {
    if (!(quantile > 0 && quantile < 1))
      throw new DomainError('pinball', `pinball: quantile ${quantile} is not in (0, 1)`)
    const r = sub(constant(targets), predictions)
    return reduce(maximum(mul(quantile, r), mul(quantile - 1, r)), reduction)
  },
)

/** Options of `expectileLoss`. */
export type ExpectileLossOptions = ReductionOptions & {
  /** The expectile level τ ∈ (0, 1). Default 0.5 (half the squared error). */
  expectile?: number
}

/**
 * The asymmetric squared loss ρ_τ(y − ŷ) = |τ − 1(y < ŷ)| (y − ŷ)² (Newey and Powell, 1987): under-predictions cost
 * τ per squared unit and over-predictions 1 − τ, so its minimiser is the conditional τ-expectile (τ = ½: the mean).
 */
export const expectileLoss = defineLoss(
  regressionInfo(
    'expectileLoss',
    'Asymmetric squared (expectile) loss',
    'the conditional τ-expectile',
    'expectile-generalised-additive-models',
  ),
  (predictions: Value, targets: Target, { reduction, expectile = 0.5 }: ExpectileLossOptions = {}): Value => {
    if (!(expectile > 0 && expectile < 1))
      throw new DomainError('expectileLoss', `expectileLoss: expectile ${expectile} is not in (0, 1)`)
    const r = sub(constant(targets), predictions)
    const r2 = square(r)
    return reduce(where(less(unwrap(r), 0), mul(1 - expectile, r2), mul(expectile, r2)), reduction)
  },
)

/** Options of `poissonNll`. */
export type PoissonNllOptions = ReductionOptions & {
  /** The prediction is the log-rate log λ (default true, the natural link); false: the rate λ > 0 itself. */
  logInput?: boolean
  /** Add log y!, so the value is the full negative log-likelihood rather than its λ-dependent part. Default false. */
  full?: boolean
}

/**
 * The Poisson negative log-likelihood of counts y under rate λ: λ − y log λ (+ log y! when `full`). From a log-rate
 * η it is e^η − yη, convex in η. Matches `torch.nn.PoissonNLLLoss` without `full`; with `full`, the exact log y! is
 * used (PyTorch uses Stirling's approximation for y > 1), and nothing is added to λ inside the log (PyTorch adds
 * 1e-8).
 */
export const poissonNll = defineLoss(
  regressionInfo('poissonNll', 'Poisson negative log-likelihood', 'the conditional mean count', 'poisson-regression'),
  (
    predictions: Value,
    targets: Target,
    { reduction, logInput = true, full = false }: PoissonNllOptions = {},
  ): Value => {
    const y = constant(targets)
    let loss = logInput ? sub(exp(predictions), mul(y, predictions)) : sub(predictions, mul(y, log(predictions)))
    if (full) loss = add(loss, logFactorial(y))
    return reduce(loss, reduction)
  },
)

/** Options of `gaussianNll`. */
export type GaussianNllOptions = ReductionOptions & {
  /** Add ½ log 2π, so the value is the full negative log-density. Default false (as PyTorch). */
  full?: boolean
}

/**
 * The Gaussian negative log-likelihood of targets y under N(μ, σ²), with the mean μ and standard deviation σ > 0
 * predicted: log σ + (y − μ)²/(2σ²) (+ ½ log 2π when `full`). Following aifn's convention the scale is a standard
 * deviation; `torch.nn.GaussianNLLLoss` takes the variance σ² and clamps it below at 1e-6, which aifn never does.
 * Computed as −log N(y; μ, σ) from `aifn-compute/probability/distributions`, minus the constant unless `full`.
 */
export const gaussianNll = defineLoss(
  regressionInfo(
    'gaussianNll',
    'Gaussian negative log-likelihood',
    'the conditional mean and standard deviation',
    'robust-and-distributional-regression-losses',
  ),
  (mean: Value, targets: Target, sd: Value, { reduction, full = false }: GaussianNllOptions = {}): Value => {
    const nll = neg(Normal(mean, sd).logProb(constant(targets)))
    return reduce(full ? nll : sub(nll, 0.5 * Math.log(2 * Math.PI)), reduction)
  },
)
