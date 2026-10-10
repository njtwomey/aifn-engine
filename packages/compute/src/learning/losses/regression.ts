/**
 * Regression losses of predictions $\hat{y}$ and targets $y$, elementwise over broadcast shapes: squared, absolute,
 * Huber, log-cosh, pinball (quantile) and asymmetric squared (expectile) losses of the residual, and the negative
 * log-likelihoods of Poisson and Gaussian predictive distributions. Each names, in its registry `target`, the statistic
 * of $y \mid \xvec$ that its population minimiser estimates. Targets are constants; the loss is differentiable in the
 * predictions, and each returns the mean over every entry by default.
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

/**
 * The registry metadata shared by the regression losses: family `regression`, inputs `values`.
 *
 * @param key The loss's key, its export name.
 * @param name The loss's display name.
 * @param target What the prediction estimates at the loss's population minimiser, in words.
 * @param note The key of the site note the loss is listed under.
 * @returns The `LossSpec` to pass to `defineLoss`.
 */
const regressionInfo = (key: string, name: string, target: string, note = 'regression-losses') =>
  ({ key, name, family: 'regression', inputs: 'values', notes: [note], target }) as const

/**
 * The squared error $(\hat{y} - y)^2$, averaged by default (`torch.nn.MSELoss`; the mean is scikit-learn's
 * `mean_squared_error`). Its minimiser is the conditional mean.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Errors 0, 2 and 3
 * print('mse =', meanSquaredErrorLoss(tensor([1, 2, 3]), tensor([1, 0, 0])), ' (0 + 4 + 9)/3 =', 13 / 3)
 */
export const meanSquaredErrorLoss = defineLoss(
  {
    ...regressionInfo('meanSquaredErrorLoss', 'Squared error', 'the conditional mean'),
    pairedMetric: 'meanSquaredError',
  },
  (predictions: Value, targets: Target, { reduction }: ReductionOptions = {}): Value =>
    reduce(square(sub(predictions, constant(targets))), reduction),
)

/**
 * The absolute error $\lvert \hat{y} - y \rvert$, averaged by default (`torch.nn.L1Loss`). Its gradient is
 * $\sgn(\hat{y} - y)$, 0 at a tie, and its minimiser the conditional median.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Errors 0, 2 and 3, and the gradient's signs
 * print('mae =', meanAbsoluteErrorLoss(tensor([1, 2, 3]), tensor([1, 0, 4])), ' (0 + 2 + 1)/3 =', 1)
 * print('grad =', grad((p) => meanAbsoluteErrorLoss(p, tensor([1, 0, 4]), { reduction: 'sum' }))(tensor([1, 2, 3])))
 */
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
  /** The residual size $\delta > 0$ where the loss turns from quadratic to linear. Default 1. */
  delta?: number
}

/**
 * The Huber loss (Huber, 1964) of the residual $r = \hat{y} - y$: $r^2/2$ for $\lvert r \rvert \le \delta$ and
 * $\delta(\lvert r \rvert - \delta/2)$ beyond, so large residuals pull with a bounded force $\delta$. Matches
 * `torch.nn.HuberLoss`.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options $\delta$ and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Quadratic inside the threshold, linear beyond it
 * print(huber(tensor([0.5, 3]), tensor([0, 0]), { reduction: 'none' }))
 * print('0.5^2 / 2 =', 0.125, ' 1 * (3 - 1/2) =', 2.5)
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
 * The log-cosh loss $\log \cosh(\hat{y} - y)$: $r^2/2$ near zero and $\lvert r \rvert - \log 2$ in the tails;
 * smooth everywhere, with gradient $\tanh r$. Computed as
 * $\lvert r \rvert + \operatorname{softplus}(-2 \lvert r \rvert) - \log 2$, which does not overflow for large
 * residuals.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Against the closed form, and a residual whose cosh would overflow
 * print(logCosh(tensor([0, 1, 1000]), 0, { reduction: 'none' }))
 * print('log cosh 1 =', Math.log(Math.cosh(1)), ' 1000 - log 2 =', 1000 - Math.log(2))
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
  /** The quantile level $\tau \in (0, 1)$. Default 0.5 (half the absolute error). */
  quantile?: number
}

/**
 * The pinball (quantile) loss $\rho_\tau(y - \hat{y}) = \max(\tau (y - \hat{y}), (\tau - 1)(y - \hat{y}))$ (Koenker &
 * Bassett, 1978): under-predictions cost $\tau$ per unit and over-predictions $1 - \tau$, so its minimiser is the
 * conditional $\tau$-quantile. As scikit-learn's `mean_pinball_loss` (`alpha` is $\tau$). A level outside $(0, 1)$
 * throws `DomainError`.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options The level $\tau$ and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example At the 0.9 quantile, under-predicting costs nine times more
 * print(pinball(tensor([8, 12]), 10, { quantile: 0.9, reduction: 'none' }))
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
  /** The expectile level $\tau \in (0, 1)$. Default 0.5 (half the squared error). */
  expectile?: number
}

/**
 * The asymmetric squared loss $\rho_\tau(y - \hat{y}) = \lvert \tau - \indicator[y < \hat{y}] \rvert (y - \hat{y})^2$
 * (Newey and Powell, 1987): under-predictions cost $\tau$ per squared unit and over-predictions $1 - \tau$, so its
 * minimiser is the conditional $\tau$-expectile ($\tau = \tfrac{1}{2}$: the mean). A level outside $(0, 1)$ throws
 * `DomainError`.
 *
 * @param predictions The predictions $\hat{y}$, of any shape.
 * @param targets The targets $y$, broadcast against `predictions`; constants.
 * @param options The level $\tau$ and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example At the 0.9 expectile, under-predicting costs nine times more
 * print(expectileLoss(tensor([8, 12]), 10, { expectile: 0.9, reduction: 'none' }))
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
  /** The prediction is the log-rate $\log \lambda$ (default true, the natural link); false: the rate $\lambda > 0$. */
  logInput?: boolean
  /** Add $\log y!$ (default false): the full negative log-likelihood, not only its $\lambda$-dependent part. */
  full?: boolean
}

/**
 * The Poisson negative log-likelihood of counts $y$ under rate $\lambda$: $\lambda - y \log \lambda$ (plus
 * $\log y!$ when `full`). From a log-rate $\eta$ it is $e^\eta - y \eta$, convex in $\eta$. Matches
 * `torch.nn.PoissonNLLLoss` without `full`; with `full`, the exact $\log y!$ is used (PyTorch uses Stirling's
 * approximation for $y > 1$), and nothing is added to $\lambda$ inside the log (PyTorch adds `1e-8`).
 *
 * @param predictions The log-rates $\eta$ (default), or the rates $\lambda > 0$ when `logInput` is false.
 * @param targets The observed counts $y$, broadcast against `predictions`; constants.
 * @param options Whether the predictions are log-rates, whether to add $\log y!$, and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Three counts under a rate of 2, against the Poisson log-probability
 * const lambda = 2
 * print('rate-dependent part:', poissonNll(lambda, 3, { logInput: false }), ' 2 - 3 log 2 =', 2 - 3 * Math.log(2))
 * print('full:', poissonNll(Math.log(lambda), 3, { full: true }), ' -log(e^-2 2^3 / 3!) =', 2 - Math.log(8 / 6))
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
  /** Add $\tfrac{1}{2} \log 2\pi$, so the value is the full negative log-density. Default false (as PyTorch). */
  full?: boolean
}

/**
 * The Gaussian negative log-likelihood of targets $y$ under $\Gauss(\mu, \sigma^2)$, with the mean $\mu$ and standard
 * deviation $\sigma > 0$ predicted: $\log \sigma + (y - \mu)^2 / (2\sigma^2)$ (plus $\tfrac{1}{2} \log 2\pi$ when
 * `full`). Following aifn's convention the scale is a standard deviation; `torch.nn.GaussianNLLLoss` takes the
 * variance $\sigma^2$ and clamps it below at `1e-6`, which aifn never does. Computed as
 * $-\log \Gauss(y; \mu, \sigma^2)$ from `aifn-compute/probability/distributions`, minus the constant unless `full`.
 * Note the order of the arguments: the mean, the targets, then the standard deviation.
 *
 * @param mean The predicted means $\mu$, of any shape.
 * @param targets The targets $y$, broadcast against `mean`; constants.
 * @param sd The predicted standard deviations $\sigma > 0$, broadcast against `mean`; differentiable like `mean`.
 * @param options Whether to add the constant, and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example One standard deviation from the mean
 * print('nll =', gaussianNll(0, 1, 1), ' 1/2 =', 0.5)
 * print('full =', gaussianNll(0, 1, 1, { full: true }), ' 1/2 + log(2 pi)/2 =', 0.5 + 0.5 * Math.log(2 * Math.PI))
 *
 * @example A wider predicted spread is cheaper for a large error, dearer for a small one
 * print('error 3:', gaussianNll(tensor([0, 0]), 3, tensor([1, 3]), { reduction: 'none' }))
 * print('error 0:', gaussianNll(tensor([0, 0]), 0, tensor([1, 3]), { reduction: 'none' }))
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
