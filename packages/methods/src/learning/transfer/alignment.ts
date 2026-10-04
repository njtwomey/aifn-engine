/**
 * Differentiable measures of how far apart two feature distributions are, the penalties of domain adaptation:
 *
 * - the squared maximum mean discrepancy with a Gaussian kernel (Gretton et al., 2012), the biased V-statistic
 *   MMD² = mean k(xₛ, xₛ′) + mean k(xₜ, xₜ′) − 2 mean k(xₛ, xₜ) with k(a, b) = exp(−‖a − b‖²/(2h²)), the penalty of deep
 *   adaptation networks (Long et al., 2015);
 * - the CORAL loss ‖Cₛ − Cₜ‖²_F / (4d²) between the feature covariances (Sun and Saenko, 2016);
 * - the gradient-reversal layer of DANN (Ganin et al., 2016): the identity going forward, −λ times the gradient going
 *   back, written as stopgrad((1 + λ) x) − λ x.
 */

import { stopGradient } from 'aifn-compute/foundation/autodiff'
import {
  add,
  div,
  exp,
  matmul,
  mean,
  mul,
  square,
  sub,
  sum,
  transpose,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** Pairwise squared distances between the rows of a [n, d] and b [m, d], as [n, m]. */
function squaredDistances(a: Value, b: Value): Value {
  const aa = sum(square(a), 1, true)
  const bb = transpose(sum(square(b), 1, true))
  return add(sub(aa, mul(2, matmul(a, transpose(b)))), bb)
}

/** The squared MMD between samples xs [n, d] and xt [m, d] with a Gaussian kernel of bandwidth h (a traced value). */
export function mmdSquared(xs: Value, xt: Value, bandwidth = 1): Value {
  const k = (a: Value, b: Value) => mean(exp(div(squaredDistances(a, b), -2 * bandwidth * bandwidth)))
  return sub(add(k(xs, xs), k(xt, xt)), mul(2, k(xs, xt)))
}

/** The covariance matrix [d, d] of rows x [n, d] (divisor n − 1). */
function covariance(x: Value, n: number): Value {
  const centred = sub(x, mean(x, 0, true))
  return div(matmul(transpose(centred), centred), n - 1)
}

/** CORAL: ‖Cov(xs) − Cov(xt)‖²_F / (4d²). */
export function coralLoss(xs: Value, xt: Value, rows: { source: number; target: number; dimension: number }): Value {
  const diff = sub(covariance(xs, rows.source), covariance(xt, rows.target))
  return div(sum(square(diff)), 4 * rows.dimension * rows.dimension)
}

/** The gradient-reversal layer: x forward, −λ ∂ backward. */
export function gradientReversal(x: Value, lambda: number): Value {
  return sub(stopGradient(mul(1 + lambda, x)), mul(lambda, x))
}
