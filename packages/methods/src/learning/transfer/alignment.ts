/**
 * Differentiable measures of how far apart two feature distributions are, the penalties of domain adaptation:
 *
 * - the squared maximum mean discrepancy with a Gaussian kernel (Gretton et al., 2012), the biased V-statistic
 *   $\mathrm{MMD}^2 = \bar k_{ss} + \bar k_{tt} - 2 \bar k_{st}$, where $\bar k_{st}$ is the mean of
 *   $k(\xvec_s, \xvec_t)$ over every source-target pair and
 *   $k(\avec, \bvec) = \exp(-\lVert \avec - \bvec \rVert^2 / (2h^2))$: the penalty of deep adaptation networks (Long
 *   et al., 2015);
 * - the CORAL loss $\lVert \Cmat_s - \Cmat_t \rVert_F^2 / (4d^2)$ between the feature covariances (Sun and Saenko,
 *   2016);
 * - the gradient-reversal layer of DANN (Ganin et al., 2016): the identity going forward, $-\lambda$ times the
 *   gradient going back, written as $\operatorname{stopgrad}((1 + \lambda) \xvec) - \lambda \xvec$.
 *
 * Samples are matrices of rows, $n \times d$ and $m \times d$; every penalty is differentiable in both.
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

/**
 * Pairwise squared distances between rows, $\lVert \avec_i \rVert^2 - 2 \avec_i^\top\bvec_j + \lVert \bvec_j \rVert^2$
 * (differentiable).
 *
 * @param a Rows $\avec_i$, $n \times d$.
 * @param b Rows $\bvec_j$, $m \times d$.
 * @returns The $n \times m$ squared distances.
 */
function squaredDistances(a: Value, b: Value): Value {
  const aa = sum(square(a), 1, true)
  const bb = transpose(sum(square(b), 1, true))
  return add(sub(aa, mul(2, matmul(a, transpose(b)))), bb)
}

/**
 * The squared MMD between two samples with a Gaussian kernel of bandwidth $h$, the biased V-statistic (every pair,
 * the diagonal included), differentiable in both samples. It is 0 for identical samples.
 *
 * @param xs The source sample, $n \times d$.
 * @param xt The target sample, $m \times d$.
 * @param bandwidth The kernel's bandwidth $h$.
 * @returns $\mathrm{MMD}^2$, a scalar.
 *
 * @example A sample against itself, a shifted copy and a slightly stretched copy
 * const a = tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
 * print('same sample:', mmdSquared(a, a))
 * print('shifted by (2, 2):', mmdSquared(a, add(a, 2)))
 * print('stretched by 1.2:', mmdSquared(a, mul(a, 1.2)))
 */
export function mmdSquared(xs: Value, xt: Value, bandwidth = 1): Value {
  const k = (a: Value, b: Value) => mean(exp(div(squaredDistances(a, b), -2 * bandwidth * bandwidth)))
  return sub(add(k(xs, xs), k(xt, xt)), mul(2, k(xs, xt)))
}

/**
 * The sample covariance matrix of rows, with divisor $n - 1$ (differentiable).
 *
 * @param x The rows, $n \times d$.
 * @param n The number of rows $n$.
 * @returns The $d \times d$ covariance.
 */
function covariance(x: Value, n: number): Value {
  const centred = sub(x, mean(x, 0, true))
  return div(matmul(transpose(centred), centred), n - 1)
}

/**
 * The CORAL loss $\lVert \Cmat_s - \Cmat_t \rVert_F^2 / (4d^2)$ between the two samples' covariances (divisor
 * $n - 1$), differentiable in both. It sees only second moments: a shift of the mean costs nothing.
 *
 * @param xs The source sample, $n \times d$.
 * @param xt The target sample, $m \times d$.
 * @param rows The sizes: `source` rows $n$, `target` rows $m$ and the `dimension` $d$ (passed in, so the loss works
 *   on traced values without reading their shapes).
 * @returns The loss, a scalar.
 *
 * @example Equal covariances cost nothing, even shifted; a sample scaled by 2 has four times the covariance
 * const a = tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
 * const rows = { source: 4, target: 4, dimension: 2 }
 * print('shifted:', coralLoss(a, add(a, 5), rows))
 * print('scaled by 2:', coralLoss(a, mul(a, 2), rows))
 */
export function coralLoss(xs: Value, xt: Value, rows: { source: number; target: number; dimension: number }): Value {
  const diff = sub(covariance(xs, rows.source), covariance(xt, rows.target))
  return div(sum(square(diff)), 4 * rows.dimension * rows.dimension)
}

/**
 * The gradient-reversal layer: $\xvec$ going forward, $-\lambda$ times the incoming gradient going back.
 *
 * @param x The input (any shape).
 * @param lambda The reversal's scale $\lambda$.
 * @returns A value equal to `x` whose derivative is $-\lambda$ times the identity.
 *
 * @example The value passes through; the gradient comes back scaled by minus a half
 * const x = tensor([1, 2])
 * print('forward:', gradientReversal(x, 0.5))
 * print('gradient of sum(w * R(x)), w = (1, 3):', grad((v) => sum(mul(gradientReversal(v, 0.5), tensor([1, 3]))))(x))
 */
export function gradientReversal(x: Value, lambda: number): Value {
  return sub(stopGradient(mul(1 + lambda, x)), mul(lambda, x))
}
