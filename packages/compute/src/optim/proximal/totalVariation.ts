/**
 * One-dimensional total variation denoising and the fused lasso signal approximator, exactly and in linear time:
 *
 * $$\hat\xvec = \argmin_{\xvec} \tfrac12 \lVert \yvec - \xvec \rVert^2 + \lambda \sum_{i=1}^{n-1} \lvert x_{i+1} -
 * x_i \rvert,$$
 *
 * the prox of $\lambda\lVert \Dmat\xvec \rVert_1$, $\Dmat$ the first-difference operator. The solution is piecewise
 * constant. It is computed by the direct algorithm of Condat (2013), "A direct algorithm for 1-D total variation
 * denoising", IEEE Signal Processing Letters 20(11), which scans the signal once, extending the current segment while
 * the dual variable $u_i$ of the taut-string condition stays within $[-\lambda, \lambda]$ and backtracking to the last
 * point where a jump is forced; it has worst-case $O(n^2)$ but runs in $O(n)$ on real signals. Adding an $\ell_1$
 * penalty gives the fused lasso signal approximator, whose solution is the soft thresholding of the total variation
 * solution (Friedman, Hastie, Höfling and Tibshirani, 2007, "Pathwise coordinate optimization", Ann. Appl. Statist.
 * 1(2), Proposition 1; Tibshirani, Saunders, Rosset, Zhu and Knight, 2005, "Sparsity and smoothness via the fused
 * lasso", JRSS B 67(1)).
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Prox } from './proximal'

/**
 * Condat's direct algorithm, writing the solution into `out`.
 *
 * @param y The signal, $n \ge 1$ values. Not modified.
 * @param lambda The weight $\lambda > 0$.
 * @param out The array the solution is written to, $n$ values; overwritten.
 */
function condat(y: ArrayLike<number>, lambda: number, out: Float64Array): void {
  const n = y.length
  let k = 0
  let k0 = 0
  let kplus = 0
  let kminus = 0
  // u is the dual variable; [vmin, vmax] bounds the value of the current segment.
  let umin = lambda
  let umax = -lambda
  let vmin = y[0] - lambda
  let vmax = y[0] + lambda
  for (;;) {
    while (k === n - 1) {
      // The right boundary: close the segment, with a jump if the bounds have crossed.
      if (umin < 0) {
        do {
          out[k0++] = vmin
        } while (k0 <= kminus)
        k = kminus = k0
        vmin = y[k]
        umin = lambda
        umax = vmin + umin - vmax
      } else if (umax > 0) {
        do {
          out[k0++] = vmax
        } while (k0 <= kplus)
        k = kplus = k0
        vmax = y[k]
        umax = -lambda
        umin = vmax + umax - vmin
      } else {
        vmin += umin / (k - k0 + 1)
        do {
          out[k0++] = vmin
        } while (k0 <= k)
        return
      }
    }
    umin += y[k + 1] - vmin
    if (umin < -lambda) {
      // A negative jump: the segment ends at the last point where vmin was set.
      do {
        out[k0++] = vmin
      } while (k0 <= kminus)
      k = kminus = kplus = k0
      vmin = y[k]
      vmax = vmin + 2 * lambda
      umin = lambda
      umax = -lambda
      continue
    }
    umax += y[k + 1] - vmax
    if (umax > lambda) {
      // A positive jump.
      do {
        out[k0++] = vmax
      } while (k0 <= kplus)
      k = kminus = kplus = k0
      vmax = y[k]
      vmin = vmax - 2 * lambda
      umin = lambda
      umax = -lambda
      continue
    }
    // No jump: extend the segment, tightening the bounds.
    k++
    if (umin >= lambda) {
      kminus = k
      vmin += (umin - lambda) / (kminus - k0 + 1)
      umin = lambda
    }
    if (umax <= -lambda) {
      kplus = k
      vmax += (umax + lambda) / (kplus - k0 + 1)
      umax = -lambda
    }
  }
}

/**
 * Check a non-negative finite weight.
 *
 * @param value The weight given.
 * @param name Its name in the message.
 * @param where The caller's name for error messages.
 */
function checkWeight(value: number, name: string, where: string): void {
  if (!(value >= 0 && Number.isFinite(value)))
    throw new DomainError(where, `${where}: ${name} must be finite and non-negative, got ${value}`)
}

/**
 * The total variation solution as an array.
 *
 * @param y The signal. Not modified.
 * @param lambda The weight $\lambda \ge 0$; 0 returns a copy of the signal.
 * @returns The solution, a new array of the signal's length.
 */
function denoise(y: Float64Array, lambda: number): dense.F64 {
  const out = new Float64Array(y.length)
  if (y.length === 0) return out
  if (lambda === 0) out.set(y)
  else condat(y, lambda, out)
  return out
}

/**
 * Exact one-dimensional total variation denoising: the $\xvec$ minimising
 * $\tfrac12 \lVert \yvec - \xvec \rVert^2 + \lambda \sum_i \lvert x_{i+1} - x_i \rvert$, by Condat's (2013) direct
 * algorithm. The solution is piecewise constant, with fewer pieces as $\lambda$ grows; it keeps the mean of $\yvec$,
 * and once $\lambda$ is large enough it is that mean everywhere. Throws `DomainError` for a negative or non-finite
 * $\lambda$.
 *
 * @param y The signal $\yvec$, $n$ values. Not modified.
 * @param lambda The weight $\lambda \ge 0$ of the total variation; 0 returns the signal.
 * @returns The denoised signal, $n$ values.
 *
 * @example A noisy step comes back as a step
 * const y = tensor([0.1, -0.2, 0.15, 0.05, 1.1, 0.9, 1.05, 0.95])
 * print('lambda 0.1 =', totalVariationDenoise1d(y, 0.1))
 * print('lambda 0.5 =', totalVariationDenoise1d(y, 0.5))
 * print('lambda 5 =', totalVariationDenoise1d(y, 5))
 */
export function totalVariationDenoise1d(y: VectorLike, lambda: number): Tensor {
  checkWeight(lambda, 'lambda', 'totalVariationDenoise1d')
  return dense.vec(denoise(dense.toF64(y, 'totalVariationDenoise1d'), lambda))
}

/** Options of `fusedLassoSignal`. */
export type FusedLassoSignalOptions = {
  /** The weight $\lambda_1 \ge 0$ of $\lVert \xvec \rVert_1$ (sparsity; default 0). */
  lambda1?: number
  /** The weight $\lambda_2 \ge 0$ of $\sum_i \lvert x_{i+1} - x_i \rvert$ (fusion; default 0). */
  lambda2?: number
}

/**
 * The fused lasso signal approximator: the $\xvec$ minimising
 * $\tfrac12 \lVert \yvec - \xvec \rVert^2 + \lambda_1 \lVert \xvec \rVert_1 + \lambda_2 \sum_i \lvert x_{i+1} - x_i
 * \rvert$, found exactly as the soft thresholding at $\lambda_1$ of the total variation solution at $\lambda_2$
 * (Friedman et al., 2007, Proposition 1). Throws `DomainError` for a negative or non-finite weight.
 *
 * @param y The signal $\yvec$, $n$ values. Not modified.
 * @param options The weights `lambda1` (sparsity) and `lambda2` (fusion), as `FusedLassoSignalOptions`.
 * @returns The fitted signal, $n$ values: piecewise constant, with the small pieces set to zero.
 *
 * @example Fusion makes pieces, the lasso part zeros the small ones
 * const y = tensor([0.1, -0.1, 0.2, 2.1, 1.9, 2, -0.05, 0.1])
 * print('fusion only =', fusedLassoSignal(y, { lambda2: 0.3 }))
 * print('fusion and sparsity =', fusedLassoSignal(y, { lambda1: 0.2, lambda2: 0.3 }))
 */
export function fusedLassoSignal(y: VectorLike, options: FusedLassoSignalOptions): Tensor {
  const { lambda1 = 0, lambda2 = 0 } = options
  checkWeight(lambda1, 'lambda1', 'fusedLassoSignal')
  checkWeight(lambda2, 'lambda2', 'fusedLassoSignal')
  const x = denoise(dense.toF64(y, 'fusedLassoSignal'), lambda2)
  return dense.vec(x.map((v) => Math.sign(v) * Math.max(Math.abs(v) - lambda1, 0)))
}

/**
 * $g(\xvec) = \lambda \sum_i \lvert x_{i+1} - x_i \rvert$, the one-dimensional total variation: its prox is
 * `totalVariationDenoise1d` at weight $\eta\lambda$. With it, proximal gradient solves total-variation regularised
 * problems, such as deblurring a piecewise-constant signal. Throws `DomainError` for a negative or non-finite
 * $\lambda$.
 *
 * @param lambda The weight $\lambda \ge 0$.
 * @returns The `Prox` of $g$.
 *
 * @example The prox at two steps, and the value
 * const g = proxTotalVariation1d(1)
 * const v = tensor([0, 0.2, 1.5, 1.3])
 * print('prox, step 0.1 =', g.prox(v, 0.1))
 * print('prox, step 1 =', g.prox(v, 1))
 * print('g(v) =', g.value(v))
 */
export function proxTotalVariation1d(lambda: number): Prox {
  checkWeight(lambda, 'lambda', 'proxTotalVariation1d')
  return {
    name: 'total-variation',
    value: (x: Vector) => {
      const a = dense.toF64(x, 'proxTotalVariation1d')
      let s = 0
      for (let i = 1; i < a.length; i++) s += Math.abs(a[i] - a[i - 1])
      return lambda * s
    },
    prox: (v: Vector, eta: number) => dense.vec(denoise(dense.toF64(v, 'proxTotalVariation1d'), eta * lambda)),
  }
}
