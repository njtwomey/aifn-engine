/**
 * Linear systems $\xvec' = \Amat\xvec$: the flow $\xvec(t) = e^{\Amat t} \xvec_0$, from the matrix exponential of
 * `aifn-compute/numerics/linalg` (Moler & Van Loan, 2003, "Nineteen dubious ways to compute the exponential of a
 * matrix, twenty-five years later", SIAM Review 45).
 */

import { expm } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The flow of the linear system $\xvec' = \Amat\xvec$: $\xvec(t) = e^{\Amat t} \xvec_0$ at each of the given times,
 * exactly (to the accuracy of `expm`) rather than by stepping. Each time is an independent `expm`, so the times need
 * not be evenly spaced, sorted or positive. Throws `ShapeError` when `x0` does not match $\Amat$.
 *
 * @param a The $n \times n$ matrix $\Amat$ of the system.
 * @param x0 The initial state $\xvec_0 = \xvec(0)$, of length $n$.
 * @param times The times $t_k$ at which to evaluate the flow, measured from the initial state at $t = 0$.
 * @returns A matrix with one row per time: row $k$ is $\xvec(t_k)$ (shape $m \times n$ for $m$ times).
 *
 * @example Exponential decay matches the closed form
 * // x′ = −x from x(0) = 1: x(t) = e^{−t}.
 * print('x(t) =', linearFlow([[-1]], [1], [0, 1, 2]))
 * print('e^{-t} =', [0, 1, 2].map((t) => Math.exp(-t)))
 *
 * @example A harmonic oscillator turns in a circle
 * // q′ = p, p′ = −q from (1, 0): (cos t, −sin t), back to the start after 2π.
 * print('x(t) =', linearFlow([[0, 1], [-1, 0]], [1, 0], [0, Math.PI / 2, Math.PI, 2 * Math.PI]))
 */
export function linearFlow(a: MatrixLike, x0: VectorLike, times: VectorLike): Matrix {
  const { data: A, n } = dense.toMatrixF64(a, 'linearFlow')
  const x = dense.toF64(x0, 'linearFlow')
  if (x.length !== n) throw new ShapeError('linearFlow', `linearFlow: x0 has length ${x.length}, A is ${n}×${n}`)
  const ts = dense.toF64(times, 'linearFlow')
  const out = new Float64Array(ts.length * n)
  ts.forEach((t, k) => {
    const At = fromData(
      A.map((v) => v * t),
      [n, n],
    )
    out.set(dense.matVec(toFlat(expm(At).value), x, n, n), k * n)
  })
  return fromData(out, [ts.length, n])
}
