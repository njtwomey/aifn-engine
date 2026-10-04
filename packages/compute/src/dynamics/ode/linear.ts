/**
 * Linear systems x′ = Ax: the flow x(t) = e^{At} x₀, from the matrix exponential of `aifn-compute/numerics/linalg`
 * (Moler & Van Loan, 2003, "Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later",
 * SIAM Review 45).
 */

import { expm } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The flow of the linear system x′ = Ax: x(t) = e^{At} x₀ at each of the given times, as a [times, n] matrix. Each
 * time is an independent `expm`, so the times need not be evenly spaced.
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
