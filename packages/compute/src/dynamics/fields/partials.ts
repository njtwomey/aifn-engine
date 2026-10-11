/**
 * Partial derivatives of a pointwise field at a batch of points, by nested forward-mode autodiff: the derivatives a
 * physics-informed loss is written in ($u_t$, $u_x$, $u_{xx}$ at every collocation point).
 *
 * A pointwise field maps points $\Xmat$ ($n \times d$, one point a row) to values whose row $i$ depends on row $i$ of
 * $\Xmat$ only, as a network applied to a batch does. Its Jacobian is then block diagonal, so one Jacobian-vector
 * product with the direction $\evec_k$ in every row gives $\partial u / \partial x_k$ at all $n$ points at once, and
 * nesting the products gives the higher derivatives (Raissi, Perdikaris & Karniadakis, 2019, "Physics-informed neural
 * networks", J. Comput. Phys. 378, §2). The results are values like any other, so a loss built on them can be
 * differentiated again, by `grad`, with respect to the field's parameters.
 */

import { jvp } from 'aifn-compute/foundation/autodiff'
import { fromData, shapeOfValue, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A field evaluated row by row: called with points $\Xmat$ ($n \times d$), it returns values with $n$ rows (shape
 * $[n]$, $[n, 1]$ or $[n, m]$) whose row $i$ depends only on row $i$ of $\Xmat$. Written with the tensor primitives, so
 * that it can be differentiated.
 */
export type PointwiseField = (x: Value) => Value

/**
 * The direction $\evec_k$ in every row of an $n \times d$ batch.
 *
 * @param n The number of rows.
 * @param d The number of columns.
 * @param k The column set to 1.
 * @returns An $n \times d$ tensor, zero except column $k$.
 */
function direction(n: number, d: number, k: number): Value {
  const e = new Float64Array(n * d)
  for (let i = 0; i < n; i++) e[i * d + k] = 1
  return fromData(e, [n, d])
}

/**
 * Partial derivatives of a pointwise field $u$ at the points $\Xmat$, one for each multi-index in `orders`: `[]` is
 * $u$ itself, `[k]` is $\partial u / \partial x_k$, `[k, k]` is $\partial^2 u / \partial x_k^2$ and `[j, k]` the
 * mixed $\partial^2 u / \partial x_j \partial x_k$. Each derivative is a chain of forward-mode products, one per index,
 * so a derivative of order $p$ costs about $2^p$ evaluations of $u$. Correct only for a pointwise field: a field that
 * mixes rows (batch normalisation, attention across the batch) gives sums of derivatives instead. Differentiable, so a
 * loss built on the results can be differentiated again with respect to the field's parameters. Throws `ShapeError`
 * when $\Xmat$ is not a matrix and `DomainError` for an index outside $0, \dots, d - 1$.
 *
 * @param u The pointwise field, written with tensor primitives.
 * @param x The points $\Xmat$, $n \times d$, one a row.
 * @param orders The derivatives wanted, each a list of input columns to differentiate along, in any order.
 * @returns One value per entry of `orders`, each with the shape of $u(\Xmat)$.
 *
 * @example First and second derivatives of $u(x, t) = t \sin x$ at three points
 * // u_x = t cos x, u_xx = −t sin x, u_t = sin x and the mixed u_xt = cos x.
 * const u = (X) => mul(slice(X, null, [1, 2]), sin(slice(X, null, [0, 1])))
 * const X = tensor([[0, 1], [1, 2], [2, 0.5]])
 * const [ux, uxx, ut, uxt] = partials(u, X, [[0], [0, 0], [1], [0, 1]])
 * print('u_x  =', ux, ' t cos x =', mul(tensor([[1], [2], [0.5]]), cos(tensor([[0], [1], [2]]))))
 * print('u_xx =', uxx)
 * print('u_t  =', ut, ' u_xt =', uxt)
 *
 * @example A residual loss, differentiated with respect to the field's parameters
 * // u = a e^{−bt} sin x solves the heat equation u_t = u_xx when b = 1: the loss and its gradient vanish there.
 * const X = tensor([[0.3, 0], [1, 0.5], [2, 1]])
 * const loss = (theta) => {
 *   const u = (Y) => mul(mul(get(theta, 0), exp(mul(neg(get(theta, 1)), slice(Y, null, [1, 2])))), sin(slice(Y, null, [0, 1])))
 *   const [ut, uxx] = partials(u, X, [[1], [0, 0]])
 *   return mean(square(sub(ut, uxx)))
 * }
 * for (const b of [0.5, 1, 2]) print(`b = ${b}: loss`, loss(tensor([1, b])), ' gradient', grad(loss)(tensor([1, b])))
 */
export function partials(u: PointwiseField, x: Value, orders: readonly (readonly number[])[]): Value[] {
  const shape = shapeOfValue(x)
  if (shape.length !== 2) throw new ShapeError('partials', 'partials: the points must be a matrix, one point a row')
  const [n, d] = shape
  const directions = new Map<number, Value>()
  const along = (k: number): Value => {
    if (!Number.isInteger(k) || k < 0 || k >= d)
      throw new DomainError('partials', `partials: no input column ${k} in points of dimension ${d}`)
    if (!directions.has(k)) directions.set(k, direction(n, d, k))
    return directions.get(k)!
  }
  return orders.map((axes) => {
    // Innermost first: g ↦ (y ↦ ∂g/∂y_k at y), applied once per index.
    const derivative = axes.reduce<PointwiseField>((g, k) => {
      const e = along(k)
      return (y) => jvp(g, y, e).tangent as Value
    }, u)
    return derivative(x)
  })
}
