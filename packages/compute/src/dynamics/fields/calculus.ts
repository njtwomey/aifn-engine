/**
 * Differential operators on vector fields by automatic differentiation (`aifn-compute/foundation/autodiff`): the
 * Jacobian, divergence and curl; and the fields built from scalar functions: gradient flows and Hamiltonian fields
 * (Marsden & Tromba, 2012, "Vector Calculus", 6th ed., §3–4; Arnold, 1989, "Mathematical Methods of Classical
 * Mechanics", §15).
 *
 * A field is a function of a rank-1 tensor $\xvec$, written with the tensor primitives so that it can be
 * differentiated; the operators evaluate its derivatives at one point and return numbers, while `gradientField` and
 * `hamiltonianField` return new fields that stay differentiable.
 */

import { grad, jacobian } from 'aifn-compute/foundation/autodiff'
import {
  concat,
  fromData,
  neg,
  shapeOfValue,
  slice,
  toFlat,
  dense,
  type Matrix,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A vector field $\fvec : \reals^n \to \reals^n$, called with the point $\xvec$ as a rank-1 tensor and returning a
 * vector of $n$ values. Written with `aifn-compute/foundation/tensor` primitives, it can be differentiated by
 * `aifn-compute/foundation/autodiff`.
 */
export type VectorField = (x: Tensor) => VectorLike | Value

/**
 * A scalar field $g : \reals^n \to \reals$, called with the point $\xvec$ as a rank-1 tensor and returning a number
 * or a rank-0 tensor.
 */
export type ScalarField = (x: Tensor) => Value

/**
 * A point as a new float64 rank-1 tensor, the form the fields are called with.
 *
 * @param x The point's coordinates.
 * @returns A copy of `x` as a vector of its length.
 */
const asInput = (x: VectorLike): Tensor => {
  const v = dense.toF64(x, 'fields')
  return fromData(v, [v.length])
}

/**
 * The Jacobian $J_{ij} = \partial f_i / \partial x_j$ of a vector field at $\xvec$ ($n \times n$), by reverse-mode
 * autodiff (one pass per output). A field whose output length differs from its input's throws `DomainError`.
 *
 * @param f The vector field, written with tensor primitives so that it can be differentiated.
 * @param x The point at which the Jacobian is taken ($n$ values).
 * @returns The $n \times n$ Jacobian, row $i$ the gradient of $f_i$.
 *
 * @example The Jacobian of a polynomial field
 * // f(x, y) = (x², xy): J = [[2x, 0], [y, x]], at (1, 2) [[2, 0], [2, 1]].
 * const f = (x) => stack([mul(get(x, 0), get(x, 0)), mul(get(x, 0), get(x, 1))])
 * print('J =', jacobianAt(f, [1, 2]))
 */
export function jacobianAt(f: VectorField, x: VectorLike): Matrix {
  const J = jacobian((y: Value) => f(y as Tensor) as Value)(asInput(x)) as Tensor
  const n = J.shape[0]
  if (J.shape.length !== 2 || J.shape[1] !== n)
    throw new DomainError('jacobianAt', 'jacobianAt: the field must map ℝⁿ to ℝⁿ')
  return fromData(Float64Array.from(toFlat(J)), [n, n])
}

/**
 * The divergence $\nabla \cdot \fvec = \sum_i \partial f_i / \partial x_i$ at $\xvec$, the trace of the Jacobian: the
 * rate at which the flow expands volume there (Liouville).
 *
 * @param f The vector field, written with tensor primitives so that it can be differentiated.
 * @param x The point at which the divergence is taken ($n$ values).
 * @returns The divergence, a number.
 *
 * @example Expanding, contracting and area-preserving fields
 * // f(x, y) = (x², xy) has divergence 2x + x = 3 at x = 1; −x contracts at rate −2; a rotation keeps area.
 * const f = (x) => stack([mul(get(x, 0), get(x, 0)), mul(get(x, 0), get(x, 1))])
 * print('(x², xy) at (1, 2):', divergence(f, [1, 2]))
 * print('−x:', divergence((x) => neg(x), [1, 2]))
 * print('rotation:', divergence((x) => stack([neg(get(x, 1)), get(x, 0)]), [1, 2]))
 */
export function divergence(f: VectorField, x: VectorLike): Scalar {
  const J = toFlat(jacobianAt(f, x))
  const n = Math.round(Math.sqrt(J.length))
  let s = 0
  for (let i = 0; i < n; i++) s += J[i * n + i]
  return s
}

/**
 * The curl of a field at $\xvec$: in two dimensions the scalar $\partial f_2/\partial x - \partial f_1/\partial y$
 * (twice the local angular velocity of the flow); in three, the vector of components
 * $\partial f_3/\partial y - \partial f_2/\partial z$, $\partial f_1/\partial z - \partial f_3/\partial x$ and
 * $\partial f_2/\partial x - \partial f_1/\partial y$, as a length-3 tensor. Any other dimension throws `DomainError`.
 *
 * @param f The vector field on $\reals^2$ or $\reals^3$, written with tensor primitives so that it can be
 *   differentiated.
 * @param x The point at which the curl is taken (2 or 3 values).
 * @returns A number in two dimensions, a vector of 3 in three.
 *
 * @example A rigid rotation has curl 2
 * // f(x, y) = (−y, x) turns at angular velocity 1; in 3D about the z axis the curl is (0, 0, 2).
 * print('2D:', curl((x) => stack([neg(get(x, 1)), get(x, 0)]), [0.3, 0.7]))
 * print('3D:', curl((x) => stack([neg(get(x, 1)), get(x, 0), mul(0, get(x, 2))]), [0.3, 0.7, 1]))
 * print('a gradient field:', curl((x) => stack([get(x, 1), get(x, 0)]), [0.3, 0.7]))
 */
export function curl(f: VectorField, x: VectorLike): Scalar | Tensor {
  const J = toFlat(jacobianAt(f, x))
  if (J.length === 4) return J[2] - J[1]
  if (J.length === 9) {
    const d = (i: number, j: number) => J[i * 3 + j]
    return fromData(Float64Array.of(d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)), [3])
  }
  throw new DomainError('curl', 'curl: defined for fields on ℝ² and ℝ³')
}

/**
 * The gradient $\nabla V(\xvec)$ of a scalar field, by reverse-mode autodiff.
 *
 * @param V The scalar field, written with tensor primitives so that it can be differentiated.
 * @param x The point at which the gradient is taken ($n$ values).
 * @returns The gradient, a vector of $n$.
 *
 * @example The gradient of a quadratic
 * // V(x, y) = x² + 3y²: ∇V = (2x, 6y), at (1, 1) (2, 6).
 * const V = (x) => add(square(get(x, 0)), mul(3, square(get(x, 1))))
 * print('grad V =', gradientAt(V, [1, 1]))
 */
export function gradientAt(V: ScalarField, x: VectorLike): Tensor {
  return grad((y: Value) => V(y as Tensor))(asInput(x)) as Tensor
}

/**
 * The gradient flow of a potential: $\xvec' = -\nabla V(\xvec)$ (or $+\nabla V$ with `ascent: true`). $V$ decreases
 * along every trajectory, its fixed points are $V$'s critical points, and the flow has zero curl. Traceable: the
 * returned field differentiates $V$ again under `aifn-compute/foundation/autodiff` (so its Jacobian is $-\nabla^2 V$).
 *
 * @param V The potential, written with tensor primitives so that it can be differentiated.
 * @param options Which way the flow runs.
 * @param options.ascent True for the ascent flow $+\nabla V$; false (default) for descent, $-\nabla V$.
 * @returns The vector field $\pm\nabla V$.
 *
 * @example Descent on a quadratic bowl, and its Jacobian
 * // V = (x² + 3y²) / 2: the descent field is (−x, −3y), and its Jacobian −∇²V = diag(−1, −3).
 * const V = (x) => mul(0.5, add(square(get(x, 0)), mul(3, square(get(x, 1)))))
 * const f = gradientField(V)
 * print('f(1, 1) =', f(tensor([1, 1])))
 * print('ascent f(1, 1) =', gradientField(V, { ascent: true })(tensor([1, 1])))
 * print('Jacobian =', jacobianAt(f, [1, 1]))
 */
export function gradientField(V: ScalarField, { ascent = false }: { ascent?: boolean } = {}): VectorField {
  const g = grad((y: Value) => V(y as Tensor))
  return (x) => (ascent ? g(x) : neg(g(x) as Value)) as Tensor
}

/**
 * The Hamiltonian field of $H(\qvec, \pvec)$ on the phase space $\xvec = (\qvec, \pvec)$ of dimension $2d$:
 * $\qvec' = \partial H/\partial \pvec$, $\pvec' = -\partial H/\partial \qvec$. $H$ is constant along its trajectories
 * and the flow preserves volume (zero divergence). Traceable, as `gradientField`. The field throws `ShapeError` when
 * called on a point of odd dimension.
 *
 * @param H The Hamiltonian, a scalar field on $\reals^{2d}$ whose first $d$ coordinates are the positions $\qvec$ and
 *   last $d$ the momenta $\pvec$; written with tensor primitives so that it can be differentiated.
 * @returns The vector field $(\partial H/\partial \pvec, -\partial H/\partial \qvec)$.
 *
 * @example The pendulum
 * // H = p²/2 − cos q: q′ = p, p′ = −sin q. The flow keeps H and area.
 * const f = hamiltonianField((x) => sub(mul(0.5, square(get(x, 1))), cos(get(x, 0))))
 * print('f(π/2, 1) =', f(tensor([Math.PI / 2, 1])))
 * print('divergence =', divergence(f, [0.4, 0.3]))
 * const end = flowMap(f, [1, 0], 5)
 * print('H before =', -Math.cos(1), ' H after =', 0.5 * toFlat(end)[1] ** 2 - Math.cos(toFlat(end)[0]))
 */
export function hamiltonianField(H: ScalarField): VectorField {
  const g = grad((y: Value) => H(y as Tensor))
  return (x) => {
    const dH = g(x) as Value
    const n = shapeOfValue(dH)[0]
    const d = n / 2
    if (!Number.isInteger(d))
      throw new ShapeError('hamiltonianField', 'hamiltonianField: the phase space must have even dimension')
    return concat([slice(dH, [d, n]) as Value, neg(slice(dH, [0, d]) as Value)]) as Tensor
  }
}
