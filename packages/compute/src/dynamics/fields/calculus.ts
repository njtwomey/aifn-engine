/**
 * Differential operators on vector fields by automatic differentiation (`aifn-compute/foundation/autodiff`): the Jacobian,
 * divergence and curl; and the fields built from scalar functions: gradient flows and Hamiltonian fields (Marsden &
 * Tromba, 2012, "Vector Calculus", 6th ed., §3–4; Arnold, 1989, "Mathematical Methods of Classical Mechanics", §15).
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
 * A vector field f: ℝⁿ → ℝⁿ. Written with `aifn-compute/foundation/tensor` primitives, it can be differentiated by
 * `aifn-compute/foundation/autodiff`.
 */
export type VectorField = (x: Tensor) => VectorLike | Value

/** A scalar field g: ℝⁿ → ℝ (a number or a rank-0 tensor). */
export type ScalarField = (x: Tensor) => Value

const asInput = (x: VectorLike): Tensor => {
  const v = dense.toF64(x, 'fields')
  return fromData(v, [v.length])
}

/** The Jacobian ∂f_i/∂x_j of a vector field at x (n × n), by reverse-mode autodiff (one pass per output). */
export function jacobianAt(f: VectorField, x: VectorLike): Matrix {
  const J = jacobian((y: Value) => f(y as Tensor) as Value)(asInput(x)) as Tensor
  const n = J.shape[0]
  if (J.shape.length !== 2 || J.shape[1] !== n)
    throw new DomainError('jacobianAt', 'jacobianAt: the field must map ℝⁿ to ℝⁿ')
  return fromData(Float64Array.from(toFlat(J)), [n, n])
}

/** The divergence ∇·f = Σᵢ ∂fᵢ/∂xᵢ at x: the rate at which the flow expands volume there (Liouville). */
export function divergence(f: VectorField, x: VectorLike): Scalar {
  const J = toFlat(jacobianAt(f, x))
  const n = Math.round(Math.sqrt(J.length))
  let s = 0
  for (let i = 0; i < n; i++) s += J[i * n + i]
  return s
}

/**
 * The curl of a field at x: in two dimensions the scalar ∂f₂/∂x − ∂f₁/∂y (twice the local angular velocity of the
 * flow); in three, the vector (∂f₃/∂y − ∂f₂/∂z, ∂f₁/∂z − ∂f₃/∂x, ∂f₂/∂x − ∂f₁/∂y) as a length-3 array.
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

/** The gradient ∇V(x) of a scalar field (length n). */
export function gradientAt(V: ScalarField, x: VectorLike): Tensor {
  return grad((y: Value) => V(y as Tensor))(asInput(x)) as Tensor
}

/**
 * The gradient flow of a potential: x′ = −∇V(x) (or +∇V with `ascent: true`). V decreases along every trajectory, its
 * fixed points are V's critical points, and the flow has zero curl. Traceable: the returned field differentiates V
 * again under `aifn-compute/foundation/autodiff` (so its Jacobian is −∇²V).
 */
export function gradientField(V: ScalarField, { ascent = false }: { ascent?: boolean } = {}): VectorField {
  const g = grad((y: Value) => V(y as Tensor))
  return (x) => (ascent ? g(x) : neg(g(x) as Value)) as Tensor
}

/**
 * The Hamiltonian field of H(q, p) on the phase space x = (q, p) of dimension 2d: q′ = ∂H/∂p, p′ = −∂H/∂q. H is
 * constant along its trajectories and the flow preserves area (zero divergence). Traceable, as `gradientField`.
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
