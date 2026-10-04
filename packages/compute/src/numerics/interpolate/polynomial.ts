/**
 * Polynomial interpolation: Newton's divided differences, the barycentric form (Berrut and Trefethen, 2004,
 * "Barycentric Lagrange interpolation", SIAM Review 46(3)), Chebyshev nodes and the Lebesgue function that measures
 * how much a node set can amplify errors (Runge's phenomenon on equispaced nodes).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const f64 = (t: Tensor) => Float64Array.from(toFlat(t))

/** An interpolating polynomial through n points (degree ≤ n − 1). */
export type InterpolatingPolynomial = {
  /** The nodes xⱼ. */
  readonly nodes: Tensor
  /** Newton coefficients aₖ = f[x₀, …, xₖ] (divided differences). */
  readonly newton: Tensor
  /** Barycentric weights wⱼ = 1/Πₖ≠ⱼ(xⱼ − xₖ). */
  readonly weights: Tensor
  /** p(t) at t [m] (barycentric formula, exact at the nodes), or its first or second derivative (Newton form). */
  evaluate(t: Tensor, derivative?: 0 | 1 | 2): Tensor
}

/** The polynomial of degree ≤ n − 1 through (xⱼ, yⱼ), distinct nodes. */
export function interpolatingPolynomial(x: Tensor, y: Tensor): InterpolatingPolynomial {
  const X = f64(x)
  const Y = f64(y)
  const n = X.length
  if (Y.length !== n)
    throw new ShapeError('interpolatingPolynomial', `interpolatingPolynomial: ${n} nodes but ${Y.length} values`)
  const a = Float64Array.from(Y)
  for (let j = 1; j < n; j++)
    for (let i = n - 1; i >= j; i--) {
      const h = X[i] - X[i - j]
      if (h === 0) throw new DomainError('interpolatingPolynomial', 'interpolatingPolynomial: nodes must be distinct')
      a[i] = (a[i] - a[i - 1]) / h
    }
  const w = Float64Array.from(X, (xj, j) => {
    let p = 1
    for (let k = 0; k < n; k++) if (k !== j) p *= xj - X[k]
    return 1 / p
  })
  const evaluate = (t: Tensor, derivative: 0 | 1 | 2 = 0) => {
    const ts = f64(t)
    const out = Float64Array.from(ts, (u) => {
      if (derivative === 0) {
        let num = 0
        let den = 0
        for (let j = 0; j < n; j++) {
          const d = u - X[j]
          if (d === 0) return Y[j]
          num += (w[j] * Y[j]) / d
          den += w[j] / d
        }
        return num / den
      }
      // Nested multiplication on the Newton form with running derivatives.
      let p = a[n - 1]
      let dp = 0
      let d2p = 0
      for (let k = n - 2; k >= 0; k--) {
        const v = u - X[k]
        d2p = d2p * v + 2 * dp
        dp = dp * v + p
        p = p * v + a[k]
      }
      return derivative === 1 ? dp : d2p
    })
    return fromData(out, t.shape)
  }
  return { nodes: fromData(X, [n]), newton: fromData(a, [n]), weights: fromData(w, [n]), evaluate }
}

/** n Chebyshev points of the first kind on [a, b], xⱼ = (a + b)/2 + (b − a)/2 · cos((2j + 1)π/(2n)), ascending. */
export function chebyshevNodes(n: Size, a: Scalar = -1, b: Scalar = 1): Tensor {
  const out = Float64Array.from(
    { length: n },
    (_, j) => (a + b) / 2 - ((b - a) / 2) * Math.cos(((2 * j + 1) * Math.PI) / (2 * n)),
  )
  return fromData(out, [n])
}

/**
 * The Lebesgue function Λ(t) = Σⱼ |ℓⱼ(t)| of a node set at t [m]; its maximum, the Lebesgue constant, bounds how much
 * worse interpolation is than the best polynomial approximation of the same degree.
 */
export function lebesgueFunction(nodes: Tensor, t: Tensor): Tensor {
  const X = f64(nodes)
  const n = X.length
  const w = Float64Array.from(X, (xj, j) => {
    let p = 1
    for (let k = 0; k < n; k++) if (k !== j) p *= xj - X[k]
    return 1 / p
  })
  const out = Float64Array.from(f64(t), (u) => {
    let num = 0
    let den = 0
    for (let j = 0; j < n; j++) {
      const d = u - X[j]
      if (d === 0) return 1
      num += Math.abs(w[j] / d)
      den += w[j] / d
    }
    return num / Math.abs(den)
  })
  return fromData(out, t.shape)
}
