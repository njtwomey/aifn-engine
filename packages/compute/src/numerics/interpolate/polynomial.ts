/**
 * Polynomial interpolation: Newton's divided differences, the barycentric form (Berrut and Trefethen, 2004,
 * "Barycentric Lagrange interpolation", SIAM Review 46(3)), Chebyshev nodes and the Lebesgue function that measures
 * how much a node set can amplify errors (Runge's phenomenon on equispaced nodes).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * Extract a flat Float64Array view from a tensor.
 *
 * @param t Input tensor.
 * @returns Flattened 64-bit float array.
 */
const f64 = (t: Tensor) => Float64Array.from(toFlat(t))

/** An interpolating polynomial through $n$ points (degree $\le n - 1$). */
export type InterpolatingPolynomial = {
  /** The interpolation nodes $x_j$. */
  readonly nodes: Tensor
  /** Newton divided difference coefficients $a_k = f[x_0, \dots, x_k]$. */
  readonly newton: Tensor
  /** Barycentric weights $w_j = 1 / \prod_{k \ne j}(x_j - x_k)$. */
  readonly weights: Tensor
  /**
   * Evaluate $p(t)$ at coordinates $t$, or its first or second derivative.
   *
   * @param t Target evaluation points tensor.
   * @param derivative Derivative order: 0 for function value, 1 for $p'(t)$, 2 for $p''(t)$.
   * @returns Evaluated polynomial or derivative tensor matching the shape of $t$.
   */
  evaluate(t: Tensor, derivative?: 0 | 1 | 2): Tensor
}

/**
 * Construct the unique interpolating polynomial of degree $\le n - 1$ through distinct points $(x_j, y_j)$.
 *
 * Evaluates in $\mathcal{O}(n)$ time per query point via the second barycentric formula
 * (Berrut & Trefethen, 2004), which is numerically stable and exact at the interpolation nodes.
 * First and second derivatives are evaluated via the Newton divided difference form.
 *
 * @param x Distinct interpolation node coordinates vector of length $n$.
 * @param y Function values vector of length $n$ at the corresponding nodes.
 * @returns An `InterpolatingPolynomial` object supporting evaluation and differentiation.
 *
 * @example Interpolate quadratic polynomial
 * const poly = interpolatingPolynomial(tensor([0, 1, 2]), tensor([0, 1, 4]))
 * const val = poly.evaluate(tensor([1.5]))
 * print('p(1.5) =', val)
 */
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

/**
 * Generate $n$ Chebyshev points of the first kind on interval $[a, b]$ in ascending order.
 *
 * Coordinates are defined by $x_j = \frac{a + b}{2} - \frac{b - a}{2} \cos\left(\frac{(2j + 1)\pi}{2n}\right)$
 * for $j = 0, \dots, n - 1$. Chebyshev nodes minimise Runge's phenomenon and keep the Lebesgue
 * constant growing only logarithmically $\mathcal{O}(\log n)$.
 *
 * @param n Number of Chebyshev nodes to generate ($n \ge 1$).
 * @param a Left interval endpoint (default -1).
 * @param b Right interval endpoint (default 1).
 * @returns 1D tensor of length $n$ containing Chebyshev nodes in $[a, b]$.
 *
 * @example Generate Chebyshev nodes on interval
 * const nodes = chebyshevNodes(5, -1, 1)
 * print('nodes count =', nodes.shape[0])
 */
export function chebyshevNodes(n: Size, a: Scalar = -1, b: Scalar = 1): Tensor {
  const out = Float64Array.from(
    { length: n },
    (_, j) => (a + b) / 2 - ((b - a) / 2) * Math.cos(((2 * j + 1) * Math.PI) / (2 * n)),
  )
  return fromData(out, [n])
}

/**
 * Evaluate the Lebesgue function $\Lambda(t) = \sum_j |\ell_j(t)|$ for a given node set at points $t$.
 *
 * The maximum of $\Lambda(t)$ is the Lebesgue constant $\Lambda_n$, which bounds the interpolation error
 * relative to the best polynomial approximation: $\|f - p_n\|_\infty \le (1 + \Lambda_n)\|f - p_n^*\|_\infty$.
 *
 * @param nodes 1D tensor of interpolation node coordinates.
 * @param t Target evaluation points tensor.
 * @returns Tensor of Lebesgue function values matching the shape of $t$.
 *
 * @example Evaluate Lebesgue function
 * const nodes = chebyshevNodes(4, -1, 1)
 * const lambda = lebesgueFunction(nodes, tensor([0]))
 * print('lebesgue at 0 =', lambda)
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
