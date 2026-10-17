/**
 * Chebyshev series on an interval $[a, b]$: interpolation at the Chebyshev nodes of the first kind, and evaluation,
 * as numpy's `chebinterpolate` and `chebval` with the interval mapped to $[-1, 1]$ by
 * $t = 2(x - a)/(b - a) - 1$.
 *
 * A series of degree $K$ is $p(x) = \sum_{k=0}^{K} c_k T_k(t)$, with $T_k$ the Chebyshev polynomials of the first
 * kind ($T_0 = 1$, $T_1 = t$, $T_{k+1} = 2tT_k - T_{k-1}$), evaluated by that recurrence. The interpolating
 * coefficients of $f$ come from its values at the $K + 1$ nodes $t_j = \cos(\pi(j + \tfrac12)/(K + 1))$ by a discrete
 * cosine transform,
 *
 * $$c_k = \frac{2 - [k = 0]}{K + 1} \sum_{j=0}^{K} f(x_j) \cos\frac{\pi k (j + \tfrac12)}{K + 1},$$
 *
 * so the series equals $f$ at the nodes, and equals $f$ everywhere when $f$ is a polynomial of degree at most $K$
 * (Trefethen 2019, "Approximation Theory and Approximation Practice", ch. 3 and 4). He, Wei and Wen (2022),
 * "Convolutional neural networks on graphs with Chebyshev approximation, revisited", NeurIPS, fit spectral graph
 * filters this way. The nodes here run from $b$ down to $a$; `chebyshevNodes` gives the same points ascending.
 */

import type { Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * Check an interval and return its ends: throws `DomainError` unless $a < b$, both finite.
 *
 * @param domain The interval $[a, b]$.
 * @param where The caller's name, for error messages.
 * @returns The ends $a$ and $b$.
 */
function interval(domain: readonly [number, number], where: string): [number, number] {
  const [a, b] = domain
  if (!(Number.isFinite(a) && Number.isFinite(b) && a < b))
    throw new DomainError(where, `${where}: the domain [${a}, ${b}] must be an interval with a < b`)
  return [a, b]
}

/**
 * The coefficients $c_0, \dots, c_K$ of the degree-$K$ Chebyshev series that interpolates $f$ at the $K + 1$ Chebyshev
 * nodes of the first kind on $[a, b]$ (see the file comment), as numpy's `chebinterpolate` on $[-1, 1]$. Throws
 * `DomainError` for a negative or fractional degree or an empty interval.
 *
 * @param f The function to interpolate, called once at each node.
 * @param order The degree $K$.
 * @param domain The interval $[a, b]$ (default $[-1, 1]$).
 * @returns The $K + 1$ coefficients as a float64 vector, $c_0$ first.
 *
 * @example A quadratic is reproduced exactly by $K = 2$
 * const f = (x) => 1 - 0.5 * x + 0.1 * x * x
 * const c = chebyshevInterpolation(f, 2, [0, 4])
 * print('coefficients:', c)
 * print('at 0, 1.7, 4:', chebyshevSeries(c, tensor([0, 1.7, 4]), [0, 4]), ' f:', [f(0), f(1.7), f(4)])
 *
 * @example A low-pass response on the spectrum $[0, 2]$ of a normalised Laplacian
 * const c = chebyshevInterpolation((l) => Math.exp(-3 * l), 4, [0, 2])
 * print('degree 4 at 0.5:', chebyshevSeries(c, 0.5, [0, 2]), ' exact:', Math.exp(-1.5))
 */
export function chebyshevInterpolation(
  f: (x: number) => number,
  order: number,
  domain: readonly [number, number] = [-1, 1],
): Tensor {
  if (!(Number.isInteger(order) && order >= 0))
    throw new DomainError('chebyshevInterpolation', `chebyshevInterpolation: order = ${order} must be a count`)
  const [a, b] = interval(domain, 'chebyshevInterpolation')
  const m = order + 1
  // Written as a + (b - a)/2 (cos + 1) so that on [0, b] the nodes are exactly b/2 (cos + 1).
  const node = (j: number) => a + ((b - a) / 2) * (Math.cos((Math.PI * (j + 0.5)) / m) + 1)
  const values = Array.from({ length: m }, (_, j) => f(node(j)))
  const out = new Float64Array(m)
  for (let k = 0; k < m; k++) {
    let s = 0
    for (let j = 0; j < m; j++) s += values[j] * Math.cos((Math.PI * k * (j + 0.5)) / m)
    out[k] = ((k === 0 ? 1 : 2) / m) * s
  }
  return fromData(out, [m])
}

/**
 * The Chebyshev series $\sum_k c_k T_k(t)$ at $x$, with $t = 2(x - a)/(b - a) - 1$, by the three-term recurrence, as
 * numpy's `chebval` on $[-1, 1]$. Points outside $[a, b]$ are evaluated too (the series extrapolates). Throws
 * `DomainError` for no coefficients or an empty interval.
 *
 * @param coefficients The coefficients $c_0, \dots, c_K$, $c_0$ first (a vector or an array).
 * @param x Where to evaluate: a number, or a vector of points.
 * @param domain The interval $[a, b]$ mapped to $[-1, 1]$ (default $[-1, 1]$).
 * @returns The value at `x`: a number for a number, else a float64 tensor of the shape of `x`.
 *
 * @example $T_2(t) = 2t^2 - 1$ at a few points
 * print(chebyshevSeries([0, 0, 1], tensor([-1, 0, 0.5, 1])))
 */
export function chebyshevSeries(coefficients: VectorLike, x: Scalar, domain?: readonly [number, number]): Scalar
export function chebyshevSeries(
  coefficients: VectorLike,
  x: Tensor | readonly number[],
  domain?: readonly [number, number],
): Tensor
export function chebyshevSeries(
  coefficients: VectorLike,
  x: Scalar | Tensor | readonly number[],
  domain?: readonly [number, number],
): Scalar | Tensor
export function chebyshevSeries(
  coefficients: VectorLike,
  x: Scalar | Tensor | readonly number[],
  domain: readonly [number, number] = [-1, 1],
): Scalar | Tensor {
  const c = dense.toF64(coefficients, 'chebyshevSeries')
  if (c.length === 0) throw new DomainError('chebyshevSeries', 'chebyshevSeries: no coefficients')
  const [a, b] = interval(domain, 'chebyshevSeries')
  const at = (v: number) => {
    const t = (2 * (v - a)) / (b - a) - 1
    let [prev, cur] = [1, t]
    let s = c[0]
    for (let k = 1; k < c.length; k++) {
      if (k >= 2) [prev, cur] = [cur, 2 * t * cur - prev]
      s += c[k] * cur
    }
    return s
  }
  if (typeof x === 'number') return at(x)
  const values = isTensor(x) ? toFlat(x) : x
  const out = Float64Array.from(values, at)
  return fromData(out, isTensor(x) ? [...x.shape] : [out.length])
}
