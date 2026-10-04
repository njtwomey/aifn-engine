/**
 * Newton–Cotes rules on equal panels (composite trapezoid and Simpson), the trapezoid rule on samples, and Romberg
 * integration (Richardson extrapolation of the trapezoid rule) as a traceable algorithm (Davis & Rabinowitz, 1984,
 * "Methods of Numerical Integration", 2nd ed., §2.1–2.4 and §6.3).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/** A real function of one real variable. */
export type Integrand = (x: Scalar) => Scalar

/**
 * $\int_a^b f(x)\,dx$ by the composite trapezoid rule on $n$ equal panels (default 100):
 * $h[f(a)/2 + f(x_1) + \dots + f(b)/2]$, error $-(b - a)h^2 f''(\xi)/12$.
 *
 * @param f The univariate real integrand function.
 * @param a The lower integration bound.
 * @param b The upper integration bound.
 * @param options Options specifying panel count.
 * @param options.n Number of equal panels (default 100).
 * @returns The approximated integral value.
 *
 * @example Integrate x^2 from 0 to 1
 * print('integral =', trapezoid((x) => x * x, 0, 1, { n: 100 }))
 */
export function trapezoid(f: Integrand, a: Scalar, b: Scalar, { n = 100 }: { n?: Size } = {}): Scalar {
  if (!(n >= 1)) throw new DomainError('trapezoid', 'trapezoid: n must be at least 1')
  const h = (b - a) / n
  let s = 0.5 * (f(a) + f(b))
  for (let i = 1; i < n; i++) s += f(a + i * h)
  return h * s
}

/**
 * $\int_a^b f(x)\,dx$ by the composite Simpson rule on $n$ equal panels ($n$ even, default 100):
 * $(h/3)[f_0 + 4f_1 + 2f_2 + \dots + f_n]$, error $-(b - a)h^4 f^{(4)}(\xi)/180$, exact for cubics.
 *
 * @param f The univariate real integrand function.
 * @param a The lower integration bound.
 * @param b The upper integration bound.
 * @param options Options specifying panel count.
 * @param options.n Number of equal panels (must be even, default 100).
 * @returns The approximated integral value.
 *
 * @example Integrate a cubic polynomial exactly
 * print('integral =', simpson((x) => x * x * x, 0, 1, { n: 2 }))
 */
export function simpson(f: Integrand, a: Scalar, b: Scalar, { n = 100 }: { n?: Size } = {}): Scalar {
  if (!(n >= 2) || n % 2 !== 0) throw new DomainError('simpson', `simpson: n must be even and at least 2, got ${n}`)
  const h = (b - a) / n
  let s = f(a) + f(b)
  for (let i = 1; i < n; i++) s += (i % 2 === 1 ? 4 : 2) * f(a + i * h)
  return (h / 3) * s
}

/**
 * The trapezoid rule on samples $y$ at points $x$ (a tensor or array of the same length; default spacing `dx` = 1), as
 * numpy's `trapezoid(y, x)`: $\sum (x_{i+1} - x_i)(y_i + y_{i+1})/2$.
 *
 * @param y Sampled function values along the integration interval.
 * @param x Optional sample evaluation points matching the length of `y`.
 * @param options Options specifying constant grid spacing when `x` is omitted.
 * @param options.dx Uniform sample step size (default 1).
 * @returns The approximated integral value.
 *
 * @example Integrate sampled function values
 * const y = [0, 1, 4, 9]
 * const x = [0, 1, 2, 3]
 * print('integral =', trapezoidSamples(y, x))
 */
export function trapezoidSamples(
  y: Tensor | ArrayLike<number>,
  x?: Tensor | ArrayLike<number>,
  { dx = 1 }: { dx?: number } = {},
): number {
  const ys = 'shape' in (y as object) ? toFlat(y as Tensor) : Array.from(y as ArrayLike<number>)
  const xs =
    x === undefined ? null : 'shape' in (x as object) ? toFlat(x as Tensor) : Array.from(x as ArrayLike<number>)
  if (xs && xs.length !== ys.length)
    throw new ShapeError('trapezoidSamples', 'trapezoidSamples: x and y differ in length')
  let s = 0
  for (let i = 0; i + 1 < ys.length; i++) s += (xs ? xs[i + 1] - xs[i] : dx) * 0.5 * (ys[i] + ys[i + 1])
  return s
}

/** The state of `romberg`. */
export type RombergState = Status & {
  /** Rows added after the first (the level $k$: the trapezoid rule on $2^k$ panels). */
  t: Size
  /**
   * The Romberg tableau $R$ ($k + 1) \times (k + 1)$, lower triangular (NaN above the diagonal): $R[i][0]$ is the
   * trapezoid rule on $2^i$ panels and $R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1])/(4^j - 1)$.
   */
  tableau: Tensor
  /** The current estimate $R[k][k]$. */
  value: number
  /** $|R[k][k] - R[k-1][k-1]|$ ($\infty$ at $t = 0$). */
  error: number
  /** Cumulative count of integrand function evaluations. */
  evaluations: number
  /** Set to true when error falls within tolerance. */
  converged: boolean
  /** True once the estimate is not finite. */
  diverged: boolean
}

/**
 * Romberg integration (Romberg, 1955): halve the trapezoid panels each step, reusing earlier evaluations, and
 * extrapolate the $h^2$ error expansion away (Richardson). For smooth $f$ the diagonal converges very fast. Stops when
 * $|R[k][k] - R[k-1][k-1]| \le \max(\text{atol}, \text{rtol} \cdot |R[k][k]|)$ (defaults 1e-12, 1e-12). `init` takes
 * `{ a, b }`.
 *
 * @param f The univariate real integrand function.
 * @param options Convergence options specifying absolute and relative tolerances.
 * @param options.atol Absolute tolerance on successive diagonal differences (default 1e-12).
 * @param options.rtol Relative tolerance on successive diagonal differences (default 1e-12).
 * @returns An `Algorithm` stepping through Romberg tableau extrapolation.
 *
 * @example Integrate e^x using Romberg extrapolation
 * const alg = romberg(Math.exp)
 * const state = run(alg, { a: 0, b: 1 }, 10)
 * print('integral =', state.value)
 */
export function romberg(
  f: Integrand,
  { atol = 1e-12, rtol = 1e-12 }: { atol?: number; rtol?: number } = {},
): Algorithm<{ a: number; b: number }, RombergState & { a: number; b: number }> {
  const tableauOf = (rows: number[][]) => {
    const k = rows.length
    const data = new Float64Array(k * k).fill(NaN)
    rows.forEach((row, i) => row.forEach((v, j) => (data[i * k + j] = v)))
    return fromData(data, [k, k])
  }
  const rowsOf = (t: Tensor) => {
    const k = t.shape[0]
    return Array.from({ length: k }, (_, i) => Array.from(t.data.subarray(i * k, i * k + i + 1)))
  }
  return {
    name: 'romberg',
    init: ({ a, b }) => {
      const r0 = 0.5 * (b - a) * (f(a) + f(b))
      return {
        a,
        b,
        t: 0,
        tableau: tableauOf([[r0]]),
        value: r0,
        error: Infinity,
        evaluations: 2,
        converged: false,
        diverged: !Number.isFinite(r0),
      }
    },
    step: (s) => {
      const rows = rowsOf(s.tableau)
      const k = rows.length
      const panels = 2 ** k
      const h = (s.b - s.a) / panels
      // The new points are the midpoints of the previous panels.
      let midpoints = 0
      for (let i = 1; i < panels; i += 2) midpoints += f(s.a + i * h)
      const row = [0.5 * rows[k - 1][0] + h * midpoints]
      for (let j = 1; j <= k; j++) row.push(row[j - 1] + (row[j - 1] - rows[k - 1][j - 1]) / (4 ** j - 1))
      rows.push(row)
      const value = row[k]
      const error = Math.abs(value - rows[k - 1][k - 1])
      return {
        ...s,
        t: s.t + 1,
        tableau: tableauOf(rows),
        value,
        error,
        evaluations: s.evaluations + panels / 2,
        converged: error <= Math.max(atol, rtol * Math.abs(value)),
        diverged: !Number.isFinite(value),
      }
    },
  }
}
