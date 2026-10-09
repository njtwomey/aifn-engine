/**
 * Linear stability of numerical methods: applied to the test equation $x' = \lambda x$ with $z = h\lambda$, a one-step
 * method gives $x_{n+1} = R(z) x_n$, and a linear multistep method a recurrence whose characteristic roots $\zeta$ must
 * satisfy $\lvert \zeta \rvert \le 1$. The stability region is where the amplification ($\lvert R(z) \rvert$, or the
 * largest $\lvert \zeta \rvert$) is at most 1 (Hairer & Wanner, 1996, §IV.2–3 and §V.1). Complex numbers are worked
 * with as pairs $[\operatorname{Re}, \operatorname{Im}]$, and a method is given by name or by its Butcher tableau.
 */

import { fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { TABLEAUX, type ButcherTableau } from './explicit'
import { DORMAND_PRINCE } from './adaptive'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A complex number as [re, im]. */
type C = [number, number]
/**
 * The complex product $ab$.
 *
 * @param a The first factor, as [re, im].
 * @param b The second factor, as [re, im].
 * @returns $ab$ as [re, im].
 */
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
/**
 * The complex quotient $a / b$ (infinite or NaN parts when $b = 0$).
 *
 * @param a The numerator, as [re, im].
 * @param b The denominator, as [re, im].
 * @returns $a / b$ as [re, im].
 */
const cdiv = (a: C, b: C): C => {
  const d = b[0] * b[0] + b[1] * b[1]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}

/** The methods whose stability can be computed by name. */
export type StabilityMethod =
  | keyof typeof TABLEAUX
  | 'dormand-prince'
  | 'implicit-euler'
  | 'implicit-trapezoid'
  | 'bdf1'
  | 'bdf2'
  | 'bdf3'
  | ButcherTableau

/**
 * The BDF methods of order $k$ by their coefficients: the characteristic polynomial on the test equation is
 * $(1 - z\beta)\zeta^k - \sum_{j=1}^{k} a_j \zeta^{k-j} = 0$ (see `implicit.ts`).
 */
const BDF: Record<string, { a: number[]; beta: number }> = {
  bdf1: { a: [1], beta: 1 },
  bdf2: { a: [4 / 3, -1 / 3], beta: 2 / 3 },
  bdf3: { a: [18 / 11, -9 / 11, 2 / 11], beta: 6 / 11 },
}

/**
 * The stability function $R(z) = 1 + z \bvec^\top (\Imat - z\Amat)^{-1} \ones$ of a Runge–Kutta tableau (explicit
 * or implicit), evaluated in complex arithmetic by Gaussian elimination without pivoting, which for a lower-triangular
 * $\Amat$ is forward substitution. A zero pivot gives infinite or NaN parts.
 *
 * @param tab The tableau: its `a` gives $\Amat$ (missing entries are 0, so rows may be short) and its `b` gives
 *   $\bvec$ and the number of stages.
 * @param z The point $z = h\lambda$, as [re, im].
 * @returns $R(z)$ as [re, im].
 */
function rkStability(tab: ButcherTableau, z: C): C {
  const s = tab.b.length
  // Solve (I − zA) k = 𝟙 for the stage vector k (complex).
  const M: C[][] = Array.from({ length: s }, (_, i) =>
    Array.from({ length: s }, (_, j): C => {
      const aij = tab.a[i]?.[j] ?? 0
      return [(i === j ? 1 : 0) - z[0] * aij, -z[1] * aij]
    }),
  )
  const k: C[] = Array.from({ length: s }, (): C => [1, 0])
  for (let c = 0; c < s; c++) {
    for (let r = c + 1; r < s; r++) {
      if (M[r][c][0] === 0 && M[r][c][1] === 0) continue
      const f = cdiv(M[r][c], M[c][c])
      for (let j = c; j < s; j++) {
        const p = cmul(f, M[c][j])
        M[r][j] = [M[r][j][0] - p[0], M[r][j][1] - p[1]]
      }
      const p = cmul(f, k[c])
      k[r] = [k[r][0] - p[0], k[r][1] - p[1]]
    }
  }
  for (let r = s - 1; r >= 0; r--) {
    let acc = k[r]
    for (let j = r + 1; j < s; j++) {
      const p = cmul(M[r][j], k[j])
      acc = [acc[0] - p[0], acc[1] - p[1]]
    }
    k[r] = cdiv(acc, M[r][r])
  }
  let sum: C = [0, 0]
  for (let i = 0; i < s; i++) sum = [sum[0] + tab.b[i] * k[i][0], sum[1] + tab.b[i] * k[i][1]]
  const zs = cmul(z, sum)
  return [1 + zs[0], zs[1]]
}

/**
 * The largest modulus among the roots of a complex polynomial, by the Durand–Kerner iteration (at most 200 sweeps,
 * stopping when no root moves by more than $10^{-14}$); a linear polynomial is solved directly.
 *
 * @param coefficients The coefficients, highest degree first, as [re, im] pairs; the leading one must be non-zero and
 *   the degree at least 1.
 * @returns The largest $\lvert \zeta \rvert$ over the roots $\zeta$.
 */
function largestRootModulus(coefficients: C[]): number {
  const n = coefficients.length - 1
  if (n === 1) {
    const r = cdiv(coefficients[1], coefficients[0])
    return Math.hypot(r[0], r[1])
  }
  const monic = coefficients.map((c) => cdiv(c, coefficients[0]))
  const value = (x: C) =>
    monic.reduce<C>(
      (acc, c) => {
        const p = cmul(acc, x)
        return [p[0] + c[0], p[1] + c[1]]
      },
      [0, 0],
    )
  let roots: C[] = Array.from({ length: n }, (_, k): C => {
    const a = (2 * Math.PI * k) / n + 0.4
    return [0.9 * Math.cos(a), 0.9 * Math.sin(a)]
  })
  for (let it = 0; it < 200; it++) {
    let change = 0
    roots = roots.map((r, i) => {
      let d: C = [1, 0]
      roots.forEach((q, j) => {
        if (j !== i) d = cmul(d, [r[0] - q[0], r[1] - q[1]])
      })
      const step = cdiv(value(r), d)
      change = Math.max(change, Math.hypot(step[0], step[1]))
      return [r[0] - step[0], r[1] - step[1]]
    })
    if (change < 1e-14) break
  }
  return Math.max(...roots.map((r) => Math.hypot(r[0], r[1])))
}

/**
 * The amplification of a method on $x' = \lambda x$ at $z = h\lambda = a + ib$: $\lvert R(z) \rvert$ for a one-step
 * method (explicit or implicit Runge–Kutta, `'implicit-euler'`, `'implicit-trapezoid'`), or the largest modulus of the
 * characteristic roots for BDF of order 1 to 3. The method is linearly stable at $z$ when this is at most 1. An
 * unknown method name throws `DomainError`.
 *
 * @param method The method: a name, or a Butcher tableau (explicit or implicit).
 * @param re The real part $a$ of $z$.
 * @param im The imaginary part $b$ of $z$.
 * @returns The amplification at $z$, a non-negative number.
 *
 * @example Explicit Euler is stable only for small steps, implicit Euler on the whole left half-plane
 * // Explicit Euler: |1 + z|. Implicit Euler: 1/|1 − z|.
 * print('euler at z = -1:', amplification('euler', -1, 0))
 * print('euler at z = -3:', amplification('euler', -3, 0))
 * print('implicit-euler at z = -3:', amplification('implicit-euler', -3, 0))
 * print('implicit-euler at z = -100:', amplification('implicit-euler', -100, 0))
 *
 * @example Higher-order methods reach further along the negative axis
 * for (const m of ['euler', 'heun', 'rk4', 'dormand-prince', 'bdf2']) {
 *   print(m, 'at z = -2.5:', amplification(m, -2.5, 0))
 * }
 *
 * @example On the imaginary axis (an undamped oscillation)
 * // The trapezoid rule is exactly neutral there; RK4 is just inside its region at z = 2i.
 * print('implicit-trapezoid at z = 2i:', amplification('implicit-trapezoid', 0, 2))
 * print('rk4 at z = 2i:', amplification('rk4', 0, 2))
 * print('heun at z = 2i:', amplification('heun', 0, 2))
 */
export function amplification(method: StabilityMethod, re: Scalar, im: Scalar): Scalar {
  const z: C = [re, im]
  if (typeof method === 'string' && method in BDF) {
    const { a, beta } = BDF[method]
    const coefficients: C[] = [[1 - beta * re, -beta * im], ...a.map((v): C => [-v, 0])]
    return largestRootModulus(coefficients)
  }
  if (method === 'implicit-euler') {
    const r = cdiv([1, 0], [1 - re, -im])
    return Math.hypot(r[0], r[1])
  }
  if (method === 'implicit-trapezoid') {
    const r = cdiv([1 + re / 2, im / 2], [1 - re / 2, -im / 2])
    return Math.hypot(r[0], r[1])
  }
  const tab =
    typeof method !== 'string'
      ? method
      : method === 'dormand-prince'
        ? DORMAND_PRINCE
        : (TABLEAUX as Record<string, ButcherTableau | undefined>)[method]
  if (!tab) throw new DomainError('amplification', `amplification: unknown method ${String(method)}`)
  const r = rkStability(tab, z)
  return Math.hypot(r[0], r[1])
}

/** A method's amplification on a grid of the complex $z$-plane. */
export type StabilityRegion = {
  /** Real parts of $z$, ascending (length `nx`). */
  real: Vector
  /** Imaginary parts of $z$, ascending (length `ny`). */
  imag: Vector
  /**
   * The amplification at each grid point (`ny` $\times$ `nx`): row $j$, column $i$ is at `real[i]` $+ i \cdot$
   * `imag[j]`. The method is stable where it is $\le 1$.
   */
  amplification: Matrix
}

/**
 * The amplification of a method over the rectangle `real` $\times$ `imag` of the $z = h\lambda$ plane on an
 * `nx` $\times$ `ny` grid (default $121 \times 121$), by `amplification` at every point. Contour it at 1 to draw the
 * boundary of the stability region.
 *
 * @param method The method: a name, or a Butcher tableau.
 * @param options The rectangle and the grid.
 * @param options.real The range of real parts, both ends included.
 * @param options.imag The range of imaginary parts, both ends included.
 * @param options.nx The number of grid points along the real axis (a single point sits at the lower end).
 * @param options.ny The number of grid points along the imaginary axis.
 * @returns The grid's axes and the amplification at each point.
 *
 * @example Explicit Euler's region is a disc of radius 1
 * // |1 + z| on a 5 × 3 grid: real parts −2 to 0, imaginary parts −1 to 1.
 * const r = stabilityRegion('euler', { real: [-2, 0], imag: [-1, 1], nx: 5, ny: 3 })
 * print('real =', r.real)
 * print('imag =', r.imag)
 * print('amplification =', r.amplification)
 *
 * @example The share of a square that is stable
 * for (const m of ['euler', 'rk4', 'implicit-euler']) {
 *   const a = toFlat(stabilityRegion(m, { real: [-4, 0], imag: [-2, 2], nx: 41, ny: 41 }).amplification)
 *   print(m, 'stable on', a.filter((v) => v <= 1).length, 'of', a.length, 'points')
 * }
 */
export function stabilityRegion(
  method: StabilityMethod,
  {
    real = [-5, 3],
    imag = [-4, 4],
    nx = 121,
    ny = 121,
  }: { real?: readonly [Scalar, Scalar]; imag?: readonly [Scalar, Scalar]; nx?: Size; ny?: Size } = {},
): StabilityRegion {
  const xs = Float64Array.from({ length: nx }, (_, i) => real[0] + ((real[1] - real[0]) * i) / Math.max(1, nx - 1))
  const ys = Float64Array.from({ length: ny }, (_, j) => imag[0] + ((imag[1] - imag[0]) * j) / Math.max(1, ny - 1))
  const out = new Float64Array(nx * ny)
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out[j * nx + i] = amplification(method, xs[i], ys[j])
  return { real: fromData(xs, [nx]), imag: fromData(ys, [ny]), amplification: fromData(out, [ny, nx]) }
}

/**
 * The boundary locus of the BDF method of order $k$: the image of the unit circle $\zeta = e^{i\theta}$ under
 * $z(\zeta) = (1 - \sum_j a_j \zeta^{-j}) / \beta$, the values of $z = h\lambda$ at which a characteristic root has
 * modulus exactly 1. The stability region is the exterior of the closed curve.
 *
 * @param method The BDF method, of order 1, 2 or 3.
 * @param n The number of points, at $\theta$ evenly spaced from 0 to $2\pi$ inclusive, so the last repeats the first
 *   and closes the curve. At least 2 (a single point is NaN).
 * @returns The real and imaginary parts of the points of the curve (length `n` each).
 *
 * @example BDF2's boundary at four angles
 * // θ = 0, π/2, π, 3π/2, 2π: z = 0 at θ = 0, and the curve reaches z = 4 on the real axis at θ = π.
 * const { real, imag } = boundaryLocus('bdf2', 5)
 * print('real =', real)
 * print('imag =', imag)
 *
 * @example BDF1's boundary is the circle of radius 1 about 1
 * // Implicit Euler is unstable only inside |1 − z| < 1.
 * const { real, imag } = boundaryLocus('bdf1', 9)
 * print('distance from 1 =', toFlat(real).map((x, k) => Math.hypot(x - 1, toFlat(imag)[k])))
 */
export function boundaryLocus(method: 'bdf1' | 'bdf2' | 'bdf3', n: Size = 400): { real: Vector; imag: Vector } {
  const { a, beta } = BDF[method]
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const theta = (2 * Math.PI * k) / (n - 1)
    let zr = 1
    let zi = 0
    a.forEach((aj, j) => {
      zr -= aj * Math.cos(-(j + 1) * theta)
      zi -= aj * Math.sin(-(j + 1) * theta)
    })
    re[k] = zr / beta
    im[k] = zi / beta
  }
  return { real: fromData(re, [n]), imag: fromData(im, [n]) }
}
