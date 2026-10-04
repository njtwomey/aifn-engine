/**
 * Linear stability of numerical methods: applied to the test equation x′ = λx with z = hλ, a one-step method gives
 * x_{n+1} = R(z) x_n, and a linear multistep method a recurrence whose characteristic roots ζ must satisfy |ζ| ≤ 1.
 * The stability region is where the amplification (|R(z)|, or the largest |ζ|) is at most 1 (Hairer & Wanner, 1996,
 * §IV.2–3 and §V.1).
 */

import { fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { TABLEAUX, type ButcherTableau } from './explicit'
import { DORMAND_PRINCE } from './adaptive'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A complex number as [re, im]. */
type C = [number, number]
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
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

// BDF_k as (1 − zβ)ζ^k − Σ_j a_j ζ^{k−j} = 0 (see implicit.ts).
const BDF: Record<string, { a: number[]; beta: number }> = {
  bdf1: { a: [1], beta: 1 },
  bdf2: { a: [4 / 3, -1 / 3], beta: 2 / 3 },
  bdf3: { a: [18 / 11, -9 / 11, 2 / 11], beta: 6 / 11 },
}

/**
 * The stability function R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙 of a Runge–Kutta tableau (explicit or implicit), evaluated in
 * complex arithmetic by forward substitution when A is lower triangular and Gaussian elimination otherwise.
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

/** The largest modulus among the roots of a complex polynomial (coefficients highest degree first), by Durand–Kerner. */
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
 * The amplification of a method on x′ = λx at z = hλ = re + i·im: |R(z)| for a one-step method (explicit or implicit
 * Runge–Kutta, `'implicit-euler'`, `'implicit-trapezoid'`), or the largest modulus of the characteristic roots for BDF1–3. The
 * method is linearly stable at z when this is at most 1.
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

/** A method's amplification on a grid of the complex z-plane. */
export type StabilityRegion = {
  /** Real parts of z (nx). */
  real: Vector
  /** Imaginary parts of z (ny). */
  imag: Vector
  /** The amplification at each grid point (ny × nx); the method is stable where it is ≤ 1. */
  amplification: Matrix
}

/**
 * The amplification of a method over the rectangle `real` × `imag` of the z = hλ plane on an nx × ny grid (default
 * 121 × 121). Contour it at 1 to draw the boundary of the stability region.
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
 * The boundary locus of BDF_k: the image of the unit circle ζ = e^{iθ} under z(ζ) = (1 − Σ_j a_j ζ^{−j})/β, the
 * values of z = hλ at which a characteristic root has modulus exactly 1. The stability region is the exterior of the
 * closed curve. Returns `n` points (default 400) as real and imaginary parts.
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
