/**
 * Stability criteria that decide closed-loop stability without solving for the poles: the Routh–Hurwitz array of a
 * characteristic polynomial, and the Nyquist plot of an open loop L with its encirclements of −1.
 *
 * Sources: Routh (1877), "A Treatise on the Stability of a Given State of Motion"; Hurwitz (1895), Math. Ann. 46;
 * Nyquist (1932), "Regeneration theory", Bell Syst. Tech. J. 11; Ogata (2010), "Modern Control Engineering", 5th ed.,
 * §5-6 (Routh's criterion, its two special cases) and §7-5 (the Nyquist criterion, Z = N + P).
 */

import { dense, fromData, toComplexFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { LtiSystem, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { poles, rationalOf, stripLeading, systemZeros } from './system'
import { responseAt } from './responses'

// ── Routh–Hurwitz ────────────────────────────────────────────────────────────────────────────────────────────────────

/** The Routh array of a polynomial and what its first column says. */
export type RouthArray = {
  /** The array, one row per power sⁿ, …, s⁰, padded with zeros to ⌈(n + 1)/2⌉ columns. */
  rows: Matrix
  /** The first column. */
  firstColumn: Vector
  /** Sign changes down the first column: the number of roots in the open right half-plane. */
  rightHalfPlane: Size
  /** Roots on the imaginary axis, counted from the auxiliary polynomials (0 unless a row vanished). */
  imaginaryAxis: Size
  /** Roots in the open left half-plane: degree − right − imaginary. */
  leftHalfPlane: Size
  /** True when every root is in the open left half-plane. */
  stable: boolean
  /** Rows whose leading entry was zero (the rest not) and was replaced by ε. */
  epsilonRows: Size[]
  /** Rows that vanished and were replaced by the derivative of the auxiliary polynomial above them. */
  auxiliaryRows: Size[]
}

/**
 * The Routh array of a(s) = a₀sⁿ + a₁sⁿ⁻¹ + … + aₙ (descending coefficients, a₀ ≠ 0 after stripping leading zeros).
 * Row k + 2 is built from the two rows above it, r[k+2][j] = (r[k+1][0]·r[k][j+1] − r[k][0]·r[k+1][j+1]) / r[k+1][0].
 * The number of sign changes in the first column is the number of roots with positive real part (Routh, 1877). Two
 * special cases: a zero leading entry with a nonzero row is replaced by a small ε > 0 (`epsilon`, default 1e-9 times
 * the largest coefficient); a row of zeros means the roots of the auxiliary polynomial of the row above (even or odd)
 * are symmetric about the origin, and the row is replaced by that polynomial's derivative. Roots of an auxiliary
 * polynomial that are not counted by sign changes below it lie on the imaginary axis.
 */
export function routhArray(coefficients: VectorLike, { epsilon }: { epsilon?: Scalar } = {}): RouthArray {
  const a = stripLeading(Array.from(dense.toF64(coefficients, 'routhArray')))
  const n = a.length - 1
  if (n < 1 || a[0] === 0) throw new DomainError('routhArray', 'routhArray: the polynomial must have degree ≥ 1')
  const scale = Math.max(...a.map(Math.abs))
  const eps = epsilon ?? 1e-9 * scale
  const width = Math.floor(n / 2) + 1
  const rows: number[][] = [
    Array.from({ length: width }, (_, j) => a[2 * j] ?? 0),
    Array.from({ length: width }, (_, j) => a[2 * j + 1] ?? 0),
  ]
  const tiny = (v: number, ref: number) => Math.abs(v) <= 1e-12 * Math.max(ref, 1e-300)
  const epsilonRows: number[] = []
  const auxiliaryRows: number[] = []
  // Each vanished row: the power of its auxiliary polynomial (the row above holds s^p, s^{p−2}, …).
  const auxiliaryDegrees: { row: number; degree: number }[] = []
  for (let k = 1; k <= n; k++) {
    const row = rows[k]
    const ref = Math.max(...rows[k - 1].map(Math.abs), ...row.map(Math.abs))
    if (row.every((v) => tiny(v, ref)) && k <= n) {
      // A row of zeros: the auxiliary polynomial of row k − 1, in powers p, p − 2, …, with p = n − (k − 1).
      const p = n - (k - 1)
      const above = rows[k - 1]
      for (let j = 0; j < width; j++) row[j] = (p - 2 * j) * (above[j] ?? 0)
      for (let j = 0; j < width; j++) if (p - 2 * j < 0) row[j] = 0
      auxiliaryRows.push(k)
      auxiliaryDegrees.push({ row: k - 1, degree: p })
    }
    if (tiny(row[0], ref)) {
      row[0] = eps
      epsilonRows.push(k)
    }
    if (k === n) break
    const next = Array.from(
      { length: width },
      (_, j) => (row[0] * (rows[k - 1][j + 1] ?? 0) - rows[k - 1][0] * (row[j + 1] ?? 0)) / row[0],
    )
    rows.push(next)
  }
  const first = rows.map((r) => r[0])
  const changes = (from: number) => {
    let c = 0
    for (let k = from + 1; k < first.length; k++) if (Math.sign(first[k]) !== Math.sign(first[k - 1])) c++
    return c
  }
  const rightHalfPlane = changes(0)
  // An auxiliary polynomial of degree d has its roots in ± pairs (or quadruples); the sign changes below it count those
  // in the right half-plane, as many lie in the left, and the rest are on the imaginary axis.
  let imaginaryAxis = 0
  if (auxiliaryDegrees.length) {
    const { row, degree } = auxiliaryDegrees[0]
    imaginaryAxis = Math.max(0, degree - 2 * changes(row))
  }
  const leftHalfPlane = n - rightHalfPlane - imaginaryAxis
  const flat = Float64Array.from(rows.flat())
  return {
    rows: fromData(flat, [rows.length, width]),
    firstColumn: fromData(Float64Array.from(first), [first.length]),
    rightHalfPlane,
    imaginaryAxis,
    leftHalfPlane,
    stable: rightHalfPlane === 0 && imaginaryAxis === 0 && epsilonRows.length === 0,
    epsilonRows,
    auxiliaryRows,
  }
}

// ── Nyquist ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A Nyquist plot of an open loop and what it says about the closed loop 1/(1 + L). */
export type Nyquist = {
  /** Contour frequencies, increasing from −ω_max to ω_max (rad/s; rad/sample from −π to π for a discrete system). */
  w: Vector
  /** Re L along the contour. */
  re: Vector
  /** Im L along the contour. */
  im: Vector
  /** Net clockwise encirclements of the point −1 by L over the closed contour: N. */
  encirclements: number
  /** Open-loop poles inside the contour (right half-plane, or outside the unit circle): P. */
  openLoopUnstable: Size
  /** Closed-loop poles inside the contour: Z = N + P. */
  closedLoopUnstable: number
  /** True when Z = 0. */
  stable: boolean
  /** Contour points closest to −1: the minimum of |1 + L|, the inverse of the peak sensitivity. */
  minimumDistance: Scalar
}

/** Options for `nyquist`. */
export type NyquistOptions = {
  /** Positive frequencies to start from (default: `frequencyGrid`-like, 400 log-spaced points). */
  w?: VectorLike
  /** Radius of the indentation around poles on the imaginary axis (or unit circle), relative to the scale. */
  indent?: Scalar
  /** Maximum angle (radians) between neighbouring points of 1 + L before the grid is refined. Default π/8. */
  maxTurn?: Scalar
}

/**
 * The Nyquist plot of an open loop L(s) under unity negative feedback (Nyquist, 1932). The contour runs up the
 * imaginary axis s = iω, ω from −ω_max to ω_max, indented to the right of any imaginary-axis pole by a small
 * semicircle, and closes through the right half-plane at infinity, where a proper L is constant. For a discrete system
 * the contour is the unit circle z = e^{iω}, indented outward around poles on it. N, the net clockwise encirclements of
 * −1, is the winding number of 1 + L, computed from its phase along a grid refined until neighbouring points turn by at
 * most `maxTurn`. The closed loop has Z = N + P poles inside the contour, P being the open loop's (Ogata, 2010, §7-5).
 */
export function nyquist(L: LtiSystem, options: NyquistOptions = {}): Nyquist {
  const discrete = L.domain === 'discrete'
  const ps = toComplexFlat(poles(L))
  const zs = toComplexFlat(systemZeros(L))
  const indentRel = options.indent ?? 1e-4
  const maxTurn = options.maxTurn ?? Math.PI / 8
  const at = responseAt(L)
  // Poles on the contour: on the imaginary axis (continuous) or the unit circle (discrete).
  const onContour = ps.filter((p) => (discrete ? Math.abs(Math.hypot(p.re, p.im) - 1) < 1e-9 : Math.abs(p.re) < 1e-9))
  const P = ps.filter((p) => (discrete ? Math.hypot(p.re, p.im) > 1 + 1e-9 : p.re > 1e-9)).length
  // The scale of the dynamics, for the frequency range and the indentation radius.
  const mods = [...ps, ...zs].map((c) => Math.hypot(c.re, c.im)).filter((m) => m > 1e-12)
  const lo = mods.length ? Math.min(...mods) : 1
  const hi = mods.length ? Math.max(...mods) : 1
  const r = indentRel * (discrete ? 1 : Math.max(lo, 1e-3))
  // L on the indented contour at parameter ω (complex point s or z, shifted off contour poles).
  const evalAt = (w: number): { re: number; im: number } => {
    if (discrete) {
      for (const p of onContour) {
        const d = Math.atan2(Math.sin(w - Math.atan2(p.im, p.re)), Math.cos(w - Math.atan2(p.im, p.re)))
        if (Math.abs(d) < r) {
          // Outward semicircle z = p + r e^{iφ}, φ sweeping through the outward normal.
          const phi = Math.atan2(p.im, p.re) + (d / r) * (Math.PI / 2)
          return evalRational(L, { re: p.re + r * Math.cos(phi), im: p.im + r * Math.sin(phi) })
        }
      }
      return at(w)
    }
    for (const p of onContour) {
      if (Math.abs(w - p.im) < r) {
        // Right semicircle s = i·p.im + r e^{iφ}, φ from −π/2 to π/2 as ω crosses p.im.
        const phi = ((w - p.im) / r) * (Math.PI / 2)
        return evalRational(L, { re: r * Math.cos(phi), im: p.im + r * Math.sin(phi) })
      }
    }
    return at(w)
  }
  // The starting grid: symmetric about 0, dense near every contour pole's indentation.
  let grid: number[]
  if (discrete) {
    grid = Array.from({ length: 801 }, (_, i) => -Math.PI + (2 * Math.PI * i) / 800)
  } else {
    const pos = options.w
      ? Array.from(dense.toF64(options.w, 'nyquist w')).filter((v) => v > 0)
      : Array.from(
          { length: 400 },
          (_, i) => 10 ** (Math.log10(lo) - 3 + ((Math.log10(hi) + 3 - Math.log10(lo) + 3) * i) / 399),
        )
    grid = [...pos.map((v) => -v).reverse(), 0, ...pos]
  }
  for (const p of onContour) {
    const centre = discrete ? Math.atan2(p.im, p.re) : p.im
    for (let k = -8; k <= 8; k++) grid.push(centre + (k / 8) * r * 1.0001)
  }
  grid = [...new Set(grid)].sort((u, v) => u - v)
  const values = grid.map(evalAt)
  // Refine where 1 + L turns too fast (or jumps), so the winding number is exact.
  const ws: number[] = []
  const vs: { re: number; im: number }[] = []
  const arg = (v: { re: number; im: number }) => Math.atan2(v.im, 1 + v.re)
  const wrap = (d: number) => d - 2 * Math.PI * Math.round(d / (2 * Math.PI))
  const push = (
    w0: number,
    v0: { re: number; im: number },
    w1: number,
    v1: { re: number; im: number },
    depth: number,
  ) => {
    const turn = Math.abs(wrap(arg(v1) - arg(v0)))
    if (depth < 30 && turn > maxTurn && Math.abs(w1 - w0) > 1e-14 * (1 + Math.abs(w0))) {
      const wm = 0.5 * (w0 + w1)
      const vm = evalAt(wm)
      push(w0, v0, wm, vm, depth + 1)
      push(wm, vm, w1, v1, depth + 1)
      return
    }
    ws.push(w1)
    vs.push(v1)
  }
  ws.push(grid[0])
  vs.push(values[0])
  for (let i = 1; i < grid.length; i++) push(grid[i - 1], values[i - 1], grid[i], values[i], 0)
  // Winding of 1 + L along the contour, closed by the arc at infinity (continuous: L constant there for proper L;
  // discrete: the circle closes on itself).
  let total = 0
  for (let i = 1; i < vs.length; i++) total += wrap(arg(vs[i]) - arg(vs[i - 1]))
  total += wrap(arg(vs[0]) - arg(vs[vs.length - 1]))
  // The contour runs counter-clockwise around the left half-plane… as traversed (ω increasing) it is clockwise
  // around the right half-plane, so a clockwise encirclement of −1 by L lowers the phase of 1 + L by 2π.
  const encirclements = Math.round(-total / (2 * Math.PI)) + 0 // + 0 turns −0 into 0
  const Z = encirclements + P
  const minimumDistance = Math.min(...vs.map((v) => Math.hypot(1 + v.re, v.im)))
  const f = (xs: number[]) => fromData(Float64Array.from(xs), [xs.length])
  return {
    w: f(ws),
    re: f(vs.map((v) => v.re)),
    im: f(vs.map((v) => v.im)),
    encirclements,
    openLoopUnstable: P,
    closedLoopUnstable: Z,
    stable: Z === 0,
    minimumDistance,
  }
}

/** L at a complex point x (s or z) off the contour: num(x)/den(x) by Horner's rule, times the delay. */
function evalRational(L: LtiSystem, x: { re: number; im: number }): { re: number; im: number } {
  const { num, den } = rationalOf(L)
  const horner = (c: readonly number[]) => {
    let re = 0
    let im = 0
    for (const a of c) {
      const r2 = re * x.re - im * x.im + a
      im = re * x.im + im * x.re
      re = r2
    }
    return { re, im }
  }
  const n = horner(num)
  const d = horner(den)
  const dd = d.re * d.re + d.im * d.im
  let re = (n.re * d.re + n.im * d.im) / dd
  let im = (n.im * d.re - n.re * d.im) / dd
  if (L.delay) {
    // e^{−sτ} (continuous) or z^{−d} (discrete).
    const discrete = L.domain === 'discrete'
    const mag = discrete ? Math.hypot(x.re, x.im) ** -L.delay : Math.exp(-x.re * L.delay)
    const ph = discrete ? -L.delay * Math.atan2(x.im, x.re) : -x.im * L.delay
    const c = mag * Math.cos(ph)
    const sn = mag * Math.sin(ph)
    const r2 = re * c - im * sn
    im = re * sn + im * c
    re = r2
  }
  return { re, im }
}
