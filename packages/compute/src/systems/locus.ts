/**
 * The root locus: the closed-loop poles of $1 + kL(s) = 0$ as the gain $k$ runs from 0 to $\infty$ (Evans, 1948,
 * "Graphical analysis of control systems", Trans. AIEE 67; Ogata, 2010, "Modern Control Engineering", 5th ed., §6-2).
 *
 * With $L = N/D$ the closed-loop poles are the roots of $D + kN$, found by `roots` at each gain; the same holds for a
 * discrete $L(z)$, whose stability boundary is the unit circle instead of the imaginary axis.
 */

import { roots } from 'aifn-compute/numerics/polynomial'
import { findRoot } from 'aifn-compute/numerics/roots'
import { dense, fromData, toComplexFlat, type Vector } from 'aifn-compute/foundation/tensor'
import type { ComplexNumber, LtiSystem, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { rationalOf, stripLeading } from './system'

/** One point where branches meet on the real axis and leave it (or arrive): `s`, the point, and `gain`, $k$ there. */
export type LocusPoint = { s: Scalar; gain: Scalar }

/**
 * A branch crossing into the other half-plane (continuous) or across the unit circle (discrete): `gain`, the $k$ of the
 * crossing, and `at`, the crossing point in the upper half-plane.
 */
export type LocusCrossing = { gain: Scalar; at: ComplexNumber }

/** The root locus of an open loop $L = N/D$ under negative feedback with gain $k$. */
export type RootLocus = {
  /** The gains, increasing from 0. */
  gains: Vector
  /** The closed-loop poles: `branches[j][i]` is branch $j$ at `gains[i]`; branches are continuous in $k$. */
  branches: ComplexNumber[][]
  /** The open-loop poles, where the branches start ($k = 0$). */
  openLoopPoles: ComplexNumber[]
  /** The open-loop zeros, where the finite branches end as $k \to \infty$. */
  openLoopZeros: ComplexNumber[]
  /**
   * The asymptotes of the $n - m$ branches that go to infinity: centroid $\sigma = (\sum p - \sum z)/(n - m)$ (NaN when
   * $n = m$) and angles (radians).
   */
  asymptotes: { centroid: Scalar; angles: Scalar[] }
  /**
   * Breakaway and break-in points, ascending: real $s$ with $dk/ds = 0$ and $k = -D(s)/N(s) > 0$.
   */
  breakaway: LocusPoint[]
  /** Gains at which a branch crosses the stability boundary, refined by Brent's method. */
  crossings: LocusCrossing[]
}

/** Options for `rootLocus`. */
export type RootLocusOptions = {
  /**
   * Gains to evaluate, finite and non-negative (sorted here; 0 is prepended if missing). Default: 0 and `points - 1`
   * log-spaced gains from $10^{-3}$ to $10^3$ times a characteristic gain.
   */
  gains?: VectorLike
  /** Points for the default grid, 0 included. Default 300. */
  points?: Size
}

/**
 * A real polynomial at a real point, by Horner's rule.
 *
 * @param c The coefficients, descending.
 * @param x The point.
 * @returns The value.
 */
const polyAt = (c: readonly number[], x: number) => c.reduce((acc, a) => acc * x + a, 0)

/**
 * The closed-loop characteristic polynomial $D + kN$, both descending, aligned on the right.
 *
 * @param num The numerator $N$, descending.
 * @param den The denominator $D$, descending.
 * @param k The gain.
 * @returns The coefficients of $D + kN$, descending, leading zeros stripped.
 */
function characteristic(num: readonly number[], den: readonly number[], k: number): number[] {
  const n = Math.max(num.length, den.length)
  const out = new Array<number>(n).fill(0)
  for (let i = 0; i < den.length; i++) out[n - den.length + i] += den[i]
  for (let i = 0; i < num.length; i++) out[n - num.length + i] += k * num[i]
  return stripLeading(out)
}

/**
 * The roots of a real polynomial as complex numbers (none for a constant).
 *
 * @param c The coefficients, descending; leading zeros are ignored.
 * @returns The roots as `{ re, im }`.
 */
function rootsOf(c: readonly number[]): ComplexNumber[] {
  const s = stripLeading([...c])
  if (s.length < 2) return []
  return toComplexFlat(roots(Float64Array.from(s)))
}

/**
 * Match the roots `next` to the previous points of each branch by a greedy nearest pairing, so branches stay
 * continuous. With equal counts every branch gets one root.
 *
 * @param prev The last point of each branch.
 * @param next The roots at the new gain.
 * @returns The new point of each branch, in the order of `prev`; a branch left without a root keeps its last point.
 */
function track(prev: readonly ComplexNumber[], next: readonly ComplexNumber[]): ComplexNumber[] {
  const out = new Array<ComplexNumber>(prev.length)
  const free = new Set(next.map((_, i) => i))
  const pairs: { i: number; j: number; d: number }[] = []
  prev.forEach((p, i) => next.forEach((q, j) => pairs.push({ i, j, d: Math.hypot(p.re - q.re, p.im - q.im) })))
  pairs.sort((a, b) => a.d - b.d)
  const used = new Set<number>()
  for (const { i, j } of pairs) {
    if (used.has(i) || !free.has(j)) continue
    out[i] = next[j]
    used.add(i)
    free.delete(j)
  }
  // Branches with no partner (a root escaped to infinity): keep them at their last position.
  prev.forEach((p, i) => {
    if (!out[i]) out[i] = p
  })
  return out
}

/**
 * A default gain grid: from $10^{-3}$ to $10^3$ times a characteristic gain $k_0$, log-spaced, with 0 in front.
 * $k_0$ is $\lvert D(r)/N(r) \rvert$ at $r$, the largest pole or zero modulus (at least 1), or 1 if that is 0.
 *
 * @param num The numerator $N$, descending.
 * @param den The denominator $D$, descending.
 * @param points The number of gains, 0 included.
 * @returns The gains, ascending.
 */
function defaultGains(num: readonly number[], den: readonly number[], points: number): number[] {
  // The characteristic gain: |D/N| at a characteristic radius of the poles and zeros.
  const r = Math.max(
    1,
    ...rootsOf(den).map((c) => Math.hypot(c.re, c.im)),
    ...rootsOf(num).map((c) => Math.hypot(c.re, c.im)),
  )
  const k0 = Math.abs(polyAt(den, r) / polyAt(num, r)) || 1
  const lo = Math.log10(k0) - 3
  const hi = Math.log10(k0) + 3
  return [0, ...Array.from({ length: points - 1 }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (points - 2)))]
}

/**
 * The root locus of a SISO open loop $L(s) = N(s)/D(s)$ (or $L(z)$): the roots of $D + kN$ for each gain $k$ (Evans,
 * 1948). Branches start at the $n$ open-loop poles; $m$ of them end at the zeros and $n - m$ leave along asymptotes at
 * angles $(2q + 1)\pi/(n - m)$ from the centroid $(\sum_i p_i - \sum_j z_j)/(n - m)$. Breakaway points are the real
 * roots of $N'D - ND' = 0$ with a positive gain $k = -D/N$ there. Crossings of the stability boundary (the imaginary
 * axis, or the unit circle for a discrete system) are bracketed on the gain grid and refined by Brent's method.
 * Throws `DomainError` for a zero or improper open loop, or a negative or non-finite gain.
 *
 * @param L The SISO open loop $L = N/D$, proper.
 * @param options The gains to evaluate, or the size of the default grid.
 * @returns The gains, the branches over them, the open-loop poles and zeros, the asymptotes, the breakaway points and
 *   the boundary crossings.
 *
 * @example $L = 1/(s(s + 1)(s + 2))$: breakaway near $-0.42$, and the crossing at $k = 6$, $s = i\sqrt 2$
 * const locus = rootLocus(transferFunction([1], [1, 3, 2, 0]))
 * print('asymptotes =', locus.asymptotes)
 * print('breakaway =', locus.breakaway)
 * print('crossings =', locus.crossings)
 * print('branches =', locus.branches.length, ' gains =', locus.gains.shape)
 *
 * @example A discrete loop $1/(z^2 - 0.7z + 0.1)$ leaves the unit circle when $0.1 + k = 1$
 * const locus = rootLocus(transferFunction([0, 0, 1], [1, -0.7, 0.1], { dt: 1 }), { points: 100 })
 * print('crossings =', locus.crossings)
 * print('breakaway =', locus.breakaway)
 */
export function rootLocus(L: LtiSystem, options: RootLocusOptions = {}): RootLocus {
  const { num: n0, den: d0 } = rationalOf(L)
  const num = stripLeading([...n0])
  const den = stripLeading([...d0])
  if (num.every((v) => v === 0)) throw new DomainError('rootLocus', 'rootLocus: the open loop is zero')
  if (num.length > den.length) throw new DomainError('rootLocus', 'rootLocus: the open loop must be proper')
  let gains = options.gains
    ? Array.from(dense.toF64(options.gains, 'rootLocus gains')).sort((a, b) => a - b)
    : defaultGains(num, den, options.points ?? 300)
  if (!gains.every((k) => k >= 0 && Number.isFinite(k)))
    throw new DomainError('rootLocus', 'rootLocus: gains must be finite and non-negative')
  if (gains[0] !== 0) gains = [0, ...gains]
  const P = rootsOf(den)
  const Z = rootsOf(num)
  const order = P.length
  let current = P
  const columns: ComplexNumber[][] = [P]
  for (let i = 1; i < gains.length; i++) {
    const rs = rootsOf(characteristic(num, den, gains[i]))
    current = track(current, rs.length === order ? rs : [...rs, ...current.slice(rs.length)])
    columns.push(current)
  }
  const branches = Array.from({ length: order }, (_, j) => columns.map((c) => c[j]))
  // Asymptotes.
  const excess = P.length - Z.length
  const centroid = excess > 0 ? (P.reduce((s, p) => s + p.re, 0) - Z.reduce((s, z) => s + z.re, 0)) / excess : NaN
  const angles = Array.from({ length: Math.max(0, excess) }, (_, q) => ((2 * q + 1) * Math.PI) / excess)
  // Breakaway points: N′D − ND′ = 0.
  const deriv = (c: readonly number[]) => c.slice(0, -1).map((a, i) => a * (c.length - 1 - i))
  const mul = (a: readonly number[], b: readonly number[]) => {
    const out = new Array<number>(a.length + b.length - 1).fill(0)
    a.forEach((x, i) => b.forEach((y, j) => (out[i + j] += x * y)))
    return out
  }
  const sub = (a: number[], b: number[]) => {
    const len = Math.max(a.length, b.length)
    const pa = [...new Array<number>(len - a.length).fill(0), ...a]
    const pb = [...new Array<number>(len - b.length).fill(0), ...b]
    return pa.map((v, i) => v - pb[i])
  }
  const condition = sub(mul(deriv(num), den), mul(num, deriv(den)))
  const breakaway: LocusPoint[] = []
  for (const r of rootsOf(condition)) {
    if (Math.abs(r.im) > 1e-7 * Math.max(1, Math.abs(r.re))) continue
    const nv = polyAt(num, r.re)
    if (Math.abs(nv) < 1e-14) continue
    const k = -polyAt(den, r.re) / nv
    if (k > 0) breakaway.push({ s: r.re, gain: k })
  }
  breakaway.sort((a, b) => a.s - b.s)
  // Crossings of the stability boundary.
  const discrete = L.domain === 'discrete'
  const margin = (k: number) => {
    const rs = rootsOf(characteristic(num, den, k))
    return Math.max(...rs.map((c) => (discrete ? Math.hypot(c.re, c.im) - 1 : c.re)))
  }
  const crossings: LocusCrossing[] = []
  for (let i = 1; i < gains.length; i++) {
    const a = margin(gains[i - 1])
    const b = margin(gains[i])
    if (Number.isFinite(a) && Number.isFinite(b) && a * b < 0) {
      const k = findRoot(margin, [gains[i - 1], gains[i]], { xtol: 1e-12 }).x
      // The crossing branch is the one that sets the margin: the root furthest toward instability.
      const rs = rootsOf(characteristic(num, den, k))
      const toward = (c: ComplexNumber) => (discrete ? Math.hypot(c.re, c.im) - 1 : c.re)
      let best = rs[0]
      for (const c of rs) if (toward(c) > toward(best)) best = c
      crossings.push({ gain: k, at: { re: best.re, im: Math.abs(best.im) } })
    }
  }
  return {
    gains: fromData(Float64Array.from(gains), [gains.length]),
    branches,
    openLoopPoles: P,
    openLoopZeros: Z,
    asymptotes: { centroid, angles },
    breakaway,
    crossings,
  }
}

/**
 * The closed-loop poles of $1 + kL = 0$ at one gain $k$ (the roots of $D + kN$).
 *
 * @param L The SISO open loop $L = N/D$.
 * @param gain The gain $k$.
 * @returns The closed-loop poles as `{ re, im }`.
 *
 * @example $L = 1/(s(s + 2))$: a double pole at $-1$ when $k = 1$, then $-1 \pm i$ at $k = 2$
 * const L = transferFunction([1], [1, 2, 0])
 * print('k = 1:', closedLoopPolesAt(L, 1))
 * print('k = 2:', closedLoopPolesAt(L, 2))
 */
export function closedLoopPolesAt(L: LtiSystem, gain: Scalar): ComplexNumber[] {
  const { num, den } = rationalOf(L)
  return rootsOf(characteristic(stripLeading([...num]), stripLeading([...den]), gain))
}
