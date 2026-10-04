/**
 * The standard normal distribution's special functions: density, cdf Φ and log Φ (accurate far into both tails), the
 * quantile Φ⁻¹, erfinv and erfcinv, and the truncated-normal moment functions v and w used by expectation propagation,
 * TrueSkill, the Bayes point machine and probit models.
 */

import { erf, erfc, erfcx, logErfc } from './erf'
import { logDiffExp } from './stable'

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI)
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI)
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)

/** e^{−z²/2}, with z² split so that its rounding error is not magnified for large |z| (see erf.ts). */
function expHalfSquare(z: number): number {
  const hi = Math.trunc(z * 16) / 16
  const lo = z - hi
  return Math.exp(-0.5 * hi * hi) * Math.exp(-0.5 * lo * (z + hi))
}

/** The standard normal density φ(z) = e^{−z²/2} / √(2π). */
export function normalPdf(z: number): number {
  return INV_SQRT_2PI * expHalfSquare(z)
}

/** log φ(z) = −z²/2 − log √(2π). */
export function normalLogPdf(z: number): number {
  return -0.5 * z * z - LOG_SQRT_2PI
}

/**
 * The standard normal cdf Φ(z) = erfc(−z/√2)/2, with full relative accuracy in the lower tail down to underflow
 * (about z = −37.5). For an upper-tail probability use Φ(−z), never 1 − Φ(z).
 */
export function normalCdf(z: number): number {
  return 0.5 * erfc(-z * Math.SQRT1_2)
}

/**
 * log Φ(z), accurate in both tails: in the lower tail through log erfc (no underflow even at z = −10⁵), in the upper
 * tail as log1p(−Φ(−z)).
 */
export function normalLogCdf(z: number): number {
  if (Number.isNaN(z)) return NaN
  if (z < -2) return -Math.LN2 + logErfc(-z * Math.SQRT1_2)
  if (z > 0) return Math.log1p(-0.5 * erfc(z * Math.SQRT1_2))
  return Math.log(0.5 * erfc(-z * Math.SQRT1_2))
}

// Wichura (1988), "Algorithm AS 241: The percentage points of the normal distribution", Applied Statistics 37,
// PPND16: rational approximations in three regions, relative accuracy about 1e-16. The published 20-digit coefficients
// are written here as the doubles they round to.
const A = [
  3.3871328727963665, 133.14166789178438, 1971.5909503065513, 13731.69376550946, 45921.95393154987, 67265.7709270087,
  33430.57558358813, 2509.0809287301227,
]
const B = [
  1, 42.31333070160091, 687.1870074920579, 5394.196021424751, 21213.794301586597, 39307.89580009271, 28729.085735721943,
  5226.495278852854,
]
const C = [
  1.4234371107496835, 4.630337846156546, 5.769497221460691, 3.6478483247632045, 1.2704582524523684, 0.2417807251774506,
  0.022723844989269184, 0.0007745450142783414,
]
const D = [
  1, 2.053191626637759, 1.6763848301838038, 0.6897673349851, 0.14810397642748008, 0.015198666563616457,
  0.0005475938084995345, 1.0507500716444169e-9,
]
const E = [
  6.657904643501103, 5.463784911164114, 1.7848265399172913, 0.29656057182850487, 0.026532189526576124,
  0.0012426609473880784, 2.7115555687434876e-5, 2.0103343992922881e-7,
]
const F = [
  1, 0.599832206555888, 0.1369298809227358, 0.014875361290850615, 0.0007868691311456133, 1.8463183175100548e-5,
  1.421511758316446e-7, 2.0442631033899397e-15,
]

function poly(c: number[], x: number): number {
  let s = c[c.length - 1]
  for (let i = c.length - 2; i >= 0; i--) s = s * x + c[i]
  return s
}

/**
 * The standard normal quantile Φ⁻¹(p) for p in [0, 1] (Wichura 1988, AS 241), relative accuracy about 1e-16;
 * Φ⁻¹(0) = −∞, Φ⁻¹(1) = +∞, NaN outside [0, 1].
 */
export function normalQuantile(p: number): number {
  if (!(p >= 0 && p <= 1)) return NaN
  if (p === 0) return -Infinity
  if (p === 1) return Infinity
  const q = p - 0.5
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q
    return (q * poly(A, r)) / poly(B, r)
  }
  let r = Math.sqrt(-Math.log(q < 0 ? p : 1 - p))
  let x: number
  if (r <= 5) {
    r -= 1.6
    x = poly(C, r) / poly(D, r)
  } else {
    r -= 5
    x = poly(E, r) / poly(F, r)
  }
  return q < 0 ? -x : x
}

/**
 * The inverse error function: erf(erfinv(y)) = y for y in [−1, 1]. Uses Φ⁻¹ for a first guess and two Newton steps on
 * erf (or on erfc for |y| > 1/2, where erf − y would cancel).
 */
export function erfinv(y: number): number {
  if (!(y >= -1 && y <= 1)) return NaN
  if (y === 0) return y
  if (Math.abs(y) > 0.5) {
    // erfinv(y) = erfcinv(1 − y); 1 − |y| is exact here (Sterbenz), so the tail keeps its accuracy.
    const x = erfcinv(1 - Math.abs(y))
    return y < 0 ? -x : x
  }
  let x = normalQuantile(0.5 + y / 2) * Math.SQRT1_2
  // Near 0, 0.5 + y/2 loses the low bits of y, so polish against erf itself (relative accuracy is kept there).
  for (let i = 0; i < 2; i++) x -= (erf(x) - y) / (TWO_OVER_SQRT_PI * Math.exp(-x * x))
  return x
}

/** The inverse complementary error function: erfc(erfcinv(z)) = z for z in [0, 2]; erfcinv(z) = −Φ⁻¹(z/2)/√2. */
export function erfcinv(z: number): number {
  if (!(z >= 0 && z <= 2)) return NaN
  // erfc(−x) = 2 − erfc(x), and 2 − z is exact for z in [1, 2].
  if (z > 1) return -erfcinv(2 - z)
  let x = -normalQuantile(z / 2) * Math.SQRT1_2
  if (!Number.isFinite(x)) return x
  // One Newton step on log erfc, which is well conditioned in the far tail: d log erfc(x)/dx = −2/(√π erfcx(x)).
  x -= (logErfc(x) - Math.log(z)) / (-TWO_OVER_SQRT_PI / erfcx(x))
  return x
}

/**
 * The truncated-normal mean function v(t) = φ(t)/Φ(t): the mean of a standard normal truncated to (−t, ∞), and the
 * derivative of log Φ(t). Computed as √(2/π)/erfcx(−t/√2), or for t < −3 by the continued fraction for the Mills
 * ratio (A&S 26.2.14), so it is accurate in both tails (v(t) → −t as t → −∞, v(t) → 0 as t → +∞).
 */
export function truncatedNormalV(t: number): number {
  if (t < -3) return -t + millsRemainder(-t)
  return SQRT_2_OVER_PI / erfcx(-t * Math.SQRT1_2)
}

/**
 * The truncated-normal variance function w(t) = v(t)(v(t) + t), in (0, 1): one minus the variance of a standard normal
 * truncated to (−t, ∞). For t < −3, v(t) + t is taken from the Mills-ratio continued fraction, avoiding the
 * cancellation of v(t) + t, so w → 1 accurately as t → −∞.
 */
export function truncatedNormalW(t: number): number {
  if (t < -3) {
    const k = millsRemainder(-t)
    return (-t + k) * k
  }
  const v = truncatedNormalV(t)
  return v * (v + t)
}

/**
 * For x > 0, 1/R(x) − x where R(x) = (1 − Φ(x))/φ(x) is the Mills ratio. From R(x) = 1/(x + 1/(x + 2/(x + 3/(x + …))))
 * (A&S 26.2.14), 1/R(x) − x = 1/(x + 2/(x + 3/(x + …))); evaluated by the modified Lentz method.
 */
function millsRemainder(x: number): number {
  let f = x
  let c = f
  let d = 0
  for (let j = 1; j < 1000; j++) {
    const a = j + 1
    d = x + a * d
    c = x + a / c
    d = 1 / d
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < 1e-16) break
  }
  return 1 / f
}

/** log(Φ(u) − Φ(l)) for l ≤ u, choosing the tail in which the difference does not cancel. */
export function normalLogIntervalProbability(l: number, u: number): number {
  if (u <= 0) return logDiffExp(normalLogCdf(u), normalLogCdf(l))
  if (l >= 0) return logDiffExp(normalLogCdf(-l), normalLogCdf(-u))
  // l < 0 < u. A wide interval: log1p(−Φ(l) − Φ(−u)), accurate as the result nears 0. A narrow one (tails above ½):
  // Φ(u) − Φ(l) = (erf(u/√2) + erf(−l/√2))/2, a sum of two positive terms, which keeps its relative accuracy where
  // 1 − Φ(l) − Φ(−u) would cancel.
  const tails = normalCdf(l) + normalCdf(-u)
  if (tails < 0.5) return Math.log1p(-tails)
  return Math.log(0.5 * (erf(u * Math.SQRT1_2) + erf(-l * Math.SQRT1_2)))
}

/**
 * The draw version of v (Herbrich, Minka and Graepel 2007, "TrueSkill", NIPS): the mean of a standard normal
 * truncated to [−ε − t, ε − t], i.e. (φ(−ε − t) − φ(ε − t)) / (Φ(ε − t) − Φ(−ε − t)). Odd in t; computed in log space
 * so it stays finite when both Φ values underflow.
 */
export function truncatedNormalVDraw(t: number, eps: number): number {
  if (t < 0) return -truncatedNormalVDraw(-t, eps)
  const u = eps - t
  const l = -eps - t
  const logZ = normalLogIntervalProbability(l, u)
  return Math.exp(normalLogPdf(l) - logZ) - Math.exp(normalLogPdf(u) - logZ)
}

/**
 * The draw version of w: one minus the variance of a standard normal truncated to [−ε − t, ε − t], i.e.
 * v² + ((ε − t)φ(ε − t) + (ε + t)φ(ε + t)) / (Φ(ε − t) − Φ(−ε − t)). Even in t; in (0, 1).
 */
export function truncatedNormalWDraw(t: number, eps: number): number {
  const a = Math.abs(t)
  const u = eps - a
  const l = -eps - a
  const logZ = normalLogIntervalProbability(l, u)
  const v = truncatedNormalVDraw(a, eps)
  return v * v + u * Math.exp(normalLogPdf(u) - logZ) - l * Math.exp(normalLogPdf(l) - logZ)
}
