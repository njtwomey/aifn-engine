/**
 * `aifn-compute/numerics/special`: special functions and numerically stable elementary forms, as primitives.
 *
 * Every function is defined once (design K §4.2): its scalar forward rule lives in the implementation files
 * (`erf.ts`, `normal.ts`, `gamma.ts`, `beta.ts`, `stable.ts`, `bessel.ts`), and each export below registers it with
 * `elementwise` from `aifn-compute/foundation/tensor` together with **one** derivative per argument, written with primitives.
 * So every function accepts numbers, tensors of any rank and traced values alike (`erf(0.5)` is a number,
 * `erf(matrix)` a matrix of the same shape; two- and three-argument functions broadcast, NumPy rules), and it is
 * differentiable to any order wherever its derivative is: logΓ → ψ → ψ₁ → ψ₂ …, Φ → φ, softplus → σ. Integer tensors
 * give float64 results.
 *
 * A partial derivative marked `null` (a shape parameter, an order, degrees of freedom) is an error when
 * differentiated (`NotDifferentiableError`), never a silent zero. Methods and sources are documented in the
 * implementation files.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import {
  add,
  cos,
  div,
  elementwise,
  exp,
  expm1,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  log1p,
  logsumexp,
  mul,
  neg,
  notEqualTo,
  shapeOfValue,
  sin,
  square,
  sub,
  where,
  type Binary,
  type Domain,
  type Ternary,
  type Tensor,
  type Traced,
  type Unary,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'
import * as B from './beta'
import * as I from './bessel'
import * as L from './elliptic'
import * as E from './erf'
import * as G from './gamma'
import * as N from './normal'
import * as S from './stable'

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)
const HALF_SQRT_PI = Math.sqrt(Math.PI) / 2
const SQRT_2PI = Math.sqrt(2 * Math.PI)

// ── Test domains ─────────────────────────────────────────────────────────────────────────────────────────────────────

const d = (lo: number, hi: number, integer = false): Domain => ({ lo, hi, integer })
const positive = d(0.5, 4)
const probability = d(0.05, 0.95)

/**
 * The input domains of the generated primitive tests, per function (one entry per argument, or one for all).
 * Functions not listed take the default [−2, 2].
 */
const domains: Readonly<Record<string, Domain | readonly Domain[]>> = {
  erfinv: d(-0.9, 0.9),
  erfcinv: d(0.1, 1.9),
  normalQuantile: probability,
  normalLogIntervalProbability: [d(-2, -0.5), d(0, 2)],
  truncatedNormalVDraw: [d(-2, 2), d(0.2, 2)],
  truncatedNormalWDraw: [d(-2, 2), d(0.2, 2)],
  logGamma: positive,
  gamma: positive,
  digamma: positive,
  trigamma: positive,
  polygamma: [d(1, 2, true), positive],
  besselRatio: d(0.1, 4),
  logBesselI0: d(0.1, 4),
  logBeta: positive,
  logFactorial: positive,
  logChoose: [d(3, 6), d(0.5, 2.5)],
  regularisedGammaP: [positive, d(0.2, 4)],
  regularisedGammaQ: [positive, d(0.2, 4)],
  logRegularisedGammaP: [positive, d(0.2, 4)],
  logRegularisedGammaQ: [positive, d(0.2, 4)],
  regularisedGammaPInverse: [positive, probability],
  regularisedGammaQInverse: [positive, probability],
  logRegularisedBeta: [positive, positive, probability],
  studentTLogCdf: [d(-2, 2), d(1, 8)],
  regularisedBeta: [positive, positive, probability],
  regularisedBetaInverse: [positive, positive, probability],
  studentTCdf: [d(-2, 2), d(1, 8)],
  studentTQuantile: [probability, d(1, 8)],
  chiSquareCdf: [d(0.2, 5), d(1, 6)],
  chiSquareSf: [d(0.2, 5), d(1, 6)],
  logit: probability,
  log1mexp: d(-3, -0.1),
  logExpm1: d(0.1, 3),
  log1pmx: d(-0.5, 2),
  logDiffExp: [d(1, 2), d(-2, 0.5)],
  xlogy: [d(-2, 2), d(0.1, 3)],
  xlog1py: [d(-2, 2), d(-0.5, 2)],
  binaryEntropy: probability,
  ellipk: d(0.05, 0.95),
  ellipe: d(0.05, 0.95),
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** A derivative written with primitives: the arguments, then the output y. */
type Derivative = ((...argsAndOutput: Value[]) => Value) | null

/** The site note that defines each function, where there is one (the primitive's `doc.note`). */
const NOTES: Readonly<Record<string, string>> = {
  gamma: 'gamma-function',
  logGamma: 'gamma-function',
  digamma: 'gamma-function',
  logFactorial: 'gamma-function',
  regularisedGammaP: 'gamma-distribution',
  regularisedGammaQ: 'gamma-distribution',
  logBeta: 'beta-distribution',
  regularisedBeta: 'beta-distribution',
  regularisedBetaInverse: 'beta-distribution',
  erf: 'gaussian-integral',
  erfc: 'gaussian-integral',
  normalCdf: 'gaussian-distribution',
  normalPdf: 'gaussian-distribution',
  normalLogPdf: 'gaussian-distribution',
  normalLogCdf: 'gaussian-distribution',
  normalQuantile: 'gaussian-distribution',
  studentTCdf: 'student-t-distribution',
  studentTQuantile: 'student-t-distribution',
  chiSquareCdf: 'chi-squared-distribution',
  chiSquareSf: 'chi-squared-distribution',
  sigmoid: 'logistic-regression',
  logSigmoid: 'logistic-regression',
  logit: 'logistic-regression',
  softplus: 'softmax-and-log-sum-exp',
  logAddExp: 'softmax-and-log-sum-exp',
  logDiffExp: 'softmax-and-log-sum-exp',
  xlogy: 'entropy',
  binaryEntropy: 'entropy',
  truncatedNormalV: 'expectation-propagation-truncated-gaussian',
  truncatedNormalW: 'expectation-propagation-truncated-gaussian',
}
const doc = (name: string) => (NOTES[name] ? { doc: { note: NOTES[name] } } : {})

/** Register a one-argument function under `numerics/special/<name>` with its derivative dy/dx(x, y). */
function unary(name: string, f: (x: number) => number, derivative: Derivative): Unary {
  return elementwise({
    id: `numerics/special/${name}`,
    f,
    derivative: [derivative],
    test: { domain: domains[name] },
    ...doc(name),
  })
}

/** Register a two-argument function with its partials ∂y/∂a(a, b, y) and ∂y/∂b(a, b, y). */
function binary(name: string, f: (a: number, b: number) => number, da: Derivative, db: Derivative): Binary {
  return elementwise({
    id: `numerics/special/${name}`,
    f,
    derivative: [da, db],
    test: { domain: domains[name] },
    ...doc(name),
  })
}

/** Register a three-argument function with its three partials. */
function ternary(
  name: string,
  f: (a: number, b: number, c: number) => number,
  derivative: readonly [Derivative, Derivative, Derivative],
): Ternary {
  return elementwise({ id: `numerics/special/${name}`, f, derivative, test: { domain: domains[name] }, ...doc(name) })
}

/**
 * `x` where `ok` holds and `safe` elsewhere, so that the unused branch of a `where` stays finite (a NaN or ∞ there
 * would turn a zero cotangent into NaN).
 */
const guarded = (ok: Value, x: Value, safe: number): Value => where(ok, x, safe)

// ── Densities used as derivatives (compositions of the primitives below) ─────────────────────────────────────────

/** log of the Gamma(a, 1) density, (a − 1) log x − x − log Γ(a), at x ≥ 0 (−∞ below). */
function logGammaDensity(a: Value, x: Value): Value {
  const ok = greaterEqual(x, 0)
  const xs = guarded(ok, x, 1)
  return where(ok, sub(sub(xlogy(sub(a, 1), xs), xs), logGamma(a)), -Infinity)
}

/** The Gamma(a, 1) density x^{a−1} e^{−x} / Γ(a) at x ≥ 0 (0 below): ∂P(a, x)/∂x. */
function gammaDensity(a: Value, x: Value): Value {
  return exp(logGammaDensity(a, x))
}

/** The χ²ₖ density at x: half the Gamma(k/2, 1) density at x/2. */
function chiSquareDensity(x: Value, k: Value): Value {
  return mul(0.5, gammaDensity(mul(0.5, k), mul(0.5, x)))
}

/** log of the Beta(a, b) density on [0, 1] (−∞ outside). */
function logBetaDensity(a: Value, b: Value, x: Value): Value {
  const ok = where(greaterEqual(x, 0), lessEqual(x, 1), 0)
  const xs = guarded(ok, x, 0.5)
  return where(ok, sub(add(xlogy(sub(a, 1), xs), xlog1py(sub(b, 1), neg(xs))), logBeta(a, b)), -Infinity)
}

/** The Beta(a, b) density on [0, 1] (0 outside): ∂I_x(a, b)/∂x. */
function betaDensity(a: Value, b: Value, x: Value): Value {
  return exp(logBetaDensity(a, b, x))
}

/** log of the Student t density with ν degrees of freedom at t (the standard normal's for ν = ∞). */
function logStudentTDensity(t: Value, df: Value): Value {
  const finite = less(df, Infinity)
  const nu = guarded(finite, df, 1)
  const half = mul(0.5, nu)
  const halfPlus = add(half, 0.5)
  const logDensity = sub(
    sub(sub(logGamma(halfPlus), logGamma(half)), mul(0.5, log(mul(Math.PI, nu)))),
    mul(halfPlus, log1p(div(square(t), nu))),
  )
  return where(finite, logDensity, normalLogPdf(t))
}

/** The Student t density with ν degrees of freedom at t (the standard normal density for ν = ∞). */
function studentTDensity(t: Value, df: Value): Value {
  return exp(logStudentTDensity(t, df))
}

// ── Error function family (erf.ts) ───────────────────────────────────────────────────────────────────────────────────

/** erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt, elementwise; relative error about 1e-15. */
export const erf: Unary = unary('erf', E.erf, (x) => mul(TWO_OVER_SQRT_PI, exp(neg(square(x)))))
/** erfc(x) = 1 − erf(x), elementwise, with full relative accuracy in the upper tail (to underflow near x = 27). */
export const erfc: Unary = unary('erfc', E.erfc, (x) => mul(-TWO_OVER_SQRT_PI, exp(neg(square(x)))))
/** erfcx(x) = e^{x²} erfc(x), the scaled complement, elementwise; no underflow for large x. */
export const erfcx: Unary = unary('erfcx', E.erfcx, (x, y) => sub(mul(mul(2, x), y), TWO_OVER_SQRT_PI))
/** log erfc(x), elementwise, accurate far beyond the underflow of erfc. */
export const logErfc: Unary = unary('logErfc', E.logErfc, (x) => div(-TWO_OVER_SQRT_PI, erfcx(x)))
/** The inverse error function on [−1, 1], elementwise. */
export const erfinv: Unary = unary('erfinv', N.erfinv, (_x, y) => mul(HALF_SQRT_PI, exp(square(y))))
/** The inverse complementary error function on [0, 2], elementwise. */
export const erfcinv: Unary = unary('erfcinv', N.erfcinv, (_x, y) => mul(-HALF_SQRT_PI, exp(square(y))))

// ── Standard normal (normal.ts) ──────────────────────────────────────────────────────────────────────────────────────

/** The standard normal density φ(z), elementwise. */
export const normalPdf: Unary = unary('normalPdf', N.normalPdf, (x, y) => neg(mul(x, y)))
/** log φ(z), elementwise. */
export const normalLogPdf: Unary = unary('normalLogPdf', N.normalLogPdf, (x) => neg(x))
/**
 * The standard normal cdf Φ(z), elementwise, relatively accurate in the lower tail to underflow; use Φ(−z) for upper
 * tails.
 */
export const normalCdf: Unary = unary('normalCdf', N.normalCdf, (x) => normalPdf(x))
/** log Φ(z), elementwise, accurate in both tails (no underflow in the lower tail). */
export const normalLogCdf: Unary = unary('normalLogCdf', N.normalLogCdf, (x) => truncatedNormalV(x))
/** The standard normal quantile Φ⁻¹(p), elementwise (Wichura 1988, AS 241). */
export const normalQuantile: Unary = unary('normalQuantile', N.normalQuantile, (_p, y) =>
  mul(SQRT_2PI, exp(mul(0.5, square(y)))),
)
/** log(Φ(u) − Φ(l)) for l ≤ u, elementwise over broadcast (l, u), without cancellation in either tail. */
export const normalLogIntervalProbability: Binary = binary(
  'normalLogIntervalProbability',
  N.normalLogIntervalProbability,
  (l, _u, y) => neg(exp(sub(normalLogPdf(l), y))),
  (_l, u, y) => exp(sub(normalLogPdf(u), y)),
)
/** v(t) = φ(t)/Φ(t), elementwise: the mean of a standard normal truncated to (−t, ∞); accurate in both tails. */
export const truncatedNormalV: Unary = unary('truncatedNormalV', N.truncatedNormalV, (t) => neg(truncatedNormalW(t)))
/**
 * w(t) = v(t)(v(t) + t) in (0, 1), elementwise: one minus the variance of the same truncation; accurate in both
 * tails.
 */
export const truncatedNormalW: Unary = unary('truncatedNormalW', N.truncatedNormalW, (t, w) => {
  // w' = v − w(2v + t), from w = v(v + t) and v' = −w.
  const v = truncatedNormalV(t)
  return sub(v, mul(w, add(mul(2, v), t)))
})
/**
 * The mean of a standard normal truncated to [−ε − t, ε − t] (the TrueSkill draw factor), elementwise over broadcast
 * (t, ε). Differentiable in t only.
 */
export const truncatedNormalVDraw: Binary = binary(
  'truncatedNormalVDraw',
  N.truncatedNormalVDraw,
  (t, eps) => neg(truncatedNormalWDraw(t, eps)),
  null,
)
/**
 * One minus the variance of a standard normal truncated to [−ε − t, ε − t], elementwise over broadcast (t, ε).
 * Differentiable in t only.
 */
export const truncatedNormalWDraw: Binary = binary(
  'truncatedNormalWDraw',
  N.truncatedNormalWDraw,
  (t, eps, w) => {
    // With X truncated to [l, u] = [−ε − t, ε − t] and Z = Φ(u) − Φ(l), d/dt E f(X) = (f(l)φ(l) − f(u)φ(u))/Z − v E f(X).
    // From f = x² and E X² = 1 − w + v², w' = (u²φ(u) − l²φ(l))/Z + v(1 + v² − 3w).
    const u = sub(eps, t)
    const l = neg(add(eps, t))
    const logZ = normalLogIntervalProbability(l, u)
    const v = truncatedNormalVDraw(t, eps)
    const tails = sub(mul(square(u), exp(sub(normalLogPdf(u), logZ))), mul(square(l), exp(sub(normalLogPdf(l), logZ))))
    return add(tails, mul(v, sub(add(1, square(v)), mul(3, w))))
  },
  null,
)

// ── Gamma family (gamma.ts) ──────────────────────────────────────────────────────────────────────────────────────────

/** log |Γ(x)|, elementwise; +∞ at the poles. */
export const logGamma: Unary = unary('logGamma', G.logGamma, (x) => digamma(x))
/** Γ(x), elementwise; NaN at the poles, +∞ above 171.62. */
export const gamma: Unary = unary('gamma', G.gamma, (x, y) => mul(y, digamma(x)))
/** ψ(x) = d/dx log Γ(x), elementwise. */
export const digamma: Unary = unary('digamma', G.digamma, (x) => trigamma(x))
/**
 * ψ₁(x) = d²/dx² log Γ(x), elementwise, for all real x except the poles. Its derivative is ψ₂(x) for x > 0 and,
 * below, the reflection ψ₁(x) = π²/sin²(πx) − ψ₁(1 − x) differentiated.
 */
export const trigamma: Unary = unary('trigamma', G.trigamma, (x) => {
  const right = greater(x, 0)
  const xr = guarded(right, x, 1)
  // x where x ≤ 0, and −1/2 (where sin(πx) = −1) elsewhere, so the unused branch stays finite.
  const xs = where(right, -0.5, x)
  const s = sin(mul(Math.PI, xs))
  const reflected = add(div(mul(-2 * Math.PI ** 3, cos(mul(Math.PI, xs))), mul(s, square(s))), polygamma(2, sub(1, xs)))
  return where(right, polygamma(2, xr), reflected)
})
/**
 * ψ⁽ⁿ⁾(x) for integer n ≥ 1 and x > 0, elementwise over broadcast (n, x). Differentiable in x (∂x = ψ⁽ⁿ⁺¹⁾), not n.
 */
export const polygamma: Binary = binary('polygamma', G.polygamma, null, (n, x) => polygamma(add(n, 1), x))
/** log B(a, b) for a, b > 0, elementwise over broadcast (a, b), accurate when either argument is large. */
export const logBeta: Binary = binary(
  'logBeta',
  G.logBeta,
  (a, b) => sub(digamma(a), digamma(add(a, b))),
  (a, b) => sub(digamma(b), digamma(add(a, b))),
)
/** log n! = log Γ(n + 1), elementwise. */
export const logFactorial: Unary = unary('logFactorial', G.logFactorial, (n) => digamma(add(n, 1)))
/** log (n choose k), elementwise over broadcast (n, k), accurate for large n and small k; −∞ outside 0 ≤ k ≤ n. */
export const logChoose: Binary = binary(
  'logChoose',
  G.logChoose,
  (n, k) => sub(digamma(add(n, 1)), digamma(add(sub(n, k), 1))),
  (n, k) => sub(digamma(add(sub(n, k), 1)), digamma(add(k, 1))),
)
/**
 * The regularised lower incomplete gamma function P(a, x), elementwise over broadcast (a, x), like
 * scipy.special.gammainc. Differentiable in x only (∂P/∂x is the Gamma(a, 1) density).
 */
export const regularisedGammaP: Binary = binary('regularisedGammaP', G.regularisedGammaP, null, (a, x) =>
  gammaDensity(a, x),
)
/** The regularised upper incomplete gamma Q(a, x) = 1 − P(a, x), accurate in the upper tail. Differentiable in x. */
export const regularisedGammaQ: Binary = binary('regularisedGammaQ', G.regularisedGammaQ, null, (a, x) =>
  neg(gammaDensity(a, x)),
)

/**
 * log P(a, x), elementwise over broadcast (a, x): accurate where P underflows (the lower tail, summed in log space) and
 * where P ≈ 1 (log1p of −Q). Differentiable in x (∂ log P/∂x is the Gamma(a, 1) density over P).
 */
export const logRegularisedGammaP: Binary = binary('logRegularisedGammaP', G.logRegularisedGammaP, null, (a, x, y) =>
  exp(sub(logGammaDensity(a, x), y)),
)
/** log Q(a, x), elementwise over broadcast (a, x): accurate in the upper tail and where Q ≈ 1. Differentiable in x. */
export const logRegularisedGammaQ: Binary = binary('logRegularisedGammaQ', G.logRegularisedGammaQ, null, (a, x, y) =>
  neg(exp(sub(logGammaDensity(a, x), y))),
)
/**
 * The inverse of P(a, x) in x, elementwise over broadcast (a, p), like scipy.special.gammaincinv: the p-quantile of
 * Gamma(a, 1), inverting Q above p = ½ so that upper quantiles keep their accuracy. Differentiable in p
 * (dx/dp = 1 / the Gamma(a, 1) density at x).
 */
export const regularisedGammaPInverse: Binary = binary(
  'regularisedGammaPInverse',
  G.regularisedGammaPInverse,
  null,
  (a, _p, y) => exp(neg(logGammaDensity(a, y))),
)
/**
 * The inverse of Q(a, x) in x, elementwise over broadcast (a, q), like scipy.special.gammainccinv: the inverse survival
 * function of Gamma(a, 1), accurate for q down to 1e-300. Differentiable in q (dx/dq = −1 / the density at x).
 */
export const regularisedGammaQInverse: Binary = binary(
  'regularisedGammaQInverse',
  G.regularisedGammaQInverse,
  null,
  (a, _q, y) => neg(exp(neg(logGammaDensity(a, y)))),
)

// ── Beta family and derived distribution functions (beta.ts) ─────────────────────────────────────────────────────────

/**
 * The regularised incomplete beta function I_x(a, b), elementwise over broadcast (a, b, x), like
 * scipy.special.betainc. Differentiable in x only (∂I/∂x is the Beta(a, b) density).
 */
export const regularisedBeta: Ternary = ternary('regularisedBeta', B.regularisedBeta, [
  null,
  null,
  (a, b, x) => betaDensity(a, b, x),
])
/**
 * The inverse of I_x(a, b) in x, elementwise over broadcast (a, b, p), like scipy.special.betaincinv. Differentiable in
 * p only (dx/dp = 1 / the Beta(a, b) density at x).
 */
export const regularisedBetaInverse: Ternary = ternary('regularisedBetaInverse', B.regularisedBetaInverse, [
  null,
  null,
  (a, b, _p, y) => div(1, betaDensity(a, b, y)),
])
/**
 * log I_x(a, b), elementwise over broadcast (a, b, x): accurate where I underflows and where I ≈ 1. Differentiable in x
 * only (the Beta(a, b) density over I).
 */
export const logRegularisedBeta: Ternary = ternary('logRegularisedBeta', B.logRegularisedBeta, [
  null,
  null,
  (a, b, x, y) => exp(sub(logBetaDensity(a, b, x), y)),
])
/** The Student t cdf, elementwise over broadcast (t, ν); ν = ∞ gives Φ. Accurate in both tails. Differentiable in t. */
export const studentTCdf: Binary = binary('studentTCdf', B.studentTCdf, (t, df) => studentTDensity(t, df), null)
/**
 * log of the Student t cdf, elementwise over broadcast (t, ν): the lower tail in log space (no underflow) and log1p of
 * the complement in the upper tail. Differentiable in t (the density over the cdf).
 */
export const studentTLogCdf: Binary = binary(
  'studentTLogCdf',
  B.studentTLogCdf,
  (t, df, y) => exp(sub(logStudentTDensity(t, df), y)),
  null,
)
/** The Student t quantile, elementwise over broadcast (p, ν). Differentiable in p. */
export const studentTQuantile: Binary = binary(
  'studentTQuantile',
  B.studentTQuantile,
  (_p, df, y) => div(1, studentTDensity(y, df)),
  null,
)
/** The chi-square cdf P(k/2, x/2), elementwise over broadcast (x, k). Differentiable in x. */
export const chiSquareCdf: Binary = binary('chiSquareCdf', B.chiSquareCdf, (x, k) => chiSquareDensity(x, k), null)
/**
 * The chi-square survival function Q(k/2, x/2), elementwise over broadcast (x, k); use it for p-values.
 * Differentiable in x.
 */
export const chiSquareSf: Binary = binary('chiSquareSf', B.chiSquareSf, (x, k) => neg(chiSquareDensity(x, k)), null)

// ── Stable elementary forms (stable.ts) ──────────────────────────────────────────────────────────────────────────────

/** softplus(x) = log(1 + eˣ), elementwise, without overflow; also known as log1pexp. */
export const softplus: Unary = unary('softplus', S.softplus, (x) => sigmoid(x))
/** log(1 + eˣ); the same primitive as {@link softplus}. */
export const log1pexp: Unary = softplus
/** The logistic sigmoid 1/(1 + e^{−x}), elementwise, stable for either sign. */
export const sigmoid: Unary = unary('sigmoid', S.sigmoid, (_x, y) => mul(y, sub(1, y)))
/** log σ(x) = −softplus(−x), elementwise. */
export const logSigmoid: Unary = unary('logSigmoid', S.logSigmoid, (x) => sigmoid(neg(x)))
/** log(p/(1 − p)), elementwise. */
export const logit: Unary = unary('logit', S.logit, (p) => div(1, mul(p, sub(1, p))))
/** log(1 − eˣ) for x ≤ 0, elementwise (Mächler 2012). */
export const log1mexp: Unary = unary('log1mexp', S.log1mexp, (x) => div(-1, expm1(neg(x))))
/** log(eˣ − 1) for x ≥ 0, elementwise. Both it and `log1mexp` have derivative −1/expm1(−x). */
export const logExpm1: Unary = unary('logExpm1', S.logExpm1, (x) => div(-1, expm1(neg(x))))
/** log(1 + x) − x, elementwise, accurate near 0. */
export const log1pmx: Unary = unary('log1pmx', S.log1pmx, (x) => neg(div(x, add(1, x))))
/** log(eᵃ + eᵇ), elementwise over broadcast (a, b), handling −∞. */
export const logAddExp: Binary = binary(
  'logAddExp',
  S.logAddExp,
  (a, _b, y) => exp(sub(a, y)),
  (_a, b, y) => exp(sub(b, y)),
)
/** log(eᵃ − eᵇ) for a ≥ b, elementwise over broadcast (a, b). */
export const logDiffExp: Binary = binary(
  'logDiffExp',
  S.logDiffExp,
  (a, _b, y) => exp(sub(a, y)),
  (_a, b, y) => neg(exp(sub(b, y))),
)

// ── Elementwise products with logarithms and Bessel functions (bessel.ts) ───────────────────────────────────────────

/**
 * 0 where `x` is 0, else `f(nonzero)`, where `nonzero` masks the entries of `x` that are not 0: the derivative of
 * x·log(·) in its second argument, which is 0 wherever x is (the scipy convention 0 · log 0 = 0).
 */
const zeroWhereZero = (x: Value, f: (nonzero: Value) => Value): Value => {
  const nonzero = notEqualTo(x, 0)
  return where(nonzero, f(nonzero), 0)
}

/** x · log y with 0 · log y = 0 (so 0 · log 0 = 0), elementwise over broadcast (x, y); as `scipy.special.xlogy`. */
export const xlogy: Binary = binary(
  'xlogy',
  (x, y) => (x === 0 && y === y ? 0 : x * Math.log(y)),
  (_x, y) => log(y),
  (x, y) => zeroWhereZero(x, (nonzero) => div(x, guarded(nonzero, y, 1))),
)
/** x · log(1 + y) with 0 · log(1 + y) = 0, elementwise over broadcast (x, y); as `scipy.special.xlog1py`. */
export const xlog1py: Binary = binary(
  'xlog1py',
  (x, y) => (x === 0 && y === y ? 0 : x * Math.log1p(y)),
  (_x, y) => log1p(y),
  (x, y) => zeroWhereZero(x, (nonzero) => div(x, add(1, guarded(nonzero, y, 0)))),
)
/** The modified Bessel function of the first kind I₀(x), elementwise (even in x). Its derivative is I₁. */
export const besselI0: Unary = unary('besselI0', I.besselI0, (x) => besselI1(x))
/** The modified Bessel function of the first kind I₁(x), elementwise (odd in x); I₁′(x) = I₀(x) − I₁(x)/x (½ at 0). */
export const besselI1: Unary = unary('besselI1', I.besselI1, (x, y) => {
  const nonzero = notEqualTo(x, 0)
  return where(nonzero, sub(besselI0(x), div(y, guarded(nonzero, x, 1))), 0.5)
})
/**
 * A(κ) = I₁(κ)/I₀(κ) for κ ≥ 0 (NaN below), elementwise: the mean resultant length of a von Mises distribution.
 * A′(κ) = 1 − A/κ − A² (½ at 0).
 */
export const besselRatio: Unary = unary('besselRatio', I.besselRatio, (x, a) => {
  const nonzero = notEqualTo(x, 0)
  return where(nonzero, sub(sub(1, div(a, guarded(nonzero, x, 1))), square(a)), 0.5)
})
/** log I₀(κ) for κ ≥ 0 (NaN below), without overflow; its derivative is `besselRatio`. */
export const logBesselI0: Unary = unary('logBesselI0', I.logBesselI0, (x) => besselRatio(x))

/**
 * The complete elliptic integral of the first kind K(m) = ∫₀^{π/2} (1 − m sin²θ)^{−½} dθ, elementwise in the parameter
 * m < 1 (as `scipy.special.ellipk`); K′(m) = (E − (1 − m)K) / (2m(1 − m)) (π/8 at 0).
 */
export const ellipk: Unary = unary('ellipk', L.ellipk, (m, k) => {
  const nonzero = notEqualTo(m, 0)
  const ms = guarded(nonzero, m, 0.5)
  return where(nonzero, div(sub(ellipe(ms), mul(sub(1, ms), k)), mul(mul(2, ms), sub(1, ms))), Math.PI / 8)
})
/**
 * The complete elliptic integral of the second kind E(m) = ∫₀^{π/2} (1 − m sin²θ)^{½} dθ, elementwise in m ≤ 1 (as
 * `scipy.special.ellipe`); E′(m) = (E − K) / (2m) (−π/8 at 0).
 */
export const ellipe: Unary = unary('ellipe', L.ellipe, (m, e) => {
  const nonzero = notEqualTo(m, 0)
  const ms = guarded(nonzero, m, 0.5)
  return where(nonzero, div(sub(e, ellipk(ms)), mul(2, ms)), -Math.PI / 8)
})
/** Scalar elliptic functions (numbers in and out): K(1 − p), F(φ | m), the Jacobi functions and Carlson's forms. */
export { carlsonRD, carlsonRF, ellipf, ellipj, ellipkm1, type Jacobi } from './elliptic'

/** Options of `softmax` and `logSoftmax`. */
export type SoftmaxOptions = {
  /** The axis normalised over (default −1, the last). */
  axis?: number
  /** Temperature T > 0 (default 1): the result is softmax(x/T). */
  temperature?: number
}

/**
 * log softmax(x/T) along `axis` (default the last): xᵢ/T − log Σⱼ e^{xⱼ/T}, same shape as x (rank ≥ 1). Entries equal
 * to −∞ give −∞; a lane that is all −∞ gives NaN (there is no distribution). A composition of primitives, so it is
 * differentiable to any order.
 */
export function logSoftmax(x: Tensor, options?: SoftmaxOptions): Tensor
export function logSoftmax(x: Traced, options?: SoftmaxOptions): Traced
export function logSoftmax(x: Value, options?: SoftmaxOptions): Value
export function logSoftmax(x: Value, { axis = -1, temperature = 1 }: SoftmaxOptions = {}): Value {
  if (shapeOfValue(x).length === 0) throw new ShapeError('logSoftmax', 'logSoftmax: needs a tensor of rank ≥ 1', [[]])
  const z = temperature === 1 ? x : div(x, temperature)
  return sub(z, logsumexp(z, axis, true))
}

/**
 * softmax(x/T) along `axis` (default the last): e^{xᵢ/T} / Σⱼ e^{xⱼ/T}, same shape as x (rank ≥ 1), each lane summing
 * to 1; stable (computed as exp of `logSoftmax`). Entries equal to −∞ get probability 0; a lane that is all −∞ gives
 * NaN.
 */
export function softmax(x: Tensor, options?: SoftmaxOptions): Tensor
export function softmax(x: Traced, options?: SoftmaxOptions): Traced
export function softmax(x: Value, options?: SoftmaxOptions): Value
export function softmax(x: Value, options: SoftmaxOptions = {}): Value {
  return exp(logSoftmax(x, options))
}

const binaryEntropyNats: Unary = unary(
  'binaryEntropy',
  (p) => S.binaryEntropy(p),
  (p) => neg(logit(p)),
)

/**
 * Binary entropy H(p) = −p log p − (1 − p) log(1 − p) elementwise, with 0 log 0 = 0 and NaN outside [0, 1], in nats,
 * or in the given `base` (2 for bits).
 */
export function binaryEntropy(p: number, base?: number): number
export function binaryEntropy(p: Tensor, base?: number): Tensor
export function binaryEntropy(p: Traced, base?: number): Traced
export function binaryEntropy(p: Value, base?: number): Value
export function binaryEntropy(p: Value, base = Math.E): Value {
  const h = binaryEntropyNats(p)
  return base === Math.E ? h : div(h, Math.log(base))
}

// ── Registry of the composite functions ──────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'numerics/special')
const SOFTMAX = ['softmax-and-log-sum-exp']

fn(
  {
    key: 'softmax',
    name: 'Softmax',
    tex: '\\operatorname{softmax}(x)_i = e^{x_i/T} / \\sum_j e^{x_j/T}',
    role: 'transform',
    notes: [...SOFTMAX, 'multinomial-logistic-regression'],
  },
  softmax,
)
fn({ key: 'logSoftmax', name: 'Log-softmax', role: 'transform', notes: SOFTMAX }, logSoftmax)
fn({ key: 'binaryEntropy', name: 'Binary entropy', tex: 'H_b(p)', role: 'property', notes: ['entropy'] }, binaryEntropy)

/** The composite functions of the module, keyed by name. */
export const specialFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', {
    softmax: softmax,
    logSoftmax: logSoftmax,
    binaryEntropy: binaryEntropy,
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
