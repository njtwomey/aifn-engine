/**
 * Numerically stable elementary forms: softplus, sigmoid, logit, logAddExp and the log1p/expm1 family. (log-sum-exp
 * and softmax are compositions of `aifn-compute/foundation/tensor` primitives; see index.ts.)
 *
 * The forms follow Mächler (2012), "Accurately computing log(1 − exp(−|a|))", CRAN Rmpfr vignette, for `log1mexp`.
 */

/** log(1 + e^x) without overflow or loss of the small-x tail. */
export function softplus(x: number): number {
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))
}

/** The logistic function 1 / (1 + e^{−x}), evaluated without overflow for either sign of x. */
export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x))
  const e = Math.exp(x)
  return e / (1 + e)
}

/** log σ(x) = −softplus(−x), accurate for large negative x where σ(x) underflows. */
export function logSigmoid(x: number): number {
  return -softplus(-x)
}

/** The log-odds log(p / (1 − p)) for p in [0, 1]; ±∞ at the ends. Uses log1p near p = 1/2 for accuracy. */
export function logit(p: number): number {
  // log(p) − log1p(−p) keeps relative accuracy for small p; near 1/2 the ratio form is exact enough either way.
  return p < 0.5 ? Math.log(p) - Math.log1p(-p) : Math.log(p / (1 - p))
}

/**
 * log(1 − e^x) for x ≤ 0 (Mächler 2012): `log(−expm1(x))` near 0, `log1p(−exp(x))` further out, switching at −ln 2.
 * Returns −∞ at x = 0 and NaN for x > 0.
 */
export function log1mexp(x: number): number {
  if (x > 0) return NaN
  return x > -Math.LN2 ? Math.log(-Math.expm1(x)) : Math.log1p(-Math.exp(x))
}

/** log(e^x − 1) for x > 0, without overflow for large x or loss for small x. NaN for x < 0. */
export function logExpm1(x: number): number {
  if (x < 0) return NaN
  return x > 36 ? x + Math.log1p(-Math.exp(-x)) : Math.log(Math.expm1(x))
}

/**
 * log(1 + x) − x for x > −1, accurate near 0 where the two terms cancel (a power series for |x| < 0.1). Used in the
 * incomplete-gamma prefactor for large shape.
 */
export function log1pmx(x: number): number {
  if (Math.abs(x) >= 0.1) return Math.log1p(x) - x
  // Σ_{k≥2} (−1)^{k+1} xᵏ/k = −x²/2 + x³/3 − …; about 16 terms reach double precision at |x| = 0.1.
  let term = x
  let sum = 0
  for (let k = 2; k < 40; k++) {
    term *= -x
    const add = term / k
    sum += add
    if (Math.abs(add) <= 1e-17 * Math.abs(sum)) break
  }
  return sum
}

/** log(e^a + e^b), handling −∞ in either argument (and both). */
export function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  const m = Math.max(a, b)
  if (m === Infinity) return Infinity
  return m + Math.log1p(Math.exp(-Math.abs(a - b)))
}

/** log(e^a − e^b) for a ≥ b; −∞ when a = b, NaN when a < b. */
export function logDiffExp(a: number, b: number): number {
  if (b === -Infinity) return a
  if (a < b) return NaN
  if (a === b) return -Infinity
  return a + log1mexp(b - a)
}

/**
 * Binary entropy H(p) = −p log p − (1 − p) log(1 − p), in nats by default or in the given `base` (2 for bits), with
 * 0 log 0 = 0. NaN outside [0, 1].
 */
export function binaryEntropy(p: number, base = Math.E): number {
  if (!(p >= 0 && p <= 1)) return NaN
  const xlogx = (q: number) => (q === 0 ? 0 : q * Math.log(q))
  const h = -xlogx(p) - (p < 0.5 ? (1 - p) * Math.log1p(-p) : xlogx(1 - p))
  return base === Math.E ? h : h / Math.log(base)
}
