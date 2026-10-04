/**
 * Numerically stable elementary forms: softplus, sigmoid, logit, logAddExp and the log1p/expm1 family. (log-sum-exp
 * and softmax are compositions of `aifn-compute/foundation/tensor` primitives; see index.ts.)
 *
 * The forms follow Mächler (2012), "Accurately computing $\log(1 - \exp(-|a|))$", CRAN Rmpfr vignette, for `log1mexp`.
 */

/**
 * $\log(1 + e^x)$ without overflow or loss of the small-$x$ tail.
 *
 * @param x - Real evaluation point.
 * @returns Value of $\operatorname{softplus}(x) = \log(1 + e^x)$.
 */
export function softplus(x: number): number {
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))
}

/**
 * The logistic function $1 / (1 + e^{-x})$, evaluated without overflow for either sign of $x$.
 *
 * @param x - Real evaluation point.
 * @returns Logistic sigmoid value $\sigma(x) \in (0, 1)$.
 */
export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x))
  const e = Math.exp(x)
  return e / (1 + e)
}

/**
 * $\log \sigma(x) = -\operatorname{softplus}(-x)$, accurate for large negative $x$ where $\sigma(x)$ underflows.
 *
 * @param x - Real evaluation point.
 * @returns Value of $\log \sigma(x)$.
 */
export function logSigmoid(x: number): number {
  return -softplus(-x)
}

/**
 * The log-odds $\log(p / (1 - p))$ for $p \in [0, 1]$; $\pm\infty$ at the ends. Uses $\operatorname{log1p}$ near $p = 1/2$ for accuracy.
 *
 * @param p - Probability $p \in [0, 1]$.
 * @returns Log-odds value $\operatorname{logit}(p)$.
 */
export function logit(p: number): number {
  // log(p) − log1p(−p) keeps relative accuracy for small p; near 1/2 the ratio form is exact enough either way.
  return p < 0.5 ? Math.log(p) - Math.log1p(-p) : Math.log(p / (1 - p))
}

/**
 * $\log(1 - e^x)$ for $x \le 0$ (Mächler 2012): `log(-expm1(x))` near 0, `log1p(-exp(x))` further out, switching at $-\ln 2$.
 * Returns $-\infty$ at $x = 0$ and `NaN` for $x > 0$.
 *
 * @param x - Non-positive evaluation point $x \le 0$.
 * @returns Value of $\log(1 - e^x)$.
 */
export function log1mexp(x: number): number {
  if (x > 0) return NaN
  return x > -Math.LN2 ? Math.log(-Math.expm1(x)) : Math.log1p(-Math.exp(x))
}

/**
 * $\log(e^x - 1)$ for $x > 0$, without overflow for large $x$ or loss for small $x$. `NaN` for $x < 0$.
 *
 * @param x - Positive evaluation point $x > 0$.
 * @returns Value of $\log(e^x - 1)$.
 */
export function logExpm1(x: number): number {
  if (x < 0) return NaN
  return x > 36 ? x + Math.log1p(-Math.exp(-x)) : Math.log(Math.expm1(x))
}

/**
 * $\log(1 + x) - x$ for $x > -1$, accurate near 0 where the two terms cancel (a power series for $|x| < 0.1$). Used in the
 * incomplete-gamma prefactor for large shape.
 *
 * @param x - Evaluation point $x > -1$.
 * @returns Value of $\log(1 + x) - x$.
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

/**
 * $\log(e^a + e^b)$, handling $-\infty$ in either argument (and both).
 *
 * @param a - First log-space argument.
 * @param b - Second log-space argument.
 * @returns Value of $\log(e^a + e^b)$.
 */
export function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  const m = Math.max(a, b)
  if (m === Infinity) return Infinity
  return m + Math.log1p(Math.exp(-Math.abs(a - b)))
}

/**
 * $\log(e^a - e^b)$ for $a \ge b$; $-\infty$ when $a = b$, `NaN` when $a < b$.
 *
 * @param a - First log-space argument.
 * @param b - Second log-space argument with $b \le a$.
 * @returns Value of $\log(e^a - e^b)$.
 */
export function logDiffExp(a: number, b: number): number {
  if (b === -Infinity) return a
  if (a < b) return NaN
  if (a === b) return -Infinity
  return a + log1mexp(b - a)
}

/**
 * Binary entropy $H(p) = -p \log p - (1 - p) \log(1 - p)$, in nats by default or in the given `base` (2 for bits), with
 * $0 \log 0 = 0$. `NaN` outside $[0, 1]$.
 *
 * @param p - Success probability $p \in [0, 1]$.
 * @param base - Logarithm base (defaults to $e$ for nats).
 * @returns Binary entropy value.
 */
export function binaryEntropy(p: number, base = Math.E): number {
  if (!(p >= 0 && p <= 1)) return NaN
  const xlogx = (q: number) => (q === 0 ? 0 : q * Math.log(q))
  const h = -xlogx(p) - (p < 0.5 ? (1 - p) * Math.log1p(-p) : xlogx(1 - p))
  return base === Math.E ? h : h / Math.log(base)
}
