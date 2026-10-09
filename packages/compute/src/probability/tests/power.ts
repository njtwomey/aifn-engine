/**
 * Power of the two-sided $z$- and $t$-tests against a standardised shift $\lambda$, the true effect in standard
 * errors: $\lambda = \delta\sqrt n$ for a one-sample test of a standardised effect $\delta$, and
 * $\lambda = \delta\sqrt{mn/(m + n)}$ for two samples. Power is the probability of rejecting when the shift is real,
 * for planning a sample size.
 */

import { logGamma, normalCdf, studentTQuantile } from 'aifn-compute/numerics/special'

/**
 * Two-sided $z$-test power for a shift $\lambda = \delta\sqrt n$ (in standard errors) and critical value $c$:
 * $\pr(\lvert Z + \lambda \rvert > c) = 1 - \Phi(c - \lambda) + \Phi(-c - \lambda)$.
 *
 * @param shift The shift $\lambda$ of the statistic's mean, in standard errors.
 * @param crit The critical value $c$ of $\lvert Z \rvert$, such as 1.96 for a 5% test.
 * @returns The power, in $[0, 1]$; at $\lambda = 0$ the test's size.
 *
 * @example A shift of 2.8 standard errors gives 80% power at 5%
 * print('power =', zTestPower(2.8, 1.96))
 * print('size =', zTestPower(0, 1.96))
 */
export function zTestPower(shift: number, crit: number): number {
  return 1 - normalCdf(crit - shift) + normalCdf(-crit - shift)
}

/**
 * Two-sided $t$-test power at level $\alpha$: $\pr(\lvert T_{\nu,\lambda} \rvert > t_{1-\alpha/2, \nu})$ for a
 * noncentral $t$ with $\nu$ degrees of freedom and noncentrality $\lambda$ = `shift`. Integrates the $z$-test power
 * over the $\chi^2_\nu$ law of $\nu s^2/\sigma^2$ (Simpson's rule, 800 panels).
 *
 * @param shift The noncentrality $\lambda$: the true effect in standard errors.
 * @param df The degrees of freedom $\nu$: $n - 1$ for one sample, $m + n - 2$ for two.
 * @param alpha The test's level $\alpha$.
 * @returns The power, at most 1.
 *
 * @example How many observations detect an effect of half a standard deviation?
 * // One sample of n: the shift is 0.5 sqrt(n) and df = n - 1 (the powers of scipy's noncentral t, `nct`).
 * for (const n of [10, 20, 34]) print('n =', n, ' power =', tTestPower(0.5 * Math.sqrt(n), n - 1, 0.05))
 * // Two groups of 64: the shift is 0.5 sqrt(64 / 2).
 * print('two groups of 64: power =', tTestPower(0.5 * Math.sqrt(32), 126, 0.05))
 */
export function tTestPower(shift: number, df: number, alpha: number): number {
  const crit = studentTQuantile(1 - alpha / 2, df)
  const logNorm = -(df / 2) * Math.log(2) - logGamma(df / 2)
  const density = (v: number) => (v <= 0 ? 0 : Math.exp(logNorm + (df / 2 - 1) * Math.log(v) - v / 2))
  const top = df + 12 * Math.sqrt(2 * df) + 20
  const steps = 800
  const h = top / steps
  let total = 0
  for (let i = 0; i <= steps; i++) {
    const v = i * h
    const f = density(v) * zTestPower(shift, crit * Math.sqrt(v / df))
    total += f * (i === 0 || i === steps ? 1 : i % 2 ? 4 : 2)
  }
  return Math.min(1, (total * h) / 3)
}
