/**
 * Power of the two-sided z- and t-tests against a standardised shift (the true effect in standard errors).
 */

import { logGamma, normalCdf, studentTQuantile } from 'aifn-compute/numerics/special'

/** Two-sided z-test power for a shift λ = δ√n (in standard errors) and critical value c: P(|Z + λ| > c). */
export function zTestPower(shift: number, crit: number): number {
  return 1 - normalCdf(crit - shift) + normalCdf(-crit - shift)
}

/**
 * Two-sided t-test power at level α: P(|T′| > t_{1−α/2, ν}) for a noncentral t with ν degrees of freedom and
 * noncentrality λ = `shift`. Integrates the z-test power over the χ²_ν law of ν s²/σ² (Simpson's rule, 800 panels).
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
