/**
 * The James–Stein estimator (James and Stein, 1961) of a normal mean in $p \ge 3$ dimensions, and its risk against
 * the maximum-likelihood estimate, simulated and in closed form: Stein's (1956) paradox.
 *
 * From one observation $\xvec \sim \Gauss(\thetavec, \sigma^2\Imat_p)$, the maximum-likelihood estimate $\xvec$ has
 * risk $\expect\norm{\xvec - \thetavec}^2 = p\sigma^2$ whatever $\thetavec$. Shrinking towards a fixed target
 * $\tvec$,
 *
 * $$
 * \hat{\thetavec}_{\mathrm{JS}} = \tvec
 *   + \left(1 - \frac{(p - 2)\sigma^2}{\norm{\xvec - \tvec}^2}\right)(\xvec - \tvec),
 * $$
 *
 * has risk $\sigma^2\left(p - (p - 2)^2\,\expect\left[1/\chi^2_p(\lambda)\right]\right)$, with $\chi^2_p(\lambda)$ a
 * noncentral chi-squared variable of noncentrality $\lambda = \norm{\thetavec - \tvec}^2/\sigma^2$, which is below
 * $p\sigma^2$ for every $\thetavec$ once $p \ge 3$: the MLE is inadmissible. At $\thetavec = \tvec$ the risk is
 * $2\sigma^2$. Shrinking towards the grand mean $\bar{x}\ones$ instead (Efron and Morris, 1973, 1975) spends one
 * dimension on the target: the constant becomes $p - 3$, the risk $\sigma^2\left(p - (p - 3)^2\,
 * \expect\left[1/\chi^2_{p-1}(\lambda)\right]\right)$ with $\lambda = \norm{\thetavec - \bar\theta\ones}^2/\sigma^2$,
 * and $3\sigma^2$ at its best. The positive-part estimator clips the factor at zero, so it never shrinks past the
 * target, and has smaller risk still. The expectation is the Poisson mixture
 * $\expect\left[1/\chi^2_\nu(\lambda)\right] = \sum_{j \ge 0} e^{-\lambda/2}(\lambda/2)^j / j! \cdot 1/(\nu + 2j - 2)$.
 */

import { child, standardNormals, type Stream } from 'aifn-compute/foundation/random'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** What `jamesStein` shrinks towards: zero, the grand mean of the observation, a constant, or a vector. */
export type ShrinkageTarget = 'zero' | 'mean' | number | ArrayLike<number>

/** Options of `jamesStein`. */
export interface JamesSteinOptions {
  /** The noise variance $\sigma^2$ of each coordinate (default 1). */
  sigma2?: number
  /**
   * The target $\tvec$: `'zero'` (default), `'mean'` (the grand mean $\bar{x}$, with constant $p - 3$), a constant,
   * or $p$ values.
   */
  target?: ShrinkageTarget
  /** Clip the shrinkage factor at zero, the positive-part estimator (default false). */
  positivePart?: boolean
}

/** A James–Stein estimate. */
export interface JamesSteinEstimate {
  /** The estimate $\hat{\thetavec}$, $p$ values. */
  readonly estimate: Float64Array
  /**
   * The factor $c = 1 - k\sigma^2 / \norm{\xvec - \tvec}^2$ applied to $\xvec - \tvec$ ($k = p - 2$, or $p - 3$
   * towards the mean); clipped at 0 for the positive part, and negative when the plain estimator overshoots the target.
   */
  readonly factor: number
  /** The target $\tvec$, $p$ values. */
  readonly target: Float64Array
  /** The constant $k$: $p - 2$, or $p - 3$ towards the mean. */
  readonly k: number
}

/**
 * The target vector and the constant $k$ for an observation. Throws `DomainError` when $p$ is too small for the
 * target, and `ShapeError` for a target vector of the wrong length.
 *
 * @param x The observation, $p$ values.
 * @param target The target.
 * @param where The caller's name, for error messages.
 * @returns The target vector and $k$.
 */
function targetOf(x: ArrayLike<number>, target: ShrinkageTarget, where: string): { t: Float64Array; k: number } {
  const p = x.length
  if (target === 'mean') {
    if (p < 4) throw new DomainError(where, `${where}: shrinking towards the mean needs p >= 4`)
    let mean = 0
    for (let i = 0; i < p; i++) mean += x[i] / p
    return { t: new Float64Array(p).fill(mean), k: p - 3 }
  }
  if (p < 3) throw new DomainError(where, `${where}: James-Stein shrinkage needs p >= 3`)
  if (target === 'zero') return { t: new Float64Array(p), k: p - 2 }
  if (typeof target === 'number') return { t: new Float64Array(p).fill(target), k: p - 2 }
  if (target.length !== p) throw new ShapeError(where, `${where}: a target of ${target.length} values for p = ${p}`)
  return { t: Float64Array.from(target), k: p - 2 }
}

/**
 * The James–Stein estimate of a normal mean from one observation $\xvec \sim \Gauss(\thetavec, \sigma^2\Imat_p)$:
 * $\hat{\thetavec} = \tvec + c(\xvec - \tvec)$ with $c = 1 - k\sigma^2/\norm{\xvec - \tvec}^2$, $k = p - 2$ for a
 * fixed target $\tvec$ (James and Stein, 1961) and $k = p - 3$ for the grand mean (Efron and Morris, 1973), and $c$
 * clipped at 0 with `positivePart`. When $\xvec = \tvec$ the estimate is the target. Throws `DomainError` for
 * $\sigma^2 \le 0$, for $p < 3$ ($p < 4$ towards the mean), and `ShapeError` for a target of the wrong length.
 *
 * @param x The observation $\xvec$, $p$ values (each an estimate of its own mean, with the same noise variance).
 * @param options The noise variance, the target and the positive-part switch, as `JamesSteinOptions`.
 * @returns The estimate, the factor $c$, the target and the constant $k$.
 *
 * @example Shrinking ten noisy means beats the observation itself
 * const theta = toFlat(normals(stream(0), [10]))
 * const noise = toFlat(normals(stream(1), [10]))
 * const x = theta.map((t, i) => t + noise[i])
 * const js = jamesStein(x, { target: 'mean', positivePart: true })
 * const loss = (e) => e.reduce((a, v, i) => a + (v - theta[i]) ** 2, 0)
 * print('factor =', js.factor.toFixed(3), ' towards', js.target[0].toFixed(3))
 * print('loss of x =', loss(x).toFixed(3), ' loss of James-Stein =', loss(js.estimate).toFixed(3))
 *
 * @example The plain estimator overshoots the target when the observation is near it; the positive part stops at it
 * const x = [0.3, -0.2, 0.1, 0.25]
 * print('plain:', jamesStein(x).estimate, ' factor', jamesStein(x).factor)
 * print('positive part:', jamesStein(x, { positivePart: true }).estimate)
 */
export function jamesStein(x: ArrayLike<number>, options: JamesSteinOptions = {}): JamesSteinEstimate {
  const { sigma2 = 1, target = 'zero', positivePart = false } = options
  if (!(sigma2 > 0) || !Number.isFinite(sigma2))
    throw new DomainError('jamesStein', 'jamesStein: sigma2 must be positive and finite')
  const { t, k } = targetOf(x, target, 'jamesStein')
  const p = x.length
  let s = 0
  for (let i = 0; i < p; i++) s += (x[i] - t[i]) ** 2
  let factor = s > 0 ? 1 - (k * sigma2) / s : -Infinity
  if (positivePart) factor = Math.max(0, factor)
  const estimate = Float64Array.from(t, (ti, i) => (s > 0 ? ti + factor * (x[i] - ti) : ti))
  return { estimate, factor, target: t, k }
}

/**
 * The mean of the reciprocal of a noncentral chi-squared variable, $\expect\left[1/\chi^2_\nu(\lambda)\right]$, as
 * the Poisson mixture $\sum_j e^{-\lambda/2}(\lambda/2)^j/j! \cdot 1/(\nu + 2j - 2)$, summed in log space so that a
 * large $\lambda$ does not underflow; $1/(\nu - 2)$ at $\lambda = 0$. Finite only for $\nu > 2$.
 *
 * @param nu The degrees of freedom $\nu > 2$.
 * @param lambda The noncentrality $\lambda \ge 0$.
 * @returns The expectation.
 */
export function inverseChiSquareMean(nu: number, lambda: number): number {
  if (lambda === 0) return 1 / (nu - 2)
  const half = lambda / 2
  const logHalf = Math.log(half)
  let logWeight = -half
  let total = 0
  for (let j = 0; ; j++) {
    if (j > 0) logWeight += logHalf - Math.log(j)
    const w = Math.exp(logWeight)
    total += w / (nu + 2 * j - 2)
    if (j > half && w < 1e-18) break
  }
  return total
}

/**
 * The exact risk $\expect\norm{\hat{\thetavec}_{\mathrm{JS}} - \thetavec}^2$ of the (plain) James–Stein estimator:
 * $\sigma^2\left(p - (p - 2)^2\expect\left[1/\chi^2_p(\lambda)\right]\right)$ towards a fixed target, with
 * $\lambda = \norm{\thetavec - \tvec}^2/\sigma^2$, and $\sigma^2\left(p - (p - 3)^2\expect\left[1/\chi^2_{p-1}(\lambda)
 * \right]\right)$ towards the grand mean, with $\lambda = \norm{\thetavec - \bar\theta\ones}^2/\sigma^2$. It rises from
 * $2\sigma^2$ (or $3\sigma^2$) at the target towards the MLE's $p\sigma^2$ as $\lambda$ grows, and never reaches it.
 * Throws `DomainError` for $p < 3$ ($p < 4$ towards the mean), $\sigma^2 \le 0$ or a negative distance.
 *
 * @param p The dimension $p$.
 * @param distance The distance $\norm{\thetavec - \tvec}$ of the mean from the target (from its own grand mean, for
 *   `'mean'`).
 * @param options The noise variance and the kind of target.
 * @param options.sigma2 The noise variance $\sigma^2$ (default 1).
 * @param options.target `'fixed'` (default; zero or any fixed vector) or `'mean'`.
 * @returns The risk.
 *
 * @example From 2 at the target towards p = 10
 * for (const d of [0, 1, 2, 4, 8, 16]) print('distance', d, ' risk =', jamesSteinExactRisk(10, d).toFixed(4))
 */
export function jamesSteinExactRisk(
  p: number,
  distance: number,
  { sigma2 = 1, target = 'fixed' }: { sigma2?: number; target?: 'fixed' | 'mean' } = {},
): number {
  const where = 'jamesSteinExactRisk'
  if (!(sigma2 > 0)) throw new DomainError(where, `${where}: sigma2 must be positive`)
  if (!(distance >= 0)) throw new DomainError(where, `${where}: distance must be non-negative`)
  const mean = target === 'mean'
  if (!(Number.isInteger(p) && p >= (mean ? 4 : 3)))
    throw new DomainError(where, `${where}: p must be an integer of at least ${mean ? 4 : 3}`)
  const k = mean ? p - 3 : p - 2
  const nu = mean ? p - 1 : p
  return sigma2 * (p - k * k * inverseChiSquareMean(nu, (distance * distance) / sigma2))
}

/** Options of `jamesSteinRisk`. */
export interface JamesSteinRiskOptions {
  /** The dimension $p$ (default 10). */
  p?: number
  /** The distances $\norm{\thetavec - \tvec}$ at which to estimate the risk (default 41, 0 to $2\sqrt{p\sigma^2}$). */
  norms?: readonly number[]
  /** The noise variance $\sigma^2$ (default 1). */
  sigma2?: number
  /** Observations drawn per distance (default 2000); the same noise serves every distance. */
  repeats?: number
  /** Shrink towards `'zero'` (default) or the grand `'mean'`. */
  target?: 'zero' | 'mean'
}

/** The risk of the MLE, James–Stein and its positive part against the distance of the mean from the target. */
export interface JamesSteinRisk {
  /** The distances $\norm{\thetavec - \tvec}$. */
  readonly norms: Float64Array
  /** The dimension $p$. */
  readonly p: number
  /** The noise variance $\sigma^2$. */
  readonly sigma2: number
  /** The simulated risk of the MLE $\xvec$ at each distance (the exact risk is $p\sigma^2$). */
  readonly mle: Float64Array
  /** The simulated risk of the James–Stein estimator. */
  readonly jamesStein: Float64Array
  /** The simulated risk of the positive-part estimator. */
  readonly positivePart: Float64Array
  /** The exact risk of the James–Stein estimator, `jamesSteinExactRisk`. */
  readonly exact: Float64Array
  /** The Monte Carlo standard errors of the three simulated risks. */
  readonly se: { readonly mle: Float64Array; readonly jamesStein: Float64Array; readonly positivePart: Float64Array }
}

/**
 * A risk simulation of Stein's paradox: for each distance $\norm{\thetavec - \tvec}$, draw `repeats` observations
 * $\xvec \sim \Gauss(\thetavec, \sigma^2\Imat_p)$ and average the squared error of the MLE, the James–Stein estimator
 * and its positive part, beside the exact risk of the James–Stein estimator. The mean lies along
 * $(\evec_1 - \evec_2)/\sqrt2$, which has grand mean zero, so the distance means the same for both targets. The noise
 * is drawn once from the stream's child `noise` and shared by every distance (common random numbers), so the curves
 * are smooth and their differences precise. Throws `DomainError` for $p$ too small for the target, a non-positive
 * $\sigma^2$, or a `repeats` that is not a positive integer.
 *
 * @param s The stream; the noise comes from its child `noise`, so `s` is not advanced.
 * @param options The dimension, distances, noise variance, repeats and target, as `JamesSteinRiskOptions`.
 * @returns The simulated risks with their standard errors, and the exact James–Stein risk, per distance.
 *
 * @example James-Stein beats the MLE everywhere in ten dimensions
 * const r = jamesSteinRisk(stream(0), { p: 10, norms: [0, 2, 4, 8], repeats: 4000 })
 * print('MLE:          ', Array.from(r.mle, (v) => v.toFixed(2)))
 * print('James-Stein:  ', Array.from(r.jamesStein, (v) => v.toFixed(2)))
 * print('exact:        ', Array.from(r.exact, (v) => v.toFixed(2)))
 * print('positive part:', Array.from(r.positivePart, (v) => v.toFixed(2)))
 */
export function jamesSteinRisk(s: Stream, options: JamesSteinRiskOptions = {}): JamesSteinRisk {
  const { p = 10, sigma2 = 1, repeats = 2000, target = 'zero' } = options
  const where = 'jamesSteinRisk'
  const mean = target === 'mean'
  if (!(Number.isInteger(p) && p >= (mean ? 4 : 3)))
    throw new DomainError(where, `${where}: p must be an integer of at least ${mean ? 4 : 3}`)
  if (!(sigma2 > 0) || !Number.isFinite(sigma2))
    throw new DomainError(where, `${where}: sigma2 must be positive and finite`)
  if (!(Number.isInteger(repeats) && repeats >= 1))
    throw new DomainError(where, `${where}: repeats must be a positive integer`)
  const sigma = Math.sqrt(sigma2)
  const norms = Float64Array.from(
    options.norms ?? Array.from({ length: 41 }, (_, i) => (i / 40) * 2 * Math.sqrt(p * sigma2)),
  )
  const z = standardNormals(child(s, 'noise'), repeats * p)
  const K = norms.length
  const out = {
    mle: new Float64Array(K),
    jamesStein: new Float64Array(K),
    positivePart: new Float64Array(K),
    exact: new Float64Array(K),
  }
  const se = { mle: new Float64Array(K), jamesStein: new Float64Array(K), positivePart: new Float64Array(K) }
  const theta = new Float64Array(p)
  const x = new Float64Array(p)
  const tgt = mean ? 'mean' : 'zero'
  for (let a = 0; a < K; a++) {
    theta.fill(0)
    theta[0] = norms[a] / Math.SQRT2
    theta[1] = -norms[a] / Math.SQRT2
    const sums = { mle: [0, 0], jamesStein: [0, 0], positivePart: [0, 0] }
    for (let r = 0; r < repeats; r++) {
      for (let i = 0; i < p; i++) x[i] = theta[i] + sigma * z[r * p + i]
      const plain = jamesStein(x, { sigma2, target: tgt })
      // The positive part rescales the same deviation, so it needs only the clipped factor.
      const clipped = Math.max(0, plain.factor)
      let lMle = 0
      let lJs = 0
      let lPp = 0
      for (let i = 0; i < p; i++) {
        lMle += (x[i] - theta[i]) ** 2
        lJs += (plain.estimate[i] - theta[i]) ** 2
        lPp += (plain.target[i] + clipped * (x[i] - plain.target[i]) - theta[i]) ** 2
      }
      sums.mle[0] += lMle
      sums.mle[1] += lMle * lMle
      sums.jamesStein[0] += lJs
      sums.jamesStein[1] += lJs * lJs
      sums.positivePart[0] += lPp
      sums.positivePart[1] += lPp * lPp
    }
    for (const key of ['mle', 'jamesStein', 'positivePart'] as const) {
      const m = sums[key][0] / repeats
      out[key][a] = m
      se[key][a] = repeats > 1 ? Math.sqrt(Math.max(0, sums[key][1] / repeats - m * m) / (repeats - 1)) : NaN
    }
    out.exact[a] = jamesSteinExactRisk(p, norms[a], { sigma2, target: mean ? 'mean' : 'fixed' })
  }
  return { norms, p, sigma2, ...out, se }
}
