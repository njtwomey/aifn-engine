/**
 * Sample autocorrelation and partial autocorrelation functions with their reference bands, part of
 * `aifn-compute/probability/stats`: the correlogram of time-series identification (Box, Jenkins and Reinsel, 2008), as
 * statsmodels' `acf` and `pacf`.
 */

import { levinsonDurbin } from 'aifn-compute/numerics/linalg'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { dense, type Vector } from 'aifn-compute/foundation/tensor'
import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { autocorrelation, autocovariance } from './sequence'
import { vectorOf } from './input'

/**
 * A vector argument as a plain array of numbers, through tensor's dense kernels (§4.3 #8: no private matrix helpers).
 *
 * @param v The vector: an array or a rank-1 tensor.
 * @param where The caller's name for error messages.
 * @returns A copy of its values.
 */
const toVec = (v: VectorLike, where: string): number[] => Array.from(dense.toF64(v, where))

/** The sample ACF with its reference bands. */
export type SampleAcf = {
  /** $\hat{\rho}(0), \dots, \hat{\rho}(\text{maxLag})$, with $\hat{\rho}(0) = 1$. */
  acf: Vector
  /**
   * The white-noise band $z/\sqrt{n}$: $\hat{\rho}(k)$ outside $\pm$band rejects $\rho(k) = 0$ at the chosen level
   * for an iid series.
   */
  band: number
  /**
   * The half-width of Bartlett's band at the chosen level: $z$ times Bartlett's standard error of $\hat{\rho}(k)$
   * under the hypothesis that the series is MA($k - 1$),
   * $z \sqrt{(1 + 2\sum_{j=1}^{k-1} \hat{\rho}(j)^2)/n}$ (Box, Jenkins & Reinsel, 2008, eq. 2.1.15), as statsmodels'
   * `acf(x, alpha=...)` intervals. Entry 0 is 0.
   */
  bartlett: Vector
}

/**
 * The sample autocorrelation $\hat{\rho}(k) = \hat{\gamma}(k)/\hat{\gamma}(0)$ for $k = 0, \dots, \text{maxLag}$,
 * with the biased (divisor $n$) autocovariance, as `aifn-compute/probability/stats`' `autocorrelation` computes it (FFT
 * for long series), plus the white-noise band $z/\sqrt{n}$ and Bartlett's band at the given `level` (default 0.95),
 * with $z$ the normal quantile at $\tfrac12 + \text{level}/2$.
 *
 * @param x The series: an array or a rank-1 tensor of $n$ values.
 * @param maxLag The largest lag; cut to $n - 1$.
 * @param options The level of the bands.
 * @param options.level The two-sided coverage of the bands, in $(0, 1)$.
 * @returns The `acf`, the white-noise `band` and Bartlett's band half-widths `bartlett`.
 *
 * @example As statsmodels' acf(x, nlags=3, alpha=0.05)
 * const r = sampleAcf([2, 4, 3, 5, 4, 6, 5, 7, 6, 8], 3)
 * print('acf =', r.acf)
 * print('white-noise band = +/-', r.band)
 * print('Bartlett half-widths =', r.bartlett)
 */
export function sampleAcf(x: VectorLike, maxLag: Size, { level = 0.95 }: { level?: number } = {}): SampleAcf {
  const xs = toVec(x, 'sampleAcf')
  const n = xs.length
  const lag = Math.min(maxLag, n - 1)
  const acf = dense.data(autocorrelation(xs, { maxLag: lag }))
  const z = normalQuantile(0.5 + level / 2) as number
  const bartlett = new Float64Array(lag + 1)
  let s = 0
  for (let k = 1; k <= lag; k++) {
    if (k > 1) s += acf[k - 1] ** 2
    bartlett[k] = z * Math.sqrt((1 + 2 * s) / n)
  }
  return { acf: vectorOf(Float64Array.from(acf)), band: z / Math.sqrt(n), bartlett: vectorOf(bartlett) }
}

/**
 * The sample partial autocorrelations $\hat{\phi}_{kk}$, $k = 1, \dots, \text{maxLag}$: the reflection coefficients
 * of Levinson–Durbin run on the sample autocovariances (the "Yule–Walker" PACF, statsmodels' `pacf(method='ywm')`).
 * Entry $k - 1$ is lag $k$.
 *
 * @param x The series: an array or a rank-1 tensor of $n$ values.
 * @param maxLag The largest lag; cut to $n - 1$.
 * @returns $\hat{\phi}_{kk}$ for $k = 1, \dots, \text{maxLag}$ (lag 0 is left out).
 *
 * @example As statsmodels' pacf(x, nlags=3, method='ywm')
 * print('pacf =', samplePacf([2, 4, 3, 5, 4, 6, 5, 7, 6, 8], 3))
 */
export function samplePacf(x: VectorLike, maxLag: Size): Vector {
  const xs = toVec(x, 'samplePacf')
  const lag = Math.min(maxLag, xs.length - 1)
  return levinsonDurbin(autocovariance(xs, { maxLag: lag }), lag).reflection
}
