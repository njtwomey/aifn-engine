/** Sample autocorrelation and partial autocorrelation functions with their reference bands, part of `aifn-compute/probability/stats`. */

import { levinsonDurbin } from 'aifn-compute/numerics/linalg'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { dense, type Vector } from 'aifn-compute/foundation/tensor'
import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { autocorrelation, autocovariance } from './sequence'
import { vectorOf } from './input'

/** A vector argument as a Float64Array, through tensor's dense kernels (§4.3 #8: no private matrix helpers). */
const toVec = (v: VectorLike, where: string): number[] => Array.from(dense.toF64(v, where))

/** The sample ACF with its reference bands. */
export type SampleAcf = {
  /** ρ̂(0) … ρ̂(maxLag), with ρ̂(0) = 1. */
  acf: Vector
  /** The white-noise band: ρ̂(k) outside ±band rejects ρ(k) = 0 at the chosen level for an iid series (z/√n). */
  band: number
  /**
   * Bartlett's standard error of ρ̂(k) under the hypothesis that the series is MA(k − 1):
   * √((1 + 2Σ_{j<k} ρ̂(j)²)/n) (Box, Jenkins & Reinsel, 2008, eq. 2.1.15). Entry 0 is 0.
   */
  bartlett: Vector
}

/**
 * The sample autocorrelation ρ̂(k) = γ̂(k)/γ̂(0) for k = 0 … maxLag, with the biased (÷ n) autocovariance, as
 * `aifn-compute/probability/stats`' `autocorrelation` computes it (FFT for long series), plus the white-noise band z/√n and Bartlett's
 * standard errors at the given `level` (default 0.95).
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
 * The sample partial autocorrelations φ̂_kk, k = 1 … maxLag: the reflection coefficients of Levinson–Durbin run on the
 * sample autocovariances (the "Yule–Walker" PACF, statsmodels' `pacf(method='ywm')`). Entry k − 1 is lag k.
 */
export function samplePacf(x: VectorLike, maxLag: Size): Vector {
  const xs = toVec(x, 'samplePacf')
  const lag = Math.min(maxLag, xs.length - 1)
  return levinsonDurbin(autocovariance(xs, { maxLag: lag }), lag).reflection
}
