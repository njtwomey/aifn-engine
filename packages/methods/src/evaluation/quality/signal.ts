/**
 * Image and audio quality: PSNR and SSIM (uniform or Gaussian window, with the local map), the remote-sensing
 * metrics SAM, ERGAS and RASE; SNR and scale-invariant SDR, and permutation-invariant scoring of separated sources.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { defineMetric, type Data, type Metric, type Rows } from 'aifn-compute/learning/metrics'
import {
  denseMatrix as dense,
  divide,
  isMatrixLike,
  sameLength,
  metricValues as values,
} from 'aifn-compute/learning/metrics'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** An image: a height × width matrix (rank-2 tensor or rows), or flat row-major data with `width` given. */
export type Image = Data | Rows

function image(x: Image, width: number | undefined, what: string): { h: number; w: number; data: Float64Array } {
  if (isMatrixLike(x)) {
    const d = dense(x as Rows, what)
    return { h: d.rows, w: d.cols, data: d.data }
  }
  const data = values(x as Data)
  const w = width ?? Math.round(Math.sqrt(data.length))
  if (data.length % w !== 0)
    throw new ShapeError('metrics', `metrics: ${what}: ${data.length} pixels do not fill rows of ${w}`)
  return { h: data.length / w, w, data }
}

/**
 * Peak signal-to-noise ratio 10 log₁₀(MAX²/MSE) in decibels (peak-signal-to-noise-ratio), with MAX = `dataRange` (255
 * for 8-bit images, 1 for images in [0, 1]); +∞ for identical images.
 */
export const psnr = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'psnr',
    name: 'Peak signal-to-noise ratio',
    inputs: 'images',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['peak-signal-to-noise-ratio'],
    capability: 'decide',
  },
  (reference: Image, distorted: Image, options: { dataRange: number }): number => {
    const a = values(isMatrixLike(reference) ? dense(reference as Rows, 'psnr').data : (reference as Data))
    const b = values(isMatrixLike(distorted) ? dense(distorted as Rows, 'psnr').data : (distorted as Data))
    sameLength(a, b, 'psnr')
    let mse = 0
    for (let i = 0; i < a.length; i++) mse += (a[i] - b[i]) ** 2
    mse /= a.length
    return 10 * Math.log10((options.dataRange * options.dataRange) / mse)
  },
)

/** Options of SSIM. */
export type SsimOptions = {
  /** The dynamic range L of the pixel values (255 for 8-bit, 1 for [0, 1]). */
  dataRange: number
  /**
   * `uniform` (default): a 7 × 7 box window with sample covariances, as scikit-image's `structural_similarity`
   * defaults. `gaussian`: Wang et al.'s 11 × 11 Gaussian window (σ = 1.5) with population covariances.
   */
  window?: 'uniform' | 'gaussian'
  /** Uniform window size (odd, default 7). */
  windowSize?: number
  /** Width, when the images are flat arrays. */
  width?: number
}

/**
 * The SSIM map and its mean (Wang et al. 2004; structural-similarity-index): at each pixel whose window lies inside the
 * image, ((2μₓμ_y + C₁)(2σₓ_y + C₂))/((μₓ² + μ_y² + C₁)(σₓ² + σ_y² + C₂)) with C₁ = (0.01L)², C₂ = (0.03L)²; border
 * pixels are NaN in the map and left out of the mean, as scikit-image crops them.
 */
export function ssimMap(reference: Image, distorted: Image, options: SsimOptions): { map: Tensor; mean: number } {
  const x = image(reference, options.width, 'ssim')
  const y = image(distorted, options.width, 'ssim')
  if (x.h !== y.h || x.w !== y.w) throw new ShapeError('metrics', 'metrics: ssim: images differ in shape')
  const gaussian = options.window === 'gaussian'
  const size = gaussian ? 11 : (options.windowSize ?? 7)
  const half = (size - 1) / 2
  const weights = new Float64Array(size * size)
  if (gaussian) {
    let t = 0
    for (let i = 0; i < size; i++)
      for (let j = 0; j < size; j++)
        t += weights[i * size + j] = Math.exp(-((i - half) ** 2 + (j - half) ** 2) / (2 * 1.5 * 1.5))
    for (let k = 0; k < weights.length; k++) weights[k] /= t
  } else weights.fill(1 / (size * size))
  // Sample covariances rescale the window's population moments by N/(N − 1) (uniform window only).
  const correction = gaussian ? 1 : (size * size) / (size * size - 1)
  const c1 = (0.01 * options.dataRange) ** 2
  const c2 = (0.03 * options.dataRange) ** 2
  const map = new Float64Array(x.h * x.w).fill(NaN)
  let sum = 0
  let used = 0
  for (let r = half; r < x.h - half; r++)
    for (let c = half; c < x.w - half; c++) {
      let mx = 0
      let my = 0
      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let i = -half; i <= half; i++)
        for (let j = -half; j <= half; j++) {
          const w = weights[(i + half) * size + j + half]
          const a = x.data[(r + i) * x.w + c + j]
          const b = y.data[(r + i) * x.w + c + j]
          mx += w * a
          my += w * b
          sxx += w * a * a
          syy += w * b * b
          sxy += w * a * b
        }
      const vx = (sxx - mx * mx) * correction
      const vy = (syy - my * my) * correction
      const cxy = (sxy - mx * my) * correction
      const s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
      map[r * x.w + c] = s
      sum += s
      used++
    }
  return { map: fromData(map, [x.h, x.w]), mean: divide(sum, used) }
}

/** The mean SSIM of two images (see `ssimMap`). */
export const ssim = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'ssim',
    name: 'Structural similarity index',
    inputs: 'images',
    direction: 'higher',
    range: [-1, 1],
    notes: ['structural-similarity-index'],
    capability: 'decide',
  },
  (reference: Image, distorted: Image, options: SsimOptions): number => ssimMap(reference, distorted, options).mean,
)

// ── Remote sensing ───────────────────────────────────────────────────────────────────────────────────────────────────

function bands(reference: Rows, estimate: Rows, what: string) {
  const R = dense(reference, what)
  const E = dense(estimate, what)
  if (R.rows !== E.rows || R.cols !== E.cols)
    throw new ShapeError('metrics', `metrics: ${what}: images differ in shape`)
  return { R, E, n: R.rows, K: R.cols }
}

/**
 * The spectral angle mapper (Kruse et al. 1993; remote-sensing-image-metrics): the mean over pixels of the angle
 * arccos(xᵀx̂/(‖x‖‖x̂‖)) between reference and estimated spectra, in radians. Pixels are rows, bands columns.
 */
export const spectralAngle = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'spectralAngle',
    name: 'Spectral angle mapper',
    inputs: 'images',
    direction: 'lower',
    range: [0, Math.PI],
    notes: ['remote-sensing-image-metrics'],
    capability: 'decide',
  },
  (reference: Rows, estimate: Rows): number => {
    const { R, E, n, K } = bands(reference, estimate, 'spectralAngle')
    let s = 0
    for (let i = 0; i < n; i++) {
      let dot = 0
      let a = 0
      let b = 0
      for (let k = 0; k < K; k++) {
        dot += R.data[i * K + k] * E.data[i * K + k]
        a += R.data[i * K + k] ** 2
        b += E.data[i * K + k] ** 2
      }
      s += Math.acos(Math.max(-1, Math.min(1, dot / Math.sqrt(a * b))))
    }
    return s / n
  },
)

/** Per-band RMSE and reference means. */
function bandErrors(reference: Rows, estimate: Rows, what: string) {
  const { R, E, n, K } = bands(reference, estimate, what)
  const rmse = new Float64Array(K)
  const mean = new Float64Array(K)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < K; k++) {
      rmse[k] += (R.data[i * K + k] - E.data[i * K + k]) ** 2 / n
      mean[k] += R.data[i * K + k] / n
    }
  return { rmse: rmse.map(Math.sqrt), mean, K }
}

/**
 * ERGAS (Wald 2002): 100 (h/l) √((1/K) Σₖ (RMSEₖ/μₖ)²), with μₖ the reference mean of band k and `ratio` = h/l the
 * ratio of high- to low-resolution pixel sizes.
 */
export const ergas = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'ergas',
    name: 'ERGAS',
    inputs: 'images',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['remote-sensing-image-metrics'],
    capability: 'decide',
  },
  (reference: Rows, estimate: Rows, options: { ratio: number }): number => {
    const { rmse, mean, K } = bandErrors(reference, estimate, 'ergas')
    let s = 0
    for (let k = 0; k < K; k++) s += (rmse[k] / mean[k]) ** 2
    return 100 * options.ratio * Math.sqrt(s / K)
  },
)

/** RASE: (100/μ) √((1/K) Σₖ RMSEₖ²), with μ the mean reference value over all bands. */
export const rase = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'rase',
    name: 'Relative average spectral error',
    inputs: 'images',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['remote-sensing-image-metrics'],
    capability: 'decide',
  },
  (reference: Rows, estimate: Rows): number => {
    const { rmse, mean, K } = bandErrors(reference, estimate, 'rase')
    const mu = mean.reduce((a, b) => a + b, 0) / K
    return (100 / mu) * Math.sqrt(rmse.reduce((s, v) => s + v * v, 0) / K)
  },
)

// ── Audio ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Signal-to-noise ratio 10 log₁₀(‖s‖²/‖s − ŝ‖²) in decibels (signal-to-noise-and-signal-to-distortion-ratios). */
export const snr = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'snr',
    name: 'Signal-to-noise ratio',
    inputs: 'signals',
    direction: 'higher',
    range: [-Infinity, Infinity],
    notes: ['signal-to-noise-and-signal-to-distortion-ratios'],
    capability: 'decide',
  },
  (reference: Data, estimate: Data): number => {
    const s = values(reference)
    const e = values(estimate)
    sameLength(s, e, 'snr')
    let signal = 0
    let noise = 0
    for (let i = 0; i < s.length; i++) {
      signal += s[i] ** 2
      noise += (s[i] - e[i]) ** 2
    }
    return 10 * Math.log10(signal / noise)
  },
)

/**
 * Scale-invariant signal-to-distortion ratio (Le Roux et al. 2019): with α = ŝᵀs/‖s‖², 10 log₁₀(‖αs‖²/‖αs − ŝ‖²) dB,
 * unchanged by any gain on the estimate. `zeroMean` removes each signal's mean first, as many toolkits do.
 */
export const siSdr = defineMetric(
  {
    module: 'applied/evaluation/quality',
    key: 'siSdr',
    name: 'Scale-invariant SDR',
    inputs: 'signals',
    direction: 'higher',
    range: [-Infinity, Infinity],
    notes: ['signal-to-noise-and-signal-to-distortion-ratios'],
    capability: 'decide',
  },
  (reference: Data, estimate: Data, options: { zeroMean?: boolean } = {}): number => {
    let s = values(reference)
    let e = values(estimate)
    sameLength(s, e, 'siSdr')
    if (options.zeroMean) {
      const ms = s.reduce((a, b) => a + b, 0) / s.length
      const me = e.reduce((a, b) => a + b, 0) / e.length
      s = s.map((v) => v - ms)
      e = e.map((v) => v - me)
    }
    let dot = 0
    let ss = 0
    for (let i = 0; i < s.length; i++) {
      dot += e[i] * s[i]
      ss += s[i] * s[i]
    }
    const alpha = dot / ss
    let target = 0
    let error = 0
    for (let i = 0; i < s.length; i++) {
      target += (alpha * s[i]) ** 2
      error += (alpha * s[i] - e[i]) ** 2
    }
    return 10 * Math.log10(target / error)
  },
)

/** Every permutation of 0 … n − 1 (Heap's algorithm). */
function permutations(n: number): number[][] {
  const a = Array.from({ length: n }, (_, i) => i)
  const out = [[...a]]
  const c = new Array<number>(n).fill(0)
  let i = 0
  while (i < n) {
    if (c[i] < i) {
      const k = i % 2 === 0 ? 0 : c[i]
      ;[a[k], a[i]] = [a[i], a[k]]
      out.push([...a])
      c[i]++
      i = 0
    } else c[i++] = 0
  }
  return out
}

/**
 * Permutation-invariant scoring (Yu et al. 2017; permutation-invariant-training): the S × S matrix of a pairwise metric
 * between estimate j (rows) and reference k (columns), and the assignment of estimates to references with the best
 * mean score (the maximum for a higher-is-better metric), by enumerating the S! permutations (S ≤ 8). Signals are
 * rows of the two S × T matrices.
 */
export function permutationInvariantScore(
  references: Rows,
  estimates: Rows,
  metric: Metric<(reference: Data, estimate: Data) => number> = siSdr,
): { score: number; permutation: number[]; matrix: Tensor } {
  const R = dense(references, 'permutationInvariantScore')
  const E = dense(estimates, 'permutationInvariantScore')
  const S = R.rows
  if (E.rows !== S || E.cols !== R.cols)
    throw new ShapeError('metrics', 'metrics: permutationInvariantScore: shapes differ')
  if (S > 8)
    throw new DomainError(
      'metrics',
      'metrics: permutationInvariantScore enumerates S! permutations; use an assignment solver for S > 8',
    )
  const T = R.cols
  const m = new Float64Array(S * S)
  for (let j = 0; j < S; j++)
    for (let k = 0; k < S; k++)
      m[j * S + k] = metric(R.data.subarray(k * T, (k + 1) * T), E.data.subarray(j * T, (j + 1) * T))
  const sign = metric.info.direction === 'higher' ? 1 : -1
  let best: number[] = []
  let bestScore = -Infinity
  for (const perm of permutations(S)) {
    // perm[j] is the reference assigned to estimate j.
    const score = perm.reduce((s, k, j) => s + m[j * S + k], 0) / S
    if (sign * score > sign * bestScore || best.length === 0) {
      bestScore = score
      best = perm
    }
  }
  return { score: bestScore, permutation: best, matrix: fromData(m, [S, S]) }
}
