/**
 * Image and audio quality: PSNR and SSIM (uniform or Gaussian window, with the local map), the remote-sensing
 * metrics SAM, ERGAS and RASE; SNR and scale-invariant SDR, and permutation-invariant scoring of separated sources.
 *
 * Every metric compares a reference with a distorted or estimated version of it, in that order. Images are
 * $\text{height} \times \text{width}$ matrices or flat row-major arrays; multispectral images are matrices with
 * pixels as rows and bands as columns; audio signals are flat arrays of samples. Ratios are in decibels,
 * $10 \log_{10}$ of a power ratio. Inputs of different shapes throw `ShapeError`.
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

/**
 * An image: a $\text{height} \times \text{width}$ matrix (rank-2 tensor or rows), or flat row-major data with
 * `width` given.
 */
export type Image = Data | Rows

/**
 * An image as its height, width and row-major pixels. A flat image of a length that `width` does not divide throws
 * `ShapeError`.
 *
 * @param x The image: a matrix, or flat row-major data.
 * @param width The width of a flat image; left out, the image is taken as square ($\sqrt{n}$ rounded). Ignored for a
 *   matrix.
 * @param what The caller's name for the image, for error messages.
 * @returns `h` and `w`, the height and width, and `data`, a row-major copy of the pixels.
 */
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
 * Peak signal-to-noise ratio $10 \log_{10}(\text{MAX}^2 / \text{MSE})$ in decibels (peak-signal-to-noise-ratio), with
 * $\text{MAX}$ = `dataRange`; $+\infty$ for identical images. It is negative when the mean squared error exceeds
 * $\text{MAX}^2$, although the metric's `info.range` starts at 0.
 *
 * @param reference The reference image, as a matrix or flat; only its pixels are read, in row-major order.
 * @param distorted The distorted image, with as many pixels as `reference`.
 * @param options `dataRange`, the largest possible pixel value $\text{MAX}$ (255 for 8-bit images, 1 for images in
 *   $[0, 1]$).
 * @returns The PSNR, in decibels.
 *
 * @example An 8-bit image off by 5 everywhere, and by 10
 * const reference = [[100, 120], [140, 160]]
 * print('off by 5:', psnr(reference, [[105, 125], [145, 165]], { dataRange: 255 }), 'dB')
 * print('off by 10:', psnr(reference, [[110, 130], [150, 170]], { dataRange: 255 }), 'dB')
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
  /** The dynamic range $L$ of the pixel values (255 for 8-bit, 1 for $[0, 1]$). */
  dataRange: number
  /**
   * `uniform` (default): a $7 \times 7$ box window with sample covariances, as scikit-image's `structural_similarity`
   * defaults. `gaussian`: Wang et al.'s $11 \times 11$ Gaussian window ($\sigma = 1.5$) with population covariances.
   */
  window?: 'uniform' | 'gaussian'
  /** Uniform window size (odd, default 7); ignored by the Gaussian window. */
  windowSize?: number
  /** Width, when the images are flat arrays (default: the images are square). */
  width?: number
}

/**
 * The SSIM map and its mean (Wang et al. 2004; structural-similarity-index): at each pixel whose window lies inside the
 * image, $\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$
 * with the window's means, variances and covariance, $C_1 = (0.01L)^2$ and $C_2 = (0.03L)^2$; border pixels are NaN in
 * the map and left out of the mean, as scikit-image crops them. Images of different shapes throw `ShapeError`.
 *
 * @param reference The reference image $x$.
 * @param distorted The distorted image $y$, of the same shape.
 * @param options The data range $L$, the window and, for flat images, the width (see `SsimOptions`).
 * @returns `map`, the local SSIM at each pixel ($\text{height} \times \text{width}$, NaN within half a window of the
 *   border), and `mean`, its mean over the pixels that have a value (NaN when the image is smaller than the window).
 *
 * @example A smooth $9 \times 9$ image and a noisy copy
 * const s = stream(0)
 * const reference = Array.from({ length: 9 }, (_, r) => Array.from({ length: 9 }, (_, c) => 10 * (r + c)))
 * const noisy = reference.map((row) => row.map((v) => v + normal(s, 0, 10)))
 * const { map, mean } = ssimMap(reference, noisy, { dataRange: 255 })
 * print('mean SSIM =', mean)
 * print('at the centre =', map.data[4 * 9 + 4], ' at a corner =', map.data[0])
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

/**
 * The mean SSIM of two images (see `ssimMap`): 1 for identical images.
 *
 * @param reference The reference image.
 * @param distorted The distorted image, of the same shape.
 * @param options The data range, the window and, for flat images, the width (see `SsimOptions`).
 * @returns The mean SSIM over the pixels whose window lies inside the image, in $[-1, 1]$.
 *
 * @example More noise, lower SSIM, with either window
 * const s = stream(0)
 * const reference = Array.from({ length: 12 }, (_, r) => Array.from({ length: 12 }, (_, c) => 10 * (r + c)))
 * const noisy = (sd) => reference.map((row) => row.map((v) => v + normal(s, 0, sd)))
 * print('noise 5:', ssim(reference, noisy(5), { dataRange: 255 }))
 * print('noise 30:', ssim(reference, noisy(30), { dataRange: 255 }))
 * print('noise 30, Gaussian window:', ssim(reference, noisy(30), { dataRange: 255, window: 'gaussian' }))
 */
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

/**
 * Two multispectral images as row-major data, checked to have the same shape (else `ShapeError`).
 *
 * @param reference The reference image, pixels as rows and bands as columns.
 * @param estimate The estimated image, of the same shape.
 * @param what The caller's name, for error messages.
 * @returns `R` and `E`, the two images, `n`, the number of pixels, and `K`, the number of bands.
 */
function bands(reference: Rows, estimate: Rows, what: string) {
  const R = dense(reference, what)
  const E = dense(estimate, what)
  if (R.rows !== E.rows || R.cols !== E.cols)
    throw new ShapeError('metrics', `metrics: ${what}: images differ in shape`)
  return { R, E, n: R.rows, K: R.cols }
}

/**
 * The spectral angle mapper (Kruse et al. 1993; remote-sensing-image-metrics): the mean over pixels of the angle
 * $\arccos(\xvec^\top\hat{\xvec} / (\lVert \xvec \rVert \lVert \hat{\xvec} \rVert))$ between reference and estimated
 * spectra, in radians. Unchanged by a gain on any pixel's spectrum; NaN when a spectrum is all zero.
 *
 * @param reference The reference image, an $n \times K$ matrix: pixels as rows, bands as columns.
 * @param estimate The estimated image, of the same shape.
 * @returns The mean angle, in $[0, \pi]$ radians.
 *
 * @example A scaled spectrum has angle 0; a changed one does not
 * const reference = [[1, 2, 3], [3, 2, 1]]
 * print('scaled:', spectralAngle(reference, [[2, 4, 6], [6, 4, 2]]))
 * print('changed:', spectralAngle(reference, [[1, 2, 3], [1, 2, 3]]))
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

/**
 * Per-band RMSE and reference means of two multispectral images.
 *
 * @param reference The reference image, pixels as rows and bands as columns.
 * @param estimate The estimated image, of the same shape.
 * @param what The caller's name, for error messages.
 * @returns `rmse` and `mean`, one value per band, and `K`, the number of bands.
 */
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
 * ERGAS (Wald 2002): $100 \frac{h}{l} \sqrt{\frac{1}{K} \sum_k (\text{RMSE}_k / \mu_k)^2}$, with $\mu_k$ the
 * reference mean of band $k$ and $h/l$ the ratio of high- to low-resolution pixel sizes. Relative errors, so a band
 * whose reference mean is 0 makes it infinite.
 *
 * @param reference The reference image, an $n \times K$ matrix: pixels as rows, bands as columns.
 * @param estimate The estimated (fused or sharpened) image, of the same shape.
 * @param options `ratio`, $h/l$ (such as $1/4$ for a pan-sharpening by 4).
 * @returns ERGAS; 0 for a perfect estimate.
 *
 * @example A 10% error in one of two bands
 * const reference = [[100, 50], [100, 50], [100, 50]]
 * const estimate = [[110, 50], [90, 50], [110, 50]]
 * print('ERGAS =', ergas(reference, estimate, { ratio: 1 / 4 }))
 * print('RASE =', rase(reference, estimate))
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

/**
 * RASE, the relative average spectral error: $\frac{100}{\mu} \sqrt{\frac{1}{K} \sum_k \text{RMSE}_k^2}$, with $\mu$
 * the mean reference value over all bands.
 *
 * @param reference The reference image, an $n \times K$ matrix: pixels as rows, bands as columns.
 * @param estimate The estimated image, of the same shape.
 * @returns RASE, a percentage; 0 for a perfect estimate.
 *
 * @example The same error in a bright and a dark image
 * print('bright:', rase([[100, 100], [100, 100]], [[105, 95], [95, 105]]))
 * print('dark:', rase([[20, 20], [20, 20]], [[25, 15], [15, 25]]))
 */
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

/**
 * Signal-to-noise ratio $10 \log_{10}(\lVert \svec \rVert^2 / \lVert \svec - \hat{\svec} \rVert^2)$ in decibels
 * (signal-to-noise-and-signal-to-distortion-ratios); $+\infty$ for a perfect estimate.
 *
 * @param reference The clean signal $\svec$, a flat array of samples.
 * @param estimate The estimate $\hat{\svec}$, with as many samples.
 * @returns The SNR, in decibels.
 *
 * @example Noise at a tenth of the signal's amplitude is 20 dB down
 * const reference = [1, -1, 1, -1]
 * print('SNR =', snr(reference, [1.1, -0.9, 1.1, -0.9]), 'dB')
 * print('SNR of a halved signal =', snr(reference, [0.5, -0.5, 0.5, -0.5]), 'dB')
 */
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
 * Scale-invariant signal-to-distortion ratio (Le Roux et al. 2019): with
 * $\alpha = \hat{\svec}^\top\svec / \lVert \svec \rVert^2$,
 * $10 \log_{10}(\lVert \alpha\svec \rVert^2 / \lVert \alpha\svec - \hat{\svec} \rVert^2)$ dB, unchanged by any
 * nonzero gain on the estimate.
 *
 * @param reference The clean signal $\svec$, a flat array of samples.
 * @param estimate The estimate $\hat{\svec}$, with as many samples.
 * @param options `zeroMean`, whether to remove each signal's mean first, as many toolkits do (default false).
 * @returns The SI-SDR, in decibels.
 *
 * @example A gain changes SNR but not SI-SDR
 * const reference = [1, -1, 1, -1]
 * const estimate = [0.55, -0.45, 0.55, -0.45]
 * print('SI-SDR =', siSdr(reference, estimate), 'dB')
 * print('SNR =', snr(reference, estimate), 'dB')
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

/**
 * Every permutation of $0, \dots, n - 1$ (Heap's algorithm), the identity first.
 *
 * @param n The number of elements.
 * @returns The $n!$ permutations, each an array of $n$ indices.
 */
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
 * Permutation-invariant scoring (Yu et al. 2017; permutation-invariant-training): the $S \times S$ matrix of a
 * pairwise metric between estimate $j$ (rows) and reference $k$ (columns), and the assignment of estimates to
 * references with the best mean score (the maximum for a higher-is-better metric, the minimum otherwise), by
 * enumerating the $S!$ permutations. Shapes that differ throw `ShapeError`, and $S > 8$ throws `DomainError`.
 *
 * @param references The $S$ reference signals, the rows of an $S \times T$ matrix.
 * @param estimates The $S$ estimated signals, the rows of an $S \times T$ matrix, in any order.
 * @param metric The pairwise metric, called as `metric(reference, estimate)`; its `info.direction` says whether the
 *   best mean is the largest or the smallest.
 * @returns `score`, the best mean score; `permutation`, where entry $j$ is the reference assigned to estimate $j$; and
 *   `matrix`, the $S \times S$ pairwise scores.
 *
 * @example Two separated sources returned in swapped order
 * const references = [[1, -1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1]]
 * const estimates = [[0.9, 1.1, -1, -0.9, 1, 1.1], [1, -0.9, 1.1, -1, 0.9, -1]]
 * const { score, permutation, matrix } = permutationInvariantScore(references, estimates)
 * print('SI-SDR =', score, 'dB with permutation', permutation)
 * print(matrix)
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
