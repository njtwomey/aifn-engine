/**
 * Corner detection from the structure tensor $\Mmat$ of `structureTensor`: the Harris response
 * $\det \Mmat - k (\trace \Mmat)^2$ (Harris and Stephens, 1988, "A combined corner and edge detector", Alvey Vision
 * Conf.), the Shi–Tomasi response $\lambda_\text{min}(\Mmat)$ (Shi and Tomasi, 1994, "Good features to track", CVPR),
 * and peak picking with non-maximum suppression. The responses are images of the input's shape; peaks are listed
 * strongest first, with row and column indices.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { structureTensor } from './filters'

/** Options shared by the corner responses. */
export interface CornerOptions {
  /** $\sigma$ of the Gaussian window that sums the structure tensor, in pixels. Default 1. */
  sigma?: number
  /** The border mode of the derivatives and the window. Default `constant`. */
  border?: Border
}

/**
 * The Harris corner response $R = \det \Mmat - k (\trace \Mmat)^2$ (Harris and Stephens, 1988), as
 * `skimage.feature.corner_harris` (method `k`): large and positive at corners (both eigenvalues large), negative along
 * edges, small in flat regions.
 *
 * @param img The image, $h \times w$.
 * @param options The structure tensor's window `sigma` and `border` (see `CornerOptions`), and $k$.
 * @param options.k The sensitivity $k$ (default 0.05); larger values make edges more negative and fewer corners.
 * @param options.options The remaining fields, `sigma` and `border`, passed to `structureTensor`.
 * @returns The response $R$ at every pixel, $h \times w$.
 *
 * @example Positive at a corner, negative on an edge, near zero where flat
 * const inside = (i) => i >= 3 && i <= 8
 * const img = Array.from({ length: 12 }, (_, r) => Array.from({ length: 12 }, (_, c) => +(inside(r) && inside(c))))
 * const R = harrisResponse(img)
 * print('corner (3, 3):', R.data[3 * 12 + 3])
 * print('edge (5, 3):', R.data[5 * 12 + 3])
 * print('flat (0, 0):', R.data[0])
 */
export function harrisResponse(img: ImageInput, { k = 0.05, ...options }: CornerOptions & { k?: number } = {}): Tensor {
  const M = structureTensor(img, options)
  const rr = M.rr.data
  const rc = M.rc.data
  const cc = M.cc.data
  return fromData(
    Float64Array.from(rr, (a, i) => a * cc[i] - rc[i] * rc[i] - k * (a + cc[i]) ** 2),
    M.rr.shape,
  )
}

/**
 * The Shi–Tomasi response, the smaller eigenvalue of the structure tensor (Shi and Tomasi, 1994), as
 * `skimage.feature.corner_shi_tomasi`: $\big(M_{rr} + M_{cc} - \sqrt{(M_{rr} - M_{cc})^2 + 4 M_{rc}^2}\big) / 2$.
 *
 * @param img The image, $h \times w$.
 * @param options The structure tensor's window and border.
 * @returns The smaller eigenvalue at every pixel, $h \times w$; never negative.
 *
 * @example Largest at a corner, small on an edge
 * const inside = (i) => i >= 3 && i <= 8
 * const img = Array.from({ length: 12 }, (_, r) => Array.from({ length: 12 }, (_, c) => +(inside(r) && inside(c))))
 * const S = shiTomasiResponse(img)
 * print('corner (3, 3):', S.data[3 * 12 + 3])
 * print('edge (5, 3):', S.data[5 * 12 + 3])
 * print('flat (0, 0):', S.data[0])
 */
export function shiTomasiResponse(img: ImageInput, options: CornerOptions = {}): Tensor {
  const M = structureTensor(img, options)
  const rr = M.rr.data
  const rc = M.rc.data
  const cc = M.cc.data
  return fromData(
    Float64Array.from(rr, (a, i) => (a + cc[i] - Math.sqrt((a - cc[i]) ** 2 + 4 * rc[i] * rc[i])) / 2),
    M.rr.shape,
  )
}

/** A detected peak: its row, column and response. */
export interface ImagePeak {
  /** The peak's row, from the top. */
  row: number
  /** The peak's column, from the left. */
  col: number
  /** The image's value at the peak. */
  value: number
}

/** Options for `imagePeaks`. */
export interface PeakOptions {
  /**
   * The distance $d$: peaks are local maxima over a $(2d + 1) \times (2d + 1)$ window, and more than $d$ pixels apart
   * in row or column. Default 1.
   */
  minDistance?: number
  /** Keep peaks with values at least this. Default `relative` times the largest value. */
  threshold?: number
  /** Threshold relative to the largest value, used when `threshold` is not given. Default 0.1. */
  relative?: number
  /** Keep at most this many, strongest first. Default all. */
  count?: number
  /** Ignore peaks within this many pixels of the border. Default `minDistance`. */
  excludeBorder?: number
}

/**
 * The local maxima of an image, strongest first (as `skimage.feature.peak_local_max` and `corner_peaks`): a pixel is a
 * peak when it equals the maximum over its $(2d + 1) \times (2d + 1)$ window and clears the threshold; a peak within
 * $d$ rows and $d$ columns of a stronger one is then dropped. Only positive values can be peaks (a response such as
 * Harris's is negative along edges), whatever the threshold. Ties are broken by row, then column.
 *
 * @param response The image to search, $h \times w$, typically a corner response.
 * @param options The window, the threshold, the most peaks and the border to ignore.
 * @returns The peaks, strongest first.
 *
 * @example Two peaks, and a weaker neighbour suppressed
 * const img = [
 *   [0, 0, 0, 0, 0, 0],
 *   [0, 5, 4, 0, 0, 0],
 *   [0, 0, 0, 0, 3, 0],
 *   [0, 0, 0, 0, 0, 0],
 * ]
 * print('peaks:', imagePeaks(img))
 * print('the strongest only:', imagePeaks(img, { count: 1 }))
 */
export function imagePeaks(response: ImageInput, options: PeakOptions = {}): ImagePeak[] {
  const { v, h, w } = readImage(response, 'imagePeaks')
  const d = options.minDistance ?? 1
  const edge = options.excludeBorder ?? d
  const max = v.reduce((a, b) => Math.max(a, b), -Infinity)
  const threshold = options.threshold ?? (options.relative ?? 0.1) * max
  const found: ImagePeak[] = []
  for (let r = edge; r < h - edge; r++)
    for (let c = edge; c < w - edge; c++) {
      const x = v[r * w + c]
      if (!(x >= threshold) || x <= 0) continue
      let isMax = true
      for (let dr = -d; dr <= d && isMax; dr++)
        for (let dc = -d; dc <= d; dc++) {
          const rr = r + dr
          const cc = c + dc
          if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue
          if (v[rr * w + cc] > x) {
            isMax = false
            break
          }
        }
      if (isMax) found.push({ row: r, col: c, value: x })
    }
  found.sort((a, b) => b.value - a.value || a.row - b.row || a.col - b.col)
  const kept: ImagePeak[] = []
  for (const p of found) {
    if (kept.some((q) => Math.max(Math.abs(q.row - p.row), Math.abs(q.col - p.col)) <= d)) continue
    kept.push(p)
    if (options.count !== undefined && kept.length >= options.count) break
  }
  return kept
}

/**
 * Corners: the peaks of the Harris (default) or Shi–Tomasi response, by `imagePeaks`.
 *
 * @param img The image, $h \times w$.
 * @param options The response's window and border, the peak picking of `PeakOptions`, the `method` (`'harris'`,
 *   the default, or `'shi-tomasi'`), and Harris's `k`.
 * @returns The corners, strongest first.
 *
 * @example The four corners of a bright square
 * const inside = (i) => i >= 3 && i <= 8
 * const img = Array.from({ length: 12 }, (_, r) => Array.from({ length: 12 }, (_, c) => +(inside(r) && inside(c))))
 * print('Harris:', detectCorners(img))
 * print('Shi-Tomasi:', detectCorners(img, { method: 'shi-tomasi' }))
 */
export function detectCorners(
  img: ImageInput,
  options: CornerOptions & PeakOptions & { method?: 'harris' | 'shi-tomasi'; k?: number } = {},
): ImagePeak[] {
  const response = options.method === 'shi-tomasi' ? shiTomasiResponse(img, options) : harrisResponse(img, options)
  return imagePeaks(response, options)
}
