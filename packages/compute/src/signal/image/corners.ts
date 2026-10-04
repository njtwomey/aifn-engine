/**
 * Corner detection from the structure tensor M: the Harris response det M − k (tr M)² (Harris & Stephens, 1988, "A
 * combined corner and edge detector", Alvey Vision Conf.), the Shi–Tomasi response λ_min(M) (Shi & Tomasi, 1994, "Good
 * features to track", CVPR), and peak picking with non-maximum suppression.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { structureTensor } from './filters'

/** Options shared by the corner responses. */
export interface CornerOptions {
  /** σ of the Gaussian window that sums the structure tensor. Default 1. */
  sigma?: number
  border?: Border
}

/**
 * The Harris corner response R = det M − k (tr M)² (Harris & Stephens, 1988), as `skimage.feature.corner_harris`
 * (method `k`): large and positive at corners (both eigenvalues large), negative along edges, small in flat regions.
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
 * The Shi–Tomasi response, the smaller eigenvalue of the structure tensor (Shi & Tomasi, 1994), as
 * `skimage.feature.corner_shi_tomasi`: ((M_rr + M_cc) − √((M_rr − M_cc)² + 4M_rc²))/2.
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
  row: number
  col: number
  value: number
}

/** Options for `imagePeaks`. */
export interface PeakOptions {
  /** Peaks are local maxima over a (2d + 1)² window and at least d pixels apart. Default 1. */
  minDistance?: number
  /** Keep peaks with value ≥ this. Default: `relative`·max. */
  threshold?: number
  /** Threshold relative to the largest value, used when `threshold` is not given. Default 0.1. */
  relative?: number
  /** Keep at most this many, strongest first. Default all. */
  count?: number
  /** Ignore peaks within this many pixels of the border. Default `minDistance`. */
  excludeBorder?: number
}

/**
 * The local maxima of an image, strongest first (as `skimage.feature.peak_local_max` and `corner_peaks`): a
 * pixel is a peak when it equals the maximum over its (2d + 1)² window and clears the threshold; peaks closer than d to
 * a stronger one are then dropped. Only positive values can be peaks (a response such as Harris's is negative along
 * edges), whatever the threshold.
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

/** Corners: the peaks of the Harris (default) or Shi–Tomasi response. */
export function detectCorners(
  img: ImageInput,
  options: CornerOptions & PeakOptions & { method?: 'harris' | 'shi-tomasi'; k?: number } = {},
): ImagePeak[] {
  const response = options.method === 'shi-tomasi' ? shiTomasiResponse(img, options) : harrisResponse(img, options)
  return imagePeaks(response, options)
}
