/**
 * Blob detection in scale space (Lindeberg, 1998, "Feature detection with automatic scale selection", IJCV 30(2)):
 * the scale-normalised Laplacian of Gaussian $-\sigma^2 \nabla^2 (G_\sigma * I)$ peaks at the centre of a bright blob
 * of radius $r$ at $\sigma = r / \sqrt{2}$; the difference of Gaussians $G_{k\sigma} * I - G_\sigma * I$ approximates
 * $(k - 1) \sigma^2 \nabla^2 G_\sigma * I$ and is what SIFT uses (Lowe, 2004, IJCV 60(2)).
 *
 * Both detectors build a cube of responses over position and scale, keep its $3 \times 3 \times 3$ local maxima above
 * a threshold, and drop the weaker of two blobs whose discs overlap too much. Only bright blobs on a dark background
 * are found; invert the image for dark ones.
 */

import { readImage, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'
import { gaussianBlur, gaussianLaplace } from './filters'

/** A blob: centre (row, column), scale $\sigma$ and the scale-normalised response there. */
export interface Blob {
  /** The centre's row. */
  row: number
  /** The centre's column. */
  col: number
  /** The scale $\sigma$ of the level the blob peaked at, in pixels. */
  sigma: number
  /** The radius $\sigma \sqrt{2}$ of the blob a 2-D Laplacian of Gaussian matches. */
  radius: number
  /** The scale-normalised response at the peak. */
  value: number
}

/** Options for the blob detectors. */
export interface BlobOptions {
  /** The smallest scale $\sigma$, in pixels. Default 1. */
  minSigma?: number
  /** The largest scale $\sigma$, in pixels. Default 8. */
  maxSigma?: number
  /** Keep blobs whose response is at least this. Default 0.2 (for images in [0, 1]). */
  threshold?: number
  /**
   * Drop the weaker (smaller response) of two blobs whose discs overlap by more than this fraction of the smaller
   * disc's area. Default 0.5. (scikit-image drops the smaller disc instead.)
   */
  overlap?: number
}

/**
 * The area of overlap of two discs over the smaller disc's area: 0 when they are apart, 1 when one holds the other.
 *
 * @param r1 The first disc's radius.
 * @param r2 The second disc's radius.
 * @param d The distance between their centres.
 * @returns The fraction of the smaller disc covered, in $[0, 1]$.
 */
function discOverlap(r1: number, r2: number, d: number): number {
  if (d >= r1 + r2) return 0
  const small = Math.min(r1, r2)
  if (d <= Math.abs(r1 - r2)) return 1
  const a1 = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
  const a2 = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
  const a3 = 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
  return (a1 + a2 - a3) / (Math.PI * small * small)
}

/**
 * Local maxima of a scale-space cube (levels $\times h \times w$) over $3 \times 3 \times 3$ neighbourhoods (cut at
 * the cube's faces, so border pixels and end levels can peak), pruned by overlap: strongest first, a blob is kept
 * unless its disc overlaps a kept one's by more than `overlap`.
 *
 * @param cube The levels, one row-major $h \times w$ array per scale.
 * @param sigmas The scale $\sigma$ of each level.
 * @param h The image height.
 * @param w The image width.
 * @param threshold The least response a blob may have.
 * @param overlap The largest overlap allowed, as a fraction of the smaller disc's area.
 * @returns The blobs kept, strongest first.
 */
function cubePeaks(
  cube: Float64Array[],
  sigmas: number[],
  h: number,
  w: number,
  threshold: number,
  overlap: number,
): Blob[] {
  const found: Blob[] = []
  for (let k = 0; k < cube.length; k++)
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) {
        const x = cube[k][r * w + c]
        if (!(x >= threshold)) continue
        let isMax = true
        for (let dk = -1; dk <= 1 && isMax; dk++)
          for (let dr = -1; dr <= 1 && isMax; dr++)
            for (let dc = -1; dc <= 1; dc++) {
              const kk = k + dk
              const rr = r + dr
              const cc = c + dc
              if (kk < 0 || kk >= cube.length || rr < 0 || rr >= h || cc < 0 || cc >= w) continue
              if (cube[kk][rr * w + cc] > x) {
                isMax = false
                break
              }
            }
        if (isMax) found.push({ row: r, col: c, sigma: sigmas[k], radius: sigmas[k] * Math.SQRT2, value: x })
      }
  found.sort((a, b) => b.value - a.value)
  const kept: Blob[] = []
  for (const b of found) {
    const clash = kept.some((q) => discOverlap(q.radius, b.radius, Math.hypot(q.row - b.row, q.col - b.col)) > overlap)
    if (!clash) kept.push(b)
  }
  return kept
}

/**
 * Bright blobs by the scale-normalised Laplacian of Gaussian (Lindeberg, 1998), as `skimage.feature.blob_log`: the
 * cube $-\sigma^2 \nabla^2 (G_\sigma * I)$ over `levels` (default 10) values of $\sigma$ evenly spaced from `minSigma`
 * (1) to `maxSigma` (8), its $3 \times 3 \times 3$ local maxima above `threshold`, and the weaker of overlapping
 * blobs dropped. The Laplacian reads beyond the edge by reflection.
 *
 * @param img The image, $h \times w$, bright blobs on a dark background, best in $[0, 1]$ for the default threshold.
 * @param options The scales, the threshold and the overlap of `BlobOptions`, and `levels`, the number of scales.
 * @returns The blobs, strongest first.
 *
 * @example Two discs, of radius 2 and 5, found at their scales
 * const disc = (r0, c0, radius) => (r, c) => (r - r0) ** 2 + (c - c0) ** 2 <= radius ** 2
 * const small = disc(6, 6, 2)
 * const large = disc(15, 15, 5)
 * const img = Array.from({ length: 24 }, (_, r) => Array.from({ length: 24 }, (_, c) => +(small(r, c) || large(r, c))))
 * for (const b of blobsLog(img, { maxSigma: 5, levels: 9 })) print(b)
 */
export function blobsLog(img: ImageInput, options: BlobOptions & { levels?: number } = {}): Blob[] {
  const { h, w } = readImage(img, 'blobsLog')
  const lo = options.minSigma ?? 1
  const hi = options.maxSigma ?? 8
  const n = options.levels ?? 10
  const sigmas = Array.from({ length: n }, (_, k) => (n === 1 ? lo : lo + ((hi - lo) * k) / (n - 1)))
  const cube = sigmas.map((s) => Float64Array.from(gaussianLaplace(img, s).data, (v) => -v * s * s))
  return cubePeaks(cube, sigmas, h, w, options.threshold ?? 0.2, options.overlap ?? 0.5)
}

/**
 * Bright blobs by the difference of Gaussians (Lowe, 2004): $\sigma$ grows geometrically by `ratio` (default 1.6) from
 * `minSigma` to at least `maxSigma`, and level $k$ is $(G_{\sigma_k} - G_{\sigma_{k+1}}) * I$ scaled by
 * $\sigma_k / (\sigma_{k+1} - \sigma_k)$, so it approximates $-\sigma_k^2 \nabla^2 G_{\sigma_k} * I$ and shares the
 * LoG's threshold scale. Faster than `blobsLog`, but with coarser scales. Throws `DomainError` unless the ratio
 * exceeds 1.
 *
 * @param img The image, $h \times w$, bright blobs on a dark background.
 * @param options The scales, the threshold and the overlap of `BlobOptions`, and `ratio`, the factor between
 *   successive scales.
 * @returns The blobs, strongest first.
 *
 * @example The same two discs, on the coarser scales of the difference of Gaussians
 * const disc = (r0, c0, radius) => (r, c) => (r - r0) ** 2 + (c - c0) ** 2 <= radius ** 2
 * const small = disc(6, 6, 2)
 * const large = disc(15, 15, 5)
 * const img = Array.from({ length: 24 }, (_, r) => Array.from({ length: 24 }, (_, c) => +(small(r, c) || large(r, c))))
 * for (const b of blobsDog(img, { maxSigma: 5 })) print(b)
 */
export function blobsDog(img: ImageInput, options: BlobOptions & { ratio?: number } = {}): Blob[] {
  const { h, w } = readImage(img, 'blobsDog')
  const lo = options.minSigma ?? 1
  const hi = options.maxSigma ?? 8
  const ratio = options.ratio ?? 1.6
  if (!(ratio > 1)) throw new DomainError('blobsDog', 'blobsDog: the ratio must exceed 1')
  const k = Math.ceil(Math.log(hi / lo) / Math.log(ratio)) + 1
  const sigmas = Array.from({ length: k + 1 }, (_, i) => lo * ratio ** i)
  const blurred = sigmas.map((s) => gaussianBlur(img, s).data)
  const cube = sigmas
    .slice(0, -1)
    .map((s, i) => Float64Array.from(blurred[i], (v, j) => ((v - blurred[i + 1][j]) * s) / (sigmas[i + 1] - s)))
  return cubePeaks(cube, sigmas.slice(0, -1), h, w, options.threshold ?? 0.2, options.overlap ?? 0.5)
}
