/**
 * Blob detection in scale space (Lindeberg, 1998, "Feature detection with automatic scale selection", IJCV 30(2)):
 * the scale-normalised Laplacian of Gaussian −σ²∇²(G_σ ∗ I) peaks at the centre of a bright blob of radius ≈ σ√2 at
 * σ = r/√2; the difference of Gaussians G_{kσ} ∗ I − G_σ ∗ I approximates (k − 1)σ²∇²G and is what SIFT uses (Lowe,
 * 2004, IJCV 60(2)).
 */

import { readImage, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'
import { gaussianBlur, gaussianLaplace } from './filters'

/** A blob: centre (row, column), scale σ and the scale-normalised response there. */
export interface Blob {
  row: number
  col: number
  sigma: number
  /** The radius σ√2 of the blob a 2-D Laplacian of Gaussian matches. */
  radius: number
  value: number
}

/** Options for the blob detectors. */
export interface BlobOptions {
  minSigma?: number
  maxSigma?: number
  /** Keep blobs whose response is at least this. Default 0.2 (for images in [0, 1]). */
  threshold?: number
  /**
   * Drop the weaker (smaller response) of two blobs whose discs overlap by more than this fraction of the smaller disc's
   * area. Default 0.5. (scikit-image drops the smaller disc instead.)
   */
  overlap?: number
}

/** The area of overlap of two discs (radii r1, r2, centres d apart) over the smaller disc's area. */
function discOverlap(r1: number, r2: number, d: number): number {
  if (d >= r1 + r2) return 0
  const small = Math.min(r1, r2)
  if (d <= Math.abs(r1 - r2)) return 1
  const a1 = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
  const a2 = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
  const a3 = 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
  return (a1 + a2 - a3) / (Math.PI * small * small)
}

/** Local maxima of a scale-space cube (levels × h × w) over 3 × 3 × 3 neighbourhoods, pruned by overlap. */
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
 * Bright blobs by the scale-normalised Laplacian of Gaussian (Lindeberg, 1998), as `skimage.feature.blob_log`: the cube
 * −σ²∇²(G_σ ∗ I) over `levels` (default 10) σ evenly spaced from `minSigma` (1) to `maxSigma` (8), its 3 × 3 × 3 local
 * maxima above `threshold`, and the weaker of overlapping blobs dropped.
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
 * Bright blobs by the difference of Gaussians (Lowe, 2004): σ grows geometrically by `ratio` (default 1.6) from
 * `minSigma` to past `maxSigma`, and level k is (G_{σ_k} − G_{σ_{k+1}}) ∗ I scaled by σ_k/(σ_{k+1} − σ_k), so it
 * approximates −σ²∇²G and shares the LoG's threshold scale.
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
