/**
 * The Canny edge detector (Canny, 1986, "A computational approach to edge detection", IEEE TPAMI 8(6)): Gaussian
 * smoothing, Sobel gradients, non-maximum suppression across the edge, and hysteresis between two thresholds. Every
 * stage is returned, so a figure can show what each one removes.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'
import { gaussianBlur, gradients } from './filters'

/** The stages of `canny`, each an image of the input's shape. */
export interface Canny {
  /** The smoothed image. */
  smoothed: Tensor
  /** The Sobel gradient magnitude of the smoothed image. */
  magnitude: Tensor
  /** The gradient direction in radians, $\operatorname{atan2}(g_r, g_c)$ ($g_r$ along rows, $g_c$ along columns). */
  direction: Tensor
  /** The magnitude where it is a local maximum across the edge, else 0. */
  suppressed: Tensor
  /** 1 where the suppressed magnitude is positive and at least `high`, else 0. */
  strong: Tensor
  /** 1 where the suppressed magnitude is positive, at least `low` and below `high`, else 0. */
  weak: Tensor
  /** The edges: strong pixels and the weak pixels 8-connected to them (0/1). */
  edges: Tensor
  /** The low threshold used, as an absolute magnitude. */
  low: number
  /** The high threshold used, as an absolute magnitude. */
  high: number
}

/** Options for `canny`. */
export interface CannyOptions {
  /** Gaussian smoothing $\sigma$ in pixels (0 for none). Default 1. */
  sigma?: number
  /**
   * The low threshold on the gradient magnitude (default 10% of the largest magnitude); with `quantile`, a quantile
   * of the nonzero suppressed magnitudes (default 0.7).
   */
  low?: number
  /** The high threshold, likewise (default 20% of the largest magnitude, or the 0.9 quantile). */
  high?: number
  /** Read `low` and `high` as quantiles in $[0, 1]$ rather than magnitudes. Default false. */
  quantile?: boolean
  /** Border mode for the filters. Default `nearest`. */
  border?: Border
}

/**
 * Canny edges (Canny, 1986). The image is blurred with $\sigma$, differentiated by Sobel, and each pixel kept only if
 * its magnitude is at least that of both neighbours along the gradient direction (rounded to the nearest of
 * $0^\circ$, $45^\circ$, $90^\circ$ and $135^\circ$). Pixels at or above `high` are edges; pixels
 * between `low` and `high` become edges when 8-connected to one, directly or through other such pixels. Thresholds
 * default to 10% and 20% of the largest magnitude; with `quantile`, they are quantiles of the suppressed nonzero
 * magnitudes (by default 0.7 and 0.9). The one-pixel frame is never an edge. Throws `DomainError` for a negative
 * $\sigma$ or a `low` above `high`.
 *
 * @param img The image, $h \times w$.
 * @param options The smoothing, the thresholds and the border mode.
 * @returns Every stage, from the smoothed image to the edges, with the thresholds used.
 *
 * @example Suppression thins a soft edge to one pixel
 * const img = [0, 1, 2, 3, 4].map(() => [0, 0, 0, 1, 2, 2, 2])
 * const { magnitude, suppressed, edges, low, high } = canny(img, { sigma: 0 })
 * print('magnitude:', magnitude)
 * print('suppressed:', suppressed)
 * print('thresholds:', low, high)
 * print('edges:', edges)
 */
export function canny(img: ImageInput, options: CannyOptions = {}): Canny {
  const { v, h, w } = readImage(img, 'canny')
  const sigma = options.sigma ?? 1
  if (!(sigma >= 0)) throw new DomainError('canny', 'canny: σ must be non-negative')
  const border = options.border ?? 'nearest'
  const smoothed = gaussianBlur(fromData(v, [h, w]), sigma, { border })
  const g = gradients(smoothed, { border })
  const gx = g.gx.data
  const gy = g.gy.data
  const mag = g.magnitude.data
  const sup = new Float64Array(h * w)
  for (let r = 1; r < h - 1; r++)
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c
      const m = mag[i]
      if (m === 0) continue
      // The direction across the edge, folded to [0, 180°) and rounded to a multiple of 45°.
      let deg = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI
      if (deg < 0) deg += 180
      const sector = Math.round(deg / 45) % 4
      // Neighbour offsets along the gradient: 0° → columns, 90° → rows; 45° → (r+1, c+1) since rows grow downward.
      const [dr, dc] = sector === 0 ? [0, 1] : sector === 1 ? [1, 1] : sector === 2 ? [1, 0] : [1, -1]
      const a = mag[(r + dr) * w + c + dc]
      const b = mag[(r - dr) * w + c - dc]
      if (m >= a && m >= b) sup[i] = m
    }
  let low: number
  let high: number
  if (options.quantile) {
    const nz = Array.from(sup.filter((x) => x > 0)).sort((x, y) => x - y)
    const q = (p: number) => (nz.length ? nz[Math.min(nz.length - 1, Math.floor(p * (nz.length - 1)))] : 0)
    low = q(options.low ?? 0.7)
    high = q(options.high ?? 0.9)
  } else {
    const max = Array.from(mag).reduce((a, b) => Math.max(a, b), 0)
    low = options.low ?? 0.1 * max
    high = options.high ?? 0.2 * max
  }
  if (low > high) throw new DomainError('canny', 'canny: the low threshold must not exceed the high one')
  const strong = new Float64Array(h * w)
  const weak = new Float64Array(h * w)
  const edges = new Float64Array(h * w)
  const stack: number[] = []
  for (let i = 0; i < h * w; i++) {
    if (sup[i] >= high && sup[i] > 0) {
      strong[i] = 1
      edges[i] = 1
      stack.push(i)
    } else if (sup[i] >= low && sup[i] > 0) weak[i] = 1
  }
  // Hysteresis: grow the strong set through 8-connected weak pixels.
  while (stack.length) {
    const i = stack.pop()!
    const r = Math.floor(i / w)
    const c = i % w
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr
        const cc = c + dc
        if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue
        const j = rr * w + cc
        if (weak[j] && !edges[j]) {
          edges[j] = 1
          stack.push(j)
        }
      }
  }
  const t = (a: Float64Array) => fromData(a, [h, w])
  return {
    smoothed,
    magnitude: g.magnitude,
    direction: g.direction,
    suppressed: t(sup),
    strong: t(strong),
    weak: t(weak),
    edges: t(edges),
    low,
    high,
  }
}
