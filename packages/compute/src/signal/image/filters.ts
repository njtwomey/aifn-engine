/**
 * Linear filters on greyscale images ([height, width] tensors, row 0 at the top), with scipy.ndimage's boundary modes,
 * on the separable filtering of `aifn-compute/foundation/convolution`: the Gaussian kernel and blur, gradient operators (Sobel,
 * Scharr, Prewitt), the Laplacian of Gaussian (Marr and Hildreth, 1980, Proc. R. Soc. Lond. B 207) and the structure
 * tensor that corner detectors read.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { readImage, separableFilter, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'

/**
 * The sampled, normalised Gaussian kernel exp(−x²/2σ²) on x = −R, …, R with R = ⌊truncate·σ + 1/2⌋, as
 * scipy.ndimage's (truncate default 4).
 */
export function gaussianKernel(sigma: number, { truncate = 4 }: { truncate?: number } = {}): Tensor {
  const radius = Math.floor(truncate * sigma + 0.5)
  const k = Float64Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2))
  const total = k.reduce((a, b) => a + b, 0)
  return fromData(k.map((v) => v / total))
}

/** Gaussian blur with standard deviation σ pixels (separable), as `scipy.ndimage.gaussian_filter`. */
export function gaussianBlur(
  img: ImageInput,
  sigma: number,
  options: { border?: Border; truncate?: number } = {},
): Tensor {
  if (sigma <= 0) {
    const I = readImage(img, 'gaussianBlur')
    return fromData(I.v, [I.h, I.w])
  }
  const k = gaussianKernel(sigma, options)
  return separableFilter(img, k, k, options)
}

/** Sobel gradients. */
export interface Gradients {
  /** Derivative along columns (x, to the right). */
  gx: Tensor
  /** Derivative along rows (downwards, row index increasing). */
  gy: Tensor
  magnitude: Tensor
  /** atan2(gy, gx), radians. */
  direction: Tensor
}

/** The smoothing taps of the gradient operators: the derivative [−1, 0, 1] is smoothed across by these. */
const SMOOTHING = { sobel: [1, 2, 1], scharr: [3, 10, 3], prewitt: [1, 1, 1] } as const

/** A gradient operator: Sobel (1968), Scharr (2000, rotation-optimised) or Prewitt (1970). */
export type GradientOperator = keyof typeof SMOOTHING

/**
 * Image gradients by a 3 × 3 operator (default Sobel; Sobel and Feldman, 1968): differences [−1, 0, 1] along one axis
 * smoothed by [1, 2, 1] (Sobel), [3, 10, 3] (Scharr) or [1, 1, 1] (Prewitt) along the other, as `scipy.ndimage.sobel`
 * and `prewitt` (unnormalised: divide by 8, 32 or 6 for a unit-spacing derivative). With `sigma`, the image is
 * Gaussian-blurred first.
 */
export function gradients(
  img: ImageInput,
  options: { border?: Border; sigma?: number; operator?: GradientOperator } = {},
): Gradients {
  const base = options.sigma ? gaussianBlur(img, options.sigma, options) : img
  const smooth = SMOOTHING[options.operator ?? 'sobel']
  const gx = separableFilter(base, [-1, 0, 1], smooth, options)
  const gy = separableFilter(base, smooth, [-1, 0, 1], options)
  const x = gx.data
  const y = gy.data
  return {
    gx,
    gy,
    magnitude: fromData(
      Float64Array.from(x, (v, i) => Math.hypot(v, y[i])),
      gx.shape,
    ),
    direction: fromData(
      Float64Array.from(x, (v, i) => Math.atan2(y[i], v)),
      gx.shape,
    ),
  }
}

/** Sobel gradients: `gradients` with the Sobel operator, as `scipy.ndimage.sobel`. */
export function sobel(img: ImageInput, options: { border?: Border; sigma?: number } = {}): Gradients {
  return gradients(img, { ...options, operator: 'sobel' })
}

/** The structure tensor's three images: the Gaussian-weighted sums of gy², gx·gy and gx² (row, column order). */
export interface StructureTensor {
  /** Σ w (∂I/∂r)², Σ w (∂I/∂r)(∂I/∂c), Σ w (∂I/∂c)². */
  rr: Tensor
  rc: Tensor
  cc: Tensor
}

/**
 * The structure tensor (second-moment matrix) M = G_σ ∗ [I_r², I_r I_c; I_r I_c, I_c²] with Sobel derivatives, as
 * `skimage.feature.structure_tensor` (default border `constant`, Gaussian truncated at 4σ). Its eigenvalues measure
 * how much the image changes along the two principal directions around each pixel.
 */
export function structureTensor(
  img: ImageInput,
  { sigma = 1, border = 'constant' }: { sigma?: number; border?: Border } = {},
): StructureTensor {
  const g = gradients(img, { border })
  const r = g.gy.data
  const c = g.gx.data
  const blur = (v: Float64Array) => gaussianBlur(fromData(v, g.gx.shape), sigma, { border })
  return {
    rr: blur(Float64Array.from(r, (v) => v * v)),
    rc: blur(Float64Array.from(r, (v, i) => v * c[i])),
    cc: blur(Float64Array.from(c, (v) => v * v)),
  }
}

/**
 * The sampled second derivative of the normalised Gaussian, (x²/σ⁴ − 1/σ²)·G(x), on x = −R … R with R = ⌊4σ + ½⌋, as
 * `scipy.ndimage`'s order-2 Gaussian kernel.
 */
function gaussianSecondDerivative(sigma: number): Float64Array {
  const g = gaussianKernel(sigma).data
  const R = (g.length - 1) / 2
  const s2 = sigma * sigma
  return Float64Array.from(g, (v, i) => v * ((i - R) ** 2 / (s2 * s2) - 1 / s2))
}

/** The Laplacian of the Gaussian-smoothed image, ∇²(G_σ ∗ I), as `scipy.ndimage.gaussian_laplace`. */
export function gaussianLaplace(
  img: ImageInput,
  sigma: number,
  { border = 'reflect' }: { border?: Border } = {},
): Tensor {
  if (!(sigma > 0)) throw new DomainError('gaussianLaplace', 'gaussianLaplace: σ must be positive')
  const g = gaussianKernel(sigma).data
  const d2 = gaussianSecondDerivative(sigma)
  const a = separableFilter(img, d2, g, { border }).data
  const b = separableFilter(img, g, d2, { border }).data
  const { h, w } = readImage(img, 'gaussianLaplace')
  return fromData(
    Float64Array.from(a, (v, i) => v + b[i]),
    [h, w],
  )
}
