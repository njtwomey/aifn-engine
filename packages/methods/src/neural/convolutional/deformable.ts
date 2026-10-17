/**
 * Deformable convolution at one output position, tap by tap: the worked form behind a figure of the sampling grid
 * (Dai et al., 2017, "Deformable Convolutional Networks", §2.1, eqs. 2 to 4; Zhu et al., 2019, "Deformable ConvNets
 * v2", §3.1, eq. 1).
 *
 * A $k \times k$ kernel normally reads the regular grid $\pvec_0 + d\pvec_n$ around the output position $\pvec_0$
 * ($d$ the dilation); deformable convolution adds an offset $\Delta\pvec_n$ to every tap and reads the single-channel
 * input there by bilinear interpolation (`bilinearSample` of `aifn-compute/signal/image`, zero outside), and the
 * modulated form scales tap $n$ by $m_n$. Each tap's sample, its four neighbours and the gradient of the output in its
 * offset, $w_n m_n \nabla x$ at the sample, are returned so a figure can show them. The whole layer, differentiable,
 * is `deformableConv2d` of `aifn-compute/nn/functional`.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData } from 'aifn-compute/foundation/tensor'
import { bilinearSample, type BilinearSample } from 'aifn-compute/signal/image'

/** One tap of a deformable kernel at one output position. */
export type DeformableTap = {
  /** Tap index $n$, row-major over the $k \times k$ kernel. */
  index: number
  /** The regular grid position $\pvec_0 + d\pvec_n$, as (row, column). */
  regular: [number, number]
  /** The sampled position $\pvec_0 + d\pvec_n + \Delta\pvec_n$. */
  position: [number, number]
  /** The bilinear sample there. */
  sample: BilinearSample
  /** The kernel weight $w_n$. */
  weight: number
  /** The modulation scalar $m_n$ (1 without a mask). */
  mask: number
}

/** Options of `deformableConvolutionAt`. */
export type DeformableOptions = {
  /** Spacing $d$ of the regular grid. Default 1. */
  dilation?: number
  /** Modulation scalars $m_n$, one per tap (DCNv2). Default all 1 (DCNv1). */
  mask?: ArrayLike<number>
}

/**
 * The output of a single-channel deformable convolution at one position, tap by tap,
 * $y(\pvec_0) = \sum_n w_n m_n x(\pvec_0 + d\pvec_n + \Delta\pvec_n)$. Throws `DomainError` for an even kernel size and
 * `ShapeError` when the weights or offsets are not $k^2$.
 *
 * @param image The input map, row-major, `height` $\times$ `width` values.
 * @param height The number of rows.
 * @param width The number of columns.
 * @param kernel The $k \times k$ weights, row-major (tap $n = uk + v$ for kernel row $u$ and column $v$).
 * @param k The kernel size, odd.
 * @param centre The output position $\pvec_0$ as (row, column).
 * @param offsets The offsets $\Delta\pvec_n$ as (row, column) pairs, one per tap.
 * @param options The dilation and the modulation scalars.
 * @returns The output value, each tap, and the gradient of the output in each tap's offset, $w_n m_n \nabla x$ at the
 *   sampled position (one-sided at integer positions).
 *
 * @example A mean kernel on a diagonal ridge, before and after bending the grid
 * const X = [0, 0, 1, 4, 8, 0, 1, 4, 8, 4, 1, 4, 8, 4, 1, 4, 8, 4, 1, 0, 8, 4, 1, 0, 0]
 * const mean = new Array(9).fill(1 / 9)
 * print('regular:', deformableConvolutionAt(X, 5, 5, mean, 3, [2, 2], new Array(9).fill([0, 0])).value)
 * const offsets = [[0.4, 0.8], [0, 0.5], [0, 0], [0.5, 0], [0, 0], [-0.5, 0], [0, 0], [0, -0.5], [-0.8, -0.4]]
 * const d = deformableConvolutionAt(X, 5, 5, mean, 3, [2, 2], offsets)
 * print('deformed:', d.value)
 * print('top-left tap:', d.taps[0].position, '->', d.taps[0].sample.value)
 * print('its offset gradient:', d.offsetGradient[0])
 */
export function deformableConvolutionAt(
  image: ArrayLike<number>,
  height: number,
  width: number,
  kernel: ArrayLike<number>,
  k: number,
  centre: readonly [number, number],
  offsets: ArrayLike<readonly [number, number]>,
  options: DeformableOptions = {},
): { value: number; taps: DeformableTap[]; offsetGradient: [number, number][] } {
  const where = 'deformableConvolutionAt'
  if (k % 2 !== 1) throw new DomainError(where, `${where}: the kernel size must be odd`)
  if (kernel.length !== k * k || offsets.length !== k * k) {
    throw new ShapeError(where, `${where}: expected ${k * k} weights and offsets`)
  }
  if (image.length !== height * width) throw new ShapeError(where, `${where}: the image is not ${height} × ${width}`)
  const img = fromData(Float64Array.from(image), [height, width])
  const d = options.dilation ?? 1
  const half = (k - 1) / 2
  const taps: DeformableTap[] = []
  for (let u = 0; u < k; u++) {
    for (let v = 0; v < k; v++) {
      const n = u * k + v
      const regular: [number, number] = [centre[0] + d * (u - half), centre[1] + d * (v - half)]
      const position: [number, number] = [regular[0] + offsets[n][0], regular[1] + offsets[n][1]]
      const sample = bilinearSample(img, position[0], position[1])
      taps.push({ index: n, regular, position, sample, weight: kernel[n], mask: options.mask?.[n] ?? 1 })
    }
  }
  const value = taps.reduce((a, t) => a + t.weight * t.mask * t.sample.value, 0)
  const offsetGradient = taps.map((t): [number, number] => [
    t.weight * t.mask * t.sample.gradient[0],
    t.weight * t.mask * t.sample.gradient[1],
  ])
  return { value, taps, offsetGradient }
}
