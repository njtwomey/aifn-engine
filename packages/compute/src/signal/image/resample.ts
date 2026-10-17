/**
 * Bilinear resampling of greyscale images: one sample at a fractional position with its four neighbours and its
 * gradient, and the resizing of a whole image with half-pixel centres.
 *
 * Bilinear interpolation reads $x(r, c) = \sum_{q} g(q_r, r)\, g(q_c, c)\, x_q$ over the four integer neighbours
 * $q$ of $(r, c)$, with $g(a, b) = \max(0, 1 - \lvert a - b \rvert)$; it is the sampling kernel of deformable
 * convolution (Dai et al., 2017, "Deformable Convolutional Networks", §2.1, eqs. 3 and 4) and of spatial transformer
 * networks. Positions are (row, column) with pixel $(i, j)$ at $(i, j)$, and positions outside the image read zero
 * in `bilinearSample`. `upsampleBilinear` is `torch.nn.functional.interpolate` with `mode='bilinear'` and
 * `align_corners=False`, which clamps at the edge instead. Neither is differentiable (they read the image's values).
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/** One of the four integer neighbours of a sampled position, with its bilinear weight and its pixel value. */
export type SampleTap = {
  /** The neighbour's row. */
  row: number
  /** The neighbour's column. */
  col: number
  /** Its bilinear weight; the four weights sum to 1. */
  weight: number
  /** Its pixel value (0 outside the image). */
  value: number
}

/** A bilinear sample: the interpolated value, its four neighbours, and the value's gradient in (row, column). */
export type BilinearSample = {
  /** The interpolated value. */
  value: number
  /** The four neighbours: top-left, top-right, bottom-left, bottom-right. */
  taps: SampleTap[]
  /** The derivative of the value in the row and in the column coordinate. */
  gradient: [number, number]
}

/**
 * Bilinear interpolation of an image at a fractional position, with zeros outside the image. With $f$ and $g$ the
 * fractional parts of the row and column, the neighbours' weights are $(1 - f)(1 - g)$, $(1 - f)g$, $f(1 - g)$ and
 * $fg$. The gradient is that of the bilinear patch of the cell whose top-left corner is the floor of the position, so
 * it is one-sided at integer positions (as torchvision's deformable convolution).
 *
 * @param img The image, $h \times w$: a tensor or an array of rows.
 * @param row The row coordinate (0 is the first row's centre); any real number.
 * @param col The column coordinate.
 * @returns The value, the four neighbours with their weights and values, and the gradient in row and column.
 *
 * @example The centre of four pixels is their mean
 * const s = bilinearSample([[0, 1], [2, 3]], 0.5, 0.5)
 * print('value =', s.value)
 * print('weights =', s.taps.map((t) => t.weight))
 * print('gradient =', s.gradient)
 *
 * @example Zero outside the image
 * print('half outside:', bilinearSample([[4, 4], [4, 4]], -0.5, 0).value)
 */
export function bilinearSample(img: ImageInput, row: number, col: number): BilinearSample {
  const { v, h, w } = readImage(img, 'bilinearSample')
  const r0 = Math.floor(row)
  const c0 = Math.floor(col)
  const f = row - r0
  const g = col - c0
  const at = (r: number, c: number) => (r >= 0 && r < h && c >= 0 && c < w ? v[r * w + c] : 0)
  const x00 = at(r0, c0)
  const x01 = at(r0, c0 + 1)
  const x10 = at(r0 + 1, c0)
  const x11 = at(r0 + 1, c0 + 1)
  const taps: SampleTap[] = [
    { row: r0, col: c0, weight: (1 - f) * (1 - g), value: x00 },
    { row: r0, col: c0 + 1, weight: (1 - f) * g, value: x01 },
    { row: r0 + 1, col: c0, weight: f * (1 - g), value: x10 },
    { row: r0 + 1, col: c0 + 1, weight: f * g, value: x11 },
  ]
  const value = taps.reduce((a, t) => a + t.weight * t.value, 0)
  const gradient: [number, number] = [(1 - g) * (x10 - x00) + g * (x11 - x01), (1 - f) * (x01 - x00) + f * (x11 - x10)]
  return { value, taps, gradient }
}

/**
 * Bilinear resizing of an $h \times w$ image to $H \times W$ with half-pixel centres: output pixel $(I, J)$ reads the
 * input at $((I + 0.5)h/H - 0.5,\ (J + 0.5)w/W - 0.5)$, clamped to the edge, as
 * `torch.nn.functional.interpolate(mode='bilinear', align_corners=False)`. Used to lay a coarse class activation map
 * over its image. Throws `DomainError` for an output side below 1.
 *
 * @param img The image, $h \times w$: a tensor or an array of rows.
 * @param shape The output size $[H, W]$.
 * @returns The resized image, $H \times W$.
 *
 * @example A 2 x 2 image to 4 x 4
 * print('upsampled:', upsampleBilinear([[1, 2], [3, 4]], [4, 4]))
 */
export function upsampleBilinear(img: ImageInput, shape: readonly [number, number]): Tensor {
  const { v, h, w } = readImage(img, 'upsampleBilinear')
  const [H, W] = shape
  if (!(Number.isInteger(H) && Number.isInteger(W) && H >= 1 && W >= 1))
    throw new DomainError('upsampleBilinear', `upsampleBilinear: the output size ${H} × ${W} is not positive`)
  const out = new Float64Array(H * W)
  const src = (o: number, n: number, N: number) => Math.min(n - 1, Math.max(0, ((o + 0.5) * n) / N - 0.5))
  for (let I = 0; I < H; I++) {
    const y = src(I, h, H)
    const y0 = Math.floor(y)
    const y1 = Math.min(h - 1, y0 + 1)
    const fy = y - y0
    for (let J = 0; J < W; J++) {
      const xx = src(J, w, W)
      const x0 = Math.floor(xx)
      const x1 = Math.min(w - 1, x0 + 1)
      const fx = xx - x0
      const top = v[y0 * w + x0] * (1 - fx) + v[y0 * w + x1] * fx
      const bottom = v[y1 * w + x0] * (1 - fx) + v[y1 * w + x1] * fx
      out[I * W + J] = top * (1 - fy) + bottom * fy
    }
  }
  return fromData(out, [H, W])
}
