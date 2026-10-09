/**
 * Two-dimensional correlation and convolution of images with border modes, as `scipy.ndimage`.
 *
 * Each filter is `pad` with the border's mode, by the kernel's extent around its centre, then a `valid` convolution of
 * the family, so the output has the image's shape and image filters are differentiable in the image and the kernel
 * and batch under `vmap`. Images and kernels are rank-2 tensors or arrays of rows.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  dense,
  fromData,
  isTensor,
  isTraced,
  reshape,
  shapeOfValue,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { conv, type ConvMethod } from './conv'
import { pad, type PadMode } from './pad'

/**
 * How samples beyond the edge are read, with `scipy.ndimage`'s names, shown for the row `a b c d`: `reflect`
 * (`d c b a | a b c d`, the edge repeated), `mirror` (`d c b | a b c d`, the edge not repeated), `nearest`
 * (`a a a | a b c d`), `constant` (zeros) or `wrap` (`b c d | a b c d`, periodic).
 */
export type Border = 'reflect' | 'mirror' | 'nearest' | 'constant' | 'wrap'

/**
 * An image or kernel: a rank-2 tensor, or an array of rows of equal length (the first row's length is the width). The
 * filters also take traced values.
 */
export type ImageInput = Tensor | readonly (readonly number[])[]

/** An image or kernel as the filters take it: an `ImageInput` or any value, traced ones included. */
type AnyImage = ImageInput | Value

/** scipy.ndimage's border names as numpy.pad's modes. */
const padMode: Record<Border, PadMode> = {
  reflect: 'symmetric',
  mirror: 'reflect',
  nearest: 'edge',
  constant: 'constant',
  wrap: 'wrap',
}

/**
 * An image as a rank-2 value: rows become a tensor, and tensors and traced values pass through. Throws `ShapeError`
 * for a number or a value that is not rank-2.
 *
 * @param x The image or kernel.
 * @param what The caller's name for error messages.
 * @returns The image as a rank-2 value.
 */
function image(x: AnyImage, what: string): Value {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a 2-D array`)
  const v = Array.isArray(x) ? rowsToTensor(x as readonly (readonly number[])[]) : (x as Value)
  if (shapeOfValue(v).length !== 2) throw new ShapeError(what, `${what}: expected a 2-D array`)
  return v
}

/**
 * A tensor from rows of numbers.
 *
 * @param rows The rows, each as long as the first (which sets the width).
 * @returns A new `[h, w]` tensor with the rows' values, row-major.
 */
function rowsToTensor(rows: readonly (readonly number[])[]): Tensor {
  const h = rows.length
  const w = rows[0]?.length ?? 0
  const v = new Float64Array(h * w)
  rows.forEach((row, r) => v.set(row, r * w))
  return fromData(v, [h, w])
}

/**
 * Filter an image: pad it with the border mode, by the kernel's centre $\lfloor k / 2 \rfloor$ before and
 * $k - 1 - \lfloor k / 2 \rfloor$ after on each axis ($k$ the kernel's length on that axis), then take the `valid`
 * convolution (kernel flipped) or correlation, which has the image's shape.
 *
 * @param img The image, `[H, W]` or rows.
 * @param kernel The kernel, `[kh, kw]` or rows.
 * @param border How samples beyond the image's edges are read.
 * @param flip True for convolution (the kernel flipped in both axes), false for correlation.
 * @param method The convolution kernel.
 * @param what The caller's name for error messages.
 * @returns The filtered image, `[H, W]`.
 */
function filter2d(img: AnyImage, kernel: AnyImage, border: Border, flip: boolean, method: ConvMethod, what: string) {
  const I = image(img, what)
  const K = image(kernel, `${what} kernel`)
  const [H, W] = shapeOfValue(I)
  const [kh, kw] = shapeOfValue(K)
  const [cy, cx] = [Math.floor(kh / 2), Math.floor(kw / 2)]
  const padded = pad(
    I,
    [
      [cy, kh - 1 - cy],
      [cx, kw - 1 - cx],
    ],
    padMode[border],
  )
  const [ph, pw] = shapeOfValue(padded)
  const y = conv(reshape(padded, [1, 1, ph, pw]), reshape(K, [1, 1, kh, kw]), { flip, method })
  return reshape(y, [H, W])
}

/**
 * Cross-correlation $y_{rc} = \sum_{i,j} k_{ij}\, x_{r + i - \lfloor k_h/2 \rfloor,\, c + j - \lfloor k_w/2 \rfloor}$
 * of an image $x$ with a $k_h \times k_w$ kernel $k$, as `scipy.ndimage.correlate` (the kernel's centre at
 * $(\lfloor k_h/2 \rfloor, \lfloor k_w/2 \rfloor)$), with samples beyond the edges read by the border mode. The
 * output has the image's shape. Differentiable in the image and the kernel. Throws `ShapeError` when either is not
 * two-dimensional.
 *
 * @param img The image $x$, `[H, W]`: a rank-2 tensor, rows of numbers or a traced value.
 * @param kernel The kernel $k$, `[kh, kw]`, in the same forms.
 * @param options How the edges are read and the convolution kernel.
 * @param options.border How samples beyond the image's edges are read (default `reflect`, the edge repeated).
 * @param options.method The convolution kernel (default `auto`).
 * @returns The filtered image $y$, `[H, W]`.
 *
 * @example A horizontal difference on a ramp
 * const img = [[1, 2, 4, 8], [1, 2, 4, 8]]
 * print('nearest  ', correlate2d(img, [[-1, 0, 1]], { border: 'nearest' }))
 * print('constant ', correlate2d(img, [[-1, 0, 1]], { border: 'constant' }))
 */
export function correlate2d(
  img: ImageInput,
  kernel: ImageInput,
  options?: { border?: Border; method?: ConvMethod },
): Tensor
export function correlate2d(img: AnyImage, kernel: AnyImage, options?: { border?: Border; method?: ConvMethod }): Value
export function correlate2d(
  img: AnyImage,
  kernel: AnyImage,
  { border = 'reflect', method = 'auto' }: { border?: Border; method?: ConvMethod } = {},
): Value {
  return filter2d(img, kernel, border, false, method, 'correlate2d')
}

/**
 * Convolution: correlation with the kernel flipped in both axes, as `scipy.ndimage.convolve` for odd kernel sizes,
 * where it is $y_{rc} = \sum_{i,j} k_{ij}\, x_{r - i + \lfloor k_h/2 \rfloor,\, c - j + \lfloor k_w/2 \rfloor}$.
 * An even-sized kernel is padded as in `correlate2d`, so on that axis the result is shifted by one sample from
 * scipy's. The output has the image's shape. Differentiable in the image and the kernel. Throws `ShapeError` when the
 * image or the kernel is not two-dimensional.
 *
 * @param img The image $x$, `[H, W]`: a rank-2 tensor, rows of numbers or a traced value.
 * @param kernel The kernel $k$, `[kh, kw]`, in the same forms.
 * @param options How the edges are read and the convolution kernel.
 * @param options.border How samples beyond the image's edges are read (default `reflect`, the edge repeated).
 * @param options.method The convolution kernel (default `auto`).
 * @returns The filtered image $y$, `[H, W]`.
 *
 * @example Convolving an impulse reproduces the kernel; correlating reverses it
 * const impulse = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
 * const k = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
 * print('convolve2d  ', convolve2d(impulse, k, { border: 'constant' }))
 * print('correlate2d ', correlate2d(impulse, k, { border: 'constant' }))
 */
export function convolve2d(
  img: ImageInput,
  kernel: ImageInput,
  options?: { border?: Border; method?: ConvMethod },
): Tensor
export function convolve2d(img: AnyImage, kernel: AnyImage, options?: { border?: Border; method?: ConvMethod }): Value
export function convolve2d(
  img: AnyImage,
  kernel: AnyImage,
  { border = 'reflect', method = 'auto' }: { border?: Border; method?: ConvMethod } = {},
): Value {
  return filter2d(img, kernel, border, true, method, 'convolve2d')
}

/**
 * Separable filtering: correlate each row with `kx`, then each column with `ky` (both by `correlate2d`, so centred at
 * $\lfloor k/2 \rfloor$), which equals correlating with the outer product of `ky` and `kx` at the cost of two
 * one-dimensional passes. Differentiable in the image and both kernels.
 *
 * @param img The image, `[H, W]`: a rank-2 tensor, rows of numbers or a traced value.
 * @param kx The kernel along each row (the horizontal axis): a tensor, a traced value or an array of numbers; its
 *   values are read in row-major order whatever its shape.
 * @param ky The kernel along each column (the vertical axis), in the same forms.
 * @param options How the edges are read.
 * @returns The filtered image, `[H, W]`.
 *
 * @example A 3-by-3 box blur as two passes
 * const img = [[0, 0, 0, 0], [0, 9, 0, 0], [0, 0, 0, 0]]
 * const box = [1 / 3, 1 / 3, 1 / 3]
 * print('separable  ', separableFilter(img, box, box, { border: 'constant' }))
 * const box2d = [box, box, box].map((r) => r.map((v) => v / 3))
 * print('2-D kernel ', correlate2d(img, box2d, { border: 'constant' }))
 */
export function separableFilter(
  img: ImageInput,
  kx: Tensor | ArrayLike<number>,
  ky: Tensor | ArrayLike<number>,
  options?: { border?: Border },
): Tensor
export function separableFilter(
  img: AnyImage,
  kx: Value | ArrayLike<number>,
  ky: Value | ArrayLike<number>,
  options?: { border?: Border },
): Value
export function separableFilter(
  img: AnyImage,
  kx: Value | ArrayLike<number>,
  ky: Value | ArrayLike<number>,
  options: { border?: Border } = {},
): Value {
  const row = (k: Value | ArrayLike<number>, shape: (n: number) => number[]): Value => {
    const v = isTraced(k) || isTensor(k) ? (k as Value) : fromData(Float64Array.from(k as ArrayLike<number>))
    return reshape(v, shape(shapeOfValue(v).reduce((a, b) => a * b, 1)))
  }
  const rows = correlate2d(
    img,
    row(kx, (n) => [1, n]),
    options,
  )
  return correlate2d(
    rows,
    row(ky, (n) => [n, 1]),
    options,
  )
}

/**
 * An image as its row-major values, height and width, for filters written over raw arrays. Throws `ShapeError` for a
 * tensor that is not two-dimensional.
 *
 * @param img The image: a rank-2 tensor or rows of numbers (not a traced value).
 * @param what The caller's name for error messages.
 * @returns `v`, a copy of the values (row `r` occupies entries `r * w` to `r * w + w - 1`), `h`, the height, and `w`,
 *   the width.
 *
 * @example Read rows into a flat array
 * const { v, h, w } = readImage([[1, 2, 3], [4, 5, 6]], 'demo')
 * print('h, w =', h, w)
 * print('v =', v)
 */
export function readImage(img: ImageInput, what: string): { v: Float64Array; h: number; w: number } {
  const t = isTensor(img) ? img : rowsToTensor(img)
  if (t.shape.length !== 2) throw new ShapeError(what, `${what}: expected a 2-D array`)
  return { v: Float64Array.from(dense.data(t)), h: t.shape[0], w: t.shape[1] }
}
