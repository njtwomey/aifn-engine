/**
 * Two-dimensional correlation and convolution with border modes, part of `aifn-compute/foundation/convolution`: `pad` with the
 * border's mode, then a `valid` convolution of the family, so image filters are differentiable in the image and the
 * kernel and batch under vmap.
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

/** How samples beyond the edge are read, as scipy.ndimage: `reflect` (d c b a | a b c d), `mirror` (d c b | a b c d), `nearest`, `constant` (zero) or `wrap`. */
export type Border = 'reflect' | 'mirror' | 'nearest' | 'constant' | 'wrap'

/** An image or kernel: a rank-2 tensor or rows. The filters also take traced values. */
export type ImageInput = Tensor | readonly (readonly number[])[]

type AnyImage = ImageInput | Value

/** scipy.ndimage's border names as numpy.pad's modes. */
const padMode: Record<Border, PadMode> = {
  reflect: 'symmetric',
  mirror: 'reflect',
  nearest: 'edge',
  constant: 'constant',
  wrap: 'wrap',
}

function image(x: AnyImage, what: string): Value {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a 2-D array`)
  const v = Array.isArray(x) ? rowsToTensor(x as readonly (readonly number[])[]) : (x as Value)
  if (shapeOfValue(v).length !== 2) throw new ShapeError(what, `${what}: expected a 2-D array`)
  return v
}

function rowsToTensor(rows: readonly (readonly number[])[]): Tensor {
  const h = rows.length
  const w = rows[0]?.length ?? 0
  const v = new Float64Array(h * w)
  rows.forEach((row, r) => v.set(row, r * w))
  return fromData(v, [h, w])
}

/** Pad by the kernel's centre ⌊size/2⌋ with the border mode, then the valid convolution (flipped kernel) or correlation. */
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
 * Cross-correlation out[r, c] = Σ_{i,j} k[i, j] img[r + i − ⌊kh/2⌋, c + j − ⌊kw/2⌋], as `scipy.ndimage.correlate`
 * (the kernel's centre at ⌊size/2⌋).
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

/** Convolution: correlation with the kernel flipped in both axes, as `scipy.ndimage.convolve` (odd kernel sizes). */
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

/** Separable filtering: correlate each row with kx, then each column with ky. */
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
 * An image (a 2-D tensor or rows of numbers) as its row-major values, height and width; `what` names the caller in
 * errors. For filters written over raw arrays.
 */
export function readImage(img: ImageInput, what: string): { v: Float64Array; h: number; w: number } {
  const t = isTensor(img) ? img : rowsToTensor(img)
  if (t.shape.length !== 2) throw new ShapeError(what, `${what}: expected a 2-D array`)
  return { v: Float64Array.from(dense.data(t)), h: t.shape[0], w: t.shape[1] }
}
