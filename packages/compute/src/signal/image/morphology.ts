/**
 * Grey-scale mathematical morphology with a flat structuring element (Serra, 1982, "Image Analysis and Mathematical
 * Morphology"; Soille, 2004, "Morphological Image Analysis", 2nd ed.): erosion is the minimum over the element placed
 * at each pixel, dilation the maximum, opening is erosion then dilation (removes bright details smaller than the
 * element) and closing the reverse (fills dark ones). On a 0/1 image these are the binary operations.
 *
 * Pixels beyond the edge are read with scipy.ndimage's border modes (default `reflect`), except that `constant` here
 * ignores them (it pads with the operation's identity, $+\infty$ for erosion and $-\infty$ for dilation), where scipy
 * pads with a constant value.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A flat structuring element: a 0/1 footprint as rows, with odd sides, centred; the operators take the pixels where it
 * is nonzero.
 */
export type StructuringElement = readonly (readonly number[])[]

/**
 * A square structuring element of side `size`. Throws `DomainError` unless `size` is an odd positive integer.
 *
 * @param size The side, in pixels: odd.
 * @returns The `size` by `size` element of ones.
 *
 * @example A 3 by 3 square
 * print(squareElement(3))
 */
export function squareElement(size: number): StructuringElement {
  if (!(size >= 1 && size % 2 === 1)) throw new DomainError('squareElement', 'squareElement: the size must be odd')
  return Array.from({ length: size }, () => new Array<number>(size).fill(1))
}

/**
 * A disc structuring element of the given radius: the offsets $(\delta_r, \delta_c)$ with
 * $\delta_r^2 + \delta_c^2 \le \text{radius}^2$, side $2 \lfloor \text{radius} \rfloor + 1$.
 *
 * @param radius The radius in pixels; it need not be an integer.
 * @returns The element, as rows of 0 and 1.
 *
 * @example Discs of radius 2 and 1.5
 * print('radius 2:', discElement(2))
 * print('radius 1.5:', discElement(1.5))
 */
export function discElement(radius: number): StructuringElement {
  const R = Math.floor(radius)
  return Array.from({ length: 2 * R + 1 }, (_, i) =>
    Array.from({ length: 2 * R + 1 }, (_, j) => ((i - R) ** 2 + (j - R) ** 2 <= radius * radius ? 1 : 0)),
  )
}

/**
 * The pixel index of $(r, c)$ under a border mode (scipy.ndimage's), or $-1$ when it is outside the image and the mode
 * is `constant`.
 *
 * @param r The row, possibly outside $0, \dots, h - 1$.
 * @param c The column, possibly outside $0, \dots, w - 1$.
 * @param h The image height.
 * @param w The image width.
 * @param border How a position beyond the edge is folded back into the image.
 * @returns The row-major index $r w + c$ of the pixel read, or $-1$.
 */
function index(r: number, c: number, h: number, w: number, border: Border): number {
  const fold = (i: number, n: number): number => {
    if (i >= 0 && i < n) return i
    switch (border) {
      case 'nearest':
        return i < 0 ? 0 : n - 1
      case 'wrap':
        return ((i % n) + n) % n
      case 'mirror': {
        if (n === 1) return 0
        const p = 2 * (n - 1)
        const k = ((i % p) + p) % p
        return k < n ? k : p - k
      }
      case 'reflect': {
        const p = 2 * n
        const k = ((i % p) + p) % p
        return k < n ? k : p - 1 - k
      }
      default:
        return -1
    }
  }
  const rr = fold(r, h)
  const cc = fold(c, w)
  return rr < 0 || cc < 0 ? -1 : rr * w + cc
}

/**
 * The minimum or maximum over a structuring element placed at each pixel (the element is not reflected here).
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element; its sides must be odd, or `DomainError` is thrown.
 * @param border How pixels beyond the edge are read.
 * @param kind `'min'` for erosion, `'max'` for dilation.
 * @param cval The value read beyond the edge when `border` is `constant`.
 * @param where The caller's name for error messages.
 * @returns The filtered image, $h \times w$.
 */
function extremum(
  img: ImageInput,
  element: StructuringElement,
  border: Border,
  kind: 'min' | 'max',
  cval: number,
  where: string,
): Tensor {
  const { v, h, w } = readImage(img, where)
  const kh = element.length
  const kw = element[0]?.length ?? 0
  if (kh % 2 === 0 || kw % 2 === 0) throw new DomainError(where, `${where}: the structuring element needs odd sides`)
  const offs: [number, number][] = []
  for (let i = 0; i < kh; i++)
    for (let j = 0; j < kw; j++) if (element[i][j]) offs.push([i - (kh - 1) / 2, j - (kw - 1) / 2])
  const out = new Float64Array(h * w)
  const min = kind === 'min'
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      let best = min ? Infinity : -Infinity
      for (const [dr, dc] of offs) {
        const k = index(r + dr, c + dc, h, w, border)
        const x = k < 0 ? cval : v[k]
        if (min ? x < best : x > best) best = x
      }
      out[r * w + c] = best
    }
  return fromData(out, [h, w])
}

/** Options for the morphological operators: the border mode (default `reflect`, as scipy.ndimage). */
export type MorphologyOptions = { border?: Border }

/**
 * Erosion: the minimum over the structuring element at each pixel, as `scipy.ndimage.grey_erosion` (footprint). A
 * `constant` border is ignored rather than read as zeros, so it does not erode the image's edge.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode.
 * @param options.border How pixels beyond the edge are read.
 * @returns The eroded image, $h \times w$.
 *
 * @example A 3 by 3 square shrinks to its centre, and a single pixel vanishes
 * const img = [
 *   [0, 0, 0, 0, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 1, 0],
 *   [0, 0, 0, 0, 0, 0, 0],
 * ]
 * print(erode(img, squareElement(3)))
 */
export function erode(
  img: ImageInput,
  element: StructuringElement,
  { border = 'reflect' }: MorphologyOptions = {},
): Tensor {
  // A constant border must not erode the image's edge: pad with +∞ (the erosion's identity).
  return extremum(img, element, border, 'min', Infinity, 'erode')
}

/**
 * Dilation: the maximum over the reflected structuring element at each pixel, as `scipy.ndimage.grey_dilation`
 * (footprint). For a symmetric element the reflection changes nothing.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode.
 * @param options.border How pixels beyond the edge are read.
 * @returns The dilated image, $h \times w$.
 *
 * @example A single pixel grows into the element
 * print(dilate([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], discElement(1)))
 */
export function dilate(
  img: ImageInput,
  element: StructuringElement,
  { border = 'reflect' }: MorphologyOptions = {},
): Tensor {
  const reflected = [...element].reverse().map((row) => [...row].reverse())
  return extremum(img, reflected, border, 'max', -Infinity, 'dilate')
}

/**
 * Opening: erosion then dilation; removes bright structures the element does not fit inside and keeps the rest.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode, for both steps.
 * @returns The opened image, $h \times w$.
 *
 * @example The square survives, the single pixel does not
 * const img = [
 *   [0, 0, 0, 0, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 1, 0],
 *   [0, 0, 0, 0, 0, 0, 0],
 * ]
 * print(opening(img, squareElement(3)))
 */
export function opening(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  return dilate(erode(img, element, options), element, options)
}

/**
 * Closing: dilation then erosion; fills dark gaps the element does not fit inside.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode, for both steps.
 * @returns The closed image, $h \times w$.
 *
 * @example A one-pixel hole is filled
 * const img = [
 *   [1, 1, 1, 1, 1],
 *   [1, 1, 1, 1, 1],
 *   [1, 1, 0, 1, 1],
 *   [1, 1, 1, 1, 1],
 *   [1, 1, 1, 1, 1],
 * ]
 * print(closing(img, squareElement(3)))
 */
export function closing(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  return erode(dilate(img, element, options), element, options)
}

/**
 * The morphological gradient, dilation minus erosion: the outline of objects, as thick as the element.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode, for both operations.
 * @returns The gradient, $h \times w$, never negative.
 *
 * @example The outline of a square
 * const img = [
 *   [0, 0, 0, 0, 0, 0],
 *   [0, 1, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 1, 0],
 *   [0, 0, 0, 0, 0, 0],
 * ]
 * print(morphologicalGradient(img, discElement(1)))
 */
export function morphologicalGradient(
  img: ImageInput,
  element: StructuringElement,
  options: MorphologyOptions = {},
): Tensor {
  const d = dilate(img, element, options).data
  const e = erode(img, element, options).data
  return fromData(
    Float64Array.from(d, (x, i) => x - e[i]),
    readShape(img),
  )
}

/**
 * The white top-hat, image minus opening: the bright details smaller than the element.
 *
 * @param img The image, $h \times w$.
 * @param element The structuring element, with odd sides.
 * @param options The border mode of the opening.
 * @returns The top-hat, $h \times w$, never negative.
 *
 * @example Only the single pixel is left
 * const img = [
 *   [0, 0, 0, 0, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 0, 0],
 *   [0, 1, 1, 1, 0, 1, 0],
 *   [0, 0, 0, 0, 0, 0, 0],
 * ]
 * print(topHat(img, squareElement(3)))
 */
export function topHat(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  const { v } = readImage(img, 'topHat')
  const o = opening(img, element, options).data
  return fromData(
    Float64Array.from(v, (x, i) => x - o[i]),
    readShape(img),
  )
}

/**
 * The height and width of an image, for the operators that build their result from raw arrays.
 *
 * @param img The image.
 * @returns $[h, w]$.
 */
const readShape = (img: ImageInput): [number, number] => {
  const { h, w } = readImage(img, 'morphology')
  return [h, w]
}
