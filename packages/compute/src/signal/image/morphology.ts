/**
 * Grey-scale mathematical morphology with a flat structuring element (Serra, 1982, "Image Analysis and Mathematical
 * Morphology"; Soille, 2004, "Morphological Image Analysis", 2nd ed.): erosion is the minimum over the element placed at
 * each pixel, dilation the maximum, opening is erosion then dilation (removes bright details smaller than the element)
 * and closing the reverse (fills dark ones). On a 0/1 image these are the binary operations.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A flat structuring element: a 0/1 footprint with odd sides, centred. */
export type StructuringElement = readonly (readonly number[])[]

/** A square structuring element of side `size` (odd). */
export function squareElement(size: number): StructuringElement {
  if (!(size >= 1 && size % 2 === 1)) throw new DomainError('squareElement', 'squareElement: the size must be odd')
  return Array.from({ length: size }, () => new Array<number>(size).fill(1))
}

/** A disc structuring element of the given radius: the offsets with (dr² + dc²) ≤ radius², side 2·radius + 1. */
export function discElement(radius: number): StructuringElement {
  const R = Math.floor(radius)
  return Array.from({ length: 2 * R + 1 }, (_, i) =>
    Array.from({ length: 2 * R + 1 }, (_, j) => ((i - R) ** 2 + (j - R) ** 2 <= radius * radius ? 1 : 0)),
  )
}

/** The pixel index of (r, c) under a border mode (scipy.ndimage's), or −1 for `constant`. */
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

/** Erosion: the minimum over the structuring element at each pixel, as `scipy.ndimage.grey_erosion` (footprint). */
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
 */
export function dilate(
  img: ImageInput,
  element: StructuringElement,
  { border = 'reflect' }: MorphologyOptions = {},
): Tensor {
  const reflected = [...element].reverse().map((row) => [...row].reverse())
  return extremum(img, reflected, border, 'max', -Infinity, 'dilate')
}

/** Opening: erosion then dilation; removes bright structures the element does not fit inside. */
export function opening(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  return dilate(erode(img, element, options), element, options)
}

/** Closing: dilation then erosion; fills dark gaps the element does not fit inside. */
export function closing(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  return erode(dilate(img, element, options), element, options)
}

/** The morphological gradient, dilation − erosion: the outline of objects, as thick as the element. */
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

/** The white top-hat, image − opening: the bright details smaller than the element. */
export function topHat(img: ImageInput, element: StructuringElement, options: MorphologyOptions = {}): Tensor {
  const { v } = readImage(img, 'topHat')
  const o = opening(img, element, options).data
  return fromData(
    Float64Array.from(v, (x, i) => x - o[i]),
    readShape(img),
  )
}

const readShape = (img: ImageInput): [number, number] => {
  const { h, w } = readImage(img, 'morphology')
  return [h, w]
}
