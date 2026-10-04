/**
 * Gaussian and Laplacian pyramids (Burt & Adelson, 1983, "The Laplacian pyramid as a compact image code", IEEE Trans.
 * Commun. 31(4)): `reduce` blurs with the five-tap generating kernel [1, 4, 6, 4, 1]/16 (a = 0.375) and keeps every other
 * row and column; `expand` inserts zeros and blurs with four times the kernel. Each Laplacian level is a Gaussian level
 * minus the expansion of the next, and adding the expansions back up the pyramid restores the image exactly.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, separableFilter, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The generating kernel of Burt and Adelson with a = 3/8: [1, 4, 6, 4, 1]/16. */
const KERNEL = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]

/** Options for the pyramid operators: the border mode (default `mirror`, OpenCV's BORDER_REFLECT_101). */
export type PyramidOptions = { border?: Border }

/** One `reduce` step: blur with the generating kernel, keep even rows and columns (size ⌈h/2⌉ × ⌈w/2⌉). */
export function pyramidReduce(img: ImageInput, { border = 'mirror' }: PyramidOptions = {}): Tensor {
  const { h, w } = readImage(img, 'pyramidReduce')
  const b = separableFilter(img, KERNEL, KERNEL, { border }).data
  const H = Math.ceil(h / 2)
  const W = Math.ceil(w / 2)
  const out = new Float64Array(H * W)
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) out[r * W + c] = b[2 * r * w + 2 * c]
  return fromData(out, [H, W])
}

/** One `expand` step to size h × w: zeros between samples, then the generating kernel times four. */
export function pyramidExpand(
  img: ImageInput,
  shape: readonly [number, number],
  { border = 'mirror' }: PyramidOptions = {},
): Tensor {
  const { v, h, w } = readImage(img, 'pyramidExpand')
  const [H, W] = shape
  if (Math.ceil(H / 2) !== h || Math.ceil(W / 2) !== w)
    throw new DomainError('pyramidExpand', `pyramidExpand: ${h} × ${w} does not expand to ${H} × ${W}`)
  const up = new Float64Array(H * W)
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) up[2 * r * W + 2 * c] = 4 * v[r * w + c]
  return separableFilter(fromData(up, [H, W]), KERNEL, KERNEL, { border })
}

/** A Gaussian pyramid: the image and `levels − 1` successive reductions (default: until a side would drop below 4). */
export function gaussianPyramid(
  img: ImageInput,
  { levels, ...options }: PyramidOptions & { levels?: number } = {},
): Tensor[] {
  const { v, h, w } = readImage(img, 'gaussianPyramid')
  const out: Tensor[] = [fromData(v, [h, w])]
  const max = levels ?? Infinity
  while (out.length < max) {
    const [a, b] = out[out.length - 1].shape
    if (Math.min(a, b) < 8) break
    out.push(pyramidReduce(out[out.length - 1], options))
  }
  return out
}

/**
 * A Laplacian pyramid: band-pass levels Lₖ = Gₖ − expand(Gₖ₊₁) and, last, the coarsest Gaussian level (the residual
 * low-pass image), so `reconstructLaplacian` restores the image exactly.
 */
export function laplacianPyramid(img: ImageInput, options: PyramidOptions & { levels?: number } = {}): Tensor[] {
  const g = gaussianPyramid(img, options)
  const out: Tensor[] = []
  for (let k = 0; k + 1 < g.length; k++) {
    const e = pyramidExpand(g[k + 1], g[k].shape as [number, number], options).data
    out.push(
      fromData(
        Float64Array.from(g[k].data, (x, i) => x - e[i]),
        g[k].shape,
      ),
    )
  }
  out.push(g[g.length - 1])
  return out
}

/** The image from its Laplacian pyramid: expand the residual and add each band-pass level, coarse to fine. */
export function reconstructLaplacian(levels: readonly Tensor[], options: PyramidOptions = {}): Tensor {
  if (!levels.length) throw new DomainError('reconstructLaplacian', 'reconstructLaplacian: no levels')
  let img = levels[levels.length - 1]
  for (let k = levels.length - 2; k >= 0; k--) {
    const e = pyramidExpand(img, levels[k].shape as [number, number], options).data
    img = fromData(
      Float64Array.from(levels[k].data, (x, i) => x + e[i]),
      levels[k].shape,
    )
  }
  return img
}
