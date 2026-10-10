/**
 * Gaussian and Laplacian pyramids (Burt and Adelson, 1983, "The Laplacian pyramid as a compact image code", IEEE Trans.
 * Commun. 31(4)): `reduce` blurs with the five-tap generating kernel $[1, 4, 6, 4, 1] / 16$ ($a = 0.375$) and keeps
 * every other row and column; `expand` inserts zeros and blurs with four times the kernel. Each Laplacian level is a
 * Gaussian level minus the expansion of the next, and adding the expansions back up the pyramid restores the image
 * exactly.
 *
 * Images of any size are allowed: a level of $h \times w$ reduces to $\lceil h/2 \rceil \times \lceil w/2 \rceil$.
 * The default border is `mirror`, as OpenCV's `pyrDown` and `pyrUp`.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readImage, separableFilter, type Border, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The generating kernel of Burt and Adelson with $a = 3/8$: $[1, 4, 6, 4, 1] / 16$. */
const KERNEL = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]

/** Options for the pyramid operators: the border mode (default `mirror`, OpenCV's BORDER_REFLECT_101). */
export type PyramidOptions = { border?: Border }

/**
 * One `reduce` step: blur with the generating kernel, keep even rows and columns (size
 * $\lceil h/2 \rceil \times \lceil w/2 \rceil$).
 *
 * @param img The image, $h \times w$.
 * @param options The border mode.
 * @param options.border How pixels beyond the edge are read.
 * @returns The reduced image, $\lceil h/2 \rceil \times \lceil w/2 \rceil$.
 *
 * @example A constant stays constant; an impulse spreads
 * print('constant:', pyramidReduce([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
 * const impulse = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
 * print('impulse:', pyramidReduce(impulse))
 */
export function pyramidReduce(img: ImageInput, { border = 'mirror' }: PyramidOptions = {}): Tensor {
  const { h, w } = readImage(img, 'pyramidReduce')
  const b = separableFilter(img, KERNEL, KERNEL, { border }).data
  const H = Math.ceil(h / 2)
  const W = Math.ceil(w / 2)
  const out = new Float64Array(H * W)
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) out[r * W + c] = b[2 * r * w + 2 * c]
  return fromData(out, [H, W])
}

/**
 * One `expand` step to size $H \times W$: zeros between samples, then the generating kernel times four. Throws
 * `DomainError` unless the image is $\lceil H/2 \rceil \times \lceil W/2 \rceil$, as `pyramidReduce` would leave it.
 *
 * @param img The image to expand, $h \times w$.
 * @param shape The target size $[H, W]$, which must reduce to $h \times w$.
 * @param options The border mode.
 * @param options.border How pixels beyond the edge are read.
 * @returns The expanded image, $H \times W$.
 *
 * @example A constant stays constant; a single pixel becomes four times the kernel's outer product
 * print('constant:', pyramidExpand([[1, 1], [1, 1]], [4, 4]))
 * print('impulse:', pyramidExpand([[0, 0, 0], [0, 1, 0], [0, 0, 0]], [5, 5]))
 */
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

/**
 * A Gaussian pyramid: the image and up to `levels` $- 1$ successive reductions. Reduction stops early once a level's
 * shorter side is below 8, so the coarsest level's shorter side is under 8 (and at least 4, unless the image's was).
 *
 * @param img The image, $h \times w$.
 * @param options The border mode of `PyramidOptions`, and `levels`, the most levels returned (default as many as the
 *   size allows).
 * @param options.levels The most levels, the image included.
 * @param options.options The remaining field, `border`, passed to `pyramidReduce`.
 * @returns The levels, finest (the image itself) first.
 *
 * @example Sizes of the levels of a 20 by 13 image, and of a 16 by 16 one
 * const img = Array.from({ length: 20 }, (_, r) => Array.from({ length: 13 }, (_, c) => r * c))
 * print('20 x 13:', gaussianPyramid(img).map((level) => level.shape))
 * const square = Array.from({ length: 16 }, () => new Array(16).fill(1))
 * print('16 x 16:', gaussianPyramid(square).map((level) => level.shape))
 */
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
 * A Laplacian pyramid: band-pass levels $L_k = G_k - \operatorname{expand}(G_{k+1})$ and, last, the coarsest Gaussian
 * level (the residual low-pass image), so `reconstructLaplacian` restores the image exactly.
 *
 * @param img The image, $h \times w$.
 * @param options The border mode and the most levels, as `gaussianPyramid`.
 * @returns As many levels as the Gaussian pyramid, finest first, the last being its coarsest level.
 *
 * @example A constant image has nothing in its band-pass levels
 * const img = Array.from({ length: 16 }, () => new Array(16).fill(5))
 * const levels = laplacianPyramid(img)
 * print('sizes:', levels.map((level) => level.shape))
 * print('band-pass levels, largest value:', levels.slice(0, -1).map((level) => max(abs(level))))
 * print('coarsest level:', levels[levels.length - 1])
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

/**
 * The image from its Laplacian pyramid: expand the residual and add each band-pass level, coarse to fine. Throws
 * `DomainError` for no levels.
 *
 * @param levels The Laplacian pyramid, finest first, as `laplacianPyramid` returns it.
 * @param options The border mode, which must be the one the pyramid was built with for an exact reconstruction.
 * @returns The image, the size of the finest level.
 *
 * @example The round trip is exact
 * const img = Array.from({ length: 20 }, (_, r) => Array.from({ length: 13 }, (_, c) => r * c))
 * const back = reconstructLaplacian(laplacianPyramid(img))
 * print('largest error:', max(abs(sub(back, tensor(img)))))
 */
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
