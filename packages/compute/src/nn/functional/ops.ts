/**
 * Convolution and pooling for neural networks, as torch.nn.functional: compositions over the convolution family of
 * `aifn-compute/foundation/convolution` (design K §8.3), so every one is differentiable to any order and batches under
 * `vmap`.
 *
 * Inputs are channels-first, `[N, C, H, W]` (or `[N, C, L]` in 1-D), and the batch axis may be left out. The
 * convolutions are cross-correlations (the family's `conv` with `flip: false`, as in every deep-learning library):
 * $y_{n,o,i,j} = \sum_{c,a,b} x_{n,\,c,\,i s_h - p_h + a d_h,\,j s_w - p_w + b d_w}\, w_{o,c,a,b}$, with zeros
 * outside the image and $s$, $p$, $d$ the stride, padding and dilation (Dumoulin & Visin, 2016, "A guide to
 * convolution arithmetic for deep learning", §2–5, for the sizes). Average pooling is a depthwise convolution with a
 * uniform kernel; max pooling gathers the windows at fixed positions and reduces them by `max`.
 */

import { conv, convOutputSize } from 'aifn-compute/foundation/convolution'
import { concat, fromData, gather, max, reshape, shapeOfValue, type Value } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

export { convOutputSize }

/** A per-axis option: one number for both axes, or a pair `[height, width]`. */
export type Pair = number | readonly [number, number]

/**
 * A per-axis option as a pair.
 *
 * @param v One number, used for both axes, or a pair, copied.
 * @returns The option for the height and width axes, in that order.
 */
const pair = (v: Pair): [number, number] => (typeof v === 'number' ? [v, v] : [v[0], v[1]])

/** Options of `conv2d` and `conv1d`. */
export type ConvOptions<P = Pair> = {
  /** Step between output positions. Default 1. */
  stride?: P
  /** Zeros added on each side of the input. Default 0. */
  padding?: P
  /** Spacing between kernel taps (1 = contiguous). Default 1. */
  dilation?: P
  /**
   * Channel groups: $C$ and $O$ must be divisible by it, and the kernels have shape `[O, C / groups, ...]`. Default 1.
   */
  groups?: number
}

/**
 * 2-D convolution (cross-correlation) of $x$, shape `[N, C, H, W]` (or `[C, H, W]`), with kernels $w$, shape
 * `[O, C / groups, KH, KW]`, giving `[N, O, H', W']` (or `[O, H', W']`) with
 * $H' = \lfloor (H + 2p - d(K_H - 1) - 1)/s \rfloor + 1$ (and likewise $W'$). The convolution family's `conv` with
 * `flip: false`, so differentiable in $x$ and $w$ to any order and batched without a loop. Matches
 * `torch.nn.functional.conv2d` (without bias; add it by broadcasting). Throws `ShapeError` when `w` is not rank 4.
 *
 * @param x The images, `[N, C, H, W]`, or one image `[C, H, W]` (the result then has no batch axis).
 * @param w The kernels, `[O, C / groups, KH, KW]`: output channel, input channel, kernel row, kernel column.
 * @param options The stride, padding, dilation and channel groups; each per-axis one is a number or
 *   `[height, width]`.
 * @returns The output, `[N, O, H', W']`, or `[O, H', W']` for one image.
 *
 * @example A $2 \times 2$ kernel of ones sums each window of a $3 \times 3$ image
 * const x = tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
 * const w = tensor([[[[1, 1], [1, 1]]]])
 * print('valid:', conv2d(x, w))
 * print('padding 1, shape:', conv2d(x, w, { padding: 1 }).shape)
 * print('stride 2 with padding 1:', conv2d(x, w, { padding: 1, stride: 2 }))
 */
export function conv2d(x: Value, w: Value, options: ConvOptions = {}): Value {
  const ws = shapeOfValue(w)
  if (ws.length !== 4)
    throw new ShapeError('conv2d', `conv2d: kernels need shape [O, C, KH, KW], got [${ws.join(', ')}]`)
  return conv(x, w, {
    layout: 'nchw',
    flip: false,
    stride: pair(options.stride ?? 1),
    padding: pair(options.padding ?? 0),
    dilation: pair(options.dilation ?? 1),
    groups: options.groups ?? 1,
    method: 'direct',
  })
}

/**
 * 1-D convolution (cross-correlation) of $x$, shape `[N, C, L]` (or `[C, L]`), with kernels $w$, shape
 * `[O, C / groups, K]`, giving `[N, O, L']` with $L' = \lfloor (L + 2p - d(K - 1) - 1)/s \rfloor + 1$. The convolution
 * family's `conv` with `flip: false`, as `torch.nn.functional.conv1d`, and differentiable to any order. Throws
 * `ShapeError` when `w` is not rank 3.
 *
 * @param x The signals, `[N, C, L]`, or one signal `[C, L]` (the result then has no batch axis).
 * @param w The kernels, `[O, C / groups, K]`.
 * @param options The stride, padding, dilation (each one number) and channel groups.
 * @returns The output, `[N, O, L']`, or `[O, L']` for one signal.
 *
 * @example The kernel $[1, -1]$ takes differences of neighbours, with no flip
 * const x = tensor([[[1, 2, 4, 7, 11]]])
 * print('differences:', conv1d(x, tensor([[[1, -1]]])))
 * print('dilation 2:', conv1d(x, tensor([[[1, -1]]]), { dilation: 2 }))
 */
export function conv1d(x: Value, w: Value, options: ConvOptions<number> = {}): Value {
  const ws = shapeOfValue(w)
  if (ws.length !== 3) throw new ShapeError('conv1d', `conv1d: kernels need shape [O, C, K]`)
  return conv(x, w, {
    layout: 'ncw',
    flip: false,
    stride: options.stride ?? 1,
    padding: options.padding ?? 0,
    dilation: options.dilation ?? 1,
    groups: options.groups ?? 1,
    method: 'direct',
  })
}

// ── Pooling ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The geometry of a 2-D pooling: `kernel`, `stride` and `padding` as `[height, width]` pairs, and `x` the shape
 * `[N, C, H, W]` of the batched input.
 */
type PoolParams = { kernel: [number, number]; stride: [number, number]; padding: [number, number]; x: number[] }

/**
 * The output shape of a 2-D pooling.
 *
 * @param p The pooling geometry and the batched input shape.
 * @returns `[N, C, H', W']`, the sizes of `convOutputSize` with dilation 1.
 */
function poolShape(p: PoolParams): number[] {
  const [N, C, H, W] = p.x
  return [
    N,
    C,
    convOutputSize(H, p.kernel[0], p.stride[0], p.padding[0], 1),
    convOutputSize(W, p.kernel[1], p.stride[1], p.padding[1], 1),
  ]
}

/** Options of the pooling functions. */
export type PoolOptions<P = Pair> = {
  /** Step between windows. Default: the kernel size (non-overlapping windows). */
  stride?: P
  /** Padding on each side: zeros for average pooling (counted in the mean), $-\infty$ for max pooling. Default 0. */
  padding?: P
}

/**
 * The common start of the 2-D poolings: the input with a batch axis, the pooling geometry, and a function that removes
 * the batch axis again when the input had none. Throws `ShapeError` unless the input is rank 3 or 4.
 *
 * @param x The input, `[N, C, H, W]` or `[C, H, W]`.
 * @param kernel The window size, one number or `[height, width]`.
 * @param options The stride (default: the kernel size) and padding (default 0).
 * @param what The caller's name, for error messages.
 * @returns `x4`, the input as `[N, C, H, W]`; `p`, the geometry with defaults filled in; `unbatch`, which drops the
 *   front axis of an output when `x` was unbatched.
 */
function pool4(x: Value, kernel: Pair, options: PoolOptions, what: string) {
  const xs = shapeOfValue(x)
  const batched = xs.length === 4
  if (!batched && xs.length !== 3) throw new ShapeError(what, `${what}: input needs shape [N, C, H, W] or [C, H, W]`)
  const x4 = batched ? x : reshape(x, [1, ...xs])
  const k = pair(kernel)
  const p: PoolParams = {
    kernel: k,
    stride: pair(options.stride ?? k),
    padding: pair(options.padding ?? 0),
    x: shapeOfValue(x4),
  }
  const unbatch = (y: Value) => (batched ? y : reshape(y, shapeOfValue(y).slice(1)))
  return { x4, p, unbatch }
}

/**
 * 2-D average pooling of `[N, C, H, W]` (or `[C, H, W]`) over kernel windows; padded zeros count in the mean (PyTorch's
 * default `count_include_pad`). A depthwise convolution (one group per channel) with the uniform kernel
 * $1/(k_h k_w)$, so linear and differentiable to any order. Throws `ShapeError` unless `x` is rank 3 or 4.
 *
 * @param x The input, `[N, C, H, W]`, or `[C, H, W]` (the result then has no batch axis).
 * @param kernel The window size $k_h \times k_w$: one number for a square window, or `[height, width]`.
 * @param options The stride (default: the kernel size, so windows do not overlap) and the zero padding.
 * @returns The window means, `[N, C, H', W']` or `[C, H', W']`.
 *
 * @example Non-overlapping $2 \times 2$ windows of a $4 \times 4$ image
 * const x = tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
 * print('means:', avgPool2d(x, 2))
 * print('stride 1, shape:', avgPool2d(x, 2, { stride: 1 }).shape)
 */
export function avgPool2d(x: Value, kernel: Pair, options: PoolOptions = {}): Value {
  const { x4, p, unbatch } = pool4(x, kernel, options, 'avgPool2d')
  const C = p.x[1]
  const [kh, kw] = p.kernel
  const w = fromData(new Float64Array(C * kh * kw).fill(1 / (kh * kw)), [C, 1, kh, kw])
  return unbatch(
    conv(x4, w, { layout: 'nchw', flip: false, groups: C, stride: p.stride, padding: p.padding, method: 'direct' }),
  )
}

/**
 * 2-D max pooling of `[N, C, H, W]` (or `[C, H, W]`) over kernel windows; padding reads $-\infty$. A composition: the
 * windows are gathered at fixed positions (padded taps read one appended $-\infty$) into a trailing axis, and `max`
 * reduces it, so the derivative is max's (split equally among tied taps) and it batches and differentiates to any
 * order. Throws `ShapeError` unless `x` is rank 3 or 4.
 *
 * @param x The input, `[N, C, H, W]`, or `[C, H, W]` (the result then has no batch axis).
 * @param kernel The window size: one number for a square window, or `[height, width]`.
 * @param options The stride (default: the kernel size, so windows do not overlap) and the padding, which reads
 *   $-\infty$ and so is never chosen unless a whole window is padding.
 * @returns The window maxima, `[N, C, H', W']` or `[C, H', W']`.
 *
 * @example The largest entry of each $2 \times 2$ window, and the gradient that routes to it
 * const x = tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
 * print('maxima:', maxPool2d(x, 2))
 * print('gradient of the sum:', grad((v) => sum(maxPool2d(v, 2)))(x))
 */
export function maxPool2d(x: Value, kernel: Pair, options: PoolOptions = {}): Value {
  const { x4, p, unbatch } = pool4(x, kernel, options, 'maxPool2d')
  const [N, C, H, W] = p.x
  const [, , HO, WO] = poolShape(p)
  const [kh, kw] = p.kernel
  const outside = N * C * H * W
  const at = new Int32Array(N * C * HO * WO * kh * kw)
  let k = 0
  for (let nc = 0; nc < N * C; nc++)
    for (let i = 0; i < HO; i++)
      for (let j = 0; j < WO; j++)
        for (let a = 0; a < kh; a++) {
          const r = i * p.stride[0] - p.padding[0] + a
          for (let b = 0; b < kw; b++) {
            const q = j * p.stride[1] - p.padding[1] + b
            at[k++] = r < 0 || r >= H || q < 0 || q >= W ? outside : (nc * H + r) * W + q
          }
        }
  const padded = concat([reshape(x4, [-1]), fromData(Float64Array.of(-Infinity))], 0)
  const windows = gather(padded, at, [N, C, HO, WO, kh * kw])
  return unbatch(max(windows, -1))
}

/**
 * 1-D average pooling of `[N, C, L]` (or `[C, L]`): `avgPool2d` over a height-1 view, so padded zeros count in the mean
 * and it is differentiable to any order.
 *
 * @param x The input, `[N, C, L]`, or `[C, L]` (the result then has no batch axis).
 * @param kernel The window length.
 * @param options The stride (default: the kernel length) and the zero padding at each end.
 * @returns The window means, `[N, C, L']` or `[C, L']`.
 *
 * @example Means of pairs, and a moving average with stride 1
 * const x = tensor([[1, 3, 5, 7, 9, 11]])
 * print('pairs:', avgPool1d(x, 2))
 * print('moving:', avgPool1d(x, 2, { stride: 1 }))
 */
export function avgPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, avgPool2d)
}

/**
 * 1-D max pooling of `[N, C, L]` (or `[C, L]`): `maxPool2d` over a height-1 view, so padding reads $-\infty$.
 *
 * @param x The input, `[N, C, L]`, or `[C, L]` (the result then has no batch axis).
 * @param kernel The window length.
 * @param options The stride (default: the kernel length) and the padding at each end.
 * @returns The window maxima, `[N, C, L']` or `[C, L']`.
 *
 * @example Maxima of windows of three
 * const x = tensor([[4, 1, 7, 2, 9, 3]])
 * print('stride 3:', maxPool1d(x, 3))
 * print('stride 1:', maxPool1d(x, 3, { stride: 1 }))
 */
export function maxPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, maxPool2d)
}

/**
 * A 1-D pooling as a 2-D one: the input is viewed as `[N, C, 1, L]`, pooled with a `[1, kernel]` window, and the
 * height axis dropped again.
 *
 * @param x The input, `[N, C, L]` or `[C, L]`.
 * @param kernel The window length.
 * @param options The stride (default: the kernel length) and padding along the length.
 * @param pool2 The 2-D pooling to apply: `avgPool2d` or `maxPool2d`.
 * @returns The pooled values, `[N, C, L']`, or `[C, L']` when `x` had no batch axis.
 */
function pool1d(
  x: Value,
  kernel: number,
  options: PoolOptions<number>,
  pool2: (x: Value, k: Pair, o: PoolOptions) => Value,
): Value {
  const xs = shapeOfValue(x)
  const batched = xs.length === 3
  const x4 = reshape(x, batched ? [xs[0], xs[1], 1, xs[2]] : [1, xs[0], 1, xs[1]])
  const y = pool2(x4, [1, kernel], {
    stride: [1, options.stride ?? kernel],
    padding: [0, options.padding ?? 0],
  })
  const [n, c, , l] = shapeOfValue(y)
  return reshape(y, batched ? [n, c, l] : [c, l])
}
