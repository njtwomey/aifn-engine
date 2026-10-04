/**
 * Convolution and pooling for neural networks, as torch.nn.functional: compositions over the convolution family of
 * `aifn-compute/foundation/convolution` (design K §8.3), so every one is differentiable to any order and batches under vmap.
 *
 * - `conv1d`, `conv2d`: the family's `conv` with `flip: false` (cross-correlation, as in every deep-learning library):
 *   y[n, o, i, j] = Σ_{c, a, b} x[n, c, i·s_h − p_h + a·d_h, j·s_w − p_w + b·d_w] · w[o, c, a, b], with zeros outside
 *   the image (Dumoulin & Visin, 2016, "A guide to convolution arithmetic for deep learning", §2–5, for the sizes).
 * - `avgPool1d`, `avgPool2d`: a depthwise convolution with a uniform kernel.
 * - `maxPool1d`, `maxPool2d`: windows gathered at fixed positions and reduced by `max`.
 */

import { conv, convOutputSize } from 'aifn-compute/foundation/convolution'
import { concat, fromData, gather, max, reshape, shapeOfValue, type Value } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

export { convOutputSize }

/** A per-axis option: one number for both axes or a pair. */
export type Pair = number | readonly [number, number]

const pair = (v: Pair): [number, number] => (typeof v === 'number' ? [v, v] : [v[0], v[1]])

/** Options of `conv2d` and `conv1d`. */
export type ConvOptions<P = Pair> = {
  /** Step between output positions. Default 1. */
  stride?: P
  /** Zeros added on each side of the input. Default 0. */
  padding?: P
  /** Spacing between kernel taps (1 = contiguous). Default 1. */
  dilation?: P
  /** Channel groups (C and O divisible by it; kernels [O, C/groups, …]). Default 1. */
  groups?: number
}

/**
 * 2-D convolution (cross-correlation) of x, shape [N, C, H, W] (or [C, H, W]), with kernels w, shape
 * [O, C/groups, KH, KW], giving [N, O, H′, W′] (or [O, H′, W′]) with H′ = ⌊(H + 2p − d(KH − 1) − 1)/s⌋ + 1. The
 * convolution family's `conv` with `flip: false`, so differentiable in x and w to any order and batched without a
 * loop. Matches `torch.nn.functional.conv2d` (without bias; add it by broadcasting).
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
 * 1-D convolution (cross-correlation) of x, shape [N, C, L] (or [C, L]), with kernels w, shape [O, C/groups, K],
 * giving [N, O, L′]. The convolution family's `conv` with `flip: false`, as `torch.nn.functional.conv1d`.
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

type PoolParams = { kernel: [number, number]; stride: [number, number]; padding: [number, number]; x: number[] }

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
  /** Padding on each side: zeros for average pooling (counted in the mean), −∞ for max pooling. Default 0. */
  padding?: P
}

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
 * 2-D average pooling of [N, C, H, W] (or [C, H, W]) over kernel windows; padded zeros count in the mean (PyTorch's
 * default `count_include_pad`). A depthwise convolution (groups = C) with a uniform kernel 1/(kh·kw), so linear and
 * differentiable to any order.
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
 * 2-D max pooling of [N, C, H, W] (or [C, H, W]) over kernel windows; padding reads −∞. A composition: the windows are
 * gathered at fixed positions (padded taps read one appended −∞) into a trailing axis, and `max` reduces it, so the
 * derivative is max's (split equally among tied taps) and it batches and differentiates to any order.
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

/** 1-D average pooling of [N, C, L] (or [C, L]); see `avgPool2d`. */
export function avgPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, avgPool2d)
}

/** 1-D max pooling of [N, C, L] (or [C, L]); see `maxPool2d`. */
export function maxPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, maxPool2d)
}

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
