/**
 * Deformable convolution (Dai et al., 2017, "Deformable Convolutional Networks", §2.1, eqs. 2 to 4) and its
 * modulated form (Zhu et al., 2019, "Deformable ConvNets v2", §3.1, eq. 1), in the layout of
 * `torchvision.ops.deform_conv2d`.
 *
 * A $K_H \times K_W$ kernel normally reads the regular grid $\pvec_0 + \pvec_n$ around an output position, with
 * $\pvec_n$ the tap's place in the strided, padded, dilated window. Deformable convolution adds a learned real
 * offset $\Delta\pvec_n$ to every tap and reads the input there by bilinear interpolation (as `bilinearSample` of
 * `aifn-compute/signal/image`), and the modulated form multiplies tap $n$ by a scalar $m_n$:
 *
 * $$ y(\pvec_0) = \sum_n w_n\, m_n\, x(\pvec_0 + \pvec_n + \Delta\pvec_n). $$
 *
 * The layer is a composition of primitives: the four integer neighbours of every sample are gathered from the input,
 * weighted by the bilinear weights of the fractional parts of the positions, and contracted with the kernels by one
 * matrix product. It is differentiable in the input, the kernels, the offsets and the masks; the gradient in an
 * offset is one-sided at integer positions, as torchvision's. The neighbours' indices are read from the offsets'
 * values, so it is not batched by `vmap` over the offsets (the batch axis $N$ is handled directly).
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  fromData,
  gather,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  sub,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { convOutputSize, type Pair } from './ops'

/** Options of `deformableConv2d`. */
export type DeformableConvOptions = {
  /** Step between output positions. Default 1. */
  stride?: Pair
  /** Zeros added on each side of the input. Default 0. */
  padding?: Pair
  /** Spacing between kernel taps of the regular grid. Default 1. */
  dilation?: Pair
  /**
   * The modulation scalars $m_n$ (DCNv2), `[N, K, H', W']` (or `[K, H', W']` for one image) for the
   * $K = K_H K_W$ taps; left out, all 1 (DCNv1).
   */
  mask?: Value
}

/**
 * A per-axis option as a pair.
 *
 * @param v One number for both axes, or a pair.
 * @returns The option for the height and width axes.
 */
const pair = (v: Pair): [number, number] => (typeof v === 'number' ? [v, v] : [v[0], v[1]])

/**
 * Deformable 2-D convolution (cross-correlation) of $x$, `[N, C, H, W]`, with kernels $w$, `[O, C, KH, KW]`, and
 * offsets `[N, 2K, H', W']` for the $K = K_H K_W$ taps (one offset group): channel $2n$ is the row offset and
 * $2n + 1$ the column offset of tap $n$ (row-major over the kernel), as `torchvision.ops.deform_conv2d`. Output
 * $(i, j)$'s tap $(u, v)$ reads the input at $(is_h - p_h + ud_h + \Delta r,\ js_w - p_w + vd_w + \Delta c)$, zero
 * outside. Zero offsets and unit masks give `conv2d`. Differentiable in `x`, `w`, `offsets` and the mask. Throws
 * `ShapeError` for shapes that do not agree.
 *
 * @param x The input, `[N, C, H, W]`, or one image `[C, H, W]` (the result then has no batch axis).
 * @param w The kernels, `[O, C, KH, KW]`.
 * @param offsets The offsets, `[N, 2K, H', W']` (or `[2K, H', W']`), with $H'$ and $W'$ the output size `conv2d`
 *   gives for the same stride, padding and dilation.
 * @param options The stride, padding, dilation and the modulation mask.
 * @returns The output, `[N, O, H', W']`, or `[O, H', W']` for one image.
 *
 * @example Zero offsets give the ordinary convolution; shifted offsets read elsewhere
 * const x = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
 * const w = ones([1, 1, 3, 3])
 * print('zero offsets:', deformableConv2d(x, w, zeros([18, 3, 3]), { padding: 1 }))
 * print('conv2d:      ', conv2d(x, w, { padding: 1 }))
 * const down = new Float64Array(18 * 9)
 * for (let n = 0; n < 9; n++) down.fill(0.5, 2 * n * 9, 2 * n * 9 + 9)
 * print('every tap half a row down:', deformableConv2d(x, w, fromData(down, [18, 3, 3]), { padding: 1 }))
 */
export function deformableConv2d(x: Value, w: Value, offsets: Value, options: DeformableConvOptions = {}): Value {
  const where = 'deformableConv2d'
  const xs = shapeOfValue(x)
  const single = xs.length === 3
  if (xs.length !== 3 && xs.length !== 4)
    throw new ShapeError(where, `${where}: x needs shape [N, C, H, W] or [C, H, W]`, [xs])
  const [N, C, H, W] = single ? [1, ...xs] : xs
  const ws = shapeOfValue(w)
  if (ws.length !== 4 || ws[1] !== C)
    throw new ShapeError(where, `${where}: kernels need shape [O, ${C}, KH, KW], got [${ws.join(', ')}]`, [ws])
  const [O, , KH, KW] = ws
  const K = KH * KW
  const [sh, sw] = pair(options.stride ?? 1)
  const [ph, pw] = pair(options.padding ?? 0)
  const [dh, dw] = pair(options.dilation ?? 1)
  const Ho = convOutputSize(H, KH, sh, ph, dh)
  const Wo = convOutputSize(W, KW, sw, pw, dw)
  if (Ho < 1 || Wo < 1) throw new ShapeError(where, `${where}: the kernel does not fit the padded input`, [xs, ws])
  const os = shapeOfValue(offsets)
  const want = [N, 2 * K, Ho, Wo]
  if (os.join() !== (single ? want.slice(1) : want).join())
    throw new ShapeError(where, `${where}: offsets need shape [${want.join(', ')}], got [${os.join(', ')}]`, [os])
  const P = Ho * Wo
  const sampleShape = [N, K, Ho, Wo]
  // The row and column offsets as [N, K, H', W'] each, and their values (which choose the neighbours read).
  const rowIdx = new Int32Array(N * K * P)
  const colIdx = new Int32Array(N * K * P)
  for (let b = 0; b < N; b++)
    for (let n = 0; n < K; n++)
      for (let q = 0; q < P; q++) {
        rowIdx[(b * K + n) * P + q] = (b * 2 * K + 2 * n) * P + q
        colIdx[(b * K + n) * P + q] = (b * 2 * K + 2 * n + 1) * P + q
      }
  const dRow = gather(offsets, rowIdx, sampleShape)
  const dCol = gather(offsets, colIdx, sampleShape)
  const offsetValues = toFlat(unwrap(offsets) as Tensor)
  // Sampling positions (primal values), their floors, and the regular grid as constants.
  const r0 = new Float64Array(N * K * P)
  const c0 = new Float64Array(N * K * P)
  const baseRow = new Float64Array(N * K * P)
  const baseCol = new Float64Array(N * K * P)
  for (let b = 0; b < N; b++)
    for (let n = 0; n < K; n++) {
      const u = Math.floor(n / KW)
      const v = n % KW
      for (let i = 0; i < Ho; i++)
        for (let j = 0; j < Wo; j++) {
          const k = (b * K + n) * P + i * Wo + j
          baseRow[k] = i * sh - ph + u * dh
          baseCol[k] = j * sw - pw + v * dw
          r0[k] = Math.floor(baseRow[k] + offsetValues[rowIdx[k]])
          c0[k] = Math.floor(baseCol[k] + offsetValues[colIdx[k]])
        }
    }
  // Fractional parts f and g, differentiable in the offsets: (base + Δ) − floor.
  const f = add(
    dRow,
    fromData(
      baseRow.map((v, k) => v - r0[k]),
      sampleShape,
    ),
  )
  const g = add(
    dCol,
    fromData(
      baseCol.map((v, k) => v - c0[k]),
      sampleShape,
    ),
  )
  const xFlat = single ? reshape(x, [1, C, H, W]) : x
  const spread = [N, 1, K, Ho, Wo]
  let sampled: Value = 0
  for (const [dr, dc] of [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ] as const) {
    // The neighbour (r0 + dr, c0 + dc) of every sample, for every channel: [N, C, K, H', W'], zero outside the input.
    const idx = new Int32Array(N * C * K * P)
    const inside = new Float64Array(N * C * K * P)
    for (let b = 0; b < N; b++)
      for (let c = 0; c < C; c++)
        for (let s = 0; s < K * P; s++) {
          const r = r0[b * K * P + s] + dr
          const cc = c0[b * K * P + s] + dc
          const at = (b * C + c) * K * P + s
          if (r >= 0 && r < H && cc >= 0 && cc < W) {
            idx[at] = ((b * C + c) * H + r) * W + cc
            inside[at] = 1
          }
        }
    const values = mul(gather(xFlat, idx, [N, C, K, Ho, Wo]), fromData(inside, [N, C, K, Ho, Wo]))
    const wr = dr === 0 ? sub(1, f) : f
    const wc = dc === 0 ? sub(1, g) : g
    sampled = add(sampled, mul(values, reshape(mul(wr, wc), spread)))
  }
  if (options.mask !== undefined) {
    const ms = shapeOfValue(options.mask)
    const wantMask = single ? [K, Ho, Wo] : sampleShape
    if (ms.join() !== wantMask.join())
      throw new ShapeError(where, `${where}: the mask needs shape [${wantMask.join(', ')}], got [${ms.join(', ')}]`)
    sampled = mul(sampled, reshape(options.mask, spread))
  }
  // [O, C·K] × [N, C·K, H'W'] → [N, O, H'W'], channel-major then tap, as w's own layout.
  const y = matmul(reshape(w, [O, C * K]), reshape(sampled, [N, C * K, P]))
  return single ? reshape(y, [O, Ho, Wo]) : reshape(y, [N, O, Ho, Wo])
}
