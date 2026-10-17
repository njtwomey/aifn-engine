/**
 * Convolution over the group $p4$ of translations and rotations by multiples of 90 degrees (Cohen and Welling 2016,
 * "Group equivariant convolutional networks", ICML, §6.1 to 6.3, eq. 10 and 11, and §7.1 for the filter
 * transformation), built on `conv2d` of `aifn-compute/nn/functional`, so every function is differentiable and batched.
 *
 * A $p4$ feature map has shape `[N, C, 4, H, W]`: channel, orientation, row, column. A rotation $r$ by $k$ quarter
 * turns acts on it by rotating every orientation slice and shifting the orientation axis cyclically,
 *
 * $$(r \cdot f)(s) = \operatorname{rot}^k f(s - k),$$
 *
 * which is `p4Act`. The lifting correlation `p4Lift` correlates an image with the four rotated copies of each filter;
 * the group correlation `p4GroupConv` correlates a $p4$ map with filters that are themselves functions on the group,
 * the filter for output orientation $r$ being `p4Act(w, r)`. Both are equivariant,
 * $\Phi(\text{rotated input}) = r \cdot \Phi(\text{input})$, exactly when the image is square and the padding
 * symmetric (none, or $(k - 1)/2$ for an odd kernel of size $k$). `orientationPool` takes the maximum over the
 * orientation axis, which the shift leaves unchanged, so its output rotates with the input. The filter transformation
 * is a `gather` with a precomputed index map, as the paper's §7.1 suggests.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { gather, max, reshape, shapeOfValue, stack, type Value } from 'aifn-compute/foundation/tensor'
import { conv2d } from 'aifn-compute/nn/functional'

/**
 * The flat source index of every element after rotating the last two axes $k$ quarter turns and, when `shift` is a
 * number, shifting the axis before them cyclically by `shift`. Throws `ShapeError` for an odd number of quarter turns
 * of a map that is not square.
 *
 * @param shape The shape of the map.
 * @param k The number of anticlockwise quarter turns (any integer; taken modulo 4).
 * @param shift The cyclic shift of axis $-3$, or null to leave it alone.
 * @param where The caller's name, for error messages.
 * @returns For every element of the result in row-major order, the index of the input element it takes.
 */
function indexMap(shape: readonly number[], k: number, shift: number | null, where: string): Int32Array {
  if (!Number.isInteger(k)) throw new DomainError(where, `${where}: k = ${k} must be an integer`)
  const rank = shape.length
  if (rank < (shift === null ? 2 : 3))
    throw new ShapeError(where, `${where}: the map has too few axes, [${shape.join(', ')}]`)
  const H = shape[rank - 2]
  const W = shape[rank - 1]
  const quarter = ((k % 4) + 4) % 4
  if (H !== W && quarter % 2 === 1)
    throw new ShapeError(where, `${where}: an odd number of quarter turns needs a square map, got ${H} × ${W}`)
  const R = shift === null ? 1 : shape[rank - 3]
  const lead = shape.slice(0, rank - (shift === null ? 2 : 3)).reduce((a, b) => a * b, 1)
  const out = new Int32Array(lead * R * H * W)
  let n = 0
  for (let l = 0; l < lead; l++)
    for (let r = 0; r < R; r++) {
      const sr = shift === null ? 0 : (((r - shift) % R) + R) % R
      for (let i = 0; i < H; i++)
        for (let j = 0; j < W; j++) {
          // Output (i, j) of k anticlockwise quarter turns reads input (a, b): out[i][j] = in[j][W - 1 - i], k times.
          let a = i
          let b = j
          for (let q = 0; q < quarter; q++) [a, b] = [b, H - 1 - a]
          out[n++] = ((l * R + sr) * H + a) * W + b
        }
    }
  return out
}

/**
 * Rotate the last two axes of `x` by $k$ quarter turns anticlockwise, as `numpy.rot90(x, k, axes=(-2, -1))`.
 * Differentiable. Throws `ShapeError` for an odd $k$ on a map that is not square.
 *
 * @param x The value to rotate, of rank at least 2.
 * @param k The number of quarter turns (negative turns clockwise).
 * @returns The rotated value, of the shape of `x`.
 *
 * @example Quarter turns of a 2 x 2 matrix
 * const x = tensor([[1, 2], [3, 4]])
 * print('once:', rot90(x, 1))
 * print('twice:', rot90(x, 2))
 */
export function rot90(x: Value, k: number): Value {
  const shape = shapeOfValue(x)
  return gather(x, indexMap(shape, k, null, 'rot90'), shape)
}

/**
 * The action of a rotation by $k$ quarter turns on a $p4$ map `[..., 4, H, W]`: every orientation slice rotates and
 * the orientations shift cyclically by $k$ (see the file comment). Differentiable. Throws `ShapeError` unless axis
 * $-3$ has length 4 and the map is square.
 *
 * @param f The $p4$ map (or filter bank) `[..., 4, H, W]`.
 * @param k The number of quarter turns.
 * @returns The transformed map, of the shape of `f`.
 *
 * @example Acting by one quarter turn three times equals acting by three
 * const f = normals(stream(0), [1, 1, 4, 3, 3])
 * const once = p4Act(p4Act(p4Act(f, 1), 1), 1)
 * print('max difference:', max(abs(sub(once, p4Act(f, 3)))))
 */
export function p4Act(f: Value, k: number): Value {
  const shape = shapeOfValue(f)
  if (shape[shape.length - 3] !== 4)
    throw new ShapeError('p4Act', `p4Act: axis -3 must have length 4, got [${shape.join(', ')}]`)
  return gather(f, indexMap(shape, k, k, 'p4Act'), shape)
}

/** Options of the $p4$ correlations. */
export type P4Options = {
  /** Zero padding on each side of both spatial axes (default 0). */
  padding?: number
}

/**
 * The lifting correlation: the image `[N, C, H, W]` with each filter `[O, C, k, k]` rotated by 0, 1, 2 and 3 quarter
 * turns, giving the $p4$ map `[N, O, 4, H', W']` (Cohen and Welling, eq. 10). Differentiable in `x` and `w`. Throws
 * `ShapeError` for filters that are not square or do not match the image's channels.
 *
 * @param x The images, `[N, C, H, W]`.
 * @param w The filters, `[O, C, k, k]`.
 * @param options The zero padding.
 * @returns The $p4$ map, `[N, O, 4, H', W']`.
 *
 * @example Lifting is equivariant: rotating the image rotates the map
 * const x = normals(stream(0), [1, 1, 5, 5])
 * const w = normals(stream(1), [2, 1, 3, 3])
 * print('error:', max(abs(sub(p4Lift(rot90(x, 1), w), p4Act(p4Lift(x, w), 1)))))
 */
export function p4Lift(x: Value, w: Value, options: P4Options = {}): Value {
  const [O, C, kh, kw] = shapeOfValue(w)
  const xs = shapeOfValue(x)
  if (xs.length !== 4 || kh !== kw || xs[1] !== C)
    throw new ShapeError('p4Lift', `p4Lift: need images [N, ${C}, H, W] and square filters [O, C, k, k]`, [
      xs,
      [O, C, kh, kw],
    ])
  const outs = [0, 1, 2, 3].map((r) => conv2d(x, rot90(w, r), { padding: options.padding ?? 0 }))
  return stack(outs, 2)
}

/**
 * The group correlation on $p4$: a map `[N, C, 4, H, W]` with filters `[O, C, 4, k, k]` (functions on the group),
 * giving `[N, O, 4, H', W']` (Cohen and Welling, eq. 11). Output orientation $r$ uses the filters transformed by
 * `p4Act(w, r)`, with the input's orientation axis folded into its channels. Differentiable in `f` and `w`. Throws
 * `ShapeError` unless both have an orientation axis of length 4, the filters are square and the channels match.
 *
 * @param f The $p4$ map, `[N, C, 4, H, W]`.
 * @param w The filters, `[O, C, 4, k, k]`.
 * @param options The zero padding.
 * @returns The $p4$ map, `[N, O, 4, H', W']`.
 *
 * @example Two layers stay equivariant
 * const x = normals(stream(0), [1, 1, 5, 5])
 * const w1 = normals(stream(1), [2, 1, 3, 3])
 * const w2 = normals(stream(2), [2, 2, 4, 3, 3])
 * const net = (v) => p4GroupConv(p4Lift(v, w1, { padding: 1 }), w2, { padding: 1 })
 * print('error:', max(abs(sub(net(rot90(x, 1)), p4Act(net(x), 1)))))
 */
export function p4GroupConv(f: Value, w: Value, options: P4Options = {}): Value {
  const fs = shapeOfValue(f)
  const ws = shapeOfValue(w)
  if (fs.length !== 5 || ws.length !== 5 || fs[2] !== 4 || ws[2] !== 4 || ws[3] !== ws[4] || fs[1] !== ws[1])
    throw new ShapeError('p4GroupConv', 'p4GroupConv: need a map [N, C, 4, H, W] and filters [O, C, 4, k, k]', [fs, ws])
  const [N, C, R, H, W] = fs
  const [O, , , k] = ws
  const flat = reshape(f, [N, C * R, H, W])
  const outs = [0, 1, 2, 3].map((r) =>
    conv2d(flat, reshape(p4Act(w, r), [O, C * R, k, k]), { padding: options.padding ?? 0 }),
  )
  return stack(outs, 2)
}

/**
 * The maximum over the orientation axis of a $p4$ map `[N, C, 4, H, W]`, giving `[N, C, H, W]`: invariant to the
 * orientation shift, so it rotates with the input. Differentiable (the gradient goes to the winning orientation).
 * Throws `ShapeError` unless the map has an orientation axis of length 4.
 *
 * @param f The $p4$ map.
 * @returns The pooled map, `[N, C, H, W]`.
 *
 * @example Pooling commutes with rotation
 * const x = normals(stream(0), [1, 1, 5, 5])
 * const w = normals(stream(1), [2, 1, 3, 3])
 * const rotatedFirst = orientationPool(p4Lift(rot90(x, 1), w))
 * print('error:', max(abs(sub(rotatedFirst, rot90(orientationPool(p4Lift(x, w)), 1)))))
 */
export function orientationPool(f: Value): Value {
  const fs = shapeOfValue(f)
  if (fs.length !== 5 || fs[2] !== 4)
    throw new ShapeError('orientationPool', `orientationPool: need a map [N, C, 4, H, W], got [${fs.join(', ')}]`)
  return max(f, 2)
}
