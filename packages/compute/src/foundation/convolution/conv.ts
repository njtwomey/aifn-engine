/**
 * The convolution family: one $n$-dimensional convolution for signals, images, volumes and neural networks (design K
 * §8.3), with stride, dilation, groups, zero padding and a `flip` switch between convolution (flip the kernel) and
 * cross-correlation (deep learning's "convolution").
 *
 * Three bilinear primitives share one trilinear form,
 * $T(x, w, y) = \sum_{n, o, i, c, a} x_{n,\, c,\, is - \mathrm{lo} + ad}\, w_{o,\, c',\, \hat a}\, y_{n,\, o,\, i}$
 * ($i$ the output position, $a$ the kernel tap, $s$ the stride, $d$ the dilation, $\mathrm{lo}$ the padding before,
 * $c$ over the group of $o$ with $c'$ its index in the group, $\hat a$ the tap reversed when `flip`): `conv` ($y$
 * from $x$ and $w$), `convTranspose` ($x$ from $y$ and $w$: the input adjoint) and `convWeight` ($w$ from $x$ and $y$:
 * the kernel adjoint). Each one's transpose in either argument is one of the three, so the family is closed under
 * differentiation (every order), jvps are derived from multilinearity, and `method` (direct, FFT, overlap-add) changes
 * only the kernel. Batching merges a batch into the image axis $N$, or into the channels with `groups` scaled by the
 * batch, so `vmap` never loops. Complex inputs are handled by real convolutions of their parts.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  avalOf,
  batchToFront,
  broadcastTo,
  complex,
  definePrimitive,
  dense,
  fromData,
  imagPart,
  permute,
  realPart,
  reshape,
  shapeOfValue,
  sub,
  type Op,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { convForward, convInputAdjoint, convWeightAdjoint, type ConvDims, type ConvMethod } from './kernels'

export type { ConvMethod } from './kernels'

/**
 * The geometry of a convolution, per spatial axis, plus groups, flip and the kernel method: the parameters of the
 * `conv` primitive.
 */
type Geometry = {
  /** The step between output positions, per spatial axis. */
  readonly stride: readonly number[]
  /** The spacing between kernel taps, per spatial axis. */
  readonly dilation: readonly number[]
  /** The zeros added before the input, per spatial axis. */
  readonly lo: readonly number[]
  /** The zeros added after the input, per spatial axis. */
  readonly hi: readonly number[]
  /** The number of channel groups. */
  readonly groups: number
  /** True for convolution (the kernel reversed), false for cross-correlation. */
  readonly flip: boolean
  /** The kernel that computes it. */
  readonly method: ConvMethod
}

/**
 * Geometry with the spatial size of the output that the shapes of the inputs do not determine: `size` is $S$ for the
 * input adjoint and $K$ for the kernel adjoint. The parameters of `convTranspose` and `convWeight`.
 */
type Sized = Geometry & { readonly size: readonly number[] }

/**
 * The output length of a convolution along one axis,
 * $\lfloor (n + \mathrm{lo} + \mathrm{hi} - d(k - 1) - 1) / s \rfloor + 1$; below 1 when the kernel does not fit.
 *
 * @param n The input length.
 * @param k The kernel length, in taps.
 * @param s The stride.
 * @param d The dilation.
 * @param lo The zeros added before the input.
 * @param hi The zeros added after the input.
 * @returns The number of output positions.
 */
const outLength = (n: number, k: number, s: number, d: number, lo: number, hi: number): number =>
  Math.floor((n + lo + hi - d * (k - 1) - 1) / s) + 1

/**
 * The output length of a convolution along one axis with symmetric padding $p$,
 * $\lfloor (n + 2p - d(k - 1) - 1) / s \rfloor + 1$ (Dumoulin and Visin, 2016, "A guide to convolution arithmetic for
 * deep learning", §2 to §5). It does not check that the kernel fits: a result below 1 means it does not.
 *
 * @param n The input length $n$.
 * @param k The kernel length $k$, in taps.
 * @param stride The step $s$ between output positions.
 * @param padding The zeros $p$ added on each side of the input.
 * @param dilation The spacing $d$ between kernel taps.
 * @returns The number of output positions.
 *
 * @example Sizes of common layers
 * print('3-tap, padding 1      ', convOutputSize(32, 3, 1, 1))
 * print('3-tap, stride 2       ', convOutputSize(32, 3, 2, 1))
 * print('3-tap, dilation 2     ', convOutputSize(32, 3, 1, 1, 2))
 * print('5-tap, no padding     ', convOutputSize(32, 5))
 */
export function convOutputSize(n: number, k: number, stride = 1, padding = 0, dilation = 1): number {
  return outLength(n, k, stride, dilation, padding, padding)
}

/**
 * The spatial shape of a convolution's output.
 *
 * @param S The spatial shape of the input.
 * @param K The spatial shape of the kernel, in taps.
 * @param g The geometry: stride, dilation and padding per axis.
 * @returns The output length of each spatial axis (below 1 where the kernel does not fit).
 */
const spatialOut = (S: readonly number[], K: readonly number[], g: Geometry): number[] =>
  S.map((n, j) => outLength(n, K[j], g.stride[j], g.dilation[j], g.lo[j], g.hi[j]))

/**
 * The dimensions of a convolution for the kernels, after checking that the kernel fits the padded input and, for an
 * adjoint, that the output gradient has the shape the forward convolution gives. Throws `ShapeError` otherwise.
 *
 * @param N The number of images.
 * @param C The number of input channels.
 * @param O The number of output channels.
 * @param S The spatial shape of the input.
 * @param K The spatial shape of the kernel, in taps.
 * @param g The geometry of the convolution.
 * @param given For an adjoint: `Y`, the spatial shape of the output gradient it was given, and `where`, the
 *   adjoint's name for error messages. Left out for the forward convolution.
 * @returns The sizes, with the output's spatial shape `Y` worked out.
 */
function dims(
  N: number,
  C: number,
  O: number,
  S: readonly number[],
  K: readonly number[],
  g: Geometry,
  given?: { readonly Y: readonly number[]; readonly where: string },
): ConvDims {
  const Y = spatialOut(S, K, g)
  if (Y.some((y) => y < 1))
    throw new ShapeError('conv', `conv: the kernel [${K.join(', ')}] does not fit the padded input [${S.join(', ')}]`)
  // An adjoint's output gradient must have the spatial shape the forward conv gives, or the kernels read past it.
  if (given && (given.Y.length !== Y.length || given.Y.some((y, i) => y !== Y[i])))
    throw new ShapeError(
      given.where,
      `${given.where}: the output gradient has spatial shape [${given.Y.join(', ')}], but conv of an input ` +
        `[${S.join(', ')}] gives [${Y.join(', ')}]`,
    )
  return {
    N,
    C,
    O,
    groups: g.groups,
    S,
    K,
    Y,
    stride: g.stride,
    dilation: g.dilation,
    lo: g.lo,
    hi: g.hi,
    flip: g.flip,
  }
}

/**
 * The geometry of a convolution without anything else its parameters carry (the `size` of an adjoint's parameters).
 *
 * @param p The parameters of a primitive of the family.
 * @returns Its stride, dilation, padding, groups, flip and method.
 */
const geometryOf = (p: Geometry): Geometry => ({
  stride: p.stride,
  dilation: p.dilation,
  lo: p.lo,
  hi: p.hi,
  groups: p.groups,
  flip: p.flip,
  method: p.method,
})

/**
 * The tensor a primitive's impl was given, refusing a plain number.
 *
 * @param x The raw input of the impl.
 * @param what The caller's name for error messages.
 * @returns `x`, known to be a tensor. Throws `ShapeError` for a number.
 */
function tensorOf(x: Raw, what: string): Tensor {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a tensor, got a number`)
  return x
}

// ── Batching ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A value with its batch axis in front, `[B, ...]`; an unbatched value is broadcast along a new front axis.
 *
 * @param v The value.
 * @param axis Where its batch axis is, or null when it is not batched.
 * @param size The batch size $B$, the length of the broadcast axis for an unbatched value.
 * @returns `v` as `[B, ...]`.
 */
function front(v: Value, axis: number | null, size: number): Value {
  if (axis !== null) return batchToFront(v, axis)
  const shape = shapeOfValue(v)
  return broadcastTo(reshape(v, [1, ...shape]), [size, ...shape])
}

/**
 * Merge a batch axis into the image axis: `[B, N, ...]` to `[B N, ...]`, so one convolution serves the whole batch.
 *
 * @param v The batched input or output gradient.
 * @param axis Where its batch axis is; it is moved to the front first.
 * @returns `v` with the batch and image axes merged, batch-major.
 */
function mergeIntoN(v: Value, axis: number): Value {
  const f = batchToFront(v, axis)
  const s = shapeOfValue(f)
  return reshape(f, [s[0] * s[1], ...s.slice(2)])
}

/**
 * Merge a batch axis into the channels: `[B, N, C, ...]` to `[N, B C, ...]`, so the batch becomes groups of channels
 * (used when the kernels are batched too, with `groups` scaled by $B$).
 *
 * @param v The value with its batch axis already in front.
 * @returns `v` with the batch and channel axes merged, batch-major.
 */
function mergeIntoChannels(v: Value): Value {
  const s = shapeOfValue(v)
  const order = [1, 0, ...s.slice(2).map((_, k) => k + 2)]
  return reshape(permute(v, order), [s[1], s[0] * s[2], ...s.slice(3)])
}

/**
 * Merge a batch of kernels into the output channels: `[B, O, ...]` to `[B O, ...]`, one group of kernels per example.
 *
 * @param w The kernels with their batch axis in front.
 * @returns The kernels with the batch and output-channel axes merged, batch-major.
 */
const mergeKernels = (w: Value): Value => {
  const s = shapeOfValue(w)
  return reshape(w, [s[0] * s[1], ...s.slice(2)])
}

/**
 * Split a batch back out of the channels: `[N, B O, ...]` to `[N, B, O, ...]`.
 *
 * @param y The output of a channel-merged convolution.
 * @param size The batch size $B$.
 * @returns The value and its batch axis, 1.
 */
function splitChannels(y: Value, size: number): [Value, number] {
  const s = shapeOfValue(y)
  return [reshape(y, [s[0], size, s[1] / size, ...s.slice(2)]), 1]
}

/**
 * Split a batch back out of the image axis: `[B N, ...]` to `[B, N, ...]`.
 *
 * @param y The output of an image-merged convolution.
 * @param size The batch size $B$.
 * @returns The value and its batch axis, 0.
 */
function splitN(y: Value, size: number): [Value, number] {
  const s = shapeOfValue(y)
  return [reshape(y, [size, s[0] / size, ...s.slice(1)]), 0]
}

// ── Primitives ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The shape of a primitive's tensor input.
 *
 * @param x The raw input of the impl.
 * @param what The caller's name for error messages.
 * @returns Its shape. Throws `ShapeError` for a number.
 */
const shape4 = (x: Raw, what: string) => tensorOf(x, what).shape

/**
 * The convolution primitive, bilinear in the input and the kernels: its transpose in the input is `convTranspose`, in
 * the kernels `convWeight`. Batched by merging the batch into the images (kernels unbatched) or into the channels.
 */
const convOp: Op<Geometry> = definePrimitive<Geometry>({
  id: 'foundation/convolution/conv',
  arity: 2,
  impl: ([x, w], g) => {
    const [N, C, ...S] = shape4(x, 'conv')
    const [O, , ...K] = shape4(w, 'conv')
    const d = dims(N, C, O, S, K, g)
    return fromData(convForward(dense.data(x as Tensor), dense.data(w as Tensor), d, g.method), [N, O, ...d.Y])
  },
  linear: 'multilinear',
  transpose: (ct, [x, w], which, g) =>
    which === 0
      ? convTransposeOp([ct, w], { ...g, size: shapeOfValue(x).slice(2) })
      : convWeightOp([x, ct], { ...g, size: shapeOfValue(w).slice(2) }),
  shape: ([x, w], g) => ({
    shape: [x.shape[0], w.shape[0], ...spatialOut(x.shape.slice(2), w.shape.slice(2), g)],
    dtype: 'float64',
    number: false,
  }),
  batch: ([x, w], [bx, bw], g, size) => {
    if (bw === null) return splitN(convOp([mergeIntoN(x, bx ?? 0), w], g), size)
    const xs = mergeIntoChannels(front(x, bx, size))
    const ws = mergeKernels(front(w, bw, size))
    return splitChannels(convOp([xs, ws], { ...g, groups: g.groups * size }), size)
  },
  dtype: 'float',
  doc: {
    note: 'convolution',
    summary:
      'N-dimensional convolution (or cross-correlation) of [N, C, ...S] inputs with [O, C/groups, ...K] kernels.',
    formula: 'y_{n,o,i} = \\sum_{c,a} x_{n,c,is-p+ad}\\, w_{o,c,\\hat a}',
  },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 4, 9]), draw([6, 2, 3])], params: geometry1([2], [2], [1, 2], 2, true) },
      { inputs: [draw([1, 2, 5, 4]), draw([3, 2, 2, 3])], params: geometry2() },
    ],
  },
})

/**
 * The input adjoint: $g_x = \Jmat^\top g$ for $\Jmat$ the linear map $x \mapsto \mathrm{conv}(x, w)$. Bilinear in
 * $(g, w)$: its transposes are $\mathrm{conv}(u, w)$ for $g$ and $\mathrm{convWeight}(u, g)$ for $w$.
 */
const convTransposeOp: Op<Sized> = definePrimitive<Sized>({
  id: 'foundation/convolution/convTranspose',
  arity: 2,
  impl: ([gy, w], p) => {
    const [N, O, ...Y] = shape4(gy, 'convTranspose')
    const [, Cg, ...K] = shape4(w, 'convTranspose')
    const d = dims(N, Cg * p.groups, O, p.size, K, p, { Y, where: 'convTranspose' })
    const out = convInputAdjoint(dense.data(gy as Tensor), dense.data(w as Tensor), d, p.method)
    return fromData(out, [N, d.C, ...p.size])
  },
  linear: 'multilinear',
  transpose: (u, [gy, w], which, p) =>
    which === 0
      ? convOp([u, w], geometryOf(p))
      : convWeightOp([u, gy], { ...geometryOf(p), size: shapeOfValue(w).slice(2) }),
  shape: ([gy, w], p) => ({ shape: [gy.shape[0], w.shape[1] * p.groups, ...p.size], dtype: 'float64', number: false }),
  batch: ([gy, w], [bg, bw], p, size) => {
    if (bw === null) return splitN(convTransposeOp([mergeIntoN(gy, bg ?? 0), w], p), size)
    const gs = mergeIntoChannels(front(gy, bg, size))
    const ws = mergeKernels(front(w, bw, size))
    return splitChannels(convTransposeOp([gs, ws], { ...p, groups: p.groups * size }), size)
  },
  dtype: 'float',
  doc: { summary: 'The transposed convolution: the adjoint of conv in its input.' },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 6, 5]), draw([6, 2, 3])], params: { ...geometry1([2], [2], [1, 2], 2, true), size: [10] } },
      { inputs: [draw([1, 3, 6, 2]), draw([3, 2, 2, 3])], params: { ...geometry2(), size: [5, 4] } },
    ],
  },
})

/**
 * The kernel adjoint, defined by $\langle g_w, v \rangle = \langle g, \mathrm{conv}(x, v) \rangle$ for every kernel
 * $v$. Bilinear in $(x, g)$: its transposes are $\mathrm{convTranspose}(g, u)$ for $x$ and $\mathrm{conv}(x, u)$
 * for $g$. It sums over the image batch.
 */
const convWeightOp: Op<Sized> = definePrimitive<Sized>({
  id: 'foundation/convolution/convWeight',
  arity: 2,
  impl: ([x, gy], p) => {
    const [N, C, ...S] = shape4(x, 'convWeight')
    const [, O, ...Y] = shape4(gy, 'convWeight')
    const d = dims(N, C, O, S, p.size, p, { Y, where: 'convWeight' })
    const out = convWeightAdjoint(dense.data(x as Tensor), dense.data(gy as Tensor), d, p.method)
    return fromData(out, [O, C / p.groups, ...p.size])
  },
  linear: 'multilinear',
  transpose: (u, [x, gy], which, p) =>
    which === 0
      ? convTransposeOp([gy, u], { ...geometryOf(p), size: shapeOfValue(x).slice(2) })
      : convOp([x, u], geometryOf(p)),
  shape: ([x, gy], p) => ({ shape: [gy.shape[1], x.shape[1] / p.groups, ...p.size], dtype: 'float64', number: false }),
  batch: ([x, gy], [bx, bg], p, size) => {
    const xs = mergeIntoChannels(front(x, bx, size))
    const gs = mergeIntoChannels(front(gy, bg, size))
    const out = convWeightOp([xs, gs], { ...p, groups: p.groups * size })
    const s = shapeOfValue(out)
    return [reshape(out, [size, s[0] / size, ...s.slice(1)]), 0]
  },
  dtype: 'float',
  doc: { summary: 'The adjoint of conv in its kernel (the weight gradient).' },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 4, 9]), draw([2, 6, 4])], params: { ...geometry1([2], [2], [1, 2], 2, true), size: [3] } },
      { inputs: [draw([1, 2, 5, 4]), draw([1, 3, 6, 2])], params: { ...geometry2(), size: [2, 3] } },
    ],
  },
})

/**
 * A one-dimensional geometry for the generated tests of the primitives, with the direct kernel.
 *
 * @param stride The stride, one value.
 * @param dilation The dilation, one value.
 * @param pad The zeros added before and after.
 * @param groups The number of channel groups.
 * @param flip True for convolution, false for cross-correlation.
 * @returns The geometry.
 */
function geometry1(
  stride: number[],
  dilation: number[],
  pad: [number, number],
  groups: number,
  flip: boolean,
): Geometry {
  return { stride, dilation, lo: [pad[0]], hi: [pad[1]], groups, flip, method: 'direct' }
}
/**
 * A fixed two-dimensional geometry for the generated tests of the primitives: stride, dilation and padding differ per
 * axis and the padding is asymmetric; cross-correlation with the direct kernel.
 *
 * @returns The geometry.
 */
function geometry2(): Geometry {
  return { stride: [1, 2], dilation: [2, 1], lo: [1, 0], hi: [2, 1], groups: 1, flip: false, method: 'direct' }
}

// ── Public API ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** One integer for every spatial axis, or a list with one per spatial axis. */
export type Ints = number | readonly number[]

/** The layout of an input: batch, channels and 1 (`ncw`), 2 (`nchw`) or 3 (`ncdhw`) spatial axes. */
export type ConvLayout = 'ncw' | 'nchw' | 'ncdhw'

/**
 * Zero padding: one integer for every side of every axis, one per axis (both sides), a `[lo, hi]` pair per axis, or
 * `valid` (none), `full` ($K_e - 1$ on each side, with $K_e = d(K - 1) + 1$ the dilated kernel extent) or `same` (the
 * centred part of `full` with the input's length when the stride is 1: $\lceil (K_e - 1)/2 \rceil$ before and
 * $\lfloor (K_e - 1)/2 \rfloor$ after, as `scipy.signal`; PyTorch puts the odd sample after).
 */
export type ConvPadding = Ints | readonly (readonly [number, number])[] | 'valid' | 'same' | 'full'

/** Options of `conv` and `convTranspose`. */
export type ConvOptions = {
  /** Spatial rank: `ncw` (1-D), `nchw` (2-D), `ncdhw` (3-D). Default: from the kernel's rank. */
  readonly layout?: ConvLayout
  /** Step between output positions. Default 1. */
  readonly stride?: Ints
  /** Spacing between kernel taps. Default 1. */
  readonly dilation?: Ints
  /** Channel groups: input and output channels split into `groups` independent blocks. Default 1. */
  readonly groups?: number
  /** Zero padding. Default `valid`. */
  readonly padding?: ConvPadding
  /** `true` (default): convolution, the kernel reversed. `false`: cross-correlation, as deep-learning libraries. */
  readonly flip?: boolean
  /** The kernel: `direct`, `fft`, `overlapAdd` (long inputs, short kernels) or `auto` (default). */
  readonly method?: ConvMethod
}

/** The number of spatial axes of each layout. */
const layoutRank: Record<ConvLayout, number> = { ncw: 1, nchw: 2, ncdhw: 3 }

/**
 * One integer per spatial axis from an option. Throws `ShapeError` (named `conv`) for the wrong number of values, or
 * for values that are not positive integers (non-negative for `padding`).
 *
 * @param v The option as given: one number for every axis, a list with one per axis, or undefined.
 * @param D The number of spatial axes.
 * @param fallback The value of every axis when `v` is undefined.
 * @param what The option's name, for error messages; `'padding'` also allows 0.
 * @returns A new list of `D` integers.
 */
function ints(v: Ints | undefined, D: number, fallback: number, what: string): number[] {
  const out =
    v === undefined
      ? new Array<number>(D).fill(fallback)
      : typeof v === 'number'
        ? new Array<number>(D).fill(v)
        : [...v]
  if (out.length !== D) throw new ShapeError('conv', `conv: ${what} needs ${D} values, got ${out.length}`)
  if (out.some((k) => !Number.isInteger(k) || k < (what === 'padding' ? 0 : 1)))
    throw new ShapeError('conv', `conv: ${what} must be ${what === 'padding' ? 'non-negative' : 'positive'} integers`)
  return out
}

/**
 * The geometry of a call from its options: stride, dilation and the padding resolved to `[lo, hi]` per axis (`same`
 * and `full` from the dilated kernel extent), groups, flip (default true) and method (default `auto`). Throws
 * `ShapeError` for invalid values.
 *
 * @param o The options of `conv` or `convTranspose`.
 * @param D The number of spatial axes.
 * @param K The spatial shape of the kernel, in taps, which `same` and `full` padding depend on.
 * @returns The geometry.
 */
function geometry(o: ConvOptions, D: number, K: readonly number[]): Geometry {
  const stride = ints(o.stride, D, 1, 'stride')
  const dilation = ints(o.dilation, D, 1, 'dilation')
  const Ke = K.map((k, j) => dilation[j] * (k - 1) + 1)
  const p = o.padding ?? 'valid'
  let lo: number[]
  let hi: number[]
  if (p === 'valid') lo = hi = new Array<number>(D).fill(0)
  else if (p === 'full') lo = hi = Ke.map((k) => k - 1)
  else if (p === 'same') {
    lo = Ke.map((k) => Math.ceil((k - 1) / 2))
    hi = Ke.map((k) => Math.floor((k - 1) / 2))
  } else if (typeof p !== 'number' && p.length > 0 && typeof p[0] !== 'number') {
    const pairs = p as readonly (readonly [number, number])[]
    lo = ints(
      pairs.map((q) => q[0]),
      D,
      0,
      'padding',
    )
    hi = ints(
      pairs.map((q) => q[1]),
      D,
      0,
      'padding',
    )
  } else lo = hi = ints(p as Ints, D, 0, 'padding')
  const groups = o.groups ?? 1
  if (!Number.isInteger(groups) || groups < 1) throw new ShapeError('conv', 'conv: groups must be a positive integer')
  return { stride, dilation, lo, hi, groups, flip: o.flip ?? true, method: o.method ?? 'auto' }
}

/**
 * The shapes of a call: `D` spatial axes, and how to restore the caller's rank. A rank-1 `x` and `w` are a signal and
 * a filter (`[1, 1, n]` and `[1, 1, m]` underneath); an `x` without the batch axis (`[C, ...S]`) gains one. Throws
 * `ShapeError` when the ranks do not fit the layout.
 *
 * @param x The input as the caller gave it (the output gradient, for `convTranspose`).
 * @param w The kernels as the caller gave them, `[O, C/groups, ...K]` or a rank-1 filter.
 * @param o The options; `layout` fixes the number of spatial axes, which otherwise comes from the kernel's rank.
 * @param what The caller's name for error messages.
 * @returns `D`, the number of spatial axes; `x` and `w` reshaped to the full layout; and `restore`, which reshapes a
 *   result back to the caller's rank.
 */
function arrange(x: Value, w: Value, o: ConvOptions, what: string) {
  const xs = shapeOfValue(x)
  const ws = shapeOfValue(w)
  if (xs.length === 1 && ws.length === 1 && (o.layout === undefined || o.layout === 'ncw'))
    return { D: 1, x: reshape(x, [1, 1, xs[0]]), w: reshape(w, [1, 1, ws[0]]), restore: (y: Value) => reshape(y, [-1]) }
  const D = o.layout ? layoutRank[o.layout] : ws.length - 2
  if (D < 1 || ws.length !== D + 2)
    throw new ShapeError(
      what,
      `${what}: kernels need shape [O, C/groups, ...K] with ${D} spatial axes, got [${ws.join(', ')}]`,
    )
  if (xs.length === D + 2) return { D, x, w, restore: (y: Value) => y }
  if (xs.length === D + 1)
    return { D, x: reshape(x, [1, ...xs]), w, restore: (y: Value) => reshape(y, shapeOfValue(y).slice(1)) }
  throw new ShapeError(what, `${what}: input needs shape [N, C, ...S] or [C, ...S] with ${D} spatial axes`)
}

/**
 * Whether a value is complex.
 *
 * @param v The value.
 * @returns True when its dtype is complex128.
 */
const isComplex = (v: Value): boolean => avalOf(v).dtype === 'complex128'

/**
 * A bilinear map $f$ of real arguments extended to complex ones, without conjugation:
 * $f(a + ib, c + id) = f(a, c) - f(b, d) + i\,(f(a, d) + f(b, c))$. One real factor needs two real calls, two complex
 * ones four. The parts are zero-copy views and `complex` is a primitive, so the result is differentiable (the
 * $\reals^2$ convention) and batched.
 *
 * @param f The real bilinear map, called on real parts.
 * @param x The first argument, real or complex.
 * @param w The second argument, real or complex.
 * @returns The complex result, or null when both arguments are real (the caller then computes it directly).
 */
function complexBilinear(f: (a: Value, b: Value) => Value, x: Value, w: Value): Value | null {
  const cx = isComplex(x)
  const cw = isComplex(w)
  if (!cx && !cw) return null
  if (!cw) return complex(f(realPart(x), w), f(imagPart(x), w))
  if (!cx) return complex(f(x, realPart(w)), f(x, imagPart(w)))
  const [a, b, c, d] = [realPart(x), imagPart(x), realPart(w), imagPart(w)]
  return complex(sub(f(a, c), f(b, d)), add(f(a, d), f(b, c)))
}

/**
 * Check that the channels split into the groups and match the kernels. Throws `ShapeError` otherwise.
 *
 * @param C The number of input channels.
 * @param O The number of output channels.
 * @param Cg The number of input channels each kernel expects (its second axis).
 * @param groups The number of channel groups.
 * @param what The caller's name for error messages.
 */
function checkChannels(C: number, O: number, Cg: number, groups: number, what: string): void {
  if (C % groups !== 0 || O % groups !== 0)
    throw new ShapeError(what, `${what}: ${C} input and ${O} output channels do not split into ${groups} groups`)
  if (C / groups !== Cg)
    throw new ShapeError(what, `${what}: the input has ${C} channels; kernels expect ${Cg * groups} (${Cg} per group)`)
}

/**
 * The convolution of $x$, shape `[N, C, ...S]` (or `[C, ...S]`, or a rank-1 signal), with kernels $w$, shape
 * `[O, C/groups, ...K]` (or a rank-1 filter):
 * $y_{n,o,i} = \sum_{c, a} x_{n,\, c,\, is - \mathrm{lo} + ad}\, w_{o,\, c',\, \hat a}$, with $\hat a$ the kernel
 * tap reversed on every spatial axis when `flip` (convolution, the default) and not reversed otherwise
 * (cross-correlation, as `torch.nn.functional.conv*d`), and $c$ running over the group of $o$ ($c'$ its index in the
 * group). The output is `[N, O, ...Y]` with $Y = \lfloor (S + \mathrm{lo} + \mathrm{hi} - d(K - 1) - 1)/s \rfloor + 1$
 * per axis. Bilinear, differentiable in $x$ and $w$ to any order, and batched by `vmap` without a loop. On signals with
 * `padding` `full`, `same` or `valid` it is `scipy.signal.convolve` (flip) or `scipy.signal.correlate` (no flip) with
 * that mode, except that `valid` needs the kernel no longer than the signal (`convolve` and `correlate` swap the
 * arguments for it); `same` keeps the signal's length, where `numpy.convolve` keeps the longer one. Complex $x$ or
 * $w$ give a complex128 result from real convolutions of the parts (no conjugation, as `numpy.convolve`). Throws
 * `ShapeError` when the shapes, channels and groups do not fit, or the kernel does not fit the padded input.
 *
 * @param x The input: `[N, C, ...S]`, `[C, ...S]` (one image, and the result then has no batch axis) or a rank-1
 *   signal with a rank-1 filter (the result is then rank-1). Real or complex.
 * @param w The kernels, `[O, C/groups, ...K]`: one per output channel, each over the input channels of its group. A
 *   rank-1 filter with a rank-1 signal. Real or complex.
 * @param options The geometry (stride, dilation, padding, groups), `flip`, the `layout` and the kernel `method`; see
 *   `ConvOptions`. By default a convolution with no padding (`valid`), stride and dilation 1, and one group.
 * @returns The output, `[N, O, ...Y]` (without the batch axis when `x` had none, rank-1 for a signal), complex when
 *   either input is.
 *
 * @example Convolution and cross-correlation of a signal
 * const x = tensor([1, 2, 3, 4])
 * const w = tensor([1, 0, -1])
 * print('convolution (full)  ', conv(x, w, { padding: 'full' }))
 * print('correlation (full)  ', conv(x, w, { padding: 'full', flip: false }))
 * print('convolution (valid) ', conv(x, w))
 *
 * @example A 2-D image with two kernels, stride 2
 * // One image [N, C, H, W] = [1, 1, 4, 4]; kernels [O, C, kH, kW] = [2, 1, 2, 2]: a box sum and a corner pick.
 * const img = tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
 * const k = tensor([[[[1, 1], [1, 1]]], [[[1, 0], [0, 0]]]])
 * print('y =', conv(img, k, { stride: 2, flip: false }))
 *
 * @example The gradient in the kernel
 * // The derivative of sum(conv(x, w)) in each tap is the sum of the samples that tap meets.
 * const x = tensor([1, 2, 3, 4, 5])
 * print('dw =', grad((w) => sum(conv(x, w, { flip: false })))(tensor([0, 0, 0])))
 */
export function conv(x: Value, w: Value, options: ConvOptions = {}): Value {
  const z = complexBilinear((u, v) => conv(u, v, options), x, w)
  if (z !== null) return z
  const a = arrange(x, w, options, 'conv')
  const [, C, ...S] = shapeOfValue(a.x)
  const [O, Cg, ...K] = shapeOfValue(a.w)
  const g = geometry(options, a.D, K)
  checkChannels(C, O, Cg, g.groups, 'conv')
  dims(1, C, O, S, K, g)
  return a.restore(convOp([a.x, a.w], g))
}

/**
 * The transposed convolution: the adjoint in $x$ of `conv(x, w, options)`, taking $y$ of shape `[N, O, ...Y]` (or
 * `[O, ...Y]`, or a rank-1 signal) to `[N, C, ...S]`. The `size` option gives $S$, by default the smallest input whose
 * convolution has output $Y$, $(Y - 1)s + d(K - 1) + 1 - \mathrm{lo} - \mathrm{hi}$ per axis. A stride $s$ makes it an
 * upsampler ($s - 1$ zeros between samples, then the filter), as in interpolation, synthesis filter banks and decoder
 * layers. Differentiable in $y$ and $w$ to any order, and batched by `vmap` without a loop. Complex inputs extend it
 * bilinearly (no conjugation), as `conv`. Throws `ShapeError` when the channels do not match the kernels, or when
 * `size` would not give output $Y$ under `conv`.
 *
 * @param y The values to transpose (an output gradient, or the coefficients to synthesise from): `[N, O, ...Y]`,
 *   `[O, ...Y]` (the result then has no batch axis) or a rank-1 signal with a rank-1 filter.
 * @param w The kernels of the forward convolution, `[O, C/groups, ...K]`, or a rank-1 filter.
 * @param options The options of the forward convolution (see `ConvOptions`), and `size`, the spatial shape $S$ of the
 *   result: one integer for every axis or one per axis. A stride above 1 makes several sizes give the same $Y$, and
 *   `size` picks one; left out, the smallest.
 * @returns The transposed convolution, `[N, C, ...S]` (without the batch axis when `y` had none, rank-1 for a
 *   signal), complex when either input is.
 *
 * @example Upsample by 2 with a hold filter
 * print('y =', convTranspose(tensor([1, 2, 3]), tensor([1, 1]), { stride: 2 }))
 *
 * @example It is the adjoint of conv
 * // <conv(x, w), y> equals <x, convTranspose(y, w)> for every x and y.
 * const x = tensor([1, -2, 3, 0, 5])
 * const w = tensor([2, 1, -1])
 * const y = tensor([1, 4, -3])
 * print('<conv(x, w), y>         ', sum(mul(conv(x, w), y)))
 * print('<x, convTranspose(y, w)>', sum(mul(x, convTranspose(y, w))))
 */
export function convTranspose(y: Value, w: Value, options: ConvOptions & { readonly size?: Ints } = {}): Value {
  const z = complexBilinear((u, v) => convTranspose(u, v, options), y, w)
  if (z !== null) return z
  const a = arrange(y, w, options, 'convTranspose')
  const [, O, ...Y] = shapeOfValue(a.x)
  const [Ow, Cg, ...K] = shapeOfValue(a.w)
  const g = geometry(options, a.D, K)
  if (O !== Ow) throw new ShapeError('convTranspose', `convTranspose: y has ${O} channels, kernels ${Ow}`)
  const smallest = Y.map((n, j) => (n - 1) * g.stride[j] + g.dilation[j] * (K[j] - 1) + 1 - g.lo[j] - g.hi[j])
  const size = options.size === undefined ? smallest : ints(options.size, a.D, 1, 'size')
  const expected = spatialOut(size, K, g)
  if (expected.some((n, j) => n !== Y[j]))
    throw new ShapeError(
      'convTranspose',
      `convTranspose: an input of size [${size.join(', ')}] gives [${expected.join(', ')}], not [${Y.join(', ')}]`,
    )
  checkChannels(Cg * g.groups, O, Cg, g.groups, 'convTranspose')
  return a.restore(convTransposeOp([a.x, a.w], { ...g, size }))
}
