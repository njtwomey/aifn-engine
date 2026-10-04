/**
 * The convolution family, part of `aifn-compute/foundation/convolution`: one n-dimensional convolution for signals, images,
 * volumes and neural networks (design K §8.3), with stride, dilation, groups, zero padding and a `flip` switch between
 * convolution (flip the kernel) and cross-correlation (deep learning's "convolution").
 *
 * Three bilinear primitives share one trilinear form T(x, w, y) = Σ x[n, c, i·s − lo + a·d] w[o, c′, â] y[n, o, i]:
 * `conv` (y from x and w), `convTranspose` (x from y and w: the input adjoint) and `convWeight` (w from x and y: the
 * kernel adjoint). Each one's transpose in either argument is one of the three, so the family is closed under
 * differentiation (every order), jvps are derived from multilinearity, and `method` (direct, FFT, overlap-add) changes
 * only the kernel. Batching merges a batch into the image axis N, or into the channels with `groups` scaled by the
 * batch, so vmap never loops.
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

/** The geometry of a convolution, per spatial axis, plus groups, flip and the kernel method. */
type Geometry = {
  readonly stride: readonly number[]
  readonly dilation: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  readonly groups: number
  readonly flip: boolean
  readonly method: ConvMethod
}

/** Geometry with the spatial size of the output that the shapes of the inputs do not determine (S or K). */
type Sized = Geometry & { readonly size: readonly number[] }

/** The output length along one axis: ⌊(n + lo + hi − d(k − 1) − 1)/s⌋ + 1. */
const outLength = (n: number, k: number, s: number, d: number, lo: number, hi: number): number =>
  Math.floor((n + lo + hi - d * (k - 1) - 1) / s) + 1

/**
 * The output length of a convolution along one axis with symmetric padding p: ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1
 * (Dumoulin & Visin, 2016, "A guide to convolution arithmetic for deep learning", §2–5).
 */
export function convOutputSize(n: number, k: number, stride = 1, padding = 0, dilation = 1): number {
  return outLength(n, k, stride, dilation, padding, padding)
}

const spatialOut = (S: readonly number[], K: readonly number[], g: Geometry): number[] =>
  S.map((n, j) => outLength(n, K[j], g.stride[j], g.dilation[j], g.lo[j], g.hi[j]))

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

const geometryOf = (p: Geometry): Geometry => ({
  stride: p.stride,
  dilation: p.dilation,
  lo: p.lo,
  hi: p.hi,
  groups: p.groups,
  flip: p.flip,
  method: p.method,
})

function tensorOf(x: Raw, what: string): Tensor {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a tensor, got a number`)
  return x
}

// ── Batching ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A value with its batch axis (or a broadcast one, when unbatched) in front: [B, ...]. */
function front(v: Value, axis: number | null, size: number): Value {
  if (axis !== null) return batchToFront(v, axis)
  const shape = shapeOfValue(v)
  return broadcastTo(reshape(v, [1, ...shape]), [size, ...shape])
}

/** [B, N, …] → [B·N, …]. */
function mergeIntoN(v: Value, axis: number): Value {
  const f = batchToFront(v, axis)
  const s = shapeOfValue(f)
  return reshape(f, [s[0] * s[1], ...s.slice(2)])
}

/** [B, N, C, …] → [N, B·C, …]: the batch becomes groups of channels. */
function mergeIntoChannels(v: Value): Value {
  const s = shapeOfValue(v)
  const order = [1, 0, ...s.slice(2).map((_, k) => k + 2)]
  return reshape(permute(v, order), [s[1], s[0] * s[2], ...s.slice(3)])
}

/** [B, O, …] → [B·O, …] (kernels). */
const mergeKernels = (w: Value): Value => {
  const s = shapeOfValue(w)
  return reshape(w, [s[0] * s[1], ...s.slice(2)])
}

/** [N, B·O, …] → [N, B, O, …]: the batch axis at 1. */
function splitChannels(y: Value, size: number): [Value, number] {
  const s = shapeOfValue(y)
  return [reshape(y, [s[0], size, s[1] / size, ...s.slice(2)]), 1]
}

/** [B·N, …] → [B, N, …]. */
function splitN(y: Value, size: number): [Value, number] {
  const s = shapeOfValue(y)
  return [reshape(y, [size, s[0] / size, ...s.slice(1)]), 0]
}

// ── Primitives ───────────────────────────────────────────────────────────────────────────────────────────────────────

const shape4 = (x: Raw, what: string) => tensorOf(x, what).shape

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

// The input adjoint: gx = Jᵀg for J = conv(·, w). Bilinear in (g, w): its transposes are conv(u, w) for g and
// convWeight(u, g) for w.
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

// The kernel adjoint: ⟨gw, v⟩ = ⟨g, conv(x, v)⟩. Bilinear in (x, g): its transposes are convTranspose(g, u) for x and
// conv(x, u) for g. It sums over the image batch.
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

/** Test geometries (registry cases). */
function geometry1(
  stride: number[],
  dilation: number[],
  pad: [number, number],
  groups: number,
  flip: boolean,
): Geometry {
  return { stride, dilation, lo: [pad[0]], hi: [pad[1]], groups, flip, method: 'direct' }
}
function geometry2(): Geometry {
  return { stride: [1, 2], dilation: [2, 1], lo: [1, 0], hi: [2, 1], groups: 1, flip: false, method: 'direct' }
}

// ── Public API ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** One integer for every spatial axis, or one per axis. */
export type Ints = number | readonly number[]

/** Batch, channels and 1, 2 or 3 spatial axes. */
export type ConvLayout = 'ncw' | 'nchw' | 'ncdhw'

/**
 * Zero padding: an integer per side for every axis, one per axis, a [lo, hi] pair per axis, or `valid` (none),
 * `full` (Kₑ − 1 on each side, Kₑ = d(K − 1) + 1) or `same` (the centred part of `full` with the input's length when
 * the stride is 1: ⌈(Kₑ − 1)/2⌉ before and ⌊(Kₑ − 1)/2⌋ after, as scipy.signal; PyTorch puts the odd sample after).
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

const layoutRank: Record<ConvLayout, number> = { ncw: 1, nchw: 2, ncdhw: 3 }

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
 * The shapes of a call: `D` spatial axes, and how to restore the caller's rank. A rank-1 x and w are a signal and a
 * filter ([1, 1, n] and [1, 1, m] underneath); an x without the batch axis ([C, ...S]) gains one.
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

const isComplex = (v: Value): boolean => avalOf(v).dtype === 'complex128'

/**
 * A bilinear map f of real arguments extended to complex ones, without conjugation: f(a + ib, c + id) =
 * f(a, c) − f(b, d) + i(f(a, d) + f(b, c)). One real factor needs two real calls, two complex ones four. The parts are
 * zero-copy views and `complex` is a primitive, so the result is differentiable (ℝ² convention) and batched.
 * Null when both arguments are real.
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

function checkChannels(C: number, O: number, Cg: number, groups: number, what: string): void {
  if (C % groups !== 0 || O % groups !== 0)
    throw new ShapeError(what, `${what}: ${C} input and ${O} output channels do not split into ${groups} groups`)
  if (C / groups !== Cg)
    throw new ShapeError(what, `${what}: the input has ${C} channels; kernels expect ${Cg * groups} (${Cg} per group)`)
}

/**
 * The convolution of x, shape [N, C, ...S] (or [C, ...S], or a rank-1 signal), with kernels w, shape
 * [O, C/groups, ...K] (or a rank-1 filter): y[n, o, i] = Σ_{c, a} x[n, c, i·s − lo + a·d] · w[o, c, â], with â the
 * kernel index reversed on every spatial axis when `flip` (convolution, the default) and not (cross-correlation, as
 * `torch.nn.functional.conv*d`) otherwise, and c running over the group of o. Output [N, O, ...Y] with
 * Y = ⌊(S + lo + hi − d(K − 1) − 1)/s⌋ + 1. Bilinear, differentiable in x and w to any order, and batched by vmap
 * without a loop. On signals with `padding` `full`, `same` or `valid` it is `numpy.convolve` (flip) or
 * `numpy.correlate` (no flip) with that mode, except that `valid` needs the kernel no longer than the signal. Complex
 * x or w give a complex128 result from real convolutions of the parts (no conjugation, as `numpy.convolve`).
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
 * The transposed convolution: the adjoint in x of `conv(x, w, options)`, taking y of shape [N, O, ...Y] (or
 * [O, ...Y], or a rank-1 signal) to [N, C, ...S]. `size` gives S (default the smallest input with output Y:
 * (Y − 1)·s + d(K − 1) + 1 − lo − hi); a stride s makes it an upsampler (zeros between samples, then the filter), as
 * in interpolation, synthesis filter banks and decoder layers. Differentiable in y and w to any order. Complex
 * inputs extend it bilinearly (no conjugation), as `conv`.
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
