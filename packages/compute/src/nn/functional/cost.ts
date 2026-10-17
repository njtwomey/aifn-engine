/**
 * Parameter counts, multiply-accumulate counts, output shapes and receptive fields of convolution layers and of whole
 * convolutional networks, by arithmetic on their geometry (no tensors).
 *
 * A grouped layer splits its $C$ input and $C'$ output channels into $g$ blocks and connects block $i$ of the input
 * only to block $i$ of the output, so it holds $CC'K/g$ weights for $K$ kernel taps; a depthwise layer is the case
 * $g = C$. A transposed layer is the adjoint of the ordinary one: the same weights, each input value scattered into
 * $K$ taps of every output channel of its group. Output sizes follow Dumoulin and Visin (2016), "A guide to
 * convolution arithmetic for deep learning", §2 to §5: ordinary $\lfloor (n + 2p - d(k - 1) - 1)/s \rfloor + 1$ (as
 * `convOutputSize`), transposed $(n - 1)s - 2p + d(k - 1) + 1 + p_{\text{out}}$. `same` padding is that of `conv`:
 * $\lceil (e - 1)/2 \rceil$ zeros before and $\lfloor (e - 1)/2 \rfloor$ after, for the dilated extent
 * $e = d(k - 1) + 1$, so a stride-$s$ layer keeps $\lceil n/s \rceil$ positions.
 *
 * Counts follow the papers and torchvision. `multiplies` counts the multiply-accumulates of convolution and dense
 * layers for one input (FLOPs are about twice it), every tap of every window including padded zeros, as framework
 * FLOP counters do; elementwise work (activations, normalisation, sums, gates, pooling) is not counted. Parameters
 * are weights, biases and the affine parameters of normalisation layers ($2C$), not running statistics.
 *
 * A network is a tree of `LayerSpec`s walked by `networkCost` from an input shape $[C, \dots]$: plain layers and the
 * combinators `sequence`, `add` (a residual sum), `concat` (Inception and DenseNet) and `gate`
 * (squeeze-and-excitation).
 * Its receptive field grows by the recurrence of `receptiveField` (Araujo, Norris and Sim, 2019), the largest over the
 * branches of a sum or concatenation, and becomes the whole input (Infinity) after a dense layer or global pooling.
 * Pure arithmetic and fast (a 150-layer network in well under a millisecond), so safe in a worker and on a drag.
 */

import { convOutputSize } from 'aifn-compute/foundation/convolution'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { growField } from './receptive'

/** One integer for every spatial axis, or one per axis. */
export type PerAxis = number | readonly number[]

/**
 * The kinds of convolution layer: ordinary, grouped (`groups` blocks), depthwise (one group per input channel) and
 * transposed.
 */
export type ConvolutionKind = 'conv' | 'grouped' | 'depthwise' | 'transposed'

/** One convolution layer, described by its geometry and the spatial shape of its input. */
export type ConvolutionLayerSpec = {
  /** The kind of layer. Default `conv`, or `grouped` when `groups` is above 1. */
  kind?: ConvolutionKind
  /** Input channels $C$. */
  inChannels: number
  /**
   * Output channels $C'$. Default $C$: for a depthwise layer one filter per channel; a multiple of $C$ gives a channel
   * multiplier.
   */
  outChannels?: number
  /** Kernel taps per spatial axis. */
  kernel: PerAxis
  /** Step between output positions (for `transposed`, the upsampling factor). Default 1. */
  stride?: PerAxis
  /**
   * Zeros added on each side of every axis, or `same`: $\lceil (e - 1)/2 \rceil$ before and
   * $\lfloor (e - 1)/2 \rfloor$ after (as `conv`), which keeps $\lceil n/s \rceil$ positions. For an odd extent this
   * is $(e - 1)/2$ on both sides; for an even extent, pass a number to pad both sides equally. Default 0.
   */
  padding?: PerAxis | 'same'
  /** Spacing between kernel taps. Default 1. */
  dilation?: PerAxis
  /** Channel groups $g$; both channel counts must be divisible by it. Default 1 ($C$ for `depthwise`). */
  groups?: number
  /** Extra positions added at the end of each axis of a transposed layer's output. Default 0. */
  outputPadding?: PerAxis
  /** Whether the layer has one bias per output channel. Default true. */
  bias?: boolean
  /** Spatial shape of the input, e.g. `[H, W]`; its length (1, 2 or 3) is the number of spatial axes. */
  input: readonly number[]
}

/** What a convolution layer costs. */
export type ConvolutionCost = {
  /** Learned numbers: weights plus biases. */
  params: number
  /** Kernel weights, $CC'K/g$. */
  weights: number
  /** Biases, $C'$ or 0. */
  biases: number
  /** Multiply-accumulates for one input example. */
  multiplies: number
  /** Output spatial shape. */
  outputShape: number[]
  /** Output channels $C'$. */
  outChannels: number
  /** The channel groups used. */
  groups: number
}

/**
 * One integer per spatial axis from a `PerAxis` value. Throws `ShapeError` (named `where`) when a list has the wrong
 * length.
 *
 * @param value The value as given: one number for every axis, a list with one per axis, or undefined.
 * @param rank The number of spatial axes.
 * @param fallback The value of every axis when `value` is undefined.
 * @param name The option's name, for the error message.
 * @param where The caller's name for error messages.
 * @returns A new list of `rank` numbers.
 */
function perAxis(value: PerAxis | undefined, rank: number, fallback: number, name: string, where: string): number[] {
  if (value === undefined) return Array.from({ length: rank }, () => fallback)
  if (typeof value === 'number') return Array.from({ length: rank }, () => value)
  if (value.length !== rank)
    throw new ShapeError(where, `${where}: ${name} has ${value.length} axes, the input ${rank}`)
  return [...value]
}

/**
 * The product of a list of numbers (1 for an empty list).
 *
 * @param xs The numbers.
 * @returns Their product.
 */
const product = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1)

/**
 * The parameter count, multiply-accumulate count and output shape of one convolution layer (ordinary, grouped,
 * depthwise or transposed) with 1, 2 or 3 spatial axes. Throws `ShapeError` for an input of another rank, a per-axis
 * option of the wrong length, or a kernel that does not fit the padded input, and `DomainError` when the channels are
 * not divisible by the groups.
 *
 * @param spec The layer: kind, channels, kernel, stride, padding, dilation, groups, output padding, bias and the
 *   input's spatial shape.
 * @returns Parameters (weights and biases), multiply-accumulates per example, output shape and channels, and groups.
 *
 * @example A ResNet 3 x 3 layer at 56 x 56
 * const c = convolutionCost({ inChannels: 64, outChannels: 64, kernel: 3, padding: 1, input: [56, 56] })
 * print('params =', c.params, ', multiplies =', c.multiplies, ', output =', c.outputShape)
 *
 * @example Standard against depthwise separable, 256 channels
 * const common = { inChannels: 256, outChannels: 256, bias: false, input: [14, 14] }
 * const full = convolutionCost({ ...common, kernel: 3, padding: 1 })
 * const dw = convolutionCost({ ...common, kind: 'depthwise', kernel: 3, padding: 1 })
 * const pw = convolutionCost({ ...common, kernel: 1 })
 * print('weights: full', full.weights, ', separable', dw.weights + pw.weights)
 * print('ratio =', full.weights / (dw.weights + pw.weights))
 *
 * @example A transposed layer doubles 7 to 14
 * const up = { kind: 'transposed', inChannels: 8, outChannels: 4, kernel: 4, stride: 2, padding: 1, input: [7, 7] }
 * print('output =', convolutionCost(up).outputShape)
 */
export function convolutionCost(spec: ConvolutionLayerSpec): ConvolutionCost {
  const where = 'convolutionCost'
  const rank = spec.input.length
  if (rank < 1 || rank > 3) throw new ShapeError(where, `${where}: the input needs 1, 2 or 3 spatial axes`)
  const kind: ConvolutionKind = spec.kind ?? ((spec.groups ?? 1) > 1 ? 'grouped' : 'conv')
  const C = spec.inChannels
  const Cout = spec.outChannels ?? C
  const g = kind === 'depthwise' ? C : (spec.groups ?? 1)
  if (C % g !== 0 || Cout % g !== 0) {
    throw new DomainError(where, `${where}: ${C} input and ${Cout} output channels are not divisible by ${g} groups`)
  }
  const k = perAxis(spec.kernel, rank, 1, 'kernel', where)
  const s = perAxis(spec.stride, rank, 1, 'stride', where)
  const d = perAxis(spec.dilation, rank, 1, 'dilation', where)
  const extent = k.map((ki, i) => d[i] * (ki - 1) + 1)
  const lo =
    spec.padding === 'same'
      ? extent.map((e) => Math.ceil((e - 1) / 2))
      : perAxis(spec.padding, rank, 0, 'padding', where)
  const hi = spec.padding === 'same' ? extent.map((e) => Math.floor((e - 1) / 2)) : lo
  const taps = product(k)
  const weights = (C * Cout * taps) / g
  const biases = spec.bias === false ? 0 : Cout
  let outputShape: number[]
  let multiplies: number
  if (kind === 'transposed') {
    const op = perAxis(spec.outputPadding, rank, 0, 'outputPadding', where)
    outputShape = spec.input.map((n, i) => (n - 1) * s[i] - lo[i] - hi[i] + extent[i] + op[i])
    // Every input value meets every tap of every output channel in its group.
    multiplies = product(spec.input) * C * (Cout / g) * taps
  } else {
    // The padding is lo before and hi after (lo - hi is 0 or 1): the size with hi on both sides of n + lo - hi.
    outputShape = spec.input.map((n, i) => convOutputSize(n + lo[i] - hi[i], k[i], s[i], hi[i], d[i]))
    // Every output value sums over the taps of the input channels in its group.
    multiplies = product(outputShape) * Cout * (C / g) * taps
  }
  if (outputShape.some((m) => m < 1)) {
    throw new ShapeError(where, `${where}: the kernel does not fit the input ${spec.input.join(' × ')}`)
  }
  return { params: weights + biases, weights, biases, multiplies, outputShape, outChannels: Cout, groups: g }
}

/** A layer of a chain for `convolutionStackCost`: its input channels and shape come from the previous layer. */
export type ChainedLayerSpec = Omit<ConvolutionLayerSpec, 'inChannels' | 'input'> & {
  /** A name for the layer in the result. Default `layer i` (from 1). */
  name?: string
}

/** One layer of a chain with its cost, input and output. */
export type ChainedLayerCost = ConvolutionCost & {
  /** The layer's name. */
  name: string
  /** The channels the layer reads. */
  inChannels: number
  /** The spatial shape the layer reads. */
  inputShape: number[]
}

/**
 * The cost of a chain of convolution layers, each fed by the previous one's output, by `convolutionCost` per layer.
 * Throws as `convolutionCost` does for the first layer that does not fit.
 *
 * @param input The channels and spatial shape of the chain's input.
 * @param layers The layers in order, without their input channels and shape.
 * @returns Each layer's cost with its input, the totals of parameters and multiply-accumulates, and the chain's output
 *   shape and channels.
 *
 * @example A 1 x 1, 3 x 3, 1 x 1 bottleneck from 256 channels at 56 x 56
 * const b = convolutionStackCost({ channels: 256, shape: [56, 56] }, [
 *   { kernel: 1, outChannels: 64, bias: false },
 *   { kernel: 3, outChannels: 64, padding: 1, bias: false },
 *   { kernel: 1, outChannels: 256, bias: false },
 * ])
 * print('params per layer =', b.layers.map((l) => l.params))
 * print('total params =', b.params, ', multiplies =', b.multiplies)
 */
export function convolutionStackCost(
  input: { channels: number; shape: readonly number[] },
  layers: readonly ChainedLayerSpec[],
): { layers: ChainedLayerCost[]; params: number; multiplies: number; outputShape: number[]; outChannels: number } {
  let channels = input.channels
  let shape = [...input.shape]
  const out: ChainedLayerCost[] = []
  layers.forEach((layer, i) => {
    const cost = convolutionCost({ ...layer, inChannels: channels, input: shape })
    out.push({ ...cost, name: layer.name ?? `layer ${i + 1}`, inChannels: channels, inputShape: shape })
    channels = cost.outChannels
    shape = cost.outputShape
  })
  return {
    layers: out,
    params: out.reduce((a, l) => a + l.params, 0),
    multiplies: out.reduce((a, l) => a + l.multiplies, 0),
    outputShape: shape,
    outChannels: channels,
  }
}

// ── Whole networks ───────────────────────────────────────────────────────────────────────────────────────────────────

/** An optional label, shown in `networkCost`'s rows. */
type Labelled = {
  /** The label of the layer or block. */
  label?: string
}

/** A convolution for `networkCost`; its input channels come from the input shape. */
export type ConvLayer = Labelled & {
  /** The kind of spec. */
  kind: 'conv'
  /** Output channels. */
  out: number
  /** Kernel taps per spatial axis. */
  kernel: PerAxis
  /** Step between outputs. Default 1. */
  stride?: PerAxis
  /** Zeros per side, or `same` as in `convolutionCost`. Default 0. */
  padding?: PerAxis | 'same'
  /** Spacing between taps. Default 1. */
  dilation?: PerAxis
  /** Channel groups; divides both channel counts. Default 1. */
  groups?: number
  /** Whether the layer has biases. Default true. */
  bias?: boolean
}

/** A depthwise convolution: one filter per input channel (groups $= C$), `multiplier` outputs per channel. */
export type DepthwiseLayer = Labelled & {
  /** The kind of spec. */
  kind: 'depthwise'
  /** Kernel taps per spatial axis. */
  kernel: PerAxis
  /** Step between outputs. Default 1. */
  stride?: PerAxis
  /** Zeros per side, or `same`. Default 0. */
  padding?: PerAxis | 'same'
  /** Spacing between taps. Default 1. */
  dilation?: PerAxis
  /** Output channels per input channel. Default 1. */
  multiplier?: number
  /** Whether the layer has biases. Default true. */
  bias?: boolean
}

/** A transposed convolution: output size $(n - 1)s - 2p + d(k - 1) + p_{\text{out}} + 1$. */
export type TransposedLayer = Labelled & {
  /** The kind of spec. */
  kind: 'transposed'
  /** Output channels. */
  out: number
  /** Kernel taps per spatial axis. */
  kernel: PerAxis
  /** The upsampling factor. Default 1. */
  stride?: PerAxis
  /** Zeros removed from each side of the output. Default 0. */
  padding?: PerAxis
  /** Extra positions at the end of each output axis. Default 0. */
  outputPadding?: PerAxis
  /** Spacing between taps. Default 1. */
  dilation?: PerAxis
  /** Channel groups. Default 1. */
  groups?: number
  /** Whether the layer has biases. Default true. */
  bias?: boolean
}

/** Max or average pooling over a window: no parameters, no counted multiplies. */
export type PoolLayer = Labelled & {
  /** The kind of spec. */
  kind: 'pool'
  /** Window taps per spatial axis. */
  kernel: PerAxis
  /** Step between windows. Default the kernel. */
  stride?: PerAxis
  /** Padding per side. Default 0. */
  padding?: PerAxis
  /** PyTorch's ceiling mode for the output size. Default false. */
  ceil?: boolean
}

/** Global average (or max) pooling: every spatial axis to 1. */
export type GlobalPoolLayer = Labelled & {
  /** The kind of spec. */
  kind: 'globalPool'
}

/** A fully connected layer on the flattened input. */
export type DenseLayer = Labelled & {
  /** The kind of spec. */
  kind: 'dense'
  /** Output units. */
  out: number
  /** Whether the layer has biases. Default true. */
  bias?: boolean
}

/** A normalisation layer with a per-channel scale and shift (batch, layer or group norm): $2C$ parameters. */
export type NormLayer = Labelled & {
  /** The kind of spec. */
  kind: 'norm'
  /** Whether it has the scale and shift. Default true. */
  affine?: boolean
}

/** A learned per-channel scale (ConvNeXt's layer scale): $C$ parameters. */
export type ChannelScaleLayer = Labelled & {
  /** The kind of spec. */
  kind: 'channelScale'
}

/** A parameter-free elementwise layer (an activation, dropout): shape and cost unchanged. */
export type ElementwiseLayer = Labelled & {
  /** The kind of spec. */
  kind: 'activation'
}

/** Layers applied in order. */
export type SequenceLayer = Labelled & {
  /** The kind of spec. */
  kind: 'sequence'
  /** The layers, first applied first. */
  layers: readonly LayerSpec[]
}

/** The sum of the branch outputs (each a sequence; an empty one is the identity). Shapes must agree. */
export type AddLayer = Labelled & {
  /** The kind of spec. */
  kind: 'add'
  /** The branches, each a list of layers. */
  branches: readonly (readonly LayerSpec[])[]
}

/** The branch outputs concatenated along channels. Spatial shapes must agree. */
export type ConcatLayer = Labelled & {
  /** The kind of spec. */
  kind: 'concat'
  /** The branches, each a list of layers. */
  branches: readonly (readonly LayerSpec[])[]
}

/** The input multiplied channelwise by the branch's output, which must have shape $[C, 1, \dots, 1]$. Depth 0. */
export type GateLayer = Labelled & {
  /** The kind of spec. */
  kind: 'gate'
  /** The gating branch. */
  branch: readonly LayerSpec[]
}

/** A layer or block of a network, for `layerCost` and `networkCost`. */
export type LayerSpec =
  | ConvLayer
  | DepthwiseLayer
  | TransposedLayer
  | PoolLayer
  | GlobalPoolLayer
  | DenseLayer
  | NormLayer
  | ChannelScaleLayer
  | ElementwiseLayer
  | SequenceLayer
  | AddLayer
  | ConcatLayer
  | GateLayer

/** The theoretical receptive field after a layer, per spatial axis. */
export type FieldSpan = {
  /** Input pixels per spatial axis that can affect one output value (Infinity once global). */
  size: number[]
  /** Input pixels between neighbouring outputs, per spatial axis. */
  jump: number[]
}

/** What a layer or network costs. */
export type LayerCost = {
  /** Learned numbers. */
  params: number
  /** Multiply-accumulates of one forward pass on one input. */
  multiplies: number
  /** Output shape $[C, \dots]$ (or $[n]$ after a dense layer). */
  outputShape: number[]
  /** Weight layers (convolution, depthwise, transposed, dense) on the longest path. */
  depth: number
  /** The receptive field and jump after the layer. */
  receptiveField: FieldSpan
}

/** What a network costs, in total and per top-level entry. */
export type NetworkCost = LayerCost & {
  /** One entry per top-level spec, in order, with its label, cost, input shape and the shape and field after it. */
  layers: (LayerCost & { label: string; inputShape: number[] })[]
}

const WHERE = 'networkCost'

/**
 * The field of the input itself: size 1 and jump 1 on every spatial axis.
 *
 * @param input The input shape $[C, \dots]$.
 * @returns The starting field.
 */
const startField = (input: readonly number[]): FieldSpan => ({
  size: input.slice(1).map(() => 1),
  jump: input.slice(1).map(() => 1),
})

/**
 * The field after a window of `span` input positions (already dilated) and stride `stride` on each axis, by
 * `growField`.
 *
 * @param rf The field before the window.
 * @param span The dilated extent per axis.
 * @param stride The stride per axis.
 * @returns The field after it.
 */
function grow(rf: FieldSpan, span: readonly number[], stride: readonly number[]): FieldSpan {
  const steps = rf.size.map((s, i) => growField(s, rf.jump[i], span[i], stride[i]))
  return { size: steps.map((t) => t.size), jump: steps.map((t) => t.jump) }
}

/**
 * The field after a layer that sees the whole input (dense, global pooling): Infinity everywhere.
 *
 * @param rf The field before the layer (only its rank is read).
 * @returns The global field.
 */
const globalField = (rf: FieldSpan): FieldSpan => ({
  size: rf.size.map(() => Infinity),
  jump: rf.jump.map(() => Infinity),
})

/**
 * The largest field over branches, per axis.
 *
 * @param fields The branches' fields (at least one).
 * @returns The per-axis maximum of size and of jump.
 */
const widest = (fields: readonly FieldSpan[]): FieldSpan => ({
  size: fields[0].size.map((_, i) => Math.max(...fields.map((f) => f.size[i]))),
  jump: fields[0].jump.map((_, i) => Math.max(...fields.map((f) => f.jump[i]))),
})

/**
 * Whether two shapes are equal.
 *
 * @param a A shape.
 * @param b Another shape.
 * @returns True when they have the same length and entries.
 */
const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((v, i) => v === b[i])

/**
 * The spatial output of a pooling window, $\lfloor (n + 2p - k)/s \rfloor + 1$ per axis, or PyTorch's ceiling mode
 * (which drops a last window that would start inside the right padding). Throws `ShapeError` when the window does not
 * fit.
 *
 * @param spatial The input's spatial shape.
 * @param kernel The window taps per axis.
 * @param stride The stride per axis.
 * @param padding The padding per side per axis.
 * @param ceil Whether to round up.
 * @returns The output's spatial shape.
 */
function pooled(
  spatial: readonly number[],
  kernel: readonly number[],
  stride: readonly number[],
  padding: readonly number[],
  ceil: boolean,
): number[] {
  return spatial.map((n, i) => {
    const raw = (n + 2 * padding[i] - kernel[i]) / stride[i]
    let m = (ceil ? Math.ceil(raw) : Math.floor(raw)) + 1
    if (ceil && (m - 1) * stride[i] >= n + padding[i]) m -= 1
    if (m < 1)
      throw new ShapeError(WHERE, `${WHERE}: a window of ${kernel[i]} does not fit ${n} with padding ${padding[i]}`)
    return m
  })
}

/**
 * Cost and output of one spec on one input, with the receptive field carried through. Throws `ShapeError` for
 * branches whose shapes do not agree and `DomainError` for a combinator without branches.
 *
 * @param spec The layer or block.
 * @param input The input shape $[C, \dots]$.
 * @param rf The receptive field of the input.
 * @returns The cost, output shape, depth and field.
 */
function walk(spec: LayerSpec, input: readonly number[], rf: FieldSpan): LayerCost {
  const [channels, ...spatial] = input
  const dims = spatial.length
  switch (spec.kind) {
    case 'conv':
    case 'depthwise':
    case 'transposed': {
      const c = convolutionCost({
        kind: spec.kind === 'conv' ? undefined : spec.kind,
        inChannels: channels,
        outChannels: spec.kind === 'depthwise' ? channels * (spec.multiplier ?? 1) : spec.out,
        kernel: spec.kernel,
        stride: spec.stride,
        padding: spec.padding,
        dilation: spec.dilation,
        groups: spec.kind === 'depthwise' ? undefined : spec.groups,
        outputPadding: spec.kind === 'transposed' ? spec.outputPadding : undefined,
        bias: spec.bias,
        input: spatial,
      })
      const kernel = perAxis(spec.kernel, dims, 1, 'kernel', WHERE)
      const stride = perAxis(spec.stride, dims, 1, 'stride', WHERE)
      const dilation = perAxis(spec.dilation, dims, 1, 'dilation', WHERE)
      const span = kernel.map((k, i) => dilation[i] * (k - 1) + 1)
      // A transposed layer's output depends on about ceil(span / s) inputs, and the jump divides by the stride: an
      // approximation, exact when s divides the span.
      const receptiveField =
        spec.kind === 'transposed'
          ? grow(
              rf,
              span.map((e, i) => Math.ceil(e / stride[i])),
              stride.map((s) => 1 / s),
            )
          : grow(rf, span, stride)
      return {
        params: c.params,
        multiplies: c.multiplies,
        outputShape: [c.outChannels, ...c.outputShape],
        depth: 1,
        receptiveField,
      }
    }
    case 'pool': {
      const kernel = perAxis(spec.kernel, dims, 1, 'kernel', WHERE)
      const stride = spec.stride === undefined ? kernel : perAxis(spec.stride, dims, 1, 'stride', WHERE)
      const padding = perAxis(spec.padding, dims, 0, 'padding', WHERE)
      return {
        params: 0,
        multiplies: 0,
        outputShape: [channels, ...pooled(spatial, kernel, stride, padding, spec.ceil ?? false)],
        depth: 0,
        receptiveField: grow(rf, kernel, stride),
      }
    }
    case 'globalPool':
      return {
        params: 0,
        multiplies: 0,
        outputShape: [channels, ...spatial.map(() => 1)],
        depth: 0,
        receptiveField: globalField(rf),
      }
    case 'dense': {
      const inputs = product(input)
      return {
        params: inputs * spec.out + ((spec.bias ?? true) ? spec.out : 0),
        multiplies: inputs * spec.out,
        outputShape: [spec.out],
        depth: 1,
        receptiveField: globalField(rf),
      }
    }
    case 'norm':
      return {
        params: (spec.affine ?? true) ? 2 * channels : 0,
        multiplies: 0,
        outputShape: [...input],
        depth: 0,
        receptiveField: rf,
      }
    case 'channelScale':
      return { params: channels, multiplies: 0, outputShape: [...input], depth: 0, receptiveField: rf }
    case 'activation':
      return { params: 0, multiplies: 0, outputShape: [...input], depth: 0, receptiveField: rf }
    case 'sequence':
      return chain(spec.layers, input, rf)
    case 'add':
    case 'concat': {
      if (spec.branches.length === 0) throw new DomainError(WHERE, `${WHERE}: ${spec.kind} needs at least one branch`)
      const results = spec.branches.map((b) => chain(b, input, rf))
      const first = results[0].outputShape
      for (const r of results) {
        const ok =
          spec.kind === 'add' ? sameShape(r.outputShape, first) : sameShape(r.outputShape.slice(1), first.slice(1))
        if (!ok)
          throw new ShapeError(
            WHERE,
            `${WHERE}: ${spec.kind} branch shapes [${r.outputShape}] and [${first}] do not agree`,
            [r.outputShape, first],
          )
      }
      const outChannels = spec.kind === 'add' ? first[0] : results.reduce((sum, r) => sum + r.outputShape[0], 0)
      return {
        params: results.reduce((s, r) => s + r.params, 0),
        multiplies: results.reduce((s, r) => s + r.multiplies, 0),
        outputShape: [outChannels, ...first.slice(1)],
        depth: Math.max(...results.map((r) => r.depth)),
        receptiveField: widest(results.map((r) => r.receptiveField)),
      }
    }
    case 'gate': {
      const r = chain(spec.branch, input, rf)
      if (r.outputShape[0] !== channels || r.outputShape.slice(1).some((v) => v !== 1))
        throw new ShapeError(WHERE, `${WHERE}: gate branch output [${r.outputShape}] is not [${channels}, 1, …]`, [
          r.outputShape,
        ])
      return {
        params: r.params,
        multiplies: r.multiplies,
        outputShape: [...input],
        // The gate's branch sits beside the main path: it is not counted in the depth.
        depth: 0,
        receptiveField: widest([rf, r.receptiveField]),
      }
    }
  }
}

/**
 * Cost of a list of specs applied in order.
 *
 * @param layers The specs.
 * @param input The input shape $[C, \dots]$.
 * @param rf The receptive field of the input.
 * @returns The summed cost, the depth along the chain, and the last output shape and field.
 */
function chain(layers: readonly LayerSpec[], input: readonly number[], rf: FieldSpan): LayerCost {
  let shape = [...input]
  let field = rf
  let params = 0
  let multiplies = 0
  let depth = 0
  for (const layer of layers) {
    const r = walk(layer, shape, field)
    params += r.params
    multiplies += r.multiplies
    depth += r.depth
    shape = r.outputShape
    field = r.receptiveField
  }
  return { params, multiplies, outputShape: shape, depth, receptiveField: field }
}

/**
 * Parameters, multiply-accumulates, output shape, depth and receptive field of one layer spec (a plain layer or a
 * block) on `input`. Throws `ShapeError` when a layer does not fit or branch shapes disagree.
 *
 * @param spec The layer or block.
 * @param input The input shape $[C, \dots]$, channels first.
 * @returns Its cost, with the field of a unit of its output measured from its input.
 *
 * @example A residual block with a projection shortcut
 * const block = {
 *   kind: 'add',
 *   branches: [
 *     [
 *       { kind: 'conv', out: 128, kernel: 3, stride: 2, padding: 1, bias: false },
 *       { kind: 'norm' },
 *       { kind: 'activation' },
 *       { kind: 'conv', out: 128, kernel: 3, padding: 1, bias: false },
 *       { kind: 'norm' },
 *     ],
 *     [{ kind: 'conv', out: 128, kernel: 1, stride: 2, bias: false }, { kind: 'norm' }],
 *   ],
 * }
 * const c = layerCost(block, [64, 56, 56])
 * print('params =', c.params, ', output =', c.outputShape, ', depth =', c.depth)
 * print('receptive field =', c.receptiveField.size)
 */
export function layerCost(spec: LayerSpec, input: readonly number[]): LayerCost {
  return walk(spec, input, startField(input))
}

/**
 * The cost of a network given as a list of specs applied in order to `input`, in total and per top-level entry: the
 * parameters, multiply-accumulates, output shape, depth and receptive field. Throws `ShapeError` when a layer does
 * not fit or branch shapes disagree.
 *
 * @param layers The top-level specs (often labelled `sequence` stages).
 * @param input The input shape $[C, \dots]$, channels first.
 * @returns The totals, and one row per top-level spec with its label, input shape and cost.
 *
 * @example Three 3 x 3 layers see 7 pixels; a stride-2 first layer makes it 11
 * const three = [1, 1, 1].map(() => ({ kind: 'conv', out: 8, kernel: 3, padding: 1 }))
 * print('stride 1:', networkCost(three, [3, 32, 32]).receptiveField.size)
 * const strided = [{ kind: 'conv', out: 8, kernel: 3, stride: 2, padding: 1 }, ...three.slice(1)]
 * print('stride 2 first:', networkCost(strided, [3, 32, 32]).receptiveField.size)
 *
 * @example LeNet-5, stage by stage
 * const lenet = [
 *   { kind: 'sequence', label: 'C1, S2', layers: [{ kind: 'conv', out: 6, kernel: 5 }, { kind: 'pool', kernel: 2 }] },
 *   { kind: 'sequence', label: 'C3, S4', layers: [{ kind: 'conv', out: 16, kernel: 5 }, { kind: 'pool', kernel: 2 }] },
 *   { kind: 'sequence', label: 'C5', layers: [{ kind: 'conv', out: 120, kernel: 5 }] },
 *   { kind: 'sequence', label: 'F6, output', layers: [{ kind: 'dense', out: 84 }, { kind: 'dense', out: 10 }] },
 * ]
 * const r = networkCost(lenet, [1, 32, 32])
 * for (const row of r.layers) print(row.label, ': params', row.params, ', output', row.outputShape)
 * print('total params =', r.params, ', multiplies =', r.multiplies, ', depth =', r.depth)
 */
export function networkCost(layers: readonly LayerSpec[], input: readonly number[]): NetworkCost {
  let shape = [...input]
  let field = startField(input)
  const rows: NetworkCost['layers'] = []
  for (const [i, layer] of layers.entries()) {
    const r = walk(layer, shape, field)
    rows.push({ ...r, label: layer.label ?? `${layer.kind} ${i + 1}`, inputShape: shape })
    shape = r.outputShape
    field = r.receptiveField
  }
  return {
    params: rows.reduce((s, r) => s + r.params, 0),
    multiplies: rows.reduce((s, r) => s + r.multiplies, 0),
    outputShape: shape,
    depth: rows.reduce((s, r) => s + r.depth, 0),
    receptiveField: field,
    layers: rows,
  }
}
