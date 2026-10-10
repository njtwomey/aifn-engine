/**
 * Layers as pairs of an initialiser and a pure forward function: `layer.init(stream)` gives the parameters (a pytree
 * of tensors) and `layer.apply(params, x, ctx)` the output. There is no hidden state: parameters are passed in, so
 * `grad` of any function of them differentiates through the layer, and the same parameters can be evaluated, perturbed
 * or traced.
 * Non-trainable state (batch norm's running statistics; torch's buffers, flax's `batch_stats` collection) is passed in
 * the same way, in `ctx.buffers`, and a layer in training mode writes its new entry to `ctx.bufferUpdates`.
 *
 * Functional forms (`linear`, `layerNorm`, `rmsNorm`, `batchNorm`, `dropout`) are exported beside the layers, which
 * only add parameter shapes and initialisation. Every layer passes its output through `ctx.tap(path, value)` when the
 * context has one, which is how `inspect` records activations and their gradients.
 *
 * Layers are named in PascalCase, like distributions: they are constructors of plain objects.
 */

import { bernoulli, child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  div,
  expandDims,
  matmul,
  mean,
  mul,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sub,
  take,
  unwrap,
  ones,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { activationFn, type Activation } from 'aifn-compute/nn/functional'
import { heUniform, lecunUniform, normalInit, zerosInit, type Initialiser } from 'aifn-compute/nn/init'
import {
  avgPool1d,
  avgPool2d,
  conv1d,
  conv2d,
  maxPool1d,
  maxPool2d,
  type ConvOptions,
  type Pair,
  type PoolOptions,
} from 'aifn-compute/nn/functional'
import type { Params } from 'aifn-compute/foundation/pytree'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── The layer protocol ───────────────────────────────────────────────────────────────────────────────────────────────

/** What a forward pass may need beyond parameters and input. */
export type Context = {
  /** Training mode: dropout is active. Default false (evaluation). */
  train?: boolean
  /** The stream dropout draws its masks from (required when training with dropout); layers take children of it. */
  stream?: Stream
  /**
   * Called with each layer's output and its path (e.g. `2`, `1.attention`); returns the value to pass on. Recording
   * activations returns them unchanged; `inspect` adds zero probes here to get their gradients.
   */
  tap?: (path: string, value: Value) => Value
  /** The path of the layer being applied, set by containers. */
  path?: string
  /**
   * The non-trainable state collection, each stateful layer's entry by its path (e.g. `{ '1': { mean, variance } }`).
   * Read in evaluation mode; absent entries take the layer's initial values.
   */
  buffers?: Buffers
  /**
   * In training mode, stateful layers write their new entries here, by path; the caller merges them into `buffers`
   * (`{ ...buffers, ...bufferUpdates }`). Without it, training-mode layers update nothing.
   */
  bufferUpdates?: Record<string, Params>
}

/**
 * Non-trainable state by layer path: plain data, never differentiated (written from primal values, so a layer applied
 * inside `grad` records concrete tensors).
 */
export type Buffers = Readonly<Record<string, Params>>

/** A layer: an initialiser and a pure forward function of parameters `P`, with a kind and a label for display. */
export interface Layer<P extends Params = Params> {
  /** The layer type, e.g. `Linear`. */
  readonly kind: string
  /** A short description, e.g. `Linear(2 → 16)`. */
  readonly label: string
  /** Fresh parameters drawn from `s` (layers without parameters return `{}`). */
  init(s: Stream): P
  /** The output for the input `x` under the parameters `params`; pure. */
  apply(params: P, x: Value, ctx?: Context): Value
}

/**
 * Pass a layer's output through the context's tap, under the layer's path (`ctx.path`, or `'output'` at the top
 * level) or a named part of it.
 *
 * @param ctx The context of the layer being applied; without a `tap`, the value is returned as it is.
 * @param value The value to record, usually the layer's output.
 * @param suffix A name for an intermediate value of the layer, appended to its path (`'1.gate'`); left out, the value
 *   is recorded under the path itself.
 * @returns What the tap returns (the value itself when recording, the value plus a zero probe for `inspect`).
 *
 * @example Record the values a layer passes on
 * const seen = []
 * const ctx = { path: '1', tap: (path, v) => (seen.push(path), v) }
 * tap(ctx, tensor([1, 2]))
 * tap(ctx, tensor([3]), 'gate')
 * print('recorded paths:', seen)
 * print('no tap, unchanged:', tap(undefined, tensor([5])))
 */
export function tap(ctx: Context | undefined, value: Value, suffix?: string): Value {
  if (!ctx?.tap) return value
  const base = ctx.path ?? ''
  const path = suffix ? (base ? `${base}.${suffix}` : suffix) : base || 'output'
  return ctx.tap(path, value)
}

/**
 * The context for a child layer at `name` below the current path, for containers to pass to their sub-layers.
 *
 * @param ctx The container's context; everything but `path` is shared with the child.
 * @param name The child's name or index, appended to the path after a dot.
 * @returns The context with the child's path, or undefined when `ctx` is.
 *
 * @example Paths below a container
 * print('below 2:', childContext({ path: '2' }, 'attention').path)
 * print('at the top:', childContext({}, 0).path)
 * print('no context:', childContext(undefined, 0))
 */
export function childContext(ctx: Context | undefined, name: string | number): Context | undefined {
  if (!ctx) return undefined
  const base = ctx.path ?? ''
  return { ...ctx, path: base ? `${base}.${name}` : String(name) }
}

/** The parameters of a layer that has none. */
type Empty = Record<string, never>
/** The one empty parameter object, shared by every layer without parameters. */
const EMPTY: Empty = {}

/**
 * A layer without parameters from a function of its input; its output goes through the context's tap.
 *
 * @param kind The layer type, e.g. `MaxPool2d`.
 * @param label The short description shown for the layer.
 * @param f The forward function of the input and the context.
 * @returns The layer, whose `init` returns `{}`.
 */
function stateless(kind: string, label: string, f: (x: Value, ctx?: Context) => Value): Layer<Empty> {
  return { kind, label, init: () => EMPTY, apply: (_p, x, ctx) => tap(ctx, f(x, ctx)) }
}

// ── Dense and embedding ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The affine map $\xvec\Wmat + \bvec$ for $\xvec$ of shape `[..., in]` and $\Wmat$ of shape `[in, out]` (the transpose
 * of PyTorch's storage), giving `[..., out]`. Differentiable in all three.
 *
 * @param x The input, `[..., in]`: one row or a batch of rows.
 * @param weight The weight matrix $\Wmat$, `[in, out]`.
 * @param bias The bias $\bvec$, `[out]`, broadcast over the rows; left out, no bias is added.
 * @returns $\xvec\Wmat + \bvec$, `[..., out]`.
 *
 * @example Two rows through a $3 \times 2$ weight
 * const x = tensor([[1, 0, 2], [0, 1, 0]])
 * const W = tensor([[1, 2], [3, 4], [5, 6]])
 * print('x W + b =', linear(x, W, tensor([0.5, -0.5])))
 * print('x W =', linear(x, W))
 */
export function linear(x: Value, weight: Value, bias?: Value): Value {
  const y = matmul(x, weight)
  return bias === undefined ? y : add(y, bias)
}

/** Parameters of `Linear`: `weight`, `[in, out]`, and `bias`, `[out]` (absent with `bias: false`). */
export type LinearParams = { weight: Tensor; bias?: Tensor }

/** Options of `Linear`. */
export type LinearOptions = {
  /** Include a bias (default true). */
  bias?: boolean
  /** Weight initialiser (default LeCun uniform, PyTorch's scale). */
  init?: Initialiser
  /** Bias initialiser (default zeros). */
  biasInit?: Initialiser
}

/**
 * A dense layer $\xvec\Wmat + \bvec$ from `inFeatures` to `outFeatures`; see `linear`. The weight is drawn by the
 * `init` initialiser with fans `inFeatures` and `outFeatures`, from the child `'weight'` of the stream.
 *
 * @param inFeatures The input width, the last axis of `x`.
 * @param outFeatures The output width.
 * @param options Whether there is a bias, and the weight and bias initialisers.
 * @returns The layer, with parameters `LinearParams`.
 *
 * @example The output is $\xvec\Wmat + \bvec$ with the drawn parameters
 * const layer = Linear(3, 2)
 * const p = layer.init(stream(0))
 * const x = tensor([1, 2, 3])
 * print(layer.label, ' W =', p.weight, ' b =', p.bias)
 * print('apply:', layer.apply(p, x))
 * print('x W + b:', add(matmul(x, p.weight), p.bias))
 */
export function Linear(inFeatures: number, outFeatures: number, options: LinearOptions = {}): Layer<LinearParams> {
  const { bias = true, init = lecunUniform(), biasInit = zerosInit() } = options
  const fans = { fanIn: inFeatures, fanOut: outFeatures }
  return {
    kind: 'Linear',
    label: `Linear(${inFeatures} → ${outFeatures})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [inFeatures, outFeatures], fans),
      ...(bias ? { bias: biasInit(child(s, 'bias'), [outFeatures], fans) } : {}),
    }),
    apply: (p, x, ctx) => tap(ctx, linear(x, p.weight, p.bias)),
  }
}

/** Parameters of `Embedding`: the table `weight`, `[vocabulary, dimension]`, one row per id. */
export type EmbeddingParams = { weight: Tensor }

/**
 * A lookup table of `vocabulary` vectors of length `dimension`: `apply(params, ids)` maps a tensor of integer ids of
 * any shape to `[...ids shape, dimension]`, and a single number id to `[1, dimension]`. The gradient adds up over
 * repeated ids. Initialised $\Gauss(0, 1)$ by default, as PyTorch. Ids outside `[0, vocabulary)` throw `ShapeError`.
 *
 * @param vocabulary The number of ids, the rows of the table.
 * @param dimension The length of each vector.
 * @param options The table's initialiser.
 * @param options.init Draws the table, with fans `vocabulary` and `dimension`; default $\Gauss(0, 1)$.
 * @returns The layer, with parameters `EmbeddingParams`.
 *
 * @example Look up rows of the table
 * const emb = Embedding(4, 3)
 * const p = emb.init(stream(0))
 * print('table:', p.weight)
 * print('ids 2, 0, 2:', emb.apply(p, tensor([2, 0, 2])))
 * print('shape for a 2 x 2 grid of ids:', emb.apply(p, tensor([[1, 2], [3, 0]])).shape)
 */
export function Embedding(
  vocabulary: number,
  dimension: number,
  { init = normalInit(1) }: { init?: Initialiser } = {},
): Layer<EmbeddingParams> {
  return {
    kind: 'Embedding',
    label: `Embedding(${vocabulary} × ${dimension})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [vocabulary, dimension], { fanIn: vocabulary, fanOut: dimension }),
    }),
    apply: (p, ids, ctx) => {
      const raw = unwrap(ids)
      if (typeof raw === 'number') return tap(ctx, take(p.weight, [raw]))
      return tap(ctx, take(p.weight, raw))
    },
  }
}

// ── Convolution and pooling ──────────────────────────────────────────────────────────────────────────────────────────

/** Parameters of a convolution: kernels `weight`, `[O, C, ...kernel]`, and `bias`, `[O]` (absent without a bias). */
export type ConvParams = { weight: Tensor; bias?: Tensor }

/** Options of `Conv1d` and `Conv2d`. */
export type ConvLayerOptions<P> = ConvOptions<P> & {
  /** Include a bias, one per output channel (default true). */
  bias?: boolean
  /** Kernel initialiser (default He uniform, for ReLU networks). */
  init?: Initialiser
}

/**
 * The input channels each kernel sees, $C / g$, after checking that $g$ is a positive integer dividing $C$ and $O$.
 *
 * @param where The layer, for the error.
 * @param inChannels The input channels $C$.
 * @param outChannels The output channels $O$.
 * @param groups The channel groups $g$ (default 1).
 * @returns $C / g$.
 */
function groupChannels(where: string, inChannels: number, outChannels: number, groups = 1): number {
  if (!Number.isInteger(groups) || groups < 1 || inChannels % groups !== 0 || outChannels % groups !== 0)
    throw new ShapeError(
      where,
      `${where}: ${inChannels} input and ${outChannels} output channels do not split into ${groups} groups`,
    )
  return inChannels / groups
}

/**
 * A 2-D convolution layer from `inChannels` to `outChannels` with a `kernel` (one size or `[KH, KW]`), plus a bias per
 * output channel; input `[N, C, H, W]` or `[C, H, W]`. See `conv2d` for the geometry. With $g$ `groups` the kernels
 * are `[O, C / g, KH, KW]`, drawn with fans $(C / g) K_H K_W$ and $O K_H K_W$ (torch's); $g$ must divide both $C$ and
 * $O$, or a `ShapeError` is thrown.
 *
 * @param inChannels The input channels $C$.
 * @param outChannels The output channels $O$.
 * @param kernel The kernel size: one number for a square kernel, or `[KH, KW]`.
 * @param options The stride, padding, dilation and groups of `conv2d`, whether there is a bias, and the kernel
 *   initialiser.
 * @returns The layer, with parameters `ConvParams`; its output is `[N, O, H', W']` (or `[O, H', W']`).
 *
 * @example Two $3 \times 3$ filters over a $5 \times 5$ image, with and without padding
 * const x = reshape(arange(25), [1, 1, 5, 5])
 * const layer = Conv2d(1, 2, 3)
 * const p = layer.init(stream(0))
 * print(layer.label, ' output shape:', layer.apply(p, x).shape)
 * const same = Conv2d(1, 2, 3, { padding: 1 })
 * print('padding 1, output shape:', same.apply(same.init(stream(0)), x).shape)
 * print('kernels:', p.weight.shape, ' bias:', p.bias)
 */
export function Conv2d(
  inChannels: number,
  outChannels: number,
  kernel: Pair,
  options: ConvLayerOptions<Pair> = {},
): Layer<ConvParams> {
  const [kh, kw] = typeof kernel === 'number' ? [kernel, kernel] : kernel
  const { bias = true, init = heUniform(), ...geometry } = options
  const perGroup = groupChannels('Conv2d', inChannels, outChannels, geometry.groups)
  const fans = { fanIn: perGroup * kh * kw, fanOut: outChannels * kh * kw }
  return {
    kind: 'Conv2d',
    label: `Conv2d(${inChannels} → ${outChannels}, ${kh}×${kw})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [outChannels, perGroup, kh, kw], fans),
      ...(bias ? { bias: zerosInit()(s, [outChannels], fans) } : {}),
    }),
    apply: (p, x, ctx) => {
      const y = conv2d(x, p.weight, geometry)
      const rank = shapeOfValue(y).length
      const b = p.bias === undefined ? undefined : reshape(p.bias, [outChannels, 1, 1])
      return tap(ctx, b === undefined ? y : add(y, rank === 4 ? expandDims(b, 0) : b))
    },
  }
}

/**
 * A 1-D convolution layer from `inChannels` to `outChannels` with a kernel of length `kernel`, plus a bias per output
 * channel; input `[N, C, L]` or `[C, L]`. See `conv1d` for the geometry. As with `Conv2d`, $g$ `groups` give kernels
 * `[O, C / g, K]` with fans $(C / g) K$ and $O K$; $g$ must divide both $C$ and $O$.
 *
 * @param inChannels The input channels $C$.
 * @param outChannels The output channels $O$.
 * @param kernel The kernel length $K$.
 * @param options The stride, padding, dilation and groups of `conv1d`, whether there is a bias, and the kernel
 *   initialiser.
 * @returns The layer, with parameters `ConvParams`; its output is `[N, O, L']` (or `[O, L']`).
 *
 * @example Three filters of length 3 over a signal of length 8
 * const layer = Conv1d(1, 3, 3)
 * const p = layer.init(stream(0))
 * const y = layer.apply(p, tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]]))
 * print(layer.label, ' kernels:', p.weight.shape, ' output:', y.shape)
 */
export function Conv1d(
  inChannels: number,
  outChannels: number,
  kernel: number,
  options: ConvLayerOptions<number> = {},
): Layer<ConvParams> {
  const { bias = true, init = heUniform(), ...geometry } = options
  const perGroup = groupChannels('Conv1d', inChannels, outChannels, geometry.groups)
  const fans = { fanIn: perGroup * kernel, fanOut: outChannels * kernel }
  return {
    kind: 'Conv1d',
    label: `Conv1d(${inChannels} → ${outChannels}, ${kernel})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [outChannels, perGroup, kernel], fans),
      ...(bias ? { bias: zerosInit()(s, [outChannels], fans) } : {}),
    }),
    apply: (p, x, ctx) => {
      const y = conv1d(x, p.weight, geometry)
      return tap(ctx, p.bias === undefined ? y : add(y, reshape(p.bias, [outChannels, 1])))
    },
  }
}

/**
 * 2-D max pooling as a layer without parameters; see `maxPool2d`.
 *
 * @param kernel The window size: one number or `[height, width]`.
 * @param options The stride (default: the kernel size) and padding.
 * @returns The layer.
 *
 * @example Halve a $4 \times 4$ image
 * const x = reshape(arange(16), [1, 1, 4, 4])
 * print('max of each 2 x 2 window:', MaxPool2d(2).apply({}, x))
 */
export const MaxPool2d = (kernel: Pair, options: PoolOptions = {}): Layer<Empty> =>
  stateless('MaxPool2d', `MaxPool2d(${String(kernel)})`, (x) => maxPool2d(x, kernel, options))
/**
 * 2-D average pooling as a layer without parameters; see `avgPool2d`.
 *
 * @param kernel The window size: one number or `[height, width]`.
 * @param options The stride (default: the kernel size) and zero padding.
 * @returns The layer.
 *
 * @example Halve a $4 \times 4$ image
 * const x = reshape(arange(16), [1, 1, 4, 4])
 * print('mean of each 2 x 2 window:', AvgPool2d(2).apply({}, x))
 */
export const AvgPool2d = (kernel: Pair, options: PoolOptions = {}): Layer<Empty> =>
  stateless('AvgPool2d', `AvgPool2d(${String(kernel)})`, (x) => avgPool2d(x, kernel, options))
/**
 * 1-D max pooling as a layer without parameters; see `maxPool1d`.
 *
 * @param kernel The window length.
 * @param options The stride (default: the kernel length) and padding.
 * @returns The layer.
 *
 * @example Maxima of pairs
 * print('pairs:', MaxPool1d(2).apply({}, tensor([[[3, 1, 4, 1, 5, 9]]])))
 */
export const MaxPool1d = (kernel: number, options: PoolOptions<number> = {}): Layer<Empty> =>
  stateless('MaxPool1d', `MaxPool1d(${kernel})`, (x) => maxPool1d(x, kernel, options))
/**
 * 1-D average pooling as a layer without parameters; see `avgPool1d`.
 *
 * @param kernel The window length.
 * @param options The stride (default: the kernel length) and zero padding.
 * @returns The layer.
 *
 * @example Means of pairs
 * print('pairs:', AvgPool1d(2).apply({}, tensor([[[3, 1, 4, 1, 5, 9]]])))
 */
export const AvgPool1d = (kernel: number, options: PoolOptions<number> = {}): Layer<Empty> =>
  stateless('AvgPool1d', `AvgPool1d(${kernel})`, (x) => avgPool1d(x, kernel, options))

/**
 * Flatten every axis from `start` on into one, as a layer without parameters.
 *
 * @param start The first axis flattened; the axes before it are kept (1 keeps the batch axis).
 * @returns The layer.
 *
 * @example Keep the batch axis, or flatten everything
 * const x = zeros([2, 3, 4])
 * print('Flatten():', Flatten().apply({}, x).shape)
 * print('Flatten(0):', Flatten(0).apply({}, x).shape)
 */
export const Flatten = (start = 1): Layer<Empty> =>
  stateless('Flatten', 'Flatten', (x) => {
    const shape = shapeOfValue(x)
    return reshape(x, [...shape.slice(0, start), -1])
  })

// ── Normalisation ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Layer normalisation (Ba, Kiros & Hinton, 2016) over the last axis:
 * $(x - \mu)/\sqrt{\sigma^2 + \varepsilon}\cdot\gamma + \beta$, with $\mu$ and the biased variance $\sigma^2$ of each
 * row. Matches `torch.nn.functional.layer_norm` over the last dimension. Differentiable.
 *
 * @param x The input, `[..., features]`; each row along the last axis is normalised on its own.
 * @param gamma The scale $\gamma$, `[features]`; left out, no scale.
 * @param beta The shift $\beta$, `[features]`; left out, no shift.
 * @param eps The $\varepsilon$ added to the variance.
 * @returns The normalised input, with the shape of `x`.
 *
 * @example Each row gets mean 0 and variance close to 1
 * const y = layerNorm(tensor([[1, 2, 3], [2, 4, 6]]))
 * print('y =', y)
 * print('row means:', mean(y, -1), ' row variances:', variance(y, -1))
 */
export function layerNorm(x: Value, gamma?: Value, beta?: Value, eps = 1e-5): Value {
  const mu = mean(x, -1, true)
  const centred = sub(x, mu)
  const variance = mean(square(centred), -1, true)
  let y = div(centred, sqrt(add(variance, eps)))
  if (gamma !== undefined) y = mul(y, gamma)
  if (beta !== undefined) y = add(y, beta)
  return y
}

/**
 * RMS normalisation (Zhang & Sennrich, 2019) over the last axis: $x/\sqrt{\operatorname{mean}(x^2) + \varepsilon}
 * \cdot\gamma$, without centring. Differentiable.
 *
 * @param x The input, `[..., features]`; each row along the last axis is scaled on its own.
 * @param gamma The scale $\gamma$, `[features]`; left out, no scale.
 * @param eps The $\varepsilon$ added to the mean square.
 * @returns The scaled input, with the shape of `x`.
 *
 * @example The row $[3, 4]$ has root mean square $\sqrt{12.5}$
 * print('rmsNorm:', rmsNorm(tensor([3, 4])))
 * print('by hand:', 3 / Math.sqrt(12.5), 4 / Math.sqrt(12.5))
 */
export function rmsNorm(x: Value, gamma?: Value, eps = 1e-6): Value {
  const y = div(x, sqrt(add(mean(square(x), -1, true), eps)))
  return gamma === undefined ? y : mul(y, gamma)
}

/** Options of `batchNorm`. */
export type BatchNormOptions = {
  /** The $\varepsilon$ added to the variance (default 1e-5). */
  eps?: number
  /** Fixed statistics (shape [C]) to normalise with, e.g. running averages at evaluation; default the batch's. */
  mean?: Value
  /** Fixed variances (shape `[C]`) to normalise with; default the batch's biased variance. */
  variance?: Value
}

/**
 * Batch normalisation (Ioffe & Szegedy, 2015) of $x$, shape `[N, C]` or `[N, C, ...spatial]`, per channel (axis 1):
 * the batch's mean and biased variance over every other axis, unless fixed statistics are given, then $\gamma$ and
 * $\beta$ (shape `[C]`). Differentiable, through the batch statistics too. Throws `ShapeError` for a rank below 2.
 *
 * @param x The input, `[N, C, ...]`.
 * @param gamma The per-channel scale $\gamma$, `[C]`; left out, no scale.
 * @param beta The per-channel shift $\beta$, `[C]`; left out, no shift.
 * @param options $\varepsilon$ and fixed statistics (`mean`, `variance`) to use in place of the batch's.
 * @returns The normalised input, with the shape of `x`.
 *
 * @example Each channel of a batch of two is normalised to $\pm 1$
 * const x = tensor([[1, 10], [3, 30]])
 * print('batch statistics:', batchNorm(x))
 * print('fixed statistics:', batchNorm(x, undefined, undefined, { mean: tensor([0, 0]), variance: tensor([1, 100]) }))
 */
export function batchNorm(x: Value, gamma?: Value, beta?: Value, options: BatchNormOptions = {}): Value {
  const { eps = 1e-5 } = options
  const shape = shapeOfValue(x)
  if (shape.length < 2) throw new ShapeError('batchNorm', 'batchNorm: needs shape [N, C, ...]')
  const axes = shape.map((_, k) => k).filter((k) => k !== 1)
  const channel = [1, shape[1], ...shape.slice(2).map(() => 1)]
  const at = (v: Value) => reshape(v, channel)
  const mu = options.mean === undefined ? mean(x, axes, true) : at(options.mean)
  const centred = sub(x, mu)
  const variance = options.variance === undefined ? mean(square(centred), axes, true) : at(options.variance)
  let y = div(centred, sqrt(add(variance, eps)))
  if (gamma !== undefined) y = mul(y, at(gamma))
  if (beta !== undefined) y = add(y, at(beta))
  return y
}

/** Parameters of a normalisation layer: the scale `gamma` ($\gamma$) and the shift `beta` ($\beta$; not RMS norm). */
export type NormParams = { gamma: Tensor; beta?: Tensor }

/**
 * Layer normalisation over a last axis of length `features`, with learned $\gamma$ (initially ones) and $\beta$
 * (initially zeros); see `layerNorm`.
 *
 * @param features The length of the last axis.
 * @param options The normalisation's settings.
 * @param options.eps The $\varepsilon$ added to the variance.
 * @returns The layer, with parameters `NormParams`.
 *
 * @example At initialisation it is `layerNorm` without scale or shift
 * const ln = LayerNorm(3)
 * const p = ln.init(stream(0))
 * print('params:', p)
 * print('apply:', ln.apply(p, tensor([1, 2, 3])))
 */
export function LayerNorm(features: number, { eps = 1e-5 }: { eps?: number } = {}): Layer<NormParams> {
  return {
    kind: 'LayerNorm',
    label: `LayerNorm(${features})`,
    init: () => ({ gamma: ones([features]), beta: zeros([features]) }),
    apply: (p, x, ctx) => tap(ctx, layerNorm(x, p.gamma, p.beta, eps)),
  }
}

/**
 * RMS normalisation over a last axis of length `features`, with learned $\gamma$ (initially ones); see `rmsNorm`.
 *
 * @param features The length of the last axis.
 * @param options The normalisation's settings.
 * @param options.eps The $\varepsilon$ added to the mean square.
 * @returns The layer, with parameters `NormParams` (no `beta`).
 *
 * @example Rows scaled to unit root mean square
 * const rn = RmsNorm(2)
 * print('apply:', rn.apply(rn.init(stream(0)), tensor([[3, 4], [1, 1]])))
 */
export function RmsNorm(features: number, { eps = 1e-6 }: { eps?: number } = {}): Layer<NormParams> {
  return {
    kind: 'RmsNorm',
    label: `RmsNorm(${features})`,
    init: () => ({ gamma: ones([features]) }),
    apply: (p, x, ctx) => tap(ctx, rmsNorm(x, p.gamma, eps)),
  }
}

/** The running statistics of `BatchNorm`, its entry in `ctx.buffers`: `mean` and `variance`, shape `[C]` each. */
export type BatchNormBuffers = { mean: Tensor; variance: Tensor }

/** Options of `BatchNorm`. */
export type BatchNormLayerOptions = {
  /** The $\varepsilon$ added to the variance (default 1e-5). */
  eps?: number
  /**
   * Weight $m$ of the new batch in the running averages, $r \leftarrow (1 - m)\,r + m\,b$ for running value $r$ and
   * batch value $b$ (torch's convention; flax's `momentum` is $1 - m$). Default 0.1.
   */
  momentum?: number
  /**
   * Keep running statistics (default true): training mode normalises with the batch's statistics and writes updated
   * running averages to `ctx.bufferUpdates`; evaluation normalises with `ctx.buffers` (mean 0 and variance 1 before
   * any training). With false, every mode normalises with the batch's statistics.
   */
  trackRunningStats?: boolean
}

/**
 * Batch normalisation of `channels` channels with learned $\gamma$ and $\beta$ and running statistics, as
 * `torch.nn.BatchNorm1d` and `BatchNorm2d`: the running mean and the running unbiased variance are exponential
 * averages of the batches seen in training, and evaluation normalises with them. The statistics live in
 * `ctx.buffers` under the layer's path (`''` at the top level); training normalises with the batch's biased variance
 * and writes the new running averages to `ctx.bufferUpdates`. Throws `ShapeError` in training for a rank below 2.
 *
 * @param channels The number of channels $C$ (axis 1 of the input).
 * @param options $\varepsilon$, the momentum of the running averages, and whether to keep them.
 * @returns The layer, with parameters `NormParams` (initially $\gamma = 1$, $\beta = 0$) and buffers
 *   `BatchNormBuffers`.
 *
 * @example A training step updates the running statistics, which evaluation then uses
 * const bn = BatchNorm(2)
 * const p = bn.init(stream(0))
 * const x = tensor([[1, 10], [3, 30]])
 * const updates = {}
 * print('train:', bn.apply(p, x, { train: true, bufferUpdates: updates }))
 * print('running statistics:', updates[''])
 * print('evaluate:', bn.apply(p, x, { buffers: updates }))
 */
export function BatchNorm(channels: number, options: BatchNormLayerOptions = {}): Layer<NormParams> {
  const { eps = 1e-5, momentum = 0.1, trackRunningStats = true } = options
  const initial = (): BatchNormBuffers => ({ mean: zeros([channels]), variance: ones([channels]) })
  return {
    kind: 'BatchNorm',
    label: `BatchNorm(${channels})`,
    init: () => ({ gamma: ones([channels]), beta: zeros([channels]) }),
    apply: (p, x, ctx) => {
      if (!trackRunningStats) return tap(ctx, batchNorm(x, p.gamma, p.beta, { eps }))
      const key = ctx?.path ?? ''
      const running = (ctx?.buffers?.[key] as BatchNormBuffers | undefined) ?? initial()
      if (!ctx?.train) return tap(ctx, batchNorm(x, p.gamma, p.beta, { eps, ...running }))
      const shape = shapeOfValue(x)
      if (shape.length < 2) throw new ShapeError('BatchNorm', 'BatchNorm: needs shape [N, C, ...]')
      const axes = shape.map((_, k) => k).filter((k) => k !== 1)
      const mu = mean(x, axes)
      const variance = mean(square(sub(x, reshape(mu, [1, channels, ...shape.slice(2).map(() => 1)]))), axes)
      if (ctx.bufferUpdates) {
        // The running variance is unbiased (torch), the normalising one biased; both from the primal values.
        const n = shape.reduce((a, b) => a * b, 1) / channels
        const blend = (old: Tensor, batch: Value, scale: number) =>
          unwrap(add(mul(1 - momentum, old), mul(momentum * scale, unwrap(batch)))) as Tensor
        const next: BatchNormBuffers = {
          mean: blend(running.mean, mu, 1),
          variance: blend(running.variance, variance, n > 1 ? n / (n - 1) : 1),
        }
        ctx.bufferUpdates[key] = next
      }
      return tap(ctx, batchNorm(x, p.gamma, p.beta, { eps, mean: mu, variance }))
    },
  }
}

// ── Dropout and activations ──────────────────────────────────────────────────────────────────────────────────────────

/**
 * Inverted dropout (Srivastava et al., 2014): each element is zeroed with probability $p$ and the survivors scaled by
 * $1/(1 - p)$, so the expectation is unchanged. The mask is drawn from `s`; it is a constant, so the gradient flows
 * through kept elements only. Throws `DomainError` unless $0 \le p < 1$.
 *
 * @param s The stream the mask is drawn from.
 * @param x The input, a number or tensor.
 * @param p The probability $p$ of zeroing an element; 0 returns `x` unchanged.
 * @returns `x` with dropped elements zeroed and the rest scaled, with the shape of `x`.
 *
 * @example Survivors are doubled at $p = 0.5$, so the mean stays near 1
 * print('ten ones:', dropout(stream(0), ones([10]), 0.5))
 * print('mean of 10000:', mean(dropout(stream(1), ones([10000]), 0.5)))
 */
export function dropout(s: Stream, x: Value, p: number): Value {
  if (!(p >= 0 && p < 1)) throw new DomainError('dropout', `dropout: p = ${p} is not in [0, 1)`)
  if (p === 0) return x
  const mask: Tensor = bernoulli(s, 1 - p, { shape: shapeOfValue(x) })
  return mul(x, div(mask, 1 - p))
}

/**
 * Dropout as a layer without parameters: active only when `ctx.train`, drawing from the child `('dropout', path)` of
 * `ctx.stream`, so each dropout layer of a network draws its own mask. Throws `DomainError` when training without
 * `ctx.stream`.
 *
 * @param p The probability of zeroing an element.
 * @returns The layer; in evaluation it passes its input on unchanged.
 *
 * @example Dropout acts in training only
 * const d = Dropout(0.5)
 * const x = ones([8])
 * print('evaluate:', d.apply({}, x))
 * print('train:', d.apply({}, x, { train: true, stream: stream(0) }))
 */
export function Dropout(p: number): Layer<Empty> {
  return {
    kind: 'Dropout',
    label: `Dropout(${p})`,
    init: () => EMPTY,
    apply: (_params, x, ctx) => {
      if (!ctx?.train || p === 0) return tap(ctx, x)
      if (!ctx.stream) throw new DomainError('Dropout', 'Dropout: training needs ctx.stream')
      return tap(ctx, dropout(child(ctx.stream, 'dropout', ctx.path ?? ''), x, p))
    },
  }
}

/**
 * An activation function as a layer without parameters.
 *
 * @param activation A name in `activationFunctions` (also the layer's label), or a function of one value.
 * @returns The layer.
 *
 * @example ReLU as a layer
 * const layer = ActivationLayer('relu')
 * print(layer.label, layer.apply({}, tensor([-1, 0, 2])))
 */
export function ActivationLayer(activation: Activation): Layer<Empty> {
  const f = activationFn(activation)
  return stateless('Activation', typeof activation === 'string' ? activation : 'activation', (x) => f(x))
}

// ── Containers ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Layers applied in order; parameters are an array, one entry per layer, and paths are the layer indices. Layer `i` is
 * initialised from the child `i` of the stream.
 *
 * @param layers The layers, each taking the output of the one before.
 * @returns The layer, with an array of the layers' parameters.
 *
 * @example A small network on two opposite inputs, with the output of each layer recorded by a tap
 * const net = Sequential(Linear(2, 3), ActivationLayer('relu'), Linear(3, 1))
 * const p = net.init(stream(0))
 * const seen = {}
 * const y = net.apply(p, tensor([[1, 2], [-1, -2]]), { tap: (path, v) => ((seen[path] = v), v) })
 * print(net.label)
 * print('y =', y)
 * print('recorded:', seen)
 */
export function Sequential(...layers: Layer<Params>[]): Layer<Params[]> {
  const ls = layers
  return {
    kind: 'Sequential',
    label: ls.map((l) => l.label).join(' → '),
    init: (s) => ls.map((l, i) => l.init(child(s, i))),
    apply: (params, x, ctx) => ls.reduce((h, l, i) => l.apply(params[i], h, childContext(ctx, i)), x),
  }
}

/** Options of `Mlp`. */
export type MlpOptions = {
  /** Hidden activation (default ReLU). */
  activation?: Activation
  /** Activation after the last layer (default none: logits or values). */
  outputActivation?: Activation
  /** Weight initialiser (default He uniform, suited to ReLU; pass `xavierUniform()` for tanh or sigmoid). */
  init?: Initialiser
  /** Dropout probability after each hidden activation (default 0). */
  dropout?: number
}

/**
 * A multilayer perceptron with layer sizes `sizes` = `[in, ...hidden, out]`: Linear layers with the activation between
 * them, a `Sequential`. Its parameters are an array alternating Linear parameters and `{}` for each activation (and
 * dropout, and an output activation when there is one). Throws `DomainError` for fewer than two sizes.
 *
 * @param sizes The widths: the input, each hidden layer, then the output.
 * @param options The hidden and output activations, the weight initialiser and the dropout probability.
 * @returns The layer, with an array of parameters.
 *
 * @example Two inputs, one hidden layer of 8, one output, on a batch of three
 * const mlp = Mlp([2, 8, 1])
 * const p = mlp.init(stream(0))
 * print(mlp.label, ' parameter entries:', p.length)
 * print('output:', mlp.apply(p, tensor([[0, 0], [1, 0], [0, 1]])))
 */
export function Mlp(sizes: readonly number[], options: MlpOptions = {}): Layer<Params[]> {
  const { activation = 'relu', outputActivation = 'identity', init = heUniform(), dropout: p = 0 } = options
  if (sizes.length < 2) throw new DomainError('Mlp', 'Mlp: needs at least an input and an output size')
  const layers: Layer<Params>[] = []
  for (let k = 0; k + 1 < sizes.length; k++) {
    layers.push(Linear(sizes[k], sizes[k + 1], { init }))
    const last = k + 2 === sizes.length
    if (!last) {
      layers.push(ActivationLayer(activation))
      if (p > 0) layers.push(Dropout(p))
    } else if (outputActivation !== 'identity') layers.push(ActivationLayer(outputActivation))
  }
  const seq = Sequential(...layers)
  return { ...seq, kind: 'Mlp', label: `Mlp(${sizes.join(' → ')})` }
}

/**
 * A residual block $x + f(x)$ (He et al., 2016); $f$ must keep the shape. The inner layer's path is `branch` below the
 * block's.
 *
 * @param inner The layer $f$; the block has its parameters.
 * @returns The layer.
 *
 * @example Around a ReLU, positive entries double and negative ones stay
 * const block = Residual(ActivationLayer('relu'))
 * print(block.label, block.apply({}, tensor([-1, 0, 2])))
 */
export function Residual<P extends Params>(inner: Layer<P>): Layer<P> {
  return {
    kind: 'Residual',
    label: `Residual(${inner.label})`,
    init: (s) => inner.init(s),
    apply: (p, x, ctx) => tap(ctx, add(x, inner.apply(p, x, childContext(ctx, 'branch')))),
  }
}
