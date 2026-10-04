/**
 * Layers: pairs of an initialiser and a pure forward function, `layer.init(stream)` → parameters (a pytree of tensors)
 * and `layer.apply(params, x, ctx)` → output. There is no hidden state: parameters are passed in, so `grad` of any
 * function of them differentiates through the layer, and the same parameters can be evaluated, perturbed or traced.
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

/** A layer: an initialiser and a pure forward function of parameters `P`. */
export interface Layer<P extends Params = Params> {
  /** The layer type, e.g. `Linear`. */
  readonly kind: string
  /** A short description, e.g. `Linear(2 → 16)`. */
  readonly label: string
  /** Fresh parameters drawn from `s` (layers without parameters return `{}`). */
  init(s: Stream): P
  /** The output for input x; pure. */
  apply(params: P, x: Value, ctx?: Context): Value
}

/** Pass a layer's output through the context's tap. */
export function tap(ctx: Context | undefined, value: Value, suffix?: string): Value {
  if (!ctx?.tap) return value
  const base = ctx.path ?? ''
  const path = suffix ? (base ? `${base}.${suffix}` : suffix) : base || 'output'
  return ctx.tap(path, value)
}

/** The context for a child layer at `name` below the current path. */
export function childContext(ctx: Context | undefined, name: string | number): Context | undefined {
  if (!ctx) return undefined
  const base = ctx.path ?? ''
  return { ...ctx, path: base ? `${base}.${name}` : String(name) }
}

type Empty = Record<string, never>
const EMPTY: Empty = {}

/** A layer without parameters from a function of its input. */
function stateless(kind: string, label: string, f: (x: Value, ctx?: Context) => Value): Layer<Empty> {
  return { kind, label, init: () => EMPTY, apply: (_p, x, ctx) => tap(ctx, f(x, ctx)) }
}

// ── Dense and embedding ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The affine map x·W + b for x of shape [..., in] and W of shape [in, out] (the transpose of PyTorch's storage), giving
 * [..., out]. `bias` may be omitted.
 */
export function linear(x: Value, weight: Value, bias?: Value): Value {
  const y = matmul(x, weight)
  return bias === undefined ? y : add(y, bias)
}

/** Parameters of `Linear`: weight [in, out] and bias [out] (absent with `bias: false`). */
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

/** A dense layer x·W + b from `inFeatures` to `outFeatures`. */
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

/** Parameters of `Embedding`: the table [vocabulary, dimension]. */
export type EmbeddingParams = { weight: Tensor }

/**
 * A lookup table of `vocabulary` vectors of length `dimension`: `apply(params, ids)` maps integer ids of any shape to
 * [...ids shape, dimension]. The gradient adds up over repeated ids. Initialised N(0, 1) by default, as PyTorch.
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

/** Parameters of a convolution: kernels [O, C, ...kernel] and bias [O]. */
export type ConvParams = { weight: Tensor; bias?: Tensor }

/** Options of `Conv1d` and `Conv2d`. */
export type ConvLayerOptions<P> = ConvOptions<P> & {
  bias?: boolean
  /** Kernel initialiser (default He uniform, for ReLU networks). */
  init?: Initialiser
}

/**
 * A 2-D convolution layer from `inChannels` to `outChannels` with a `kernel` (one size or [KH, KW]); input [N, C, H, W]
 * or [C, H, W]. See `conv2d` for the geometry.
 */
export function Conv2d(
  inChannels: number,
  outChannels: number,
  kernel: Pair,
  options: ConvLayerOptions<Pair> = {},
): Layer<ConvParams> {
  const [kh, kw] = typeof kernel === 'number' ? [kernel, kernel] : kernel
  const { bias = true, init = heUniform(), ...geometry } = options
  const fans = { fanIn: inChannels * kh * kw, fanOut: outChannels * kh * kw }
  return {
    kind: 'Conv2d',
    label: `Conv2d(${inChannels} → ${outChannels}, ${kh}×${kw})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [outChannels, inChannels, kh, kw], fans),
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

/** A 1-D convolution layer; input [N, C, L] or [C, L]. See `conv1d`. */
export function Conv1d(
  inChannels: number,
  outChannels: number,
  kernel: number,
  options: ConvLayerOptions<number> = {},
): Layer<ConvParams> {
  const { bias = true, init = heUniform(), ...geometry } = options
  const fans = { fanIn: inChannels * kernel, fanOut: outChannels * kernel }
  return {
    kind: 'Conv1d',
    label: `Conv1d(${inChannels} → ${outChannels}, ${kernel})`,
    init: (s) => ({
      weight: init(child(s, 'weight'), [outChannels, inChannels, kernel], fans),
      ...(bias ? { bias: zerosInit()(s, [outChannels], fans) } : {}),
    }),
    apply: (p, x, ctx) => {
      const y = conv1d(x, p.weight, geometry)
      return tap(ctx, p.bias === undefined ? y : add(y, reshape(p.bias, [outChannels, 1])))
    },
  }
}

/** 2-D max pooling as a layer. */
export const MaxPool2d = (kernel: Pair, options: PoolOptions = {}): Layer<Empty> =>
  stateless('MaxPool2d', `MaxPool2d(${String(kernel)})`, (x) => maxPool2d(x, kernel, options))
/** 2-D average pooling as a layer. */
export const AvgPool2d = (kernel: Pair, options: PoolOptions = {}): Layer<Empty> =>
  stateless('AvgPool2d', `AvgPool2d(${String(kernel)})`, (x) => avgPool2d(x, kernel, options))
/** 1-D max pooling as a layer. */
export const MaxPool1d = (kernel: number, options: PoolOptions<number> = {}): Layer<Empty> =>
  stateless('MaxPool1d', `MaxPool1d(${kernel})`, (x) => maxPool1d(x, kernel, options))
/** 1-D average pooling as a layer. */
export const AvgPool1d = (kernel: number, options: PoolOptions<number> = {}): Layer<Empty> =>
  stateless('AvgPool1d', `AvgPool1d(${kernel})`, (x) => avgPool1d(x, kernel, options))

/** Flatten every axis from `start` on into one (default 1: keep the batch axis). */
export const Flatten = (start = 1): Layer<Empty> =>
  stateless('Flatten', 'Flatten', (x) => {
    const shape = shapeOfValue(x)
    return reshape(x, [...shape.slice(0, start), -1])
  })

// ── Normalisation ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Layer normalisation (Ba, Kiros & Hinton, 2016) over the last axis: (x − μ)/√(σ² + ε)·γ + β, with μ and the biased
 * variance σ² of each row. Matches `torch.nn.functional.layer_norm` over the last dimension.
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

/** RMS normalisation (Zhang & Sennrich, 2019) over the last axis: x/√(mean(x²) + ε)·γ. */
export function rmsNorm(x: Value, gamma?: Value, eps = 1e-6): Value {
  const y = div(x, sqrt(add(mean(square(x), -1, true), eps)))
  return gamma === undefined ? y : mul(y, gamma)
}

/** Options of `batchNorm`. */
export type BatchNormOptions = {
  eps?: number
  /** Fixed statistics (shape [C]) to normalise with, e.g. running averages at evaluation; default the batch's. */
  mean?: Value
  variance?: Value
}

/**
 * Batch normalisation (Ioffe & Szegedy, 2015) of x, shape [N, C] or [N, C, ...spatial], per channel (axis 1): the
 * batch's mean and biased variance over every other axis, unless fixed statistics are given, then γ and β (shape [C]).
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

/** Parameters of a normalisation layer: scale γ and (except RMS norm) shift β. */
export type NormParams = { gamma: Tensor; beta?: Tensor }

/** Layer normalisation over a last axis of length `features`, with learned γ (ones) and β (zeros). */
export function LayerNorm(features: number, { eps = 1e-5 }: { eps?: number } = {}): Layer<NormParams> {
  return {
    kind: 'LayerNorm',
    label: `LayerNorm(${features})`,
    init: () => ({ gamma: ones([features]), beta: zeros([features]) }),
    apply: (p, x, ctx) => tap(ctx, layerNorm(x, p.gamma, p.beta, eps)),
  }
}

/** RMS normalisation over a last axis of length `features`, with learned γ (ones). */
export function RmsNorm(features: number, { eps = 1e-6 }: { eps?: number } = {}): Layer<NormParams> {
  return {
    kind: 'RmsNorm',
    label: `RmsNorm(${features})`,
    init: () => ({ gamma: ones([features]) }),
    apply: (p, x, ctx) => tap(ctx, rmsNorm(x, p.gamma, eps)),
  }
}

/** The running statistics of `BatchNorm` (shape [C] each), its entry in `ctx.buffers`. */
export type BatchNormBuffers = { mean: Tensor; variance: Tensor }

/** Options of `BatchNorm`. */
export type BatchNormLayerOptions = {
  eps?: number
  /**
   * Weight of the new batch in the running averages, running ← (1 − m)·running + m·batch (torch's convention; flax's
   * `momentum` is 1 − m). Default 0.1.
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
 * Batch normalisation of `channels` channels with learned γ and β and running statistics, as `torch.nn.BatchNorm1d`
 * and `BatchNorm2d`: the running mean and the running unbiased variance are exponential averages of the batches seen
 * in training, and evaluation normalises with them.
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
 * Inverted dropout (Srivastava et al., 2014): each element is zeroed with probability p and the survivors scaled by
 * 1/(1 − p), so the expectation is unchanged. The mask is drawn from `s`; it is a constant, so the gradient flows
 * through kept elements only.
 */
export function dropout(s: Stream, x: Value, p: number): Value {
  if (!(p >= 0 && p < 1)) throw new DomainError('dropout', `dropout: p = ${p} is not in [0, 1)`)
  if (p === 0) return x
  const mask: Tensor = bernoulli(s, 1 - p, { shape: shapeOfValue(x) })
  return mul(x, div(mask, 1 - p))
}

/** Dropout as a layer: active only when `ctx.train`, drawing from `ctx.stream.child(path)`. */
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

/** An activation function as a layer. */
export function ActivationLayer(activation: Activation): Layer<Empty> {
  const f = activationFn(activation)
  return stateless('Activation', typeof activation === 'string' ? activation : 'activation', (x) => f(x))
}

// ── Containers ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Layers applied in order; parameters are an array, one entry per layer, and paths are the layer indices. */
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
 * A multilayer perceptron with layer sizes `sizes` = [in, hidden…, out]: Linear layers with the activation between
 * them. Its parameters are an array alternating Linear parameters and `{}` for each activation (and dropout).
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

/** A residual block x + f(x) (He et al., 2016); `f` must keep the shape. */
export function Residual<P extends Params>(inner: Layer<P>): Layer<P> {
  return {
    kind: 'Residual',
    label: `Residual(${inner.label})`,
    init: (s) => inner.init(s),
    apply: (p, x, ctx) => tap(ctx, add(x, inner.apply(p, x, childContext(ctx, 'branch')))),
  }
}
