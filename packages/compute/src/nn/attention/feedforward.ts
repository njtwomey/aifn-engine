/**
 * The position-wise feed-forward network of a transformer block (Vaswani et al., 2017, §3.3), plain or gated, applied
 * to each token's vector on its own. A gated layer multiplies an activated projection by a linear one,
 * $(\phi(\xvec\Wmat_{\mathrm{gate}}) \odot \xvec\Wmat_{\mathrm{up}})\Wmat_{\mathrm{down}}$ (Shazeer, 2020, "GLU
 * variants improve transformer"): SwiGLU with $\phi$ the SiLU, GeGLU with GELU, ReGLU with ReLU. Gating adds a third
 * matrix, so the hidden width is usually cut to $2/3$ of the plain $4d$ to keep the parameter count.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { mul, type Value } from 'aifn-compute/foundation/tensor'
import { activationFn, type Activation } from 'aifn-compute/nn/functional'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Linear, linear, tap, type Context, type Layer, type LinearParams } from 'aifn-compute/nn/layers'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The kinds of feed-forward layer: `mlp`, a plain two-layer network, or a gated one, `swiglu` (SiLU gate), `geglu`
 * (GELU gate) or `reglu` (ReLU gate).
 */
export type FeedForwardKind = 'mlp' | 'swiglu' | 'geglu' | 'reglu'

/**
 * Parameters of a feed-forward layer: `up` `[d, hidden]`, `down` `[hidden, d]` and, for gated kinds, `gate`
 * `[d, hidden]`.
 */
export type FeedForwardParams = { up: LinearParams; gate?: LinearParams; down: LinearParams }

/** Options of `feedForward`. */
export type FeedForwardOptions = {
  /** Plain or gated, and which gate (default `mlp`). */
  kind?: FeedForwardKind
  /** The activation of a plain network (default GELU); gated kinds fix theirs. */
  activation?: Activation
}

/** The activation of each gated kind's gate. */
const gateActivation = { swiglu: 'silu', geglu: 'gelu', reglu: 'relu' } as const

/**
 * The feed-forward layer of $\xvec$ (`[..., d]`): $\mathrm{down}(\phi(\mathrm{up}(\xvec)))$ for `mlp`,
 * $\mathrm{down}(\phi(\mathrm{gate}(\xvec)) \odot \mathrm{up}(\xvec))$ for the gated kinds, each of up, gate and down
 * an affine map. The hidden activations `[..., hidden]` (what `down` reads) are tapped at `<path>.hidden`. Throws
 * `DomainError` for a gated kind without `gate` parameters.
 *
 * @param params The `up`, `down` and (gated kinds) `gate` projections.
 * @param x The input `[..., d]`; every leading axis is a token or batch axis.
 * @param options The kind of layer and, for `mlp`, its activation.
 * @param ctx The forward-pass context, read only for its tap: the hidden activations are passed through it.
 * @returns The output `[..., d]`.
 *
 * @example By hand: with identity up and down, ReGLU zeroes the coordinate whose gate is negative
 * const params = {
 *   up: { weight: tensor([[1, 0], [0, 1]]) },
 *   gate: { weight: tensor([[1, 0], [0, -1]]) },
 *   down: { weight: tensor([[1, 0], [0, 1]]) },
 * }
 * const x = tensor([[2, 3]])
 * print('mlp, relu:', feedForward(params, x, { activation: 'relu' }))
 * print('reglu, relu([2, -3]) * [2, 3]:', feedForward(params, x, { kind: 'reglu' }))
 */
export function feedForward(
  params: FeedForwardParams,
  x: Value,
  options: FeedForwardOptions = {},
  ctx?: Context,
): Value {
  const { kind = 'mlp', activation = 'gelu' } = options
  const up = linear(x, params.up.weight, params.up.bias)
  const hidden = (h: Value) => tap(ctx, h, 'hidden')
  if (kind === 'mlp') return linear(hidden(activationFn(activation)(up)), params.down.weight, params.down.bias)
  if (!params.gate) throw new DomainError('feedForward', `feedForward: a ${kind} layer needs gate parameters`)
  const gate = activationFn(gateActivation[kind])(linear(x, params.gate.weight, params.gate.bias))
  return linear(hidden(mul(gate, up)), params.down.weight, params.down.bias)
}

/**
 * The usual hidden width: $4d$ for a plain network, $\lceil 8d/3 \rceil$ for a gated one, so both have about $8d^2$
 * weights.
 *
 * @param dModel The model width $d$.
 * @param kind The kind of layer: `mlp`, or any gated kind.
 * @returns The hidden width.
 *
 * @example Plain and gated widths for d = 4 and d = 768
 * print('d = 4:', feedForwardWidth(4), feedForwardWidth(4, 'swiglu'))
 * print('d = 768:', feedForwardWidth(768), feedForwardWidth(768, 'swiglu'))
 */
export function feedForwardWidth(dModel: Size, kind: FeedForwardKind = 'mlp'): Size {
  return kind === 'mlp' ? 4 * dModel : Math.ceil((8 * dModel) / 3)
}

/** Options of the `FeedForward` layer. */
export type FeedForwardLayerOptions = FeedForwardOptions & {
  /** Hidden width (default `feedForwardWidth(d, kind)`). */
  hidden?: Size
  /** Biases on the projections (default true for `mlp`, false for gated kinds, as in LLaMA). */
  bias?: boolean
}

/**
 * The feed-forward network as a layer over `[..., d]`: `feedForward` with Xavier-uniform projections. Its output is
 * tapped at the layer's path and the hidden activations at `<path>.hidden`.
 *
 * @param dModel The model width $d$, the width of input and output.
 * @param options The kind, activation, hidden width and whether the projections have biases.
 * @returns The layer: `init` draws `up`, `down` and (gated kinds) `gate`; `apply` runs `feedForward`.
 *
 * @example A SwiGLU layer of width 4 on three tokens, and the paths it taps
 * const layer = FeedForward(4, { kind: 'swiglu' })
 * const params = layer.init(stream(0))
 * print(layer.label)
 * print('up, gate, down:', shapeOf(params.up.weight), shapeOf(params.gate.weight), shapeOf(params.down.weight))
 * const tapped = []
 * const y = layer.apply(params, normals(stream(1), [3, 4]), { tap: (path, v) => (tapped.push(path), v) })
 * print('output:', shapeOf(y))
 * print('tapped:', tapped)
 */
export function FeedForward(dModel: Size, options: FeedForwardLayerOptions = {}): Layer<FeedForwardParams> {
  const { kind = 'mlp' } = options
  const hidden = options.hidden ?? feedForwardWidth(dModel, kind)
  const bias = options.bias ?? kind === 'mlp'
  const up = Linear(dModel, hidden, { init: xavierUniform(), bias })
  const down = Linear(hidden, dModel, { init: xavierUniform(), bias })
  return {
    kind: 'FeedForward',
    label: `FeedForward(${dModel} → ${hidden}, ${kind})`,
    init: (s: Stream) => ({
      up: up.init(child(s, 'up')),
      ...(kind === 'mlp' ? {} : { gate: up.init(child(s, 'gate')) }),
      down: down.init(child(s, 'down')),
    }),
    apply: (p, x, ctx) => tap(ctx, feedForward(p, x, options, ctx)),
  }
}
