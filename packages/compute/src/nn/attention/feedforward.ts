/**
 * The position-wise feed-forward network of a transformer block (Vaswani et al., 2017, §3.3), plain or gated. A gated
 * layer multiplies an activated projection by a linear one, down(act(x·W_gate) ⊙ x·W_up) (Shazeer, 2020, "GLU
 * variants improve transformer"): SwiGLU with SiLU, GeGLU with GELU, ReGLU with ReLU. Gating adds a third matrix, so
 * the hidden width is usually cut to about ⅔ of the plain 4·d to keep the parameter count.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { mul, type Value } from 'aifn-compute/foundation/tensor'
import { activationFn, type Activation } from 'aifn-compute/nn/functional'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Linear, linear, tap, type Context, type Layer, type LinearParams } from 'aifn-compute/nn/layers'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The kinds of feed-forward layer: a plain two-layer network or a gated one. */
export type FeedForwardKind = 'mlp' | 'swiglu' | 'geglu' | 'reglu'

/** Parameters of a feed-forward layer: `up` [d, hidden], `down` [hidden, d] and for gated kinds `gate` [d, hidden]. */
export type FeedForwardParams = { up: LinearParams; gate?: LinearParams; down: LinearParams }

/** Options of `feedForward`. */
export type FeedForwardOptions = {
  /** Default `mlp`. */
  kind?: FeedForwardKind
  /** The activation of a plain network (default GELU); gated kinds fix theirs. */
  activation?: Activation
}

const gateActivation = { swiglu: 'silu', geglu: 'gelu', reglu: 'relu' } as const

/**
 * The feed-forward layer of x [..., d]: down(act(up(x))) for `mlp`, down(act(gate(x)) ⊙ up(x)) for the gated kinds.
 * The hidden activations [..., hidden] (what `down` reads) are tapped at `<path>.hidden`.
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

/** The usual hidden width: 4·d for a plain network, ⌈(8/3)·d⌉ for a gated one (the same parameter count). */
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

/** A feed-forward layer over [..., d] as a layer. */
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
