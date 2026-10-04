/**
 * A mixture-of-experts layer: a router (a linear map to N logits) chooses experts per token with `route`, and the
 * output is Σᵢ wₜᵢ Eᵢ(xₜ) over the experts with non-zero weight. The experts are any layers with the same input and
 * output widths (a `Linear` gives the classic mixture of linear experts, an `Mlp` or `FeedForward` the sparse layer of
 * a transformer).
 *
 * Every expert is evaluated on every token and the combine weights zero the unchosen ones. That is the dense
 * simulation of sparse dispatch: the output, the dropping and every gradient are those of a sparse implementation, and
 * at the sizes of a notebook the gather and scatter would cost more than they save.
 */

import { child } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { add, fromData, matmul, mul, reshape, shapeOfValue, type Value } from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { zerosInit } from 'aifn-compute/nn/init'
import { childContext, Linear, linear, tap, type Context, type Layer, type LinearParams } from 'aifn-compute/nn/layers'
import { route, type Routing, type RoutingOptions } from './routing'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Parameters of `MixtureOfExperts`: the router, the noise map of a noisy gate, and each expert's parameters. */
export type MixtureOfExpertsParams = { router: LinearParams; noise?: LinearParams; experts: Params[] }

/** Options of `MixtureOfExperts`: the routing options other than the noise, which the layer makes itself. */
export type MixtureOfExpertsOptions = Omit<RoutingOptions, 'noiseScale' | 'stream'> & {
  /** Initialise the router's weights to zero (default false: LeCun uniform), so every expert starts equally likely. */
  zeroRouter?: boolean
}

/** What a forward pass computes besides the output. */
export type MixtureOfExpertsResult = {
  /** Σᵢ wₜᵢ Eᵢ(xₜ), shaped like the input with the experts' output width last. */
  readonly output: Value
  readonly routing: Routing
  /** Each expert's output on every token, [T, out] (flattened tokens). */
  readonly expertOutputs: readonly Value[]
}

/** A mixture-of-experts layer with a `forward` that also returns the routing (for the auxiliary losses). */
export interface MixtureOfExpertsLayer extends Layer<MixtureOfExpertsParams> {
  readonly experts: readonly Layer[]
  readonly options: MixtureOfExpertsOptions
  forward(params: MixtureOfExpertsParams, x: Value, ctx?: Context): MixtureOfExpertsResult
}

/**
 * A mixture of the given experts over inputs [..., inFeatures]: a `Linear(inFeatures → N)` router and, for the
 * `noisy-top-k` gate, a second linear map whose softplus is the noise scale (Shazeer et al., 2017, eq. 4), drawn only in
 * training mode from `ctx.stream`. `apply` returns the output; `forward` adds the routing and each expert's output.
 */
export function MixtureOfExperts(
  inFeatures: Size,
  experts: readonly Layer[],
  options: MixtureOfExpertsOptions = {},
): MixtureOfExpertsLayer {
  const N = experts.length
  if (N < 1) throw new DomainError('MixtureOfExperts', 'MixtureOfExperts: needs at least one expert')
  const { zeroRouter = false, ...routing } = options
  const noisy = routing.gate === 'noisy-top-k'
  const router = Linear(inFeatures, N, zeroRouter ? { init: zerosInit() } : {})
  const noise = Linear(inFeatures, N, { init: zerosInit() })
  const forward = (p: MixtureOfExpertsParams, x: Value, ctx?: Context): MixtureOfExpertsResult => {
    const shape = shapeOfValue(x)
    const flat = shape.length === 2 ? x : reshape(x, [-1, inFeatures])
    const logits = linear(flat, p.router.weight, p.router.bias)
    const draw = noisy && ctx?.train && ctx.stream ? child(ctx.stream, 'router-noise', ctx.path ?? '') : undefined
    const noiseScale = noisy && p.noise ? softplus(linear(flat, p.noise.weight, p.noise.bias)) : undefined
    const r = route(logits, { ...routing, noiseScale, stream: draw })
    const outputs = experts.map((e, i) => e.apply(p.experts[i], flat, childContext(ctx, `experts.${i}`)))
    let y: Value | null = null
    for (let i = 0; i < N; i++) {
      // Column i of the combine weights, [T, 1], as a product with a constant one-hot vector.
      const pick = fromData(
        Float64Array.from({ length: N }, (_, j) => (j === i ? 1 : 0)),
        [N, 1],
      )
      const term = mul(matmul(r.combine, pick), outputs[i])
      y = y === null ? term : add(y, term)
    }
    const width = shapeOfValue(outputs[0]).at(-1)!
    const output = shape.length === 2 ? y! : reshape(y!, [...shape.slice(0, -1), width])
    return { output: tap(ctx, output), routing: r, expertOutputs: outputs }
  }
  return {
    kind: 'MixtureOfExperts',
    label: `MixtureOfExperts(${N} × ${experts[0].label}, ${routing.gate ?? 'top-k'})`,
    experts,
    options,
    init: (s) => ({
      router: router.init(child(s, 'router')),
      ...(noisy ? { noise: noise.init(child(s, 'noise')) } : {}),
      experts: experts.map((e, i) => e.init(child(s, 'experts', i))),
    }),
    apply: (p, x, ctx) => forward(p, x, ctx).output,
    forward,
  }
}
