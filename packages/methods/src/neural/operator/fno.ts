/**
 * A one-dimensional Fourier neural operator (Li et al., 2021, "Fourier Neural Operator for Parametric Partial
 * Differential Equations", §4): the input channels and the grid coordinate are lifted pointwise to `width` channels by
 * a `Linear` layer, `layers` Fourier layers $\vvec \leftarrow \sigma(\mathcal{K}\vvec + \Wmat\vvec + \bvec)$
 * follow ($\mathcal{K}$ the spectral convolution `SpectralConv1d` of `aifn-compute/nn/layers`, $\Wmat$ a pointwise
 * linear map; no activation after the last), and a pointwise two-layer projection maps to the output channels. The
 * spectral weights are indexed by frequency, so one network applies at any grid size.
 *
 * The grid coordinate is an extra input channel, as in the paper. The paper uses four layers with ReLU and batch
 * normalisation; here the activation is an option (GELU by default, as in the reference implementation) and there is
 * no batch normalisation. An Adam step at batch 64 takes 30 to 55 ms.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  broadcastTo,
  concat,
  fromData,
  shapeOfValue,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Params } from 'aifn-compute/foundation/pytree'
import { activationFunctions } from 'aifn-compute/nn/functional'
import {
  Linear,
  Mlp,
  SpectralConv1d,
  type Layer,
  type LinearParams,
  type SpectralConvParams,
} from 'aifn-compute/nn/layers'

/** Options of `FourierNeuralOperator1d`. */
export type FnoOptions = {
  /** The input channels of the function $a(x)$ (default 1). */
  inChannels?: number
  /** The output channels of $u(x)$ (default 1). */
  outChannels?: number
  /** The channels inside the Fourier layers (default 16). */
  width?: number
  /** The frequencies kept in each spectral convolution (default 12). */
  modes?: number
  /** The number of Fourier layers (default 4). */
  layers?: number
  /** The hidden width of the projection (default 4 times `width`). */
  projection?: number
  /** The activation $\sigma$ (default `'gelu'`). */
  activation?: 'gelu' | 'relu' | 'tanh'
}

/** Parameters of `FourierNeuralOperator1d`. */
export type FnoParams = {
  /** The pointwise lift of $(a(x), x)$ to `width` channels. */
  lift: LinearParams
  /** Each Fourier layer's spectral convolution and pointwise linear map. */
  blocks: { spectral: SpectralConvParams; pointwise: LinearParams }[]
  /** The two-layer pointwise projection. */
  project: Params[]
}

/**
 * The grid coordinate $x_j = j/n$ as a channel.
 *
 * @param batch The batch size.
 * @param n The number of grid points.
 * @returns The coordinate, $[\text{batch}, n, 1]$.
 */
function gridChannel(batch: number, n: number): Tensor {
  return broadcastTo(
    fromData(
      Float64Array.from({ length: n }, (_, j) => j / n),
      [1, n, 1],
    ),
    [batch, n, 1],
  ) as Tensor
}

/**
 * A one-dimensional Fourier neural operator: a pointwise lift of $(a(x), x)$ to `width` channels, `layers` Fourier
 * layers $\sigma(\mathcal{K}\vvec + \Wmat\vvec + \bvec)$, and a pointwise two-layer projection to the output.
 * Parameters are drawn from children of the stream (`lift`, `spectral` and `pointwise` with the layer's index,
 * `project`). Differentiable in its parameters.
 *
 * @param options The channels, width, modes, number of layers, projection width and activation.
 * @returns The layer; `apply(p, a)` maps $[\text{batch}, n, c_{\text{in}}]$ sampled at $x_j = j/n$ to
 *   $[\text{batch}, n, c_{\text{out}}]$, for any $n \ge 2(\text{modes} - 1)$.
 *
 * @example One network at two resolutions
 * const fno = FourierNeuralOperator1d({ width: 4, modes: 4, layers: 2 })
 * const p = fno.init(stream(0))
 * const at = (n) => tensor([Array.from({ length: n }, (_, j) => [Math.sin((2 * Math.PI * j) / n)])])
 * print(fno.label, shapeOf(fno.apply(p, at(16))), shapeOf(fno.apply(p, at(32))))
 */
export function FourierNeuralOperator1d(options: FnoOptions = {}): Layer<FnoParams> {
  const { inChannels = 1, outChannels = 1, width = 16, modes = 12, layers = 4, activation = 'gelu' } = options
  const projection = options.projection ?? 4 * width
  const sigma = activationFunctions[activation]
  const lift = Linear(inChannels + 1, width)
  const blocks = Array.from({ length: layers }, () => ({
    spectral: SpectralConv1d(width, width, modes),
    pointwise: Linear(width, width),
  }))
  const project = Mlp([width, projection, outChannels], { activation })
  return {
    kind: 'FourierNeuralOperator1d',
    label: `FNO1d(width ${width}, ${modes} modes, ${layers} layers)`,
    init(s: Stream) {
      return {
        lift: lift.init(child(s, 'lift')),
        blocks: blocks.map((b, i) => ({
          spectral: b.spectral.init(child(s, 'spectral', i)),
          pointwise: b.pointwise.init(child(s, 'pointwise', i)),
        })),
        project: project.init(child(s, 'project')) as Params[],
      }
    },
    apply(p: FnoParams, a: Value) {
      const [batch, n] = shapeOfValue(a)
      let v = lift.apply(p.lift, concat([a, gridChannel(batch, n)], -1))
      p.blocks.forEach((b, i) => {
        const z = add(blocks[i].spectral.apply(b.spectral, v), blocks[i].pointwise.apply(b.pointwise, v))
        v = i < layers - 1 ? sigma(z) : z
      })
      return project.apply(p.project, v)
    },
  }
}
