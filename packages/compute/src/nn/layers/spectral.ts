/**
 * The spectral convolution of a Fourier layer (Li et al., 2021, "Fourier Neural Operator for Parametric Partial
 * Differential Equations", §4): for an input $[\text{batch}, n, c_{\text{in}}]$ sampled on a uniform periodic grid,
 * take the real FFT along the grid axis, keep the lowest `modes` frequencies, multiply the channel vector at each kept
 * frequency $k$ by a learned complex matrix $\Rmat_k$ ($c_{\text{in}} \times c_{\text{out}}$), set the other
 * frequencies to zero and transform back to $n$ points:
 *
 * $$\vvec \mapsto \mathcal{F}^{-1}\big(\Rmat_k \, (\mathcal{F}\vvec)_k\big)_{k < \text{modes}}.$$
 *
 * The weights are indexed by frequency, not by grid point, so the same parameters apply at any
 * $n \ge 2(\text{modes} - 1)$; with NumPy's conventions (`rfft` unscaled, `irfft` divided by $n$) a band-limited input
 * gives the same function at every resolution. The complex weights are stored as two real tensors (`real`, `imag`),
 * so any real optimiser trains them, and the layer differentiates through `rfft`, `einsum` and `irfft` by the
 * complex rules of autodiff. The Fourier neural operator built from it is `FourierNeuralOperator1d` of
 * `aifn-methods/neural/operator`.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { irfft, rfft } from 'aifn-compute/foundation/fourier'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { complex, einsum, shapeOfValue, slice, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Layer } from './layers'

/** Parameters of `SpectralConv1d`: the real and imaginary parts of the weights $\Rmat_k$. */
export type SpectralConvParams = {
  /** $\operatorname{Re}\Rmat_k$ for each kept frequency, $[\text{modes}, c_{\text{in}}, c_{\text{out}}]$. */
  real: Tensor
  /** $\operatorname{Im}\Rmat_k$, of the same shape. */
  imag: Tensor
}

/**
 * The spectral convolution of a Fourier layer: `rfft` along axis 1, the lowest `modes` frequencies multiplied by
 * learned complex channel matrices, the rest set to zero, `irfft` back to the input's length. The weights are drawn
 * uniformly from $[0, 1/(c_{\text{in}} c_{\text{out}}))$, as the reference implementation does (real and imaginary
 * parts from the child streams `real` and `imag`). `apply` throws `ShapeError` when the input has too few points for
 * the modes.
 *
 * @param inChannels The input channels $c_{\text{in}}$, the last axis of $[\text{batch}, n, c_{\text{in}}]$.
 * @param outChannels The output channels $c_{\text{out}}$.
 * @param modes The frequencies kept, $k = 0, \dots, \text{modes} - 1$; at most $\lfloor n/2 \rfloor + 1$ for an
 *   input of $n$ points.
 * @returns The layer; `apply(p, v)` maps $[\text{batch}, n, c_{\text{in}}]$ to $[\text{batch}, n, c_{\text{out}}]$.
 *
 * @example The same weights give the same band-limited function at 16 and 64 points
 * const layer = SpectralConv1d(1, 1, 4)
 * const p = layer.init(stream(0))
 * const at = (n) => tensor([Array.from({ length: n }, (_, j) => [Math.sin((2 * Math.PI * j) / n)])])
 * print('16 points, every fourth:', slice(layer.apply(p, at(16)), 0, [0, 16, 4], 0))
 * print('64 points, every sixteenth:', slice(layer.apply(p, at(64)), 0, [0, 64, 16], 0))
 */
export function SpectralConv1d(inChannels: Size, outChannels: Size, modes: Size): Layer<SpectralConvParams> {
  const scale = 1 / (inChannels * outChannels)
  return {
    kind: 'SpectralConv1d',
    label: `SpectralConv1d(${inChannels} → ${outChannels}, ${modes} modes)`,
    init(s: Stream) {
      const draw = (name: string) =>
        uniform(child(s, name), 0, scale, { shape: [modes, inChannels, outChannels] }) as unknown as Tensor
      return { real: draw('real'), imag: draw('imag') }
    },
    apply(p: SpectralConvParams, v: Value) {
      const n = shapeOfValue(v)[1]
      if (modes > Math.floor(n / 2) + 1)
        throw new ShapeError(
          'SpectralConv1d',
          `SpectralConv1d: ${modes} modes need at least ${2 * (modes - 1)} points (got ${n})`,
        )
      const spectrum = slice(rfft(v, { axis: 1 }), null, [0, modes]) // [batch, modes, in]
      const mixed = einsum('bki,kio->bko', spectrum, complex(p.real, p.imag)) // [batch, modes, out]
      return irfft(mixed, { axis: 1, n }) // zero-padded to n/2 + 1 frequencies, then n points
    },
  }
}
