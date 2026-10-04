/**
 * The signal-processing objects (design S §2.13–2.14): signals with a time axis, spectra, time–frequency rasters,
 * linear time-invariant systems in four representations, and additive decompositions. None exists in this form yet:
 * today `aifn-compute/signal` passes bare tensors and returns `{ f, psd }`, `IirFilter`, `Zpk`, `EmdResult` and
 * `WaveletDecomposition`, and `aifn-compute/systems` has `TransferFunction` and `StateSpace`. Phase 1 moves them here.
 */

import type { Kinded } from './kinds'
import type { Scalar, Size, Tensor } from './numbers'

/**
 * Samples with a time axis. Every dsp function takes `Signal | Tensor` (a bare tensor means `fs = 1`) and returns a
 * `Signal` wherever the output lives in time, so axes and units carry through a chain.
 */
export interface Signal extends Kinded<'signal'> {
  /** [n] or [channels, n]; complex allowed (analytic signals, baseband). */
  readonly data: Tensor
  /** Samples per second (1 for "samples"). */
  readonly fs: Scalar
  /** Time of sample 0, in seconds. */
  readonly t0: Scalar
  /** Unit of the values, e.g. `V`, `Pa`, `a.u.`. */
  readonly unit?: string
  readonly channels?: readonly string[]
}

/**
 * One frequency-domain type for periodograms, Welch, multitaper, DFT magnitudes, cross-spectra, coherence and
 * frequency responses. dB conversion reads `quantity`, so the power rule is never applied to an amplitude.
 */
export interface Spectrum extends Kinded<'spectrum'> {
  /** Hz for sampled data, rad/s for continuous systems (see `axis`). */
  readonly f: Tensor
  readonly axis: 'hz' | 'rad/s' | 'rad/sample' | 'cycles/sample'
  /** Real (power, amplitude, coherence) or complex (a DFT, a response H(f)). */
  readonly values: Tensor
  readonly quantity: 'psd' | 'power' | 'amplitude' | 'complex' | 'coherence' | 'response'
  /** One-sided spectra double every bin except DC and Nyquist. */
  readonly sided: 'one' | 'two'
  readonly fs?: Scalar
  /** e.g. `V²/Hz`. */
  readonly unit?: string
}

/** One raster type for STFT, spectrogram, CWT scalogram, constant-Q, mel, Wigner–Ville and Hilbert spectra. */
export interface TimeFrequency extends Kinded<'time-frequency'> {
  readonly t: Tensor
  /** Frequencies, or centre frequencies of scales or bands. */
  readonly f: Tensor
  /** [f, t], real or complex. */
  readonly values: Tensor
  /** `distribution` may be negative (Wigner–Ville). */
  readonly quantity: 'power' | 'amplitude' | 'complex' | 'distribution'
  readonly method: 'stft' | 'cwt' | 'cqt' | 'mel' | 'wvd' | 'hht' | 'synchrosqueezed' | 'reassigned'
  readonly frequencyScale: 'linear' | 'log' | 'mel' | 'bark' | 'erb'
  /** For resolution and uncertainty readouts. */
  readonly window?: { readonly name: string; readonly length: Size; readonly hop: Size }
}

/**
 * A representation of an LTI system. `tf`: continuous systems in descending powers of s, discrete in ascending powers
 * of z⁻¹. `zpk`: complex zeros and poles (complex128 tensors) and a gain. `sos`: second-order sections [k, 6] as
 * b0 b1 b2 a0 a1 a2. `ss`: state-space matrices.
 */
export type Representation =
  | { readonly form: 'tf'; readonly b: Tensor; readonly a: Tensor }
  | { readonly form: 'zpk'; readonly zeros: Tensor; readonly poles: Tensor; readonly gain: Scalar }
  | { readonly form: 'sos'; readonly sections: Tensor }
  | { readonly form: 'ss'; readonly A: Tensor; readonly B: Tensor; readonly C: Tensor; readonly D: Tensor }

/**
 * A linear time-invariant system (after `scipy.signal`'s `lti`/`dlti`), shared by dsp filters and control. It keeps
 * the form it was built in; the other forms are converted on demand, exactly.
 */
export interface LtiSystem extends Kinded<'lti'> {
  readonly domain: 'continuous' | 'discrete'
  /** Sampling interval of a discrete system; null for a continuous one. */
  readonly dt: Scalar | null
  /** Input delay (continuous: seconds; discrete: samples). */
  readonly delay: Scalar
  readonly repr: Representation
}

/**
 * An additive split of a signal or function into named parts: EMD modes, wavelet multiresolution levels, STL
 * components, filter-bank subbands, a GAM's partial effects. Their sum (plus `residual`) reconstructs `original`.
 */
export interface Decomposition extends Kinded<'decomposition'> {
  /** `emd`, `dwt-mra`, `stl`, `gam`, `filter-bank`, … */
  readonly method: string
  /** Time or x, shared by every component. */
  readonly axis: Tensor
  readonly components: readonly {
    readonly name: string
    readonly values: Tensor
    readonly band?: readonly [number, number]
    readonly meta?: object
  }[]
  readonly residual?: Tensor
  readonly original?: Tensor
}
