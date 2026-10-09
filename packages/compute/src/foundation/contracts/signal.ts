/**
 * The signal-processing objects (design S §2.13–2.14): signals with a time axis, spectra, time–frequency rasters,
 * linear time-invariant systems in four representations, and additive decompositions. `aifn-compute/signal` builds
 * the signals, spectra and rasters, `aifn-compute/systems` the LTI systems (and their frequency responses as spectra),
 * and the EMD and VMD of `aifn-compute/signal/decompositions` return decompositions.
 */

import type { Kinded } from './kinds'
import type { Scalar, Size, Tensor } from './numbers'

/**
 * Samples with a time axis. Every dsp function takes `Signal | Tensor` (a bare tensor means `fs = 1`) and returns a
 * `Signal` wherever the output lives in time, so axes and units carry through a chain.
 */
export interface Signal extends Kinded<'signal'> {
  /** `[n]` or `[channels, n]`; complex allowed (analytic signals, baseband). */
  readonly data: Tensor
  /** Samples per second (1 for "samples"). */
  readonly fs: Scalar
  /** Time of sample 0, in seconds. */
  readonly t0: Scalar
  /** Unit of the values, e.g. `V`, `Pa`, `a.u.`. */
  readonly unit?: string
  /** A name per channel, for a `[channels, n]` signal. */
  readonly channels?: readonly string[]
}

/**
 * One frequency-domain type for periodograms, Welch, multitaper, DFT magnitudes, cross-spectra, coherence and
 * frequency responses. dB conversion reads `quantity`, so the power rule is never applied to an amplitude.
 */
export interface Spectrum extends Kinded<'spectrum'> {
  /** Hz for sampled data, rad/s for continuous systems (see `axis`). */
  readonly f: Tensor
  /** The unit of `f`: hertz, radians per second, radians per sample or cycles per sample. */
  readonly axis: 'hz' | 'rad/s' | 'rad/sample' | 'cycles/sample'
  /** Real (power, amplitude, coherence) or complex (a DFT, a response $H(f)$). */
  readonly values: Tensor
  /**
   * What `values` holds: a power spectral density, a power, an amplitude, complex coefficients, a coherence or a
   * frequency response.
   */
  readonly quantity: 'psd' | 'power' | 'amplitude' | 'complex' | 'coherence' | 'response'
  /** One-sided spectra double every bin except DC and Nyquist. */
  readonly sided: 'one' | 'two'
  /** The sample rate of the data the spectrum came from, in hertz. */
  readonly fs?: Scalar
  /** e.g. `V²/Hz`. */
  readonly unit?: string
}

/** One raster type for STFT, spectrogram, CWT scalogram, constant-Q, mel, Wigner–Ville and Hilbert spectra. */
export interface TimeFrequency extends Kinded<'time-frequency'> {
  /** The time of each column, in seconds (in samples when the signal's `fs` is 1). */
  readonly t: Tensor
  /** Frequencies, or centre frequencies of scales or bands. */
  readonly f: Tensor
  /** `[f, t]`: one row per frequency, one column per time; real or complex. */
  readonly values: Tensor
  /** `distribution` may be negative (Wigner–Ville). */
  readonly quantity: 'power' | 'amplitude' | 'complex' | 'distribution'
  /** The transform that made it. */
  readonly method: 'stft' | 'cwt' | 'cqt' | 'mel' | 'wvd' | 'hht' | 'synchrosqueezed' | 'reassigned'
  /** How the frequencies are spaced, for drawing the axis. */
  readonly frequencyScale: 'linear' | 'log' | 'mel' | 'bark' | 'erb'
  /** For resolution and uncertainty readouts. */
  readonly window?: { readonly name: string; readonly length: Size; readonly hop: Size }
}

/**
 * A representation of an LTI system. `tf`: continuous systems in descending powers of $s$, discrete in ascending
 * powers of $z^{-1}$. `zpk`: complex zeros and poles (complex128 tensors) and a gain. `sos`: second-order sections
 * `[k, 6]`, each row $b_0, b_1, b_2, a_0, a_1, a_2$. `ss`: state-space matrices.
 */
export type Representation =
  /** A transfer function: numerator coefficients `b` over denominator coefficients `a`. */
  | { readonly form: 'tf'; readonly b: Tensor; readonly a: Tensor }
  /** Zeros and poles (complex128 tensors) and the gain that scales their ratio of products. */
  | { readonly form: 'zpk'; readonly zeros: Tensor; readonly poles: Tensor; readonly gain: Scalar }
  /** A cascade of second-order sections, one row of `sections` each. */
  | { readonly form: 'sos'; readonly sections: Tensor }
  /**
   * State space: $\dot{\xvec} = \Amat\xvec + \Bmat\uvec$ (the next state, when discrete) and
   * $\yvec = \Cmat\xvec + \Dmat\uvec$.
   */
  | { readonly form: 'ss'; readonly A: Tensor; readonly B: Tensor; readonly C: Tensor; readonly D: Tensor }

/**
 * A linear time-invariant system (after `scipy.signal`'s `lti`/`dlti`), shared by dsp filters and control. It keeps
 * the form it was built in; the other forms are converted on demand, exactly.
 */
export interface LtiSystem extends Kinded<'lti'> {
  /** Whether the system acts in continuous or discrete time. */
  readonly domain: 'continuous' | 'discrete'
  /** Sampling interval of a discrete system; null for a continuous one. */
  readonly dt: Scalar | null
  /** Input delay (continuous: seconds; discrete: samples). */
  readonly delay: Scalar
  /** The representation it was built in. */
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
  /** The parts, in order. */
  readonly components: readonly {
    /** The part's label, e.g. `imf 1`. */
    readonly name: string
    /** The part's values along `axis`. */
    readonly values: Tensor
    /** The frequency band the part covers, low and high, where it has one. */
    readonly band?: readonly [number, number]
    /** Anything else the method reports about the part. */
    readonly meta?: object
  }[]
  /** What the components leave over: the original minus their sum. */
  readonly residual?: Tensor
  /** The signal or function decomposed. */
  readonly original?: Tensor
}
