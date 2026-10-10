/**
 * `aifn-compute/signal/time-frequency`: the analytic signal, the constant-Q transform, Wigner–Ville distributions and
 * sharpened spectrograms.
 *
 * - Analytic signal: `hilbert` (as `scipy.signal.hilbert`), `instantaneous` amplitude, phase and frequency,
 *   `envelope`, and `hilbertSpectrum`, the Hilbert spectrum of intrinsic mode functions.
 * - Constant-Q: `cqt`, bins geometrically spaced from $f_{\min}$, each window spanning $Q$ cycles of its frequency.
 * - Quadratic distributions of Cohen's class: `wignerVille` (sharpest, with cross-terms), `pseudoWignerVille` (a lag
 *   window smooths along frequency) and `smoothedPseudoWignerVille` (a time window too, removing most cross-terms).
 * - Sharpened spectrograms: `reassignedSpectrogram` (power moved in time and frequency) and `synchrosqueeze` (STFT
 *   coefficients moved in frequency only).
 * - `timeFrequencyFunctions` lists the functions with the notes they serve.
 *
 * Every function reads a single-channel `Signal` or bare samples, takes its sample rate from the signal or an `fs`
 * option (frequencies in Hz, or cycles per sample at the default of 1), and returns a `TimeFrequency` raster
 * $[f, t]$ or tensors of the signal's length.
 */

export { envelope, hilbert, hilbertSpectrum, instantaneous, type HilbertSpectrum, type Instantaneous } from './hilbert'
export { cqt, type CqtOptions } from './cqt'
export {
  pseudoWignerVille,
  smoothedPseudoWignerVille,
  wignerVille,
  type WignerOptions,
  type WignerVille,
} from './wigner'
export { reassignedSpectrogram, synchrosqueeze, type ReassignOptions, type Sharpened } from './reassign'
export { timeFrequencyFunctions } from './registry'
