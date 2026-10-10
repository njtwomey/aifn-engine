/**
 * `aifn-methods/data/signals`: test signals, and seeded series whose spectrum or structure is known.
 *
 * - Deterministic signals, for filters and transforms: `uniformTimes`, `chirp` (a linear, quadratic or logarithmic
 *   frequency sweep, as scipy's) and `tones` (a sum of sinusoids); `testFunction` (the Donoho–Johnstone blocks, bumps,
 *   HeaviSine and Doppler, standardised) for denoising; `voicedSound` (a glottal pulse train through formants,
 *   returned with its true $f_0$) for pitch estimation.
 * - Series with a known power spectrum, for spectral estimation, each a `SignalDataset` whose `meta.truth` is a
 *   `SpectralTruth`: `sinusoidsInNoise` (lines on a flat floor), `arProcess` (a peak), `armaProcess` (a peak and a
 *   notch), `unevenSinusoids` (uneven times with per-sample noise, for Lomb–Scargle) and `coupledProcesses` (a pair
 *   with known cross-spectrum and coherence).
 * - A synthetic electrocardiogram for sparse coding and dictionary learning: `syntheticEcg` (beats of a few shapes,
 *   some ectopic, on a baseline wander with noise, with every beat's start and kind) and `syntheticBeat` (one beat).
 *
 * The seeded generators take a stream first and are registered as datasets (kind `dataset`), so recipes can replay
 * them; the deterministic functions are registered as functions, listed in `signalGeneratorFunctions` and
 * `testSignalFunctions`. Frequencies are in cycles per unit of time: cycles per sample at the default rate of 1.
 */

export { chirp, uniformTimes, tones } from './signals'
export { signalGeneratorFunctions } from './registry'
export {
  testFunction,
  testSignalFunctions,
  voicedSound,
  type Formant,
  type TestFunctionName,
  type VoicedOptions,
} from './test-signals'
export {
  armaProcess,
  arProcess,
  coupledProcesses,
  sinusoidsInNoise,
  unevenSinusoids,
  type ArmaProcessOptions,
  type ArProcessOptions,
  type CoupledProcessesOptions,
  type SignalDataset,
  type SinusoidsInNoiseOptions,
  type UnevenSampling,
  type UnevenSinusoidsOptions,
} from './processes'
export { syntheticBeat, syntheticEcg, type SyntheticEcg, type SyntheticEcgOptions } from './heartbeat'
