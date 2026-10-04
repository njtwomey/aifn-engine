/**
 * `aifn-methods/data/signals`: test signals: chirps and tones; and registered generators of series with a known power
 * spectrum (`SpectralTruth`): sinusoids in noise, AR and ARMA processes, unevenly sampled sinusoids and a coupled
 * pair with known coherence; deterministic test signals: the Donoho–Johnstone test functions and a synthetic voiced
 * sound with its true f₀.
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
