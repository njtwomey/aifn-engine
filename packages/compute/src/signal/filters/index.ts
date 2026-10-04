/**
 * `aifn-compute/signal/filters`: digital filters as `LtiSystem`s: FIR design by windowing (`firwin`, with Kaiser estimates),
 * IIR design (`butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter`), equiripple FIR design (`remez`), filtering as compositions over `linearFilter` (`lfilter`,
 * `sosfilt`, `lfilterZi`, `sosfiltZi`, `filtfilt`; differentiable in the coefficients), frequency response (`freqz`,
 * a complex128 `Spectrum`), group delay and phase unwrapping; smoothing and detection (`savgolFilter`, `medfilt`,
 * `wiener`, `wienerDenoise`, `matchedFilter`). `filterDesignRegistry` lists the design methods.
 */

export {
  bessel,
  butter,
  cheby1,
  cheby2,
  ellip,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  iirfilter,
  kaiserAttenuation,
  kaiserBeta,
  kaiserOrder,
  lfilter,
  lfilterZi,
  sosfilt,
  sosfiltZi,
  unwrapPhase,
  type FilterCoefficients,
  type FilterOptions,
  type FilterSpec,
  type Filtered,
  type FiltfiltOptions,
  type FirwinOptions,
  type GroupDelay,
  type IirOptions,
  type ResponseOptions,
} from './filters'
export { equiripple, remez, type EquirippleOptions, type RemezOptions, type RemezResult } from './remez'
export {
  matchedFilter,
  medfilt,
  savgolCoeffs,
  savgolFilter,
  wiener,
  wienerDenoise,
  type Matched,
  type SavgolOptions,
  type WienerDenoised,
} from './smoothing'
export { filterDesignRegistry, filtersFunctions } from './registry'
