/**
 * `aifn-compute/signal/filters`: digital filter design and filtering, as scipy.signal, on `LtiSystem`s.
 *
 * - FIR design: `firwin` (the window method), with Kaiser's estimates `kaiserOrder`, `kaiserBeta` and
 *   `kaiserAttenuation`; `remez` (Parks–McClellan, minimax over arbitrary bands) and `equiripple` (the same, specified
 *   like `firwin` by cutoff and transition width).
 * - IIR design by the bilinear transform: `butter`, `cheby1`, `cheby2`, `ellip` and `bessel`, or `iirfilter` with the
 *   family as an option. They return zeros, poles and gain unless `output` asks for `tf` or `sos`.
 * - Filtering: `lfilter` (the difference equation) and `sosfilt` (second-order sections, better conditioned), with
 *   their steady-state initial states `lfilterZi` and `sosfiltZi`, and `filtfilt` (zero phase, forwards and
 *   backwards). They are compositions over `linearFilter`, so differentiable in the coefficients, the state and the
 *   signal.
 * - Responses: `freqz` (the complex frequency response, a `Spectrum`), `groupDelay` and `unwrapPhase`.
 * - Smoothing and detection on concrete samples: `savgolFilter` (with its weights `savgolCoeffs`), `medfilt`, `wiener`
 *   (local adaptive), `wienerDenoise` (frequency-domain shrinkage) and `matchedFilter`.
 *
 * Without `fs`, frequencies are fractions of the Nyquist frequency in $(0, 1)$, as in scipy (`remez` alone takes
 * cycles per sample, up to 0.5); with `fs`, they are in its units, and the designed system's `dt` is $1/f_s$. A
 * `Signal` input gives a `Signal` output on the same time axis. `filterDesignRegistry` lists the design methods and
 * `filtersFunctions` the rest.
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
