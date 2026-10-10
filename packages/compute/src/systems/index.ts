/**
 * `aifn-compute/systems`: the one linear time-invariant system type, `LtiSystem` (tf, zpk, ss, sos; continuous or
 * discrete), shared by `aifn-compute/signal`'s filters and `aifn-compute/dynamics`' control.
 *
 * - Construction: `transferFunction`, `zerosPolesGain`, `stateSpace`, `secondOrderSections`.
 * - Conversion (exact): `toTransferFunction`, `toZerosPolesGain`, `toStateSpace`, `toSecondOrderSections`, `convert`.
 * - Analysis: `poles`, `systemZeros`, `stability`, `dimensions`; `controllability`, `observability` (Kalman rank
 *   tests), `controllabilityGramian`, `observabilityGramian`.
 * - Responses: `frequencyResponse` (a `Spectrum`), `responseAt`, `frequencyGrid`, `bode`, `margins`; `simulate` (a
 *   traceable algorithm; exact between samples for held inputs), `respond`, `stepResponse`, `impulseResponse`,
 *   `initialResponse`.
 * - New systems: `discretise` (zoh, Euler, Tustin; its $\bar\Amat$ and $\bar\Bmat$ by the differentiable
 *   `discretiseSsm`, which `aifn-compute/nn/sequence` uses too), `stateFeedback`, `series`, `parallel`, `feedback`.
 * - Feedback design: `placePoles` (multi-input robust pole placement, Kautsky–Nichols–Van Dooren).
 * - Stability criteria: `routhArray` (Routh–Hurwitz, with the $\varepsilon$ and auxiliary-polynomial cases), `nyquist`
 *   (the plot with indentations and the encirclement count, $Z = N + P$), `rootLocus` (branches, asymptotes, breakaway
 *   points, stability crossings) and `closedLoopPolesAt`.
 * - Identification: `arx` (least squares) and `arxOrderSelection`, the prediction-error method for the polynomial
 *   family (`predictionErrorMethod`, a step algorithm; `polynomialModel`, `armax`, `outputError`) and `n4sid`
 *   (subspace identification of a state-space model).
 * - Registries: `systemsFunctions` and `systemsAlgorithms`, the module's functions and algorithms keyed by name.
 *
 * Zeros, poles and responses are complex128 tensors (`ComplexLike` inputs, from `aifn-compute/numerics/polynomial`).
 * A continuous transfer function is in descending powers of $s$, a discrete one in ascending powers of $z^{-1}$, as
 * scipy. Functions that need a state realise any representation in controllable canonical form first, and those that
 * take a SISO view read channel 0 to 0 of a MIMO state-space system unless told otherwise. Invalid arguments throw
 * `DomainError` or `ShapeError`.
 */

export type { LtiSystem, Representation } from 'aifn-compute/foundation/contracts'
export type { ComplexLike } from 'aifn-compute/numerics/polynomial'
export {
  convert,
  dimensions,
  poles,
  secondOrderSections,
  stability,
  stateSpace,
  toSecondOrderSections,
  toStateSpace,
  toTransferFunction,
  toZerosPolesGain,
  transferFunction,
  systemZeros,
  zerosPolesGain,
  type ChannelOptions,
  type LtiOf,
  type SecondOrderSectionsForm,
  type Stability,
  type StateSpaceForm,
  type StateSpaceInput,
  type SystemOptions,
  type TransferFunctionForm,
  type ZerosPolesGainForm,
} from './system'
export {
  controllability,
  controllabilityGramian,
  observability,
  observabilityGramian,
  type Gramian,
  type RankTest,
} from './structure'
export {
  bode,
  frequencyGrid,
  frequencyResponse,
  impulseResponse,
  initialResponse,
  margins,
  respond,
  responseAt,
  simulate,
  stepResponse,
  type Bode,
  type Input,
  type Margins,
  type RespondOptions,
  type Response,
  type SimulationOptions,
  type SimulationState,
  type StandardResponseOptions,
} from './responses'
export {
  discretise,
  discretiseSsm,
  feedback,
  parallel,
  series,
  stateFeedback,
  type DiscreteSsm,
  type DiscretisationMethod,
  type SsmDiscretisation,
} from './transform'
export { placePoles, type PlacePolesOptions, type PolePlacementResult } from './placement'
export { nyquist, routhArray, type Nyquist, type NyquistOptions, type RouthArray } from './criteria'
export {
  closedLoopPolesAt,
  rootLocus,
  type LocusCrossing,
  type LocusPoint,
  type RootLocus,
  type RootLocusOptions,
} from './locus'
export {
  armax,
  arx,
  arxOrderSelection,
  n4sid,
  outputError,
  polynomialModel,
  predictionErrorMethod,
  type N4sidOptions,
  type PolynomialModel,
  type PolynomialOrders,
  type PredictionErrorOptions,
  type PredictionErrorState,
  type SubspaceModel,
} from './identification'
export { systemsAlgorithms, systemsFunctions } from './registry'
