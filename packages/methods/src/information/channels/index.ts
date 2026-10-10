/**
 * `aifn-methods/information/channels`: channel capacity and rate–distortion of discrete distributions by
 * Blahut–Arimoto.
 *
 * - Channel capacity $C = \max_p I(X; Y)$ of a discrete memoryless channel: `blahutArimotoCapacity` (traceable, with
 *   the bounds $\text{lower} \le C \le \text{upper}$ at every step) and `channelCapacity` (run to convergence, in
 *   any base).
 * - The rate–distortion function $R(D)$: `blahutArimotoRateDistortion` (traceable, one point per slope $-\beta$),
 *   `rateDistortion` (that point, run to convergence) and `rateDistortionCurve` (a point per $\beta$).
 * - The registry: `channelsAlgorithms` and `channelsFunctions`.
 *
 * Conventions: a channel $W(y \mid x)$ or a distortion $d(x, \hat x)$ is a matrix with one row per input or source
 * symbol. Information is computed in nats; the solvers take a `base` (2 for bits). A channel with a negative entry or
 * a row that does not sum to 1, or a negative $\beta$, throws `DomainError`; a distortion of the wrong shape throws
 * `ShapeError`.
 */

export {
  blahutArimotoCapacity,
  blahutArimotoRateDistortion,
  channelCapacity,
  rateDistortion,
  rateDistortionCurve,
  type CapacityOptions,
  type CapacityStart,
  type CapacityState,
  type ChannelCapacity,
  type RateDistortionOptions,
  type RateDistortionPoint,
  type RateDistortionState,
} from './channel'
export { channelsAlgorithms, channelsFunctions } from './registry'
