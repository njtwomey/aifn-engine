/**
 * `aifn-methods/information/channels`: channel capacity and rate–distortion by Blahut–Arimoto.
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
